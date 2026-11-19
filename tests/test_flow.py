# -*- coding: utf-8 -*-
"""The simple flow: source folder -> package names -> prepare -> review -> Finish (skipped when nothing changed).
   .venv\\Scripts\\python -m unittest discover -s tests -v"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from agent import config, server  # noqa: E402
from agent import clips as clipsmod  # noqa: E402
import package as pkg  # noqa: E402  (pipeline)
import run_pipeline as rp  # noqa: E402
from test_agent import make_video  # noqa: E402


class TestPackageNames(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(pkg.title_slug("The Hidden Blade!"), "the_hidden_blade")
        self.assertEqual(pkg.title_slug("  隐剑  "), "")

    def test_numbering_continues_and_keeps_existing(self):
        names = {"a": "the_hidden_blade-1", "b": "the_hidden_blade-7", "x": "other-3"}
        plan = pkg.propose_names(["a", "c", "d"], "The hidden blade", names)
        self.assertEqual(plan["a"], ("the_hidden_blade-1", False))
        self.assertEqual(plan["c"], ("the_hidden_blade-8", True))
        self.assertEqual(plan["d"], ("the_hidden_blade-9", True))
        self.assertTrue(all(pkg.PKG_RE.match(v[0]) for v in plan.values()))

    def test_no_title_no_name(self):
        self.assertEqual(pkg.propose_names(["c"], "", {}), {"c": (None, False)})

    def test_assign_writes_only_new(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "names.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"a": "film-1"}, fh)
            with mock.patch.object(pkg, "NAMES_P", p):
                new = pkg.assign_names(["a", "b"], "film")
                self.assertEqual(new, {"b": "film-2"})
                self.assertEqual(pkg.package_names(), {"a": "film-1", "b": "film-2"})
                self.assertEqual(pkg.assign_names(["a", "b"], "film"), {})      # never renames
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestMarksFolder(unittest.TestCase):
    def test_default_source_unchanged(self):
        c = {"source_root": os.path.join(rp.AGENT, "SOURCE")}
        self.assertEqual(rp.marks_base(c), os.path.join(rp.AGENT, "output"))
        self.assertEqual(rp.marks_dir(os.path.join(rp.AGENT, "SOURCE", "x.mp4"), c), os.path.join(rp.AGENT, "output", "x"))

    def test_other_folders_do_not_share_joints(self):
        a = {"source_root": r"D:\shoot\day1"}; b = {"source_root": r"E:\copy\day1"}
        da, db = rp.marks_dir(r"D:\shoot\day1\x.mp4", a), rp.marks_dir(r"E:\copy\day1\x.mp4", b)
        self.assertNotEqual(da, db)
        self.assertIn(os.path.join("output", "_sources", "day1_"), da)


class TestClipIds(unittest.TestCase):
    def test_same_file_name_other_folder_other_id(self):
        d = tempfile.mkdtemp()
        try:
            for sub in ("a", "b"):
                os.makedirs(os.path.join(d, sub)); make_video(os.path.join(d, sub, "x.mp4"), n=5)
            ids = [clipsmod.Library(dict(config.load(), source_root=os.path.join(d, sub), work_root=os.path.join(d, "w"))).clips[0].id
                   for sub in ("a", "b")]
            self.assertNotEqual(ids[0], ids[1])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestRemoveFromList(unittest.TestCase):
    def test_remove_and_put_back_keeps_the_file_and_the_names(self):
        d = tempfile.mkdtemp(); data = config.DATA
        try:
            config.DATA = os.path.join(d, "data")
            src = os.path.join(d, "src", "FilmA"); os.makedirs(src)
            for n in ("a.mp4", "b.mp4"):
                make_video(os.path.join(src, n), n=5)
            lib = clipsmod.Library(dict(config.load(), source_root=os.path.join(d, "src"), work_root=os.path.join(d, "w")))
            names = sorted(cl.name for cl in lib.clips)
            a = next(cl for cl in lib.clips if cl.video.endswith("a.mp4"))
            lib.remove(a)
            self.assertEqual([cl.video for cl in lib.clips], [os.path.join(src, "b.mp4")])
            self.assertTrue(os.path.exists(a.video), "the file stays where it is")
            self.assertEqual([r["path"] for r in lib.removed], [a.video])
            lib.restore(a.video)
            self.assertEqual(sorted(cl.name for cl in lib.clips), names)
            self.assertEqual(lib.removed, [])
        finally:
            config.DATA = data
            shutil.rmtree(d, ignore_errors=True)


class TestFinishFingerprint(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        src = os.path.join(self.d, "src", "FilmA"); os.makedirs(src)
        make_video(os.path.join(src, "1-1.mp4"), n=10)
        self.c = dict(config.load(), source_root=os.path.join(self.d, "src"), work_root=os.path.join(self.d, "w"),
                      submit_root=os.path.join(self.d, "s"), marks_root=os.path.join(self.d, "m"))
        self.cl = clipsmod.Library(self.c).clips[0]
        self.f = clipsmod.folders(self.c, self.cl)
        os.makedirs(self.f["marks"]); os.makedirs(self.f["work"]); os.makedirs(self.c["submit_root"])
        self.write(self.f["review"], {"confirmed": True, "people": {"1": {"role": "main"}}, "saved": "t1"})

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    @staticmethod
    def write(p, d):
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(d, fh)

    def test_skip_until_something_changes(self):
        c, cl, f = self.c, self.cl, self.f
        fp = clipsmod.fingerprint(c, cl)
        self.write(f["review"], {"confirmed": False, "people": {"1": {"role": "main"}}, "saved": "t2"})
        self.assertEqual(clipsmod.fingerprint(c, cl), fp, "save time and confirm flag are not changes")
        open(f["zip"], "w").close()
        clipsmod.record_finish(c, cl, fp)
        self.write(f["review"], {"confirmed": True, "people": {"1": {"role": "main"}}, "saved": "t3"})
        self.assertTrue(clipsmod.status(c, cl)["done"])
        self.assertEqual(clipsmod.status(c, cl)["label"], "finished")
        self.write(os.path.join(f["marks"], "checked.json"), {"frames": {}, "saved": "t5"})  # the editor saved ...
        clipsmod.record_finish(c, cl, clipsmod.fingerprint(c, cl))                             # ... and was finished
        self.write(os.path.join(f["marks"], "checked.json"), {"frames": {}, "saved": "t6"})  # saved again, nothing new
        self.assertTrue(clipsmod.status(c, cl)["done"], "a save without a new fix is not a change")
        self.write(os.path.join(f["marks"], "checked.json"), {"frames": {"0": []}})     # a joint fix
        st = clipsmod.status(c, cl)
        self.assertFalse(st["done"]); self.assertTrue(st["changed"])
        self.assertEqual(st["label"], "changed since Finish")
        os.remove(os.path.join(f["marks"], "checked.json"))
        self.write(f["review"], {"confirmed": True, "people": {"1": {"role": "ignore"}}, "saved": "t4"})   # a role
        self.assertFalse(clipsmod.status(c, cl)["done"])


class TestProgressAndFlags(unittest.TestCase):
    def test_progress_from_log(self):
        d = tempfile.mkdtemp()
        try:
            lp = os.path.join(d, "job_fit.log")
            cl = mock.Mock(); cl.name = "x"
            with mock.patch.object(server, "folders", return_value={"work": d}):
                with open(lp, "w", encoding="utf-8") as fh:
                    fh.write("  detected 40 frames (frame 79/137, people 2)  (139.6s)\n")
                p = server.job_progress({}, cl, {"stage": "fit", "log": lp})
                self.assertTrue(10 < p["pct"] < 20, p); self.assertIn("40 of 137", p["step"])
                with open(lp, "a", encoding="utf-8") as fh:
                    fh.write("  detected 86 frames (frame 40/137, people 2)  (170.2s)\n")
                p2 = server.job_progress({}, cl, {"stage": "fit", "log": lp})
                self.assertGreater(p2["pct"], p["pct"])                       # never steps back
                with open(lp, "a", encoding="utf-8") as fh:
                    fh.write("  marks: x\n  lens: 40.7 mm\n  [shot 1: frames 0-136]\n  camera ...\n")
                p = server.job_progress({}, cl, {"stage": "fit", "log": lp})
                self.assertEqual(p["pct"], 58); self.assertIn("camera", p["step"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_person_flags(self):
        rep = {"quality": {"people": [
            {"track": 3, "flags": ["pose never changes: a statue or a picture?", "jerky (12 sudden movements per 100 frames)"], "jump_frames": [5, 40]},
            {"track": 1, "flags": []}]}, "left_out": [{"track": 7, "why": "short and unclear (0.5 s)"}]}
        f = server.person_flags(rep)
        self.assertIn("statue", f["3"]["not_person"]); self.assertEqual(len(f["3"]["flags"]), 1)
        self.assertIsNone(f["1"]["not_person"])
        self.assertIn("short and unclear", f["7"]["not_person"])
        self.assertEqual(server.jump_frames(rep), {"3": [5, 40]})
        self.assertEqual(server.jump_frames(rep, {"9": 3}), {"3": [5, 40], "9": [5, 40]})   # joined track: old editor id too


class TestSubmissionsPage(unittest.TestCase):
    def test_v21_meta_and_package_name(self):
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, "clip_a", "meta"))
            with open(os.path.join(d, "clip_a", "meta", "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"duration": 4.5, "output_video": {"resolution": [1920, 1080]}}, fh)
            with mock.patch.object(pkg, "package_names", return_value={"clip_a": "film-3"}):
                row = server.submissions({"submit_root": d})[0]
            self.assertEqual((row["package"], row["duration"], row["resolution"]), ("film-3", 4.5, [1920, 1080]))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_download_name(self):
        self.assertEqual(server.download_name("the_hidden_blade-3.zip"),
                         "attachment; filename=\"the_hidden_blade-3.zip\"; filename*=UTF-8''the_hidden_blade-3.zip")
        self.assertIn("filename*=UTF-8''%E9%9A%90", server.download_name("隐剑.zip"))   # non-ASCII kept in the UTF-8 form

    def test_character_file_follows_style(self):
        self.assertEqual(server.char_file({"character_style": "master"}), "human_master.json")
        self.assertEqual(server.char_file({"character_style": "base21"}), "base_character_v21.json")
        self.assertTrue(os.path.exists(os.path.join(ROOT, "assets", server.char_file(config.load()))))


class TestEditorColours(unittest.TestCase):
    def test_role_colours_and_joined_tracks(self):
        rv = {"people": {"1": {"role": "main", "colour": "red"}, "2": {"role": "ignore", "colour": "purple"}}}
        out = server.editor_colours(rv, {"3": 1})
        self.assertEqual(out["1"], {"hex": "#ff4d4d", "role": "Main"})
        self.assertEqual(out["2"]["hex"], server.DRAW_HEX["grey"], "not a person: grey")
        self.assertEqual(out["3"], out["1"], "a joined track keeps the colour of the track it joined")
        self.assertEqual(server.editor_colours(None), {})


class TestCheckPage(unittest.TestCase):
    def test_current_template_used(self):
        d = tempfile.mkdtemp()
        try:
            c = {}; cl = mock.Mock()
            with open(os.path.join(d, "check.html"), "w", encoding="utf-8") as fh:
                fh.write('<html><head></head><script>\nconst PAGE = {"clip": "x", "frames": []};\nconst cv = 1;</script></html>')
            with mock.patch.object(server, "folders", return_value={"marks": d}):
                html = server.check_page(c, cl)
            self.assertIn('const PAGE = {"clip": "x", "frames": []};', html)
            self.assertIn("nextFlagged", html)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

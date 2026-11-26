# -*- coding: utf-8 -*-
"""The simple flow: source folder -> package names -> one automatic Movement run (skipped while nothing changed).
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
        self.assertEqual(pkg.title_slug("  \u9690\u5251  "), "")

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


class TestMemberNames(unittest.TestCase):
    def test_members_keep_their_own_names_and_never_share_a_number(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "names.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"x": "film-1"}, fh)                                  # from before members: stays
            anna, ben = {"_user": "anna"}, {"_user": "ben"}
            with mock.patch.object(pkg, "NAMES_P", p):
                self.assertEqual(pkg.assign_names(["a", "b"], "film", anna), {"a": "film-2", "b": "film-3"})
                self.assertEqual(pkg.assign_names(["a"], "film", ben), {"a": "film-4"}, "Ben's clip a is not Anna's")
                self.assertEqual(pkg.package_names(anna), {"a": "film-2", "b": "film-3"})
                self.assertEqual(pkg.package_name("a", ben), "film-4")
                self.assertEqual(pkg.package_names(), {"x": "film-1"})
                with mock.patch.dict(os.environ, {"BA_USER": "ben"}):          # inside a job run for Ben
                    self.assertEqual(pkg.package_name("a"), "film-4")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_member_folders(self):
        import common
        c = common.member_cfg({"work_root": r"D:\w", "submit_root": r"D:\s", "source_root": r"D:\SOURCE"}, "anna", r"E:\anna")
        self.assertEqual((c["source_root"], c["work_root"], c["submit_root"]), (r"E:\anna", r"D:\w\_users\anna", r"D:\s\_users\anna"))
        self.assertEqual(c["marks_root"], os.path.join(rp.AGENT, "output", "_users", "anna"))
        self.assertEqual(common.output_dir(c), os.path.join(rp.AGENT, "output", "_users", "anna"))
        with self.assertRaises(ValueError):
            common.member_cfg(c, "ben", "")


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


class TestTaskZips(unittest.TestCase):
    """the client's task zips (v2.1 layout, the source at target/original.mp4): the video is written next to the zip
    under the zip's name, listed as a clip, and the zip name is its delivery name"""
    def setUp(self):
        import zipfile
        self.d = tempfile.mkdtemp(); self.data = config.DATA
        config.DATA = os.path.join(self.d, "data"); os.makedirs(config.DATA)
        self.src = os.path.join(self.d, "src"); os.makedirs(self.src)
        v = os.path.join(self.d, "v.mp4"); make_video(v, n=5)
        with open(v, "rb") as fh:
            self.vid = fh.read()
        self.zp = os.path.join(self.src, "p867bed92_t022-1.zip")
        task = {"taskId": "p-1-T022", "taskCode": "T022", "taskName": "11\u670821\u65e5 (3)(21).mp4", "projectTitle": "\u4fa0\u63a2\u6770\u514b",
                "sourceVideo": {"name": "11\u670821\u65e5 (3)(21).mp4", "durationSec": 5.08}}
        with zipfile.ZipFile(self.zp, "w") as z:
            for n in ("comparison/", "target/", "white_model/", "model_3d/", "meta/"):
                z.writestr(n, b"")
            z.writestr("target/original.mp4", self.vid)
            z.writestr("model_3d/p867bed92_t022-1.blend", b"BLENDER")
            z.writestr("meta/task.json", json.dumps(task, ensure_ascii=False).encode("utf-8"))
        done = os.path.join(self.src, "done-1.zip")                      # a finished submission is not a task
        with zipfile.ZipFile(done, "w") as z:
            z.writestr("target/original.mp4", self.vid); z.writestr("white_model/blendervid.mp4", self.vid)
        self.names = os.path.join(self.d, "names.json")
        self.c = dict(config.load(), source_root=self.src, work_root=os.path.join(self.d, "w"), submit_root=os.path.join(self.d, "s"))

    def tearDown(self):
        config.DATA = self.data
        shutil.rmtree(self.d, ignore_errors=True)

    def test_video_comes_out_under_the_zip_name_with_its_delivery_name(self):
        with mock.patch.object(pkg, "NAMES_P", self.names):
            lib = clipsmod.Library(self.c)
            self.assertEqual([cl.rel for cl in lib.clips], ["p867bed92_t022-1.mp4"])
            out = os.path.join(self.src, "p867bed92_t022-1.mp4")
            with open(out, "rb") as fh:
                self.assertEqual(fh.read(), self.vid, "byte for byte")
            cl = lib.clips[0]
            self.assertEqual(cl.to_dict()["zip"], "p867bed92_t022-1.zip")
            self.assertEqual(cl.task["code"], "T022"); self.assertEqual(cl.task["project"], "\u4fa0\u63a2\u6770\u514b")
            self.assertEqual(pkg.package_name(cl.name, self.c), "p867bed92_t022-1")
            m = os.path.getmtime(out)
            lib.rescan()                                                    # not written again
            self.assertEqual(os.path.getmtime(out), m)
            self.assertEqual(len(lib.clips), 1)
            self.assertEqual(lib.zip_problems, [])
            self.assertFalse([f for f in os.listdir(self.src) if f.endswith(".part")])

    def test_a_video_of_that_name_that_is_not_from_the_zip_is_left_alone(self):
        out = os.path.join(self.src, "p867bed92_t022-1.mp4")
        with open(out, "wb") as fh:
            fh.write(b"mine")
        with mock.patch.object(pkg, "NAMES_P", self.names):
            lib = clipsmod.Library(self.c)
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(), b"mine")
        self.assertEqual(len(lib.zip_problems), 1)

    def test_preview_lists_without_writing(self):
        with mock.patch.object(pkg, "NAMES_P", self.names):
            lib = clipsmod.Library(self.c, preview=True)
            self.assertEqual([cl.rel for cl in lib.clips], ["p867bed92_t022-1.mp4"])
            self.assertFalse(os.path.exists(os.path.join(self.src, "p867bed92_t022-1.mp4")))
            self.assertEqual(pkg.package_names(self.c), {})
            scan = server.folder_scan(self.c, self.src)
            self.assertEqual([(r["file"], r["package"]) for r in scan["videos"]], [("p867bed92_t022-1.mp4", "p867bed92_t022-1")])
            self.assertEqual(scan["unnamed"], 0)


class TestColourCard(unittest.TestCase):
    def test_card_values(self):
        import colour_card as CC
        self.assertEqual(CC.CARD["red"], [214, 62, 61]); self.assertEqual(CC.CARD["bg"], [184, 184, 184])
        self.assertEqual(CC.CARD["near"], [223, 135, 47]); self.assertEqual(CC.CARD["special"], [122, 78, 194])

    def test_roles(self):
        """the role rules stay for set pieces (Stage 2); Stage 1's floor is always background grey"""
        import colour_card as CC
        from rig import JOINTS
        N = 3
        ch = {"height": 1.75, "style": "sample", "track_id": 1, "visible": [True] * N, "root": [[0.0, 5.0, 0.0, 0.0]] * N,
              "pelvis_z": [0.94] * N, "rot": {j: [[0, 0, 0]] * N for j in JOINTS}, "colour": "purple"}
        box = lambda c, s: [[c[0] + sx * s / 2, c[1] + sy * s / 2, c[2] + sz * s / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        mk = lambda obj, v, **kw: dict({"name": obj, "obj": obj, "v": v, "f": [[0, 1, 2]], "colour": "set", "layer": 2}, **kw)
        data = {"res": [1920, 1080], "lens": 35.0, "sensor": 36.0, "frames": N, "cam": [[0.0, 0.0, 1.2, 90.0, 0.0, 0.0]] * N,
                "chars": {"P1": ch}, "palette": {}, "props": [],
                "meshes": [mk("ground", box((0, 0, 0), 9), colour="ground", layer=1),
                           mk("wall", box((0, 9, 1.5), 3), cls="wall", kind="patch"),
                           mk("lamp", box((3, 7, 1), 0.4), kind="lamp"),
                           mk("table", box((0, 3.5, 0.9), 0.7), kind="table"),                 # between camera and person
                           mk("door", box((-4, 12, 1), 1), kind="door"),                       # far from him
                           mk("cup", box((0.45, 5.0, 0.95), 0.1), kind="box", what="cup"),     # by his hand
                           mk("special", box((5, 9, 0.5), 1), role="special")]}
        CC.apply(data, log=None)
        col = {m["obj"]: m["colour"] for m in data["meshes"]}
        self.assertEqual(col, {"ground": "bg", "wall": "bg", "lamp": "lamp", "table": "fg", "door": "far_i", "cup": "near",
                               "special": "special"})
        self.assertEqual(data["chars"]["P1"]["colour"], "cyan", "purple is the special colour: a person gets another")
        self.assertEqual(data["palette"]["fg"], [154, 107, 59])
        self.assertEqual(data["palette"]["bg"], [184, 184, 184])


class TestFingerprint(unittest.TestCase):
    """a Movement zip is up to date while the video, its delivery name and the Stage 1 settings are unchanged"""
    def setUp(self):
        self.d = tempfile.mkdtemp()
        src = os.path.join(self.d, "src", "FilmA"); os.makedirs(src)
        make_video(os.path.join(src, "1-1.mp4"), n=10)
        self.c = dict(config.load(), source_root=os.path.join(self.d, "src"), work_root=os.path.join(self.d, "w"),
                      submit_root=os.path.join(self.d, "s"), marks_root=os.path.join(self.d, "m"))
        self.cl = clipsmod.Library(self.c).clips[0]
        self.f = clipsmod.folders(self.c, self.cl)
        os.makedirs(self.f["marks"]); os.makedirs(self.f["work"]); os.makedirs(self.c["submit_root"])

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_done_until_a_stage1_setting_changes(self):
        c, cl, f = self.c, self.cl, self.f
        open(f["zip"], "w").close()
        clipsmod.record_finish(c, cl, clipsmod.fingerprint(c, cl))
        with open(os.path.join(f["marks"], "review.json"), "w", encoding="utf-8") as fh:   # an old review file: no input
            json.dump({"confirmed": True, "people": {"1": {"role": "ignore"}}}, fh)
        self.assertTrue(clipsmod.status(c, cl)["done"])
        c2 = dict(c, person_height=c["person_height"] + 0.1)
        self.assertFalse(clipsmod.status(c2, cl)["done"])


class TestJobResult(unittest.TestCase):
    """a job's result is the pipeline's exit code: 0 done, anything else failed (tried again once)"""
    def setUp(self):
        from agent import jobs
        self.d = tempfile.mkdtemp()
        src = os.path.join(self.d, "src", "FilmA"); os.makedirs(src)
        make_video(os.path.join(src, "1-1.mp4"), n=10)
        c = dict(config.load(), source_root=os.path.join(self.d, "src"), work_root=os.path.join(self.d, "w"),
                 submit_root=os.path.join(self.d, "s"), marks_root=os.path.join(self.d, "m"), max_attempts=2)
        self.store = jobs.JobStore(os.path.join(self.d, "j.db"))
        lib = clipsmod.Library(c)
        self.cl = lib.clips[0]
        app = mock.Mock(c=c, store=self.store)
        app.ws = lambda user: mock.Mock(user=user, c=c, lib=lib)          # the member's workspace
        self.r = jobs.Runner(app)
        self.r.after = lambda *a: None

    def tearDown(self):
        self.store.db.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def zip_maker(self):
        """python code that writes the clip's zip (a successful run leaves its package)"""
        zp = clipsmod.folders(self.r.app.c, self.cl)["zip"]
        return "import os; os.makedirs(os.path.dirname(%r), exist_ok=True); open(%r, 'w').close(); " % (zp, zp)

    def run_job(self, code, stage="movement"):
        make = self.zip_maker() if code == 0 else ""
        self.r._cmd = lambda job, cl: [sys.executable, "-c", make + "print('working'); raise SystemExit(%d)" % code]
        jid = self.store.enqueue(self.cl.id, stage, user="anna")
        self.r._run(self.store.take())
        return jid

    def test_job_runs_in_the_members_folders(self):
        """the job's program gets the member and their folder (BA_USER / BA_SOURCE)"""
        make = self.zip_maker()
        self.r._cmd = lambda job, cl: [sys.executable, "-c", make + "print('member=' + os.environ['BA_USER'] + '|' + os.environ['BA_SOURCE'])"]
        self.store.enqueue(self.cl.id, "movement", user="anna")
        self.r._run(self.store.take())
        with open(os.path.join(clipsmod.folders(self.r.app.c, self.cl)["work"], "job_movement.log"), encoding="utf-8") as fh:
            self.assertIn("member=anna|" + self.r.app.c["source_root"], fh.read())
        self.store.enqueue(self.cl.id, "movement")                            # no member: refused, not run as someone
        self.r._run(self.store.take())
        self.assertIn("no member", self.store.latest_for(self.cl.id)["error"])

    def test_exit_codes(self):
        self.run_job(0)
        self.assertEqual(self.store.latest_for(self.cl.id)["state"], "done")
        self.run_job(1)
        self.assertEqual(self.store.latest_for(self.cl.id)["state"], "queued", "a failure is tried once more")
        self.r._run(self.store.take())
        self.assertEqual(self.store.latest_for(self.cl.id)["state"], "failed")
        work = clipsmod.folders(self.r.app.c, self.cl)["work"]
        self.assertTrue(os.path.exists(os.path.join(work, "job_movement.previous.log")), "the first attempt's log is kept")

    def test_movement_command(self):
        """the contract with run_pipeline.py: one movement stage, the clip's name, no review gate"""
        cmd = self.r._cmd({"stage": "movement"}, self.cl)
        self.assertIn("run_pipeline.py", os.path.basename(cmd[1]))
        self.assertEqual(cmd[cmd.index("--stage") + 1], "movement")
        self.assertEqual(cmd[cmd.index("--name") + 1], self.cl.name)
        self.assertNotIn("--require-confirm", cmd)
        self.assertIn("--force", cmd)


class TestStartOver(unittest.TestCase):
    """Start over deletes what a run left for one clip, and nothing else"""
    def setUp(self):
        self.d = tempfile.mkdtemp()
        src = os.path.join(self.d, "src", "FilmA"); os.makedirs(src)
        make_video(os.path.join(src, "1-1.mp4"), n=10); make_video(os.path.join(src, "1-2.mp4"), n=10)
        self.c = dict(config.load(), source_root=os.path.join(self.d, "src"), work_root=os.path.join(self.d, "w"),
                      submit_root=os.path.join(self.d, "s"), marks_root=os.path.join(self.d, "m"))
        self.a, self.b = clipsmod.Library(self.c).clips

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def fill(self, cl):
        f = clipsmod.folders(self.c, cl)
        for p in (f["pose"], f["preview"], f["scene"], os.path.join(f["submit"], "meta", "meta.json"), f["zip"]):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write("{}")
        return f

    def test_reset_one_clip(self):
        fa, fb = self.fill(self.a), self.fill(self.b)
        self.assertTrue(clipsmod.status(self.c, self.a)["done"])
        self.assertEqual(len(clipsmod.prepared_paths(self.c, self.a)), 4)        # marks, work, submit folder, zip
        r = clipsmod.reset(self.c, self.a)
        self.assertEqual((len(r["removed"]), r["failed"]), (4, []))
        self.assertEqual(clipsmod.prepared_paths(self.c, self.a), [])
        self.assertFalse(clipsmod.status(self.c, self.a)["done"])
        self.assertTrue(os.path.exists(self.a.video), "the video stays")
        self.assertTrue(os.path.exists(fb["pose"]) and os.path.exists(fb["zip"]), "the other clip is untouched")
        for root in (self.c["work_root"], self.c["submit_root"], self.c["marks_root"]):
            self.assertTrue(os.path.isdir(root), "the folders themselves stay")

    def test_never_outside_the_app_folders(self):
        self.assertFalse(clipsmod._inside(self.c["work_root"], [self.c["work_root"]]), "a root itself is not inside")
        self.assertFalse(clipsmod._inside(self.a.video, [self.c["work_root"], self.c["submit_root"]]))
        self.assertTrue(clipsmod._inside(os.path.join(self.c["work_root"], "x"), [self.c["work_root"]]))


class TestProgress(unittest.TestCase):
    """the app reads the pipeline's log lines: run_pipeline must keep writing them"""
    def test_progress_from_log(self):
        d = tempfile.mkdtemp()
        try:
            lp = os.path.join(d, "job_movement.log")
            cl = mock.Mock(); cl.name = "x"
            with mock.patch.object(server, "folders", return_value={"work": d}):
                with open(lp, "w", encoding="utf-8") as fh:
                    fh.write("  marks: x\n  lens: 40.7 mm\n  [shot 1: frames 0-136]\n  camera ...\n")
                p = server.job_progress({}, cl, {"stage": "movement", "log": lp})
                self.assertIn("camera", p["step"])
                with open(lp, "a", encoding="utf-8") as fh:
                    fh.write("  people ...\n  set ...  (Stage 1: the floor only)\n  quality:\n  blender build + render (10 frames) ...\n")
                p2 = server.job_progress({}, cl, {"stage": "movement", "log": lp})
                self.assertGreater(p2["pct"], p["pct"]); self.assertIn("rendering", p2["step"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_pipeline_still_writes_the_parsed_lines(self):
        with open(os.path.join(ROOT, "pipeline", "run_pipeline.py"), encoding="utf-8") as fh:
            src = fh.read()
        for line in ('"  marks: %s"', '"  camera ..."', '"  people ..."', '"  set ...', '"  quality:"',
                     '"  [shot %d: frames %d-%d]"', '"  blender build + render (%d frames) ..."', '"  videos + zip ..."'):
            self.assertIn(line, src)


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


if __name__ == "__main__":
    unittest.main()

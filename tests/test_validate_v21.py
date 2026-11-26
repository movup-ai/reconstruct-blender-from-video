# -*- coding: utf-8 -*-
"""The v2.1 package check (agent/validate.py) on small synthetic packages made with ffmpeg."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config  # noqa: E402,F401  (puts the pipeline on sys.path)
from agent import validate  # noqa: E402
from common import ffmpeg_exe  # noqa: E402  (pipeline)

PKG = "test_film-1"
FF = ffmpeg_exe()


def ffmpeg(*args):
    p = subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)


def encode(path, size, source):
    """a 1 s H.264 video of <size> carrying the source's audio stream unchanged"""
    ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=%dx%d:r=24:d=1" % size, "-i", source, "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "copy", path)


def zip_folder(sub, zp):
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(sub):
            for fn in sorted(fs):
                full = os.path.join(dp, fn)
                z.write(full, os.path.relpath(full, sub).replace(os.sep, "/"))


@unittest.skipUnless(os.path.exists(FF) or shutil.which(FF), "ffmpeg is needed")
class TestPackageCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.source = os.path.join(cls.d, "source.mp4")
        ffmpeg("-f", "lavfi", "-i", "testsrc=s=320x180:r=24:d=1", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
               "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", cls.source)
        cls.videos = {}
        for name, size in (("wm_h", (1920, 1080)), ("cmp_h", (1920, 2160)), ("wm_v", (1080, 1920)), ("cmp_v", (2160, 1920))):
            cls.videos[name] = os.path.join(cls.d, name + ".mp4")
            encode(cls.videos[name], size, cls.source)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def package(self, wm="wm_h", cmp_="cmp_h", root=None):
        root = root or tempfile.mkdtemp(dir=self.d)
        sub = os.path.join(root, "clip_one")
        for d in ("comparison", "meta", "model_3d/reconstruction", "target", "white_model"):
            os.makedirs(os.path.join(sub, d))
        shutil.copy(self.source, os.path.join(sub, "target", "original.mp4"))
        shutil.copy(self.videos[wm], os.path.join(sub, "white_model", "blendervid.mp4"))
        shutil.copy(self.videos[cmp_], os.path.join(sub, "comparison", "comparison.mp4"))
        with open(os.path.join(sub, "model_3d", PKG + ".blend"), "wb") as fh:
            fh.write(b"BLENDER-v500" + b"\0" * 64)
        with open(os.path.join(sub, "model_3d", "reconstruction", "scene.json"), "w") as fh:
            json.dump({"scene": {"fps": 24}, "objects": [{"name": "P1"}]}, fh)
        meta = {"id": PKG, "source": "target/original.mp4", "white_model": "white_model/blendervid.mp4",
                "comparison": "comparison/comparison.mp4", "model_3d": "model_3d/%s.blend" % PKG,
                "scene_json_path": "model_3d/reconstruction/scene.json", "blender_version": "5.0.0",
                "frame_start": 1, "frame_end": 24, "frames": 24, "fps": 24, "fps_base": 1, "duration": 1.0}
        with open(os.path.join(sub, "meta", "meta.json"), "w") as fh:
            json.dump(meta, fh)
        zip_folder(sub, sub + ".zip")
        return sub

    def test_good_package_passes(self):
        sub = self.package()
        problems, facts = validate.check_package_v21(sub, FF, source=self.source)
        self.assertEqual(problems, [])
        self.assertEqual((facts["white_model"]["width"], facts["comparison"]["height"]), (1920, 2160))
        self.assertEqual(validate.check_zip(sub + ".zip", sub), [])

    def test_vertical_package_passes(self):
        sub = self.package("wm_v", "cmp_v")
        self.assertEqual(validate.check_package_v21(sub, FF, source=self.source)[0], [])

    def test_wrong_orientation_pairing_fails(self):
        for wm, cmp_ in (("wm_h", "cmp_v"), ("wm_v", "cmp_h")):
            problems = validate.check_package_v21(self.package(wm, cmp_), FF)[0]
            self.assertTrue(any(p.startswith("comparison is ") for p in problems), (wm, cmp_, problems))

    def test_white_model_must_be_1080p(self):
        problems = validate.check_package_v21(self.package("cmp_h", "cmp_h"), FF)[0]
        self.assertTrue(any(p.startswith("white_model is 1920x2160") for p in problems), problems)

    def test_modified_target_fails_with_source(self):
        sub = self.package()
        with open(os.path.join(sub, "target", "original.mp4"), "ab") as fh:
            fh.write(b"\0" * 16)
        self.assertNotIn("target/original.mp4 is not byte-identical to the source video",
                         validate.check_package_v21(sub, FF)[0], "without a source nothing is compared")
        problems = validate.check_package_v21(sub, FF, source=self.source)[0]
        self.assertIn("target/original.mp4 is not byte-identical to the source video", problems)

    def test_missing_file_and_bad_id_fail(self):
        sub = self.package()
        os.remove(os.path.join(sub, "model_3d", "reconstruction", "scene.json"))
        with open(os.path.join(sub, "meta", "meta.json")) as fh:
            meta = json.load(fh)
        meta["id"] = "Not A Name"
        with open(os.path.join(sub, "meta", "meta.json"), "w") as fh:
            json.dump(meta, fh)
        problems = validate.check_package_v21(sub, FF)[0]
        self.assertIn("missing model_3d/reconstruction/scene.json", problems)
        self.assertTrue(any("is not a v2.1 package name" in p for p in problems))

    def test_zip_must_match_the_folder(self):
        sub = self.package()
        with open(os.path.join(sub, "meta", "extra.txt"), "w") as fh:
            fh.write("x")
        problems = validate.check_zip(sub + ".zip", sub)
        self.assertEqual(len(problems), 1)
        self.assertIn("missing: meta/extra.txt", problems[0])
        with open(sub + ".zip", "wb") as fh:
            fh.write(b"not a zip")
        self.assertTrue(validate.check_zip(sub + ".zip", sub)[0].startswith("zip unreadable"))

    def test_check_all_with_an_old_folder_does_not_crash(self):
        root = tempfile.mkdtemp(dir=self.d)
        self.package(root=root)
        old = os.path.join(root, "older_clip", "meta")
        os.makedirs(old)
        with open(os.path.join(old, "meta.jsonl"), "w") as fh:
            fh.write("{}\n")
        rows = {r["package"]: r for r in validate.check_all(root)}
        self.assertTrue(rows["clip_one"]["ok"], rows["clip_one"]["problems"])
        self.assertFalse(rows["older_clip"]["ok"])
        self.assertIn("not the v2.1 layout", rows["older_clip"]["problems"][0])
        self.assertEqual(rows["older_clip"]["white_model"], "NonexNone")


if __name__ == "__main__":
    unittest.main()

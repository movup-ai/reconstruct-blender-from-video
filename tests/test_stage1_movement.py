"""Stage 1 Movement end to end with the heavy models mocked: marks -> camera -> people -> floor -> scene_data.json ->
Blender (colour card) -> zip -> v2.1 package check.  No GPU, Blender or ffmpeg is used."""
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline")]
import run_pipeline as RP  # noqa: E402
import colour_card as CC  # noqa: E402
import quality  # noqa: E402
import scene_depth  # noqa: E402
from common import DEFAULT_CFG, jload, jsave  # noqa: E402
from rig import JOINTS  # noqa: E402

N, W, H = 12, 64, 48


def pose_json(video):
    k = np.zeros((17, 3)); k[:, 2] = 0.9
    k[:, 0] = np.linspace(28, 36, 17); k[:, 1] = np.linspace(10, 40, 17)
    frames = {str(f): k.round(1).tolist() for f in range(N)}
    boxes = {str(f): [26.0, 8.0, 38.0, 42.0] for f in range(N)}
    return {"video": video, "frames": N, "fps": 24.0, "width": W, "height": H, "picture_area": [0, 0, W, H],
            "tracker_backend": "sam3", "people": [{"id": 1, "sam3_id": 5, "frames": frames, "boxes": boxes}]}


def character(n):
    return {"label": "Main character_red", "colour": "red", "height": 1.72, "track_id": 1, "style": "sample",
            "root": [[0.0, 6.0, 0.0, 0.0] for _ in range(n)], "pelvis_z": [0.94] * n,
            "rot": {j: [[0.0, 0.0, 0.0] for _ in range(n)] for j in JOINTS}, "visible": [True] * n,
            "human_backend": "sam3d_body", "body_constraints": {"version": 2}}


def fake_fit_all(world, y0, people, n, c):
    return {"P1": character(n)}, [{"person": "P1", "track": 1, "colour": "red", "frames": n,
                                   "median_error_px": 1.0, "worst_error_px": 2.0}]


def fake_camera(frames, pose, people, cam, y0, Hp, c):
    camj = {"R": [np.eye(3).tolist() for _ in frames], "zoom": [1.0] * len(frames), "warnings": [],
            "max_rotation_deg": 0.0, "parallax_px": 0.0, "steady": "locked"}
    return camj, np.array([0.0, -1.0, 0.0]), 1.6, {"source": "test", "camera_height": 1.6}


class Stage1RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = self.tmp.name
        self.video = os.path.join(d, "src", "clip.mp4"); os.makedirs(os.path.dirname(self.video))
        open(self.video, "wb").close()
        self.marks = os.path.join(d, "marks"); os.makedirs(self.marks)
        self.pose_p = os.path.join(self.marks, "pose.json"); jsave(self.pose_p, pose_json(self.video))
        self.c = dict(DEFAULT_CFG, source_root=os.path.join(d, "src"), work_root=os.path.join(d, "work"),
                      submit_root=os.path.join(d, "submit"), marks_root=self.marks, lens_auto=False, metric_scene=False,
                      device="cuda")
        self.args = types.SimpleNamespace(stage="movement", force=True, device="cuda", name="clip_a")
        self.work = os.path.join(self.c["work_root"], "clip_a")
        self.calls = {}

    def tearDown(self):
        self.tmp.cleanup()

    def blender(self, cmd, **kw):
        """stands in for Blender: the colour card has been applied to the scene it is given"""
        self.calls["blender"] = jload(cmd[cmd.index("--") + 1])
        open(os.path.join(self.work, "render_done.txt"), "w").close()
        return subprocess.CompletedProcess(cmd, 0)

    def package(self, name, video, render_dir, submit_root, ci, c, desc=""):
        sub = os.path.join(submit_root, name); os.makedirs(sub, exist_ok=True)
        zp = os.path.join(submit_root, name + ".zip"); open(zp, "wb").close()
        self.calls["package"] = (name, video)
        return sub, zp

    def run_clip(self, problems=(), calibrated=False):
        frames = [np.full((H, W, 3), 90, np.uint8) for _ in range(N)]
        body = ({}, {"calibrated": calibrated, "meters_per_unit": 1.0, "heights_m": {}})
        with patch.object(RP, "find_marks", return_value=self.pose_p) as marks, \
             patch.object(RP, "read_frames", return_value=(frames, 24.0)), \
             patch.object(RP, "find_shots", return_value=[(0, N)]), \
             patch.object(RP, "estimate_shot", side_effect=fake_camera), \
             patch("sam3d_body.recover", return_value=body), \
             patch.object(RP, "fit_all", side_effect=fake_fit_all), \
             patch.object(quality, "floor_shift", return_value=None) as floor, \
             patch.object(RP.pkg, "package_name", return_value="film-1"), \
             patch.object(RP, "blender_exe", return_value="blender.exe"), \
             patch.object(RP.subprocess, "run", side_effect=self.blender), \
             patch.object(RP.pkg, "package", side_effect=self.package), \
             patch("agent.validate.check_package_v21", return_value=(list(problems), {})) as check, \
             patch("agent.validate.check_zip", return_value=[]) as zcheck:
            try:
                return RP.run_clip(self.video, dict(self.c), self.args)
            finally:
                self.calls.update(marks=marks, floor=floor, check=check, zcheck=zcheck)

    def test_one_straight_run_to_a_checked_zip(self):
        rep = self.run_clip()
        self.assertEqual(rep["status"], "ok"); self.assertEqual(rep["package_check"], "ok")
        data = jload(os.path.join(self.work, "scene_data.json"))
        self.assertEqual([m["name"] for m in data["meshes"]], ["ground"], "Stage 1: the floor only, no set, no backdrop")
        self.assertEqual(data["props"], [])
        self.assertEqual(data["meshes"][0]["colour"], "bg"); self.assertEqual(data["palette"]["bg"], [184, 184, 184])
        self.assertEqual(data["chars"]["P1"]["colour"], "red"); self.assertEqual(data["palette"]["red"], CC.CARD["red"])
        self.assertEqual(data["palette"]["marker"], [255, 255, 255])
        self.assertTrue(data["colour_card"])
        for key in ("background_policy", "previs_standard", "validation_mode"):
            self.assertNotIn(key, data)
        self.assertEqual(self.calls["blender"]["meshes"][0]["colour"], "bg", "Blender gets the recoloured scene")
        self.calls["check"].assert_called_once()
        args, kwargs = self.calls["check"].call_args
        self.assertEqual(args[0], os.path.join(self.c["submit_root"], "clip_a"))
        self.assertEqual(kwargs, {"source": self.video})
        self.calls["zcheck"].assert_called_once_with(os.path.join(self.c["submit_root"], "clip_a.zip"),
                                                     os.path.join(self.c["submit_root"], "clip_a"))
        for fn in ("camera.json", "clip_info.json", "report.json", "preview.jpg", "human_alignment.json", "render_done.txt"):
            self.assertTrue(os.path.isfile(os.path.join(self.work, fn)), fn)
        report = jload(os.path.join(self.work, "report.json"))
        self.assertEqual(report["status"], "ok"); self.assertIn("human_alignment", report); self.assertIn("quality", report)
        self.calls["floor"].assert_called_once()          # SAM 3D Body not calibrated: the half-body floor shift runs

    def test_calibrated_body_scale_skips_the_floor_shift(self):
        self.run_clip(calibrated=True)
        self.calls["floor"].assert_not_called()

    def test_package_problems_fail_the_clip(self):
        with self.assertRaisesRegex(RuntimeError, "v2.1 package check failed: comparison is 1920x1080"):
            self.run_clip(problems=["comparison is 1920x1080, must be 1920x2160 or 2160x1920"])
        self.assertNotEqual(jload(os.path.join(self.work, "report.json"))["status"], "ok")

    def test_skips_a_finished_clip_without_force(self):
        os.makedirs(self.c["submit_root"]); open(os.path.join(self.c["submit_root"], "clip_a.zip"), "wb").close()
        self.args.force = False
        self.assertEqual(RP.run_clip(self.video, dict(self.c), self.args)["status"], "skipped")


class ColourCardTests(unittest.TestCase):
    def scene(self, n_people=6):
        chars = {"P%d" % (i + 1): dict(character(2), colour=c) for i, c in
                 enumerate(["red", "blue", "green", "yellow", "purple", "orange"][:n_people])}
        return {"frames": 2, "res": [64, 48], "lens": 35.0, "sensor": 36.0, "cam": [[0, 0, 1.6, 90, 0, 0]] * 2,
                "palette": {"ground": [242, 242, 242]}, "chars": chars, "props": [],
                "meshes": scene_depth.flat_set(None, 2, 1500.0)}

    def test_grey_floor_white_marks_and_card_people(self):
        data = self.scene()
        count = CC.apply(data, log=None)
        self.assertEqual(count, {"bg": 1})
        self.assertEqual(data["meshes"][0]["colour"], "bg"); self.assertEqual(data["palette"]["bg"], [184, 184, 184])
        self.assertEqual(data["palette"]["marker"], [255, 255, 255])
        self.assertEqual([data["chars"]["P%d" % i]["colour"] for i in range(1, 7)],
                         ["red", "blue", "green", "yellow", "cyan", "magenta"], "5th on: the card's extra colours")
        self.assertFalse(any(k.startswith("source_bg_") for k in data["palette"]), "no colours sampled from the video")

    def test_flat_set_is_one_ground_disc(self):
        meshes = scene_depth.flat_set(None, 5, 1500.0)
        self.assertEqual(len(meshes), 1)
        self.assertEqual((meshes[0]["name"], meshes[0]["colour"], meshes[0]["layer"]), ("ground", "ground", 1))
        self.assertTrue(all(v[2] == 0 for v in meshes[0]["v"]))

    def test_colour_card_failure_fails_the_render(self):
        with tempfile.TemporaryDirectory() as work:
            dp = os.path.join(work, "scene_data.json")
            jsave(dp, dict(self.scene(1), name="clip", work_dir=work, submit_dir=os.path.join(work, "s", "clip")))
            with patch.object(RP, "blender_exe", return_value="blender.exe"), \
                 patch.object(CC, "apply", side_effect=ValueError("bad palette")), \
                 patch.object(RP.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "bad palette"):
                    RP.blender_render({"submit_root": os.path.join(work, "s")}, work, dp, 2)
            run.assert_not_called()


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, os.path.join(ROOT, "pipeline", "run_pipeline.py")] + list(args),
                              capture_output=True, text=True, timeout=120)

    def test_only_movement_render_and_background_stages(self):
        for stage in ("fit", "blend", "all", "depth"):
            r = self.run_cli("x.mp4", "--stage", stage)
            self.assertEqual(r.returncode, 2, stage); self.assertIn("invalid choice", r.stderr)
        r = self.run_cli("x.mp4", "--require-confirm")
        self.assertEqual(r.returncode, 2); self.assertIn("unrecognized arguments", r.stderr)
        self.assertEqual(RP.STAGES, ("movement", "render", "background"))


if __name__ == "__main__":
    unittest.main()

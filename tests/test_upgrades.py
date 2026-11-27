"""Regressions for foot contact and per-clip timing."""
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline")]
import profiling
from common import Camera


class FootContactTests(unittest.TestCase):
    def test_contact_fit_preserves_visible_joint_alignment(self):
        from ground_solve import World
        from rig import Rig
        from fit_people import fit_track
        rig = Rig(1.72)
        world = World(Camera(640, 360, 35), [np.eye(3)] * 4, np.array([0, -1, 0]), 1.6)
        joints = rig.coco_points(rig.fk([0, 6, 0], 0, {}, rig.J[0][2][2]))
        uv, _ = world.project(joints, 0)
        keys = np.repeat(np.c_[uv, np.full(17, 0.95)][None], 4, axis=0)
        with patch("fit_people.log"):
            _, poses, errors = fit_track(world, 0, list(range(4)), keys, 1.72, step=1,
                                          hint={"foot_contact": True, "lying": False})
        self.assertEqual(len(poses), 4)
        self.assertLess(max(errors.values()), 5)


class ProfilingTests(unittest.TestCase):
    def test_failed_clip_still_writes_timings(self):
        @profiling.stage("broken")
        def fail():
            raise ValueError("expected")
        @profiling.profile_clip
        def clip(video, config, args):
            fail()
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaises(ValueError):
                clip("input.mp4", {"work_root": work}, types.SimpleNamespace(name="clip", stage="run"))
            with open(os.path.join(work, "clip", "profile_run.json")) as stream:
                result = json.load(stream)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stages"]["broken"]["calls"], 1)


if __name__ == '__main__':
    unittest.main()

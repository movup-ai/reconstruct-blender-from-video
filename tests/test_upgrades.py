"""Regressions for tracking, adaptive sampling, camera alignment and timing."""
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
import automark
import env_auto
import da3_camera
import profiling
from camera_solve import Camera


def person(x, confidence=0.9):
    keys = np.tile([x + 10, 50.0], (17, 1))
    return keys, np.full(17, confidence), [x, 10, x + 20, 110]


class TrackingTests(unittest.TestCase):
    def test_cut_starts_new_identity(self):
        tracks = automark.track([[person(0)], [person(0)]], cuts=[1])
        self.assertEqual([list(t) for t in tracks.values()], [[0], [1]])

    def test_low_confidence_recovers_without_starting_noise_tracks(self):
        tracks = automark.track([[person(0)], [person(1, 0.4), person(200, 0.1)]])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(list(tracks[1]), [0, 1])

    def test_previous_uncertain_joints_do_not_break_track(self):
        old = person(0)
        old[0][:12] = 10000
        old[1][:12] = 0.1
        tracks = automark.track([[old], [person(1)]])
        self.assertEqual(len(tracks), 1)

    def test_color_distinguishes_people_with_same_geometry(self):
        red = np.zeros((120, 100, 3), np.uint8); red[:, :, 2] = 255
        blue = np.zeros_like(red); blue[:, :, 0] = 255
        first = red.copy(); first[:, 40:] = blue[:, 40:]
        second = blue.copy(); second[:, 40:] = red[:, 40:]
        # Both candidates are near both previous tracks. Color must overcome the swapped positions.
        p = person(20); q = person(40)
        tracks = automark.track([[p, q], [p, q]], frames=[first, second])
        self.assertEqual(tracks[1][1][2], q[2])


class SamplingTests(unittest.TestCase):
    def test_static_camera_keeps_two_views(self):
        data = {"cam": [[0, 0, 1.6, 90, 0, 0]] * 100}
        self.assertEqual(env_auto.sample_set_frames(data, 0, 99, {}), [0, 49])

    def test_translation_and_zoom_add_bounded_views(self):
        data = {"cam": [[i / 10, 0, 1.6, 90, 0, 0] for i in range(100)]}
        selected = env_auto.sample_set_frames(data, 0, 99, {"set_max_frames": 4})
        self.assertEqual(len(selected), 4)
        self.assertIn(99, selected)
        data["cam"] = [[0, 0, 1.6, 90, 0, 0]] * 100
        data["zoom"] = np.linspace(1, 2, 100).tolist()
        self.assertGreater(len(env_auto.sample_set_frames(data, 0, 99, {})), 2)
        self.assertEqual(env_auto.sample_set_frames(data, 0, 99, {}, True), [0])

    def test_newly_uncovered_background_adds_a_view(self):
        data = {"cam": [[0, 0, 1.6, 90, 0, 0]] * 100}
        boxes = {str(f): [0, 0, 160, 180] for f in range(80)}
        pose = {"people": [{"boxes": boxes}]}
        selected = env_auto.sample_set_frames(data, 0, 99, {}, pose=pose, picture=(320, 180, 0))
        self.assertTrue(any(f >= 80 for f in selected))


class CameraTests(unittest.TestCase):
    def test_explicit_scale_and_missing_scale(self):
        cam = Camera(320, 180, 35)
        result = {"travel_cam0": [[0, 0, 0], [1, 0, 0]], "warnings": []}
        da3_camera.align_scale(result, np.ones((90, 160)), {}, cam, 0, np.array([0, -1, 0]), 1.6,
                               {"da3_meters_per_unit": 2})
        self.assertEqual(result["travel_cam0"][1], [2, 0, 0])
        da3_camera.align_scale(result, np.ones((90, 160)), {}, cam, 0, np.array([0, -1, 0]), 1.6, {})
        self.assertNotIn("travel_cam0", result)
        self.assertTrue(result["warnings"])

    def test_pose_interpolation_and_coordinate_conversion(self):
        from scipy.spatial.transform import Rotation
        ext = np.repeat(np.eye(4)[None], 2, axis=0)
        ext[1, :3, :3] = Rotation.from_euler("y", 20, degrees=True).as_matrix()
        ext[1, :3, 3] = -ext[1, :3, :3] @ [1, 0, 0]
        prediction = types.SimpleNamespace(extrinsics=ext, depth=np.ones((2, 90, 160)))
        model = types.SimpleNamespace(inference=lambda **kw: prediction)
        module = types.ModuleType("depth_anything_3.api"); module.DepthAnything3 = object
        with tempfile.TemporaryDirectory() as work, patch.dict(sys.modules, {"depth_anything_3.api": module}), patch.dict(da3_camera._models, {work: model}):
            result, _ = da3_camera.solve([np.zeros((180, 320, 3), np.uint8)] * 3, Camera(320, 180, 35), 0, 180,
                                         {"da3_model": work, "da3_max_views": 2, "da3_python": ""})
        self.assertAlmostEqual(result["travel_cam0"][1][0], 0.5)
        self.assertAlmostEqual(result["max_rotation_deg"], 20)

    def test_translation_scale_uses_later_visible_ankles(self):
        cam = Camera(320, 180, 35)
        keys = np.zeros((17, 3)); keys[15] = [100, 150, 0.9]; keys[16] = [140, 150, 0.9]
        result = {"travel_cam0": [[0, 0, 0], [0, 0.2, 0]], "R": [np.eye(3).tolist()] * 2, "warnings": []}
        da3_camera.align_scale(result, {1: np.full((90, 160), 2.0)}, {1: ([1], [keys])}, cam, 0,
                               np.array([0, -1, 0]), 1.6, {})
        self.assertAlmostEqual(result["meters_per_unit"], 1.6 / (0.2 + 2 * 60 / cam.f))


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
                clip("input.mp4", {"work_root": work}, types.SimpleNamespace(name="clip", stage="fit"))
            with open(os.path.join(work, "clip", "profile_fit.json")) as stream:
                result = json.load(stream)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stages"]["broken"]["calls"], 1)


class SettingsTests(unittest.TestCase):
    def test_camera_scale_validation_and_auto_reset(self):
        from agent import config
        with tempfile.TemporaryDirectory() as work, patch.object(config, "CFG_PATH", os.path.join(work, "config.json")):
            self.assertEqual(config.save({"da3_meters_per_unit": 2})["da3_meters_per_unit"], 2)
            self.assertIsNone(config.save({"da3_meters_per_unit": None})["da3_meters_per_unit"])
            for value in (0, -1, float("nan")):
                with self.assertRaises(ValueError):
                    config.save({"da3_meters_per_unit": value})

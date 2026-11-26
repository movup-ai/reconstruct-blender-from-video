# -*- coding: utf-8 -*-
"""quality.py: the junk-track filter and the per-track numbers"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import quality  # noqa: E402


def track(n, sure, move=0.0, seed=0):
    """n frames; `sure` joints with a high score; move = how much the pose itself changes"""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 200, (17, 2))
    K = np.zeros((n, 17, 3))
    for f in range(n):
        K[f, :, :2] = base + 30 * f + move * 40 * np.sin(f / 3.0 + np.arange(17))[:, None]   # walks across + moves
        K[f, :sure, 2] = 0.9
        K[f, sure:, 2] = 0.1
    return list(range(n)), K


class TestJunk(unittest.TestCase):
    def test_short_and_unclear_is_junk(self):
        fs, K = track(16, 5)
        self.assertIsNotNone(quality.junk_reason(quality.track_stats(fs, K, 30.0)))

    def test_long_unclear_person_is_kept(self):                 # a far, half-hidden person seen the whole shot
        fs, K = track(129, 5)
        self.assertIsNone(quality.junk_reason(quality.track_stats(fs, K, 25.0)))

    def test_short_clear_person_is_kept(self):                  # someone walking in at the end of the shot
        fs, K = track(16, 15)
        self.assertIsNone(quality.junk_reason(quality.track_stats(fs, K, 30.0)))

    def test_persistent_track_is_kept_and_junk_is_reported(self):
        fs, K = track(16, 5)
        self.assertIn(7, quality.drop_junk({7: (fs, K)}, {"_persistent_tracks": True}, 30.0))   # SAM 3 tracks
        c = {}
        self.assertEqual(quality.drop_junk({7: (fs, K)}, c, 30.0), {})
        self.assertEqual(c["_junk"][0]["track"], 7)

    def test_filter_can_be_switched_off(self):
        fs, K = track(16, 5)
        self.assertIn(7, quality.drop_junk({7: (fs, K)}, {"junk_filter": False}, 30.0))


class TestStats(unittest.TestCase):
    def test_rigid_pose_scores_lower_than_moving(self):
        still = quality.track_stats(*track(60, 17, move=0.0), fps=30.0)["pose_change"]
        moving = quality.track_stats(*track(60, 17, move=1.0), fps=30.0)["pose_change"]
        self.assertLess(still, quality.LIMITS["pose_change"])
        self.assertGreater(moving, quality.LIMITS["pose_change"])

    def test_feet_seen(self):
        self.assertEqual(quality.track_stats(*track(30, 17), fps=30.0)["feet_seen"], 1.0)
        self.assertEqual(quality.track_stats(*track(30, 11), fps=30.0)["feet_seen"], 0.0)

    def test_jumps(self):
        n = 40
        rot = np.zeros((n, 3)); rot[:, 0] = np.linspace(0, 20, n)          # 0.5 deg a frame: smooth
        ch = {"root": [[0, 0, 0, 0]] * n, "rot": {"elbow_L": rot.tolist()}, "visible": [True] * n}
        self.assertEqual(quality.jumps(ch), [])
        ch["root"] = [[0.05 * f, 0, 0, 0] for f in range(n)]                 # walking 5 cm a frame: not a jump
        self.assertEqual(quality.jumps(ch), [])
        rot[20:, 0] += 30                                                    # one sudden 30 deg step
        ch["rot"]["elbow_L"] = rot.tolist()
        self.assertIn(20, quality.jumps(ch))


class TestFloorShift(unittest.TestCase):
    def setUp(self):
        self._gap = quality.foot_gap

    def tearDown(self):
        quality.foot_gap = self._gap

    def test_half_body_people_get_the_floor_under_them(self):
        quality.foot_gap = lambda ch: -0.28                     # feet 28 cm under the guessed floor
        fs, K = track(60, 11)                                   # ankles never seen
        self.assertAlmostEqual(quality.floor_shift({"P1": {"track_id": 2}}, {2: (fs, K)}), 0.28)

    def test_seen_feet_keep_the_floor(self):
        quality.foot_gap = lambda ch: -0.28
        half, full = track(60, 11), track(60, 17)
        self.assertIsNone(quality.floor_shift({"P1": {"track_id": 2}, "P2": {"track_id": 3}}, {2: half, 3: full}))

    def test_small_gap_is_left_alone(self):
        quality.foot_gap = lambda ch: 0.02
        self.assertIsNone(quality.floor_shift({"P1": {"track_id": 2}}, {2: track(60, 11)}))


class TestLens(unittest.TestCase):
    def test_fov_to_lens(self):
        import lens_auto
        self.assertAlmostEqual(lens_auto.lens_from_fov(90.0), 18.0, places=3)        # 36 mm sensor, 90 deg -> 18 mm
        self.assertAlmostEqual(lens_auto.lens_from_fov(54.6), 34.9, places=0)        # the old fixed lens

    def test_three_frames_a_shot(self):
        import lens_auto
        self.assertEqual(lens_auto.sample_frames([(0, 100), (100, 140)]), [[25, 50, 75], [110, 120, 130]])

    def test_clip_lens_is_the_median_and_differing_shots_are_reported(self):
        import lens_auto
        lens, info = lens_auto.summarise([[50.0, 55.0, 60.0]])
        self.assertEqual(lens, 55.0); self.assertNotIn("warning", info)
        lens, info = lens_auto.summarise([[20.0, 21.0, 22.0], [50.0, 50.0, 50.0], [80.0, 85.0, 90.0]])
        self.assertEqual(lens, 50.0); self.assertIn("warning", info)
        self.assertIsNone(lens_auto.summarise([[]])[0])


if __name__ == "__main__":
    unittest.main()

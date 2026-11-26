"""Track cleaning (pipeline/track_clean.py) and the short-gap fill (fit_people.visible_frames)."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import track_clean as T          # noqa: E402
from fit_people import visible_frames   # noqa: E402

W, Y0, Y1, FPS = 1920, 0, 1080, 24.0
# a standing body in box-relative units (x, y as shares of the box height), COCO-17
BODY = np.array([[.25, .06], [.27, .05], [.23, .05], [.30, .06], [.20, .06], [.35, .2], [.15, .2], [.4, .38], [.1, .38],
                 [.42, .52], [.08, .52], [.32, .52], [.18, .52], [.33, .75], [.17, .75], [.33, .97], [.17, .97]])


def track(frames, box, conf=0.8, move=0.04, step=(0.0, 0.0), seed=0, core=None):
    """a person: box [x0, y0, x1, y1] (drifting by `step` px a frame), joints wobbling `move` of the height"""
    rs = np.random.default_rng(seed)
    out = {}
    for i, f in enumerate(frames):
        x0, y0, x1, y1 = box[0] + step[0] * i, box[1] + step[1] * i, box[2] + step[0] * i, box[3] + step[1] * i
        h = y1 - y0
        k = BODY * h + [x0, y0] + rs.normal(0, move * h, (17, 2))
        s = np.full(17, conf)
        if core is not None:
            s[list(T.CORE)] = core
        out[f] = (k, s, [x0, y0, x1, y1])
    return out


class FalsePeople(unittest.TestCase):
    def test_fragment_with_no_head_shoulders_or_hips_is_dropped(self):
        hand = track(range(60), [1800, 0, 1915, 120], conf=0.5, core=0.1)
        kept, gone = T.drop_false({7: hand}, W, Y0, Y1, FPS)
        self.assertNotIn(7, kept); self.assertIn("fragment", gone[7])

    def test_poster_never_moving_fully_in_view_is_dropped(self):
        poster = track(range(100), [900, 100, 1000, 380], conf=0.35, move=0.002)
        kept, gone = T.drop_false({3: poster}, W, Y0, Y1, FPS)
        self.assertNotIn(3, kept); self.assertIn("picture or statue", gone[3])

    def test_poster_cut_by_the_top_edge_is_still_dropped(self):
        poster = track(range(100), [900, 0, 1000, 280], conf=0.35, move=0.002)          # touches only the top
        kept, gone = T.drop_false({4: poster}, W, Y0, Y1, FPS)
        self.assertNotIn(4, kept, gone)

    def test_still_person_cut_by_the_frame_edge_is_kept(self):
        legs = track(range(100), [1700, 120, 1920, 1080], conf=0.35, move=0.002)       # touches the right / bottom edge
        kept, gone = T.drop_false({2: legs}, W, Y0, Y1, FPS)
        self.assertIn(2, kept, gone)

    def test_seated_person_with_strong_joints_is_kept_even_when_still(self):
        sitting = track(range(150), [600, 300, 800, 980], conf=0.77, move=0.002)
        kept, gone = T.drop_false({1: sitting}, W, Y0, Y1, FPS)
        self.assertIn(1, kept, gone)

    def test_weak_joints_jumping_around_are_dropped(self):
        blob = track(range(40), [1500, 0, 1700, 250], conf=0.35, move=0.4)
        kept, gone = T.drop_false({6: blob}, W, Y0, Y1, FPS)
        self.assertNotIn(6, kept); self.assertIn("jump", gone[6])

    def test_second_id_on_the_same_person_is_dropped_and_the_stronger_kept(self):
        real = track(range(60), [500, 200, 700, 900], conf=0.8, seed=1)
        twin = {f: (k + 2.0, s * 0.6, b) for f, (k, s, b) in real.items()}           # same joints, weaker
        kept, gone = T.drop_false({1: real, 2: twin}, W, Y0, Y1, FPS)
        self.assertIn(1, kept); self.assertNotIn(2, kept); self.assertIn("second id on person 1", gone[2])

    def test_two_real_people_side_by_side_are_both_kept(self):
        a = track(range(60), [300, 200, 500, 900], seed=1); b = track(range(60), [700, 200, 900, 900], seed=2)
        kept, gone = T.drop_false({1: a, 2: b}, W, Y0, Y1, FPS)
        self.assertEqual(set(kept), {1, 2}, gone)


class Joining(unittest.TestCase):
    SIG = np.full((18, 8), 1.0 / 144, np.float32)

    def looks(self, *ids, other=None):
        out = {i: (self.SIG, self.SIG) for i in ids}
        if other is not None:
            sig = np.zeros((18, 8), np.float32); sig[0, 0] = 1.0                       # completely different clothes
            out[other] = (sig, sig)
        return out

    def test_actor_hidden_longer_than_sam_memory_is_joined_when_alone(self):
        a = track(range(0, 50), [800, 200, 1000, 900])
        b = track(range(50 + 8 * 24, 50 + 8 * 24 + 60), [840, 210, 1040, 910])           # back 8 s later, nearby
        out, merged, notes = T.join_pieces({1: a, 2: b}, self.looks(1, 2), [(0, 2000)], FPS, min_gap=144)
        self.assertEqual(merged, {2: 1}, notes); self.assertEqual(len(out[1]), 110)

    def test_short_hide_while_alone_is_joined_even_after_walking_towards_the_camera(self):
        a = track(range(0, 30), [900, 300, 1000, 600])                                   # small, at the back
        b = track(range(102, 122), [800, 100, 1100, 1000])                               # 3 s later, 3x bigger, close
        out, merged, notes = T.join_pieces({1: a, 2: b}, self.looks(1, 2), [(0, 2000)], FPS)
        self.assertEqual(merged, {2: 1}, notes)

    def test_not_joined_when_someone_else_is_in_the_shot(self):
        a = track(range(0, 50), [800, 200, 1000, 900]); b = track(range(250, 300), [820, 200, 1020, 900])
        c = track(range(0, 300), [100, 200, 300, 900], seed=5)                         # a third person all along
        _, merged, notes = T.join_pieces({1: a, 2: b, 3: c}, self.looks(1, 2, 3), [(0, 2000)], FPS, min_gap=144)
        self.assertEqual(merged, {}); self.assertTrue(any("someone else" in n for n in notes))

    def test_not_joined_when_the_clothes_differ(self):
        a = track(range(0, 50), [800, 200, 1000, 900]); b = track(range(250, 300), [820, 200, 1020, 900])
        _, merged, _ = T.join_pieces({1: a, 2: b}, self.looks(1, other=2), [(0, 2000)], FPS, min_gap=144)
        self.assertEqual(merged, {})

    def test_not_joined_across_a_cut_or_when_much_smaller(self):
        a = track(range(0, 50), [800, 200, 1000, 900])
        cut = track(range(250, 300), [820, 200, 1020, 900])
        _, m1, _ = T.join_pieces({1: a, 2: cut}, self.looks(1, 2), [(0, 150), (150, 2000)], FPS, min_gap=144)
        small = track(range(62, 110), [820, 600, 900, 900])                             # under half the size 0.5 s later
        _, m2, _ = T.join_pieces({1: a, 2: small}, self.looks(1, 2), [(0, 2000)], FPS)
        self.assertEqual(m1, {}); self.assertEqual(m2, {})

    def test_not_joined_when_the_return_is_too_far_for_the_time_hidden(self):
        a = track(range(0, 50), [1700, 200, 1900, 900])
        b = track(range(50 + 145, 260), [0, 200, 200, 900])                             # 1700 px = 2.4 heights in 6 s: ok
        quick = track(range(60, 120), [0, 200, 200, 900])                               # same distance in 0.4 s: not a walk
        _, m1, _ = T.join_pieces({1: a, 2: b}, self.looks(1, 2), [(0, 2000)], FPS, min_gap=144)
        _, m2, _ = T.join_pieces({1: a, 2: quick}, self.looks(1, 2), [(0, 2000)], FPS, min_gap=0)
        self.assertEqual(m1, {2: 1}); self.assertEqual(m2, {})

    def test_relabel_puts_joined_masks_under_one_id_without_clashes(self):
        bundle = {"frames": np.array([0, 0, 5, 9]), "ids": np.array([1, 3, 3, 2]), "packed": np.arange(4)[:, None],
                  "probabilities": np.ones(4, np.float32), "shape": np.array([2, 2])}
        out = T.relabel(bundle, {2: 1})
        self.assertEqual(sorted(zip(out["frames"].tolist(), out["ids"].tolist())), [(0, 1), (0, 3), (5, 3), (9, 1)])
        with self.assertRaises(ValueError):
            T.relabel(bundle, {3: 1})                                                   # both at frame 0: would clash


class GapFill(unittest.TestCase):
    def test_short_gaps_are_filled_long_ones_stay_hidden(self):
        fs = list(range(0, 10)) + list(range(15, 20)) + list(range(60, 70))           # 5-frame and 40-frame gaps
        v = visible_frames(fs, 80, 24.0, 0.5)                                          # fill up to 12 frames at 24 fps
        self.assertTrue(all(v[0:20]))                                                  # the 5-frame dropout: shown
        self.assertFalse(any(v[20:60]))                                                # really out of sight: hidden
        self.assertTrue(all(v[60:70])); self.assertFalse(any(v[70:]))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'pipeline')]

import object_checks as oc

def box(w, h):
    return {"size": [w, w, h]}


class Kinds(unittest.TestCase):
    def test_size_class_of_each_model(self):
        self.assertEqual(oc.role("coffee_table"), "table")
        self.assertEqual(oc.role("office_chair"), "chair")
        self.assertEqual(oc.role("fridge"), "fridge")
        self.assertEqual(oc.role("spaceship_console"), "spaceship_console")      # no range: not size-checked
        self.assertIsNone(oc.size_range("spaceship_console"))
        self.assertTrue(oc.is_seat("armchair") and oc.is_seat("stool") and not oc.is_seat("table"))


class EvenSplit(unittest.TestCase):
    def test_a_row_with_no_gap_is_cut_into_real_widths(self):
        rng = np.random.default_rng(0)
        P = np.c_[rng.uniform(-1.05, 1.05, 3000), rng.normal(0, .01, 3000), rng.uniform(0, 1.0, 3000)]   # 3 washers, touching
        def fit(Q):
            return {"size": [float(np.ptp(Q[:, 0])), 0.6, float(np.ptp(Q[:, 2]))], "yaw": 0.0,
                    "centre": [float(Q[:, 0].mean()), 0.0, 0.0]}
        parts = oc.split_if_too_wide("washer", P, fit)
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(0.5 <= np.ptp(q[:, 0]) <= 0.9 for q in parts))
        self.assertEqual(len(oc.split_if_too_wide("washer", P[np.abs(P[:, 0]) < 0.35], fit)), 1, "one washer stays whole")


class Sizes(unittest.TestCase):
    def test_real_world_ranges(self):
        self.assertIsNone(oc.size_problem("chair", box(0.55, 0.9)))
        self.assertIn("wide", oc.size_problem("chair", box(2.08, 0.96)))
        self.assertIn("wide", oc.size_problem("chair", box(0.12, 1.0)))
        self.assertIn("tall", oc.size_problem("door", box(0.7, 0.5)))
        self.assertIn("tall", oc.size_problem("coffee_table", box(1.0, 1.0)), "a type's own range beats its role's")
        self.assertIsNone(oc.size_problem("spaceship_console", box(9, 9)), "unknown types are not checked")
        self.assertIsNone(oc.size_problem("chair", box(2.0, 0.9), {"size": [[1, 3], [0.5, 1]]}), "objects.json size wins")

    def test_a_row_of_chairs_is_split_only_into_real_chairs(self):
        rng = np.random.default_rng(0)
        chairs = [np.c_[rng.uniform(x, x + 0.5, 400), rng.uniform(0, 0.5, 400), rng.uniform(0, 0.9, 400)] for x in (0, 0.9, 1.8)]
        P = np.vstack(chairs)
        fit = lambda Q: {"size": [float(np.ptp(Q[:, 0])), float(np.ptp(Q[:, 1])), float(np.ptp(Q[:, 2]))]}
        parts = oc.split_if_too_wide("chair", P, fit)
        self.assertEqual(len(parts), 3)
        sofa = np.c_[rng.uniform(0, 2.2, 1200), rng.uniform(0, 0.9, 1200), rng.uniform(0, 0.9, 1200)]
        self.assertEqual(len(oc.split_if_too_wide("sofa", sofa, fit)), 1, "a sofa is not too wide: never split")
        long = np.c_[rng.uniform(0, 3.4, 2000), rng.uniform(0, 0.9, 2000), rng.uniform(0, 0.9, 2000)]
        gap = long[(long[:, 0] < 1.5) | (long[:, 0] > 1.9)]          # two 1.5 m pieces: no chair is that wide
        self.assertEqual(len(oc.split_if_too_wide("chair", gap, fit)), 1, "pieces without a real chair size: kept whole")


class People(unittest.TestCase):
    def test_people_inside(self):
        self.assertIsNone(oc.people_inside_problem("sofa", 120, 136), "sitting on a seat is expected")
        self.assertIsNone(oc.people_inside_problem("table", 14, 142))
        self.assertIn("42 of 136", oc.people_inside_problem("table", 42, 136))


if __name__ == "__main__":
    unittest.main()

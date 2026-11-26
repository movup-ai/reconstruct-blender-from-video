from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'pipeline')]

import object_checks as oc

NAMES = ["wall", "floor", "cabinet", "person", "door", "chair", "table", "swivel chair", "desk", "sofa"]


def box(w, h):
    return {"size": [w, w, h]}


class Agreement(unittest.TestCase):
    def test_types_map_to_mask2former_classes(self):
        self.assertEqual(oc.role("coffee_table"), "table")
        self.assertEqual(oc.role("couch"), "sofa")
        self.assertEqual(oc.role("fridge"), "fridge")
        self.assertIsNone(oc.role("spaceship_console"))
        ids = oc.accepted_ids("office_chair", NAMES)
        self.assertEqual(sorted(NAMES[i] for i in ids), ["chair", "swivel chair"], "related classes count")
        self.assertIsNone(oc.accepted_ids("bush", NAMES), "a type with no class is not checked")
        self.assertEqual(oc.accepted_ids("door", NAMES, {"labels": ["cabinet"]}), [2], "objects.json labels win")

    def test_a_cabinet_called_door_disagrees(self):
        labels = np.full((20, 20), 2)                  # all cabinet
        labels[:, :5] = 3                              # a person in front: not counted
        mask = np.ones((40, 40), bool)
        share, top = oc.agreement(mask, labels, oc.accepted_ids("door", NAMES), NAMES)
        self.assertEqual((share, top), (0.0, "cabinet"))
        labels[:, 10:] = 4                             # most of it is a door
        share, _ = oc.agreement(mask, labels, oc.accepted_ids("door", NAMES), NAMES)
        self.assertGreater(share, oc.AGREE_MIN)


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

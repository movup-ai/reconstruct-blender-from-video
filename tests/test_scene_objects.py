# -*- coding: utf-8 -*-
"""The objects step of the run: the object library, the depth scale, placing a box, duplicates, and Mask2Former's
name maps for the depth step's surfaces.  No model, no Blender: synthetic cameras and points."""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, ROOT)
import assets_lib  # noqa: E402
import scene_objects as BG  # noqa: E402
from camera_model import Cam  # noqa: E402

UNIT_BOX = {"v": [[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (0, 1)], "f": [[0, 1, 3, 2]]}


def scene(cam=(0, -5, 1.6, 90, 0, 0), W=640, H=360):
    """a camera 5 m in front of the origin (on -y), 1.6 m up, looking along +y"""
    return {"res": [W, H], "lens": 36.0, "sensor": 36.0, "cam": [list(cam)], "zoom": [1.0]}


class Library(unittest.TestCase):
    def test_catalog_types_roles_and_hanging(self):
        with tempfile.TemporaryDirectory() as d:
            for fn in ("Coffee Table.blend", "pillar.blend", "painting.blend", "floor.blend", "notes.txt"):
                open(os.path.join(d, fn), "wb").close()
            with open(os.path.join(d, "objects.json"), "w", encoding="utf-8") as fh:
                json.dump({"pillar": {"role": "bg", "on_floor": True}, "painting": {"on_floor": True}}, fh)
            cat = assets_lib.catalog({"object_assets": d})
            self.assertEqual(sorted(cat), ["coffee_table", "painting", "pillar"])        # the floor is the run's own disc
            self.assertEqual(cat["pillar"]["role"], "bg")
            self.assertIsNone(cat["coffee_table"]["role"])
            self.assertTrue(cat["painting"]["on_floor"])                                 # objects.json wins
        self.assertFalse(assets_lib.on_floor("wall_mirror"))
        self.assertTrue(assets_lib.on_floor("sofa"))
        self.assertEqual(assets_lib.catalog({"object_assets": os.path.join(ROOT, "no_such_folder")}), {})

    def test_normalise_and_place(self):
        raw = {"v": [[1, 2, 3], [3, 3, 3], [1, 2, 5], [3, 3, 5]], "f": [[0, 1, 3, 2]]}
        m = assets_lib.normalise(raw)
        V = np.asarray(m["v"])
        self.assertEqual(m["size"], [2.0, 1.0, 2.0])
        np.testing.assert_allclose(V.min(0), [-.5, -.5, 0]); np.testing.assert_allclose(V.max(0), [.5, .5, 1])
        P = assets_lib.place(m, [10, 0, 0], math.pi / 2, [4, 2, 3])       # turned 90 degrees: x size along world y
        np.testing.assert_allclose(P.min(0), [9, -2, 0], atol=1e-9); np.testing.assert_allclose(P.max(0), [11, 2, 3], atol=1e-9)
        with self.assertRaises(ValueError):
            assets_lib.normalise({"v": [], "f": []})


class Geometry(unittest.TestCase):
    def test_floor_scale_recovers_the_depth_factor(self):
        cam = Cam(scene())
        h, w = 90, 160
        vv, uu = np.mgrid[0:h, 0:w]
        C, D = BG.pixel_dirs(cam, 0, (uu + .5) * cam.W / w, (vv + .5) * cam.H / h)
        zt = np.where(D[..., 2] < -1e-4, -C[2] / np.minimum(D[..., 2], -1e-4), 0)
        floor = (D[..., 2] < -1e-3) & (zt < 50)
        depth = np.where(floor, zt / 2.0, 1.0)                              # the depth model says half the distance
        s, n = BG.floor_scale(cam, 0, depth, floor)
        self.assertAlmostEqual(s, 2.0, places=3); self.assertGreater(n, 300)
        self.assertIsNone(BG.floor_scale(cam, 0, depth, np.zeros_like(floor)))

    def test_object_points_land_where_the_depth_says(self):
        cam = Cam(scene())
        h, w = 36, 64
        depth = np.full((h, w), 5.0, np.float32)                            # a flat wall 5 m away: the plane y = 0
        mask = np.zeros((cam.H, cam.W), bool); mask[100:260, 200:440] = True
        P = BG.object_points(cam, 0, depth, np.ones((h, w), bool), mask, 1.0)
        self.assertGreater(len(P), 50)
        np.testing.assert_allclose(P[:, 1], 0.0, atol=1e-6)
        self.assertTrue(np.all(np.abs(P[:, 0]) < 3) and np.all((P[:, 2] > 0) & (P[:, 2] < 3.5)))

    def front(self, w=1.0, h=1.0, y=0.0, z0=0.0, n=3000):
        rng = np.random.default_rng(1)
        return np.c_[rng.uniform(-w / 2, w / 2, n), np.full(n, y) + rng.normal(0, .005, n), rng.uniform(z0, z0 + h, n)]

    def test_box_from_the_visible_side_faces_the_camera(self):
        model = {"size": [1.0, 0.5, 1.0]}                       # half as deep as wide
        b = BG.fit_box(self.front(), [[0, -5, 1.6]], model)
        self.assertTrue(b["grounded"])
        self.assertAlmostEqual(b["size"][0], 1.0, delta=0.04)    # the whole width (a 1 m wide front)
        self.assertAlmostEqual(b["size"][2], 1.0, delta=0.04)
        P = np.r_[self.front(), [[0, 30, 9], [8, 0, 0.5]] * 40]    # stray depth points far off: left out
        self.assertAlmostEqual(BG.fit_box(P, [[0, -5, 1.6]], model)["size"][0], 1.0, delta=0.04)
        self.assertAlmostEqual(b["size"][1], b["size"][0] * 0.5, delta=0.05)
        self.assertGreater(b["centre"][1], 0.1)                  # pushed back from its front, away from the camera
        front = (math.sin(b["yaw"]), -math.cos(b["yaw"]))       # where the model's -Y points after the turn
        self.assertLess(front[1], -0.99)                         # toward the camera (on -y)

    def test_a_flat_thing_keeps_its_own_thinness(self):
        P = self.front(w=2.0, h=1.0)
        P[:, 1] += np.random.default_rng(2).uniform(0, 0.8, len(P))   # depth bleeding behind a window: noise
        window = BG.fit_box(P, [[0, -5, 1.6]], {"size": [1.0, 0.05, 1.0]}, stands=False)
        self.assertAlmostEqual(window["size"][1], window["size"][0] * 0.05, places=3)   # not 0.8 m deep
        box = BG.fit_box(P, [[0, -5, 1.6]], {"size": [1.0, 1.0, 1.0]})
        self.assertGreater(box["size"][1], 0.6)                       # a solid thing keeps what it measured

    def test_standing_types_reach_the_floor_hanging_ones_do_not(self):
        model = {"size": [1.0, 0.1, 1.0]}
        P = self.front(h=0.5, z0=0.6)                            # its bottom 0.6 m up: hidden behind people
        self.assertEqual(BG.fit_box(P, [[0, -5, 1.6]], model, stands=True)["centre"][2], 0.0)
        hang = BG.fit_box(P, [[0, -5, 1.6]], model, stands=False)
        self.assertFalse(hang["grounded"]); self.assertAlmostEqual(hang["centre"][2], 0.6, delta=0.05)
        high = BG.fit_box(self.front(h=0.4, z0=1.6), [[0, -5, 1.6]], model, stands=True)
        self.assertFalse(high["grounded"])                       # all of it well above the floor: on a shelf
        self.assertIsNone(BG.fit_box(self.front(h=0.02, z0=-0.5), [[0, -5, 1.6]], model))   # under the floor
        self.assertIsNone(BG.fit_box(self.front(n=20), [[0, -5, 1.6]], model))             # too few points

    def test_duplicates_compare_the_seen_heights(self):
        m = {"size": [1.0, 1.0, 1.0]}
        counter = BG.fit_box(self.front(h=0.3, z0=0.8), [[0, -5, 1.6]], m)       # seen 0.8-1.1 m, box to the floor
        stool = BG.fit_box(self.front(w=0.4, h=0.3, z0=0.2), [[0, -5, 1.6]], m)  # seen 0.2-0.5 m
        fo, zo = BG.overlap(counter, stool)
        self.assertGreater(fo, 0.5); self.assertEqual(zo, 0.0)                   # same spot, different heights: two things
        fo, zo = BG.overlap(counter, counter)
        self.assertGreater(fo, 0.99); self.assertGreater(zo, 0.99)


class Mask2FormerFinder(unittest.TestCase):
    """the run finds objects and surfaces with Mask2Former's 150 names (ade_find.py, ade_objects.py)"""

    def test_every_one_of_the_150_names_has_a_place_in_the_table(self):
        import ade_objects
        with open(os.path.join(ROOT, "models", "mask2former_ade_panoptic", "config.json"), encoding="utf-8") as fh:
            names = {v.strip() for v in json.load(fh)["id2label"].values()}         # the same 150 ADE20K names
        self.assertEqual(len(names), 150)
        self.assertEqual(set(ade_objects.ASSET), names)
        self.assertIsNone(ade_objects.asset_of("person"))                           # people: their own step
        self.assertIsNone(ade_objects.asset_of("sky"))                              # scenery: not placed
        self.assertEqual(ade_objects.asset_of("windowpane"), "window")
        self.assertEqual(ade_objects.asset_of("television receiver"), "tv")
        self.assertEqual(ade_objects.asset_of("not a name"), None)

    def test_library_holds_only_models_the_table_uses(self):
        import ade_objects
        have = set(assets_lib.catalog({}))
        self.assertTrue(have <= set(ade_objects.wanted()), sorted(have - set(ade_objects.wanted())))

    def test_samples_cover_each_shot_with_its_last_frame(self):
        import ade_find
        f = ade_find.sample_frames([(0, 20), (20, 31)], 25.0, {"bg_detect_fps": 5})
        self.assertEqual(f, [0, 5, 10, 15, 19, 20, 25, 30])

    def test_an_object_keeps_its_id_from_frame_to_frame(self):
        import ade_find
        def sq(x, y, s=10):
            m = np.zeros((60, 60), bool); m[y:y + s, x:x + s] = True; return m
        active, nid = {}, 1
        out, nid = ade_find.link(active, [(0, sq(5, 5), .9), (0, sq(40, 40), .8), (1, sq(20, 5), .7)], .3, nid)
        ids = {(t, tuple(np.argwhere(m)[0])): i for t, i, _, m in out}
        self.assertEqual(len(set(i for _, i, _, _ in out)), 3)
        out2, nid = ade_find.link(active, [(0, sq(6, 6), .9), (1, sq(41, 41), .6)], .3, nid)   # moved a little
        self.assertEqual(out2[0][1], ids[(0, (5, 5))], "the same chair keeps its id")
        self.assertNotEqual(out2[1][1], ids[(0, (40, 40))], "another type never takes a chair's id")
        self.assertEqual(nid, 5)
        out3, nid = ade_find.link(active, [(0, sq(40, 40), .8)], .3, nid)                    # seen again later
        self.assertEqual(out3[0][1], ids[(0, (40, 40))], "an object hidden for a while keeps its id")


class Surfaces(unittest.TestCase):
    """the depth step names its pixels from the finder's name maps (no second segmentation model)"""

    def test_labels_at_scales_a_frames_name_map(self):
        import ade_find
        lab = np.zeros((2, 4, 6), np.int16); lab[0, :, 3:] = 5; lab[1] = 7
        bundle = {"label_frames": np.asarray([10, 20]), "labels": lab}
        got = ade_find.labels_at(bundle, 10, 12, 8)
        self.assertEqual(got.shape, (8, 12)); self.assertEqual(got[0, 0], 0); self.assertEqual(got[0, -1], 5)
        self.assertTrue((ade_find.labels_at(bundle, 20, 3, 2) == 7).all())
        self.assertIsNone(ade_find.labels_at(bundle, 15, 12, 8))           # not looked at

    def test_floor_names_are_ade_names(self):
        with open(os.path.join(ROOT, "models", "mask2former_ade_panoptic", "config.json"), encoding="utf-8") as fh:
            names = {v.strip() for v in json.load(fh)["id2label"].values()}
        import metric_scene
        self.assertTrue(BG.FLOOR_NAMES <= names, sorted(BG.FLOOR_NAMES - names))
        self.assertTrue((metric_scene.FLOOR - {"ground"}) <= names)
        self.assertTrue(metric_scene.WALL <= names and "ceiling" in names)

    def test_the_depth_views_are_among_the_looked_at_frames(self):
        import ade_find, metric_scene
        shots, c = [(0, 40), (40, 97)], {"metric_max_views": 3}
        sample = set(ade_find.sample_frames(shots, 25.0, c))
        for a, b in shots:
            sample |= {a + int(f) for f in metric_scene.view_frames(b - a, c)}
        for a, b in shots:
            self.assertTrue({a + int(f) for f in metric_scene.view_frames(b - a, c)} <= sample)
        self.assertEqual(list(metric_scene.view_frames(40, c)), [0, 19, 39])


if __name__ == "__main__":
    unittest.main()

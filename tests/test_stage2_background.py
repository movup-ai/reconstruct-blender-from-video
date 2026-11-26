# -*- coding: utf-8 -*-
"""Stage 2 (Background): the object library, the depth scale, placing a box, duplicates and the clip status.
No model, no Blender: synthetic cameras and points."""
import json
import math
import os
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, ROOT)
import assets_lib  # noqa: E402
import background as BG  # noqa: E402
from camera_model import Cam  # noqa: E402

UNIT_BOX = {"v": [[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (0, 1)], "f": [[0, 1, 3, 2]]}


def scene(cam=(0, -5, 1.6, 90, 0, 0), W=640, H=360):
    """a camera 5 m in front of the origin (on -y), 1.6 m up, looking along +y"""
    return {"res": [W, H], "lens": 36.0, "sensor": 36.0, "cam": [list(cam)], "zoom": [1.0]}


class Library(unittest.TestCase):
    def test_catalog_types_prompts_and_hanging(self):
        with tempfile.TemporaryDirectory() as d:
            for fn in ("Coffee Table.blend", "pillar.blend", "painting.blend", "floor.blend", "notes.txt"):
                open(os.path.join(d, fn), "wb").close()
            with open(os.path.join(d, "objects.json"), "w", encoding="utf-8") as fh:
                json.dump({"pillar": {"prompt": "column", "role": "bg", "on_floor": True}, "painting": {"on_floor": True}}, fh)
            cat = assets_lib.catalog({"object_assets": d})
            self.assertEqual(sorted(cat), ["coffee_table", "painting", "pillar"])        # the floor is Stage 1's
            self.assertEqual(cat["coffee_table"]["prompt"], "coffee table")
            self.assertEqual((cat["pillar"]["prompt"], cat["pillar"]["role"]), ("column", "bg"))
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


class StagesRunOnTheirOwn(unittest.TestCase):
    """Movement and Background: each has its own Start / Pause, its own jobs and its own status"""

    def setUp(self):
        from agent.jobs import JobStore, Runner
        import types
        self.d = tempfile.mkdtemp()
        self.store = JobStore(os.path.join(self.d, "jobs.db"))
        self.runner = Runner(types.SimpleNamespace(store=self.store))

    def tearDown(self):
        self.store.db.close()

    def test_take_runs_only_the_started_stages(self):
        self.store.enqueue("a", "movement"); self.store.enqueue("b", "background")
        self.assertIsNone(self.store.take(stages=set()))
        self.assertEqual(self.store.take(stages={"background"})["clip"], "b")
        self.assertIsNone(self.store.take(stages={"background"}))
        self.assertEqual(self.store.take(stages={"movement"})["clip"], "a")
        self.assertEqual(self.store.counts_by_stage(), {"movement": {"running": 1}, "background": {"running": 1}})

    def test_pause_and_resume_one_stage(self):
        r = self.runner
        self.assertTrue(r.is_paused() and r.is_paused("movement") and r.is_paused("background"))
        r.resume("background")
        self.assertFalse(r.is_paused("background")); self.assertTrue(r.is_paused("movement")); self.assertFalse(r.is_paused())
        r.resume("movement"); r.pause("background")
        self.assertEqual(r.active_stages, {"movement"})
        r.pause()
        self.assertTrue(r.is_paused())
        r.active = True                                      # all stages at once
        self.assertEqual(r.active_stages, {"movement", "background"})

    def test_cancel_queued_of_one_stage(self):
        self.store.enqueue("a", "movement"); self.store.enqueue("b", "background")
        self.assertEqual(self.store.cancel_queued(stage="movement"), 1)
        self.assertEqual(self.store.active_clips("background"), {"b"})
        self.assertEqual(self.store.active_clips("movement"), set())

    def test_each_stage_has_its_own_status(self):
        from agent import clips as clipsmod
        import types
        jid = self.store.enqueue("a", "movement"); self.store.take(); self.store.finish(jid, True)
        jid = self.store.enqueue("a", "background"); self.store.take()
        stages = self.store.latest_stages()["a"]
        self.assertEqual({k: v["state"] for k, v in stages.items()}, {"movement": "done", "background": "running"})
        with tempfile.TemporaryDirectory() as d:
            c = {"work_root": d, "submit_root": d, "source_root": d, "marks_root": d}
            cl = types.SimpleNamespace(id="a", name="film_a", video=os.path.join(d, "film", "a.mp4"))
            st = clipsmod.status(c, cl, self.store.latest()["a"], stages)
            self.assertEqual(st["label"], "new")                       # Movement: its own job (done, but no zip here)
            self.assertEqual(st["background"]["state"], "processing")  # Background runs: shown in its own status
            self.assertEqual(st["job"]["stage"], "background")         # the clip's latest job (Cancel acts on it)


class Status(unittest.TestCase):
    def test_background_status_follows_its_zip_and_job(self):
        from agent.clips import background_status
        with tempfile.TemporaryDirectory() as d:
            f = {"zip": os.path.join(d, "a.zip"), "bg_zip": os.path.join(d, "bg.zip")}
            self.assertEqual(background_status(f, None, False)["state"], "none")
            self.assertEqual(background_status(f, {"stage": "background", "state": "running"}, True)["state"], "processing")
            self.assertEqual(background_status(f, {"stage": "movement", "state": "running"}, True)["state"], "none")
            failed = background_status(f, {"stage": "background", "state": "failed", "error": "no models"}, True)
            self.assertEqual((failed["state"], failed["error"]), ("failed", "no models"))
            open(f["zip"], "w").close(); open(f["bg_zip"], "w").close()
            self.assertEqual(background_status(f, None, True)["state"], "done")
            later = time.time() + 5
            os.utime(f["zip"], (later, later))                                     # Stage 1 made again after it
            self.assertEqual(background_status(f, None, True)["state"], "outdated")
            self.assertEqual(background_status(f, None, False)["state"], "outdated")


if __name__ == "__main__":
    unittest.main()

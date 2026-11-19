# -*- coding: utf-8 -*-
"""Blender Agent tests:   .venv\\Scripts\\python -m unittest discover -s tests -v"""
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config  # noqa: E402
from agent import clips as clipsmod, review, jobs  # noqa: E402
import run_pipeline as rp  # noqa: E402  (pipeline, on sys.path via agent.config)


def make_video(path, n=40, cut=None, size=(320, 180)):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, size)
    for i in range(n):
        img = np.full((size[1], size[0], 3), (40, 120, 200) if (cut is None or i < cut) else (200, 60, 30), np.uint8)
        cv2.rectangle(img, (20 + i, 40), (60 + i, 140), (255, 255, 255), -1)
        vw.write(img)
    vw.release()


class TestNaming(unittest.TestCase):
    def test_real_source_names_unique(self):
        c = config.load()
        if not os.path.isdir(c["source_root"]):
            self.skipTest("no source folder")
        lib = clipsmod.Library(c)
        names = [cl.name for cl in lib.clips]
        ids = [cl.id for cl in lib.clips]
        self.assertGreater(len(names), 0)
        self.assertEqual(len(names), len(set(names)), "clip names must be unique (they become folder / zip names)")
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [cl.id for cl in clipsmod.Library(c).clips], "ids must be stable between scans")


class TestShots(unittest.TestCase):
    def test_cut_detected(self):
        frames = []
        for i in range(60):
            img = np.full((90, 160, 3), (30, 100, 180) if i < 25 else (190, 70, 20), np.uint8)
            img[10:40, i:i + 20] = 255
            frames.append(img)
        self.assertEqual(rp.split_shots(frames, 0, 90), [(0, 25), (25, 60)])

    def test_no_cut(self):
        frames = [np.full((90, 160, 3), (30 + i, 100, 180), np.uint8) for i in range(30)]
        self.assertEqual(rp.split_shots(frames, 0, 90), [(0, 30)])

    def test_slice_and_widen(self):
        pose = {"frames": 10, "people": [{"id": 1, "frames": {str(f): [[0, 0, 1]] * 17 for f in range(10)},
                                          "boxes": {str(f): [0, 0, 1, 1] for f in range(10)}}]}
        sp = rp.slice_pose(pose, 4, 8)
        self.assertEqual(sorted(int(f) for f in sp["people"][0]["frames"]), [0, 1, 2, 3])
        ch = {"root": [[i, 0, 0, 0] for i in range(4)], "pelvis_z": [0.9] * 4, "rot": {"head": [[0, 0, i] for i in range(4)]},
              "visible": [True] * 4}
        w = rp.widen(ch, 4, 8, 10)
        self.assertEqual(len(w["root"]), 10)
        self.assertEqual(w["visible"], [False] * 4 + [True] * 4 + [False] * 2)
        self.assertEqual(w["root"][5][0], 1)


class TestStitch(unittest.TestCase):
    @staticmethod
    def track(pid, f0, f1, x, w=60, h=180):
        kp = [[0, 0, 0.9]] * 17
        return {"id": pid, "frames": {str(f): kp for f in range(f0, f1)}, "boxes": {str(f): [x, 100, x + w, 100 + h] for f in range(f0, f1)}}

    def test_broken_track_is_joined(self):
        pose = {"people": [self.track(1, 0, 40, 500), self.track(2, 46, 90, 510), self.track(3, 0, 90, 100)]}
        m = rp.stitch_tracks(pose, [(0, 90)])
        self.assertEqual(m, {2: 1}, "P2 continues P1 in the same place")
        self.assertEqual(len(pose["people"]), 2)
        self.assertEqual(len(pose["people"][0]["boxes"]), 84)

    def test_duplicate_detection_is_joined(self):
        pose = {"people": [self.track(1, 0, 80, 500), self.track(2, 50, 120, 505)]}
        self.assertEqual(rp.stitch_tracks(pose, [(0, 120)]), {2: 1})

    def test_never_across_a_cut_or_far_apart(self):
        pose = {"people": [self.track(1, 0, 40, 500), self.track(2, 46, 90, 510)]}
        self.assertEqual(rp.stitch_tracks(pose, [(0, 43), (43, 90)]), {}, "a cut in between")
        pose = {"people": [self.track(1, 0, 40, 100), self.track(2, 46, 90, 900)]}
        self.assertEqual(rp.stitch_tracks(pose, [(0, 90)]), {}, "somewhere else in the picture")


class TestCamera(unittest.TestCase):
    def test_wandering_noise_is_locked(self):
        from scipy.spatial.transform import Rotation
        t = np.linspace(0, 2 * np.pi, 150)
        R = [Rotation.from_rotvec([0, np.radians(1.4) * (1 - np.cos(a)), 0]).as_matrix().tolist() for a in t]   # out 2.8 deg and back
        out = rp.steady_camera({"R": R, "max_rotation_deg": 2.8}, {})
        self.assertTrue(np.allclose(out["R"], np.eye(3)))

    def test_still_camera_is_locked(self):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(0)
        R = [Rotation.from_rotvec(rng.normal(0, 0.0005, 3)).as_matrix().tolist() for _ in range(40)]
        out = rp.steady_camera({"R": R, "max_rotation_deg": 0.1}, {})
        self.assertTrue(np.allclose(out["R"], np.eye(3)), "tiny tracking noise -> camera locked")

    def test_pan_kept_but_smoothed(self):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(1)
        ang = np.radians(np.linspace(0, 10, 60))                       # a real 10 degree pan
        noisy = [Rotation.from_rotvec([0, a, 0] + rng.normal(0, 0.003, 3)).as_matrix().tolist() for a in ang]
        out = rp.steady_camera({"R": noisy, "max_rotation_deg": 10}, {})
        rv = Rotation.from_matrix(np.array(out["R"])).as_rotvec()
        self.assertAlmostEqual(np.degrees(rv[-1][1]), 10, delta=1.0, msg="the pan is kept")
        jit_in = np.median(np.abs(np.diff(Rotation.from_matrix(np.array(noisy)).as_rotvec(), 2, axis=0)))
        jit_out = np.median(np.abs(np.diff(rv, 2, axis=0)))
        self.assertLess(jit_out, jit_in / 3, "frame-to-frame shake reduced")


class TestJobStore(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.s = jobs.JobStore(os.path.join(self.d, "j.db"))

    def tearDown(self):
        self.s.db.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_dedupe_priority_recover(self):
        self.assertIsNotNone(self.s.enqueue("a", "fit"))
        self.assertIsNone(self.s.enqueue("a", "fit"), "one job per clip at a time")
        self.assertIsNotNone(self.s.enqueue("b", "all"))
        j = self.s.take()
        self.assertEqual((j["clip"], j["stage"]), ("b", "all"), "renders go first")
        self.assertEqual(self.s.recover(), 1)
        self.assertEqual(self.s.take()["clip"], "b")

    def test_force_replaces_queued(self):
        self.s.enqueue("a", "fit")
        self.assertIsNotNone(self.s.enqueue("a", "all", force=True))
        j = self.s.take()
        self.assertEqual(j["stage"], "all")
        self.assertIsNone(self.s.take())

    def test_finish_and_latest(self):
        jid = self.s.enqueue("a", "fit")
        self.s.take()
        self.s.finish(jid, False, "boom")
        self.assertEqual(self.s.latest()["a"]["state"], "failed")
        self.assertEqual(self.s.latest()["a"]["error"], "boom")

    def test_one_ai_at_a_time(self):
        self.s.enqueue("a", "ai"); self.s.enqueue("b", "ai"); self.s.enqueue("c", "fit")
        self.assertEqual(self.s.take()["stage"], "fit", "preparation goes before AI")
        self.assertEqual(self.s.take()["clip"], "a")
        self.assertIsNone(self.s.take(), "second AI job waits")

    def test_bad_stage(self):
        with self.assertRaises(ValueError):
            self.s.enqueue("a", "nope")


class TestReviewValidation(unittest.TestCase):
    def test_good(self):
        d = review.validate({"confirmed": True, "people": {"1": {"role": "main", "height": 1.8}}, "camera": {"type": "pan", "lens": 35},
                             "environment": {"props": [{"name": "bag", "kind": "orange"}, {"name": ""}]}, "notes": "x"})
        self.assertTrue(d["confirmed"])
        self.assertEqual(d["people"]["1"]["colour"], "red")
        self.assertEqual(len(d["environment"]["props"]), 1)

    def test_bad(self):
        for bad in ({"people": {"1": {"role": "boss"}}}, {"people": {"1": {"role": "main", "height": 9}}}, {"camera": {"type": "drone"}}, []):
            with self.assertRaises(ValueError):
                review.validate(bad)

    def test_guess_camera(self):
        self.assertEqual(review.guess_camera({"max_rotation_deg": 0.1, "warnings": []}), "static")
        self.assertEqual(review.guess_camera({"max_rotation_deg": 5, "warnings": []}), "pan")
        self.assertEqual(review.guess_camera({"max_rotation_deg": 0, "warnings": ["camera seems to MOVE (parallax 3 px)"]}), "moving")


class TestStatus(unittest.TestCase):
    def test_labels(self):
        d = tempfile.mkdtemp()
        try:
            src = os.path.join(d, "src", "FilmA"); os.makedirs(src)
            make_video(os.path.join(src, "1-1.mp4"), n=10)
            c = dict(config.load(), source_root=os.path.join(d, "src"), work_root=os.path.join(d, "w"), submit_root=os.path.join(d, "s"),
                     marks_root=os.path.join(d, "m"))
            cl = clipsmod.Library(c).clips[0]
            self.assertEqual(clipsmod.status(c, cl)["label"], "not prepared")
            self.assertEqual(clipsmod.status(c, cl, {"stage": "fit", "state": "running"})["label"], "preparing")
            self.assertEqual(clipsmod.status(c, cl, {"stage": "all", "state": "failed", "error": "x"})["label"], "failed")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestProps(unittest.TestCase):
    def test_every_type_builds_on_the_floor(self):
        import props
        for t, ((w, h, d), _) in props.TYPES.items():
            parts = props.build(t, w, h, d)
            self.assertTrue(parts, t)
            zs = [v[2] for _, V, _, _ in parts for v in V]
            self.assertGreaterEqual(min(zs), -1e-6, t)
            self.assertAlmostEqual(max(zs), h, delta=h * 0.35 + 0.35, msg=t)

    def test_aliases_and_place(self):
        import props
        self.assertEqual(props.norm_type("Refrigerator"), "fridge")
        self.assertEqual(props.norm_type("spaceship"), "other")
        out = props.place(props.build("table", 1.2, 0.75, 0.8), (10.0, 5.0), 90, z0=1.0)
        xs = [v[0] for _, V, _, _ in out for v in V]; zs = [v[2] for _, V, _, _ in out for v in V]
        self.assertAlmostEqual((min(xs) + max(xs)) / 2, 10.0, places=2)
        self.assertAlmostEqual(min(zs), 1.0, places=3)


class TestZip(unittest.TestCase):
    def test_gbk_names_like_the_sample(self):
        import zipfile
        import package
        d = tempfile.mkdtemp()
        try:
            name = "人生一世_11月12日(1)"
            sub = os.path.join(d, name)
            for s in package.SUBDIRS:
                os.makedirs(os.path.join(sub, s))
            with open(os.path.join(sub, "meta", "meta.jsonl"), "w") as fh:
                fh.write("{}\n")
            zp = sub + ".zip"
            package.zip_gbk(sub, name, zp)
            with zipfile.ZipFile(zp) as z:
                infos = z.infolist()
            self.assertTrue(all(not i.flag_bits & 0x800 for i in infos), "GBK names, UTF-8 flag off")
            names = [i.filename.encode("cp437").decode("gbk") for i in infos]
            self.assertIn(name + "/meta/meta.jsonl", names)
            self.assertIn(name + "/comparison/", names)
            self.assertNotIn(name + "/", names, "no root folder entry (as the sample)")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestConfig(unittest.TestCase):
    def test_rejects_bad_values(self):
        for bad in ({"parallel": 0}, {"autopilot": "sometimes"}, {"unknown": 1}, {"work_root": "relative\\path"},
                    {"deepseek_key": "short"}, {"deepseek_key": "sk-has a space in it 1234567890"}):
            with self.assertRaises(ValueError):
                config.save(bad)

    def test_ai_status_without_key(self):
        from agent import ai
        c = dict(config.load(), deepseek_key="", ai_enabled=True)
        if ai.api_key(c):
            self.skipTest("a key is configured on this PC")
        self.assertFalse(ai.status(c)["ready"])
        self.assertFalse(ai.status(dict(c, ai_enabled=False))["ready"])


class FakeCam:
    W, H, f = 1000, 500, 1000.0

    def norm(self, x, y):
        return np.array([(x - self.W / 2) / self.f, (y - self.H / 2) / self.f])


class FakeWorld:
    """camera 1.6 m high, level, looking along +Y"""
    cam = FakeCam()
    C = np.array([0.0, 0.0, 1.6])

    def R_cw(self, k):
        return np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])     # world -> camera (x right, y down, z forward)


class TestEnvCorrections(unittest.TestCase):
    def test_drop_and_add(self):
        import scene_depth as sd
        meshes = [{"name": "wall_a"}, {"name": "blk_1"}, {"name": "blk_2"}]
        regions = [{"id": "W0", "bbox": [0, 0, 100, 200], "meshes": ["wall_a"]},
                   {"id": "B0", "bbox": [400, 0, 600, 100], "meshes": ["blk_1", "blk_2"]}]
        env = {"drop": [{"id": "B0", "what": "sky", "bbox": [405, 0, 600, 95]}],
               "objects": [{"what": "parked car", "kind": "car", "bbox": [450, 230, 550, 300]}]}
        plate = np.full((500, 1000, 3), 120, np.uint8)
        info, pal = {}, {}
        out, added = sd.apply_env_ai(meshes, regions, env, plate, FakeWorld(), 0, pal, None, info)
        names = [m["name"] for m in out]
        self.assertIn("wall_a", names)
        self.assertNotIn("blk_1", names, "region the AI called sky is removed")
        self.assertEqual(len(added), 1)
        # bottom edge 50 px below the centre with f=1000 and a 1.6 m camera -> 32 m away; 70 px tall -> 2.24 m high
        self.assertAlmostEqual(added[0]["distance_m"], 32.0, delta=0.5)
        self.assertAlmostEqual(added[0]["size_m"][2], 2.24, delta=0.1)
        self.assertEqual(added[0]["kind"], "car")
        car = [m for m in out if m["name"].startswith("obj_car")]
        self.assertTrue(any("wheel" in m["name"] for m in car), "a car is built with wheels")
        self.assertAlmostEqual(min(v[2] for m in car for v in m["v"]), 0.0, places=3, msg="stands on the ground")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class TestApi(unittest.TestCase):
    """real server on a temporary library with one synthetic clip"""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        src = os.path.join(cls.d, "src", "电影Film")
        os.makedirs(src)
        make_video(os.path.join(src, "11月12日(1).mp4"), n=30, cut=15)
        cls._data = config.DATA
        config.DATA = os.path.join(cls.d, "data")
        from agent.server import App, Server, make_handler
        c = dict(config.load(), source_root=os.path.join(cls.d, "src"), work_root=os.path.join(cls.d, "w"),
                 submit_root=os.path.join(cls.d, "s"), marks_root=os.path.join(cls.d, "m"), autopilot="off", ai_enabled=False)
        cls.app = App(c)
        cls.port = free_port()
        cls.app.server = Server(("127.0.0.1", cls.port), make_handler(cls.app))
        threading.Thread(target=cls.app.server.serve_forever, daemon=True).start()
        cls.cid = cls.app.lib.clips[0].id

    @classmethod
    def tearDownClass(cls):
        cls.app.server.shutdown()
        cls.app.store.db.close()
        config.DATA = cls._data
        shutil.rmtree(cls.d, ignore_errors=True)

    def get(self, path, headers=None):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def post(self, path, body=None):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=json.dumps(body or {}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_new_pages_and_endpoints(self):
        for p in ("/submissions", "/settings", "/help", "/favicon.svg"):
            self.assertEqual(self.get(p)[0], 200, p)
        code, body, _ = self.get("/api/submissions")
        self.assertEqual((code, json.loads(body)), (200, []))
        self.assertEqual(self.post("/api/open", {"path": "C:\\Windows"})[0], 400, "only the app's own folders")
        self.assertEqual(self.get("/download/nothing")[0], 404)
        from agent.server import advice
        self.assertIn("Settings", advice("Blender not found"))

    def test_pages_and_health(self):
        for p in ("/", "/clip/" + self.cid, "/static/app.css", "/static/common.js"):
            self.assertEqual(self.get(p)[0], 200, p)
        code, body, _ = self.get("/api/health")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["app"], "Blender Agent")

    def test_clips_and_info(self):
        code, body, _ = self.get("/api/clips")
        d = json.loads(body)
        self.assertEqual(len(d["clips"]), 1)
        self.assertEqual(d["clips"][0]["name"], "电影_11月12日(1)")
        self.assertEqual(d["clips"][0]["status"]["label"], "not prepared")
        code, body, _ = self.get("/api/clip/" + self.cid)
        self.assertEqual(code, 200)
        self.assertIsNone(json.loads(body)["review"])

    def test_nothing_runs_before_start(self):
        q = json.loads(self.get("/api/clips")[1])["queue"]
        self.assertTrue(q["paused"], "the app starts paused")
        try:
            code, d = self.post("/api/start")
            self.assertEqual((code, d["queued"]), (200, 1))
            self.assertFalse(json.loads(self.get("/api/clips")[1])["queue"]["paused"])
            self.assertEqual(self.post("/api/start")[1]["queued"], 0, "a queued clip is not queued twice")
            self.post("/api/pause")
            self.assertTrue(json.loads(self.get("/api/clips")[1])["queue"]["paused"])
        finally:
            self.app.runner.pause()
            self.post("/api/cancel_queued")

    def test_remove_from_list_and_put_back(self):
        code, d = self.post("/api/clip/%s/remove" % self.cid)
        self.assertEqual(code, 200)
        try:
            lst = json.loads(self.get("/api/clips")[1])
            self.assertEqual(lst["clips"], [])
            self.assertEqual(len(lst["removed"]), 1)
            self.assertTrue(os.path.exists(lst["removed"][0]["path"]), "the video file stays")
            self.assertEqual(self.post("/api/restore", {"path": "C:\\not\\removed.mp4"})[0], 400)
        finally:
            code, d = self.post("/api/restore", {"path": lst["removed"][0]["path"]})
        self.assertEqual((code, d["clips"]), (200, 1))
        self.assertEqual(json.loads(self.get("/api/clips")[1])["clips"][0]["id"], self.cid)

    def test_zip_downloads_under_its_package_name(self):
        import package as pkg
        os.makedirs(self.app.c["submit_root"], exist_ok=True)
        zp = os.path.join(self.app.c["submit_root"], "clipX.zip")
        with open(zp, "wb") as fh:
            fh.write(b"PK\x05\x06" + b"\0" * 18)
        try:
            with unittest.mock.patch.object(pkg, "package_names", return_value={"clipX": "film-7"}):
                code, body, h = self.get("/download/clipX")
            self.assertEqual(code, 200)
            self.assertIn('filename="film-7.zip"', h["Content-Disposition"])
        finally:
            os.remove(zp)

    def test_media(self):
        self.assertEqual(self.get("/media/clipthumb/%s.jpg" % self.cid)[0], 200)
        code, body, h = self.get("/media/video/%s.mp4" % self.cid, {"Range": "bytes=0-99"})
        self.assertEqual(code, 206)
        self.assertEqual(len(body), 100)
        self.assertEqual(self.get("/media/blocks/%s/..%%5Cconfig.json" % self.cid)[0], 404, "no path traversal")

    def test_errors(self):
        self.assertEqual(self.get("/api/clip/nope")[0], 404)
        self.assertEqual(self.post("/api/clip/%s/review" % self.cid, {"people": {"1": {"role": "boss"}}})[0], 400)
        self.assertEqual(self.post("/api/clip/%s/run?stage=all" % self.cid)[0], 409, "cannot render before confirming")
        self.assertEqual(self.post("/api/settings", {"parallel": 99})[0], 400)

    def upload(self, film, name, data):
        req = urllib.request.Request("http://127.0.0.1:%d/api/upload?film=%s&name=%s" % (
            self.port, urllib.request.quote(film), urllib.request.quote(name)), data=data, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_upload(self):
        p = os.path.join(self.d, "up.mp4")
        make_video(p, n=12)
        with open(p, "rb") as fh:
            data = fh.read()
        code, d = self.upload("新片Test", "clip A.mp4", data)
        self.assertEqual(code, 200, d)
        self.assertTrue(os.path.exists(os.path.join(self.d, "src", "新片Test", "clip A.mp4")))
        self.assertEqual(d["clip"]["film"], "新片Test")
        code, d = self.upload("新片Test", "clip A.mp4", data)
        self.assertEqual(d["file"], "clip A (2).mp4", "never overwrites an existing clip")
        self.assertEqual(self.upload("新片Test", "notes.txt", b"x")[0], 400)
        self.assertEqual(self.upload("..\\evil", "a.mp4", data)[0], 400)
        self.assertEqual(self.upload("", "a.mp4", data)[0], 400)

    def test_objects_endpoints(self):
        code, body, _ = self.get("/api/clip/%s/objects" % self.cid)
        d = json.loads(body)
        self.assertEqual(code, 200)
        self.assertIn("door", d["types"])
        self.assertEqual(d["shots"], [], "not prepared yet: no shots to draw on")
        self.assertEqual(self.post("/api/clip/%s/objects" % self.cid, {"shots": {"_s9_000": {"objects": []}}})[0], 400)

    def test_key_never_returned(self):
        code, body, _ = self.get("/api/settings")
        d = json.loads(body)
        self.assertNotIn("deepseek_key", d)
        self.assertIn("deepseek_key_set", d)

    def test_review_roundtrip(self):
        code, d = self.post("/api/clip/%s/review" % self.cid, {"confirmed": False, "people": {}, "camera": {"type": "static"}, "notes": "test"})
        self.assertEqual(code, 200)
        self.assertFalse(d["queued"])
        code, body, _ = self.get("/api/clip/" + self.cid)
        self.assertEqual(json.loads(body)["review"]["notes"], "test")


if __name__ == "__main__":
    unittest.main()

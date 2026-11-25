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

    def test_timing_from_start_to_package(self):
        def row(stage, state, started, ended):
            self.s.db.execute("INSERT INTO jobs(clip, stage, state, priority, created, started, ended) VALUES ('a',?,?,0,?,?,?)",
                              (stage, state, started, started, ended))
        row("fit", "done", 100, 400)                  # prepare: 300 s
        row("all", "done", 1000, 1200)                # review in between, then Finish: 200 s
        t = self.s.timing(now=5000)["a"]
        self.assertEqual(t["done"], {"start": 100, "end": 1200, "wall": 1100, "machine": 500})
        self.assertIsNone(t["open"])
        row("fit", "running", 2000, None)             # prepared again after the package: a new run, still going
        t = self.s.timing(now=2600)["a"]
        self.assertEqual(t["done"]["wall"], 1100, "the last finished run stays")
        self.assertEqual(t["open"], {"start": 2000, "so_far": 600, "machine": 600})
        self.assertEqual(t["last"], {"stage": "fit", "state": "running", "ran": 600}, "the card shows this job's own time")

    def test_dedupe_priority_recover(self):
        self.assertIsNotNone(self.s.enqueue("a", "fit", user="u"))
        self.assertIsNone(self.s.enqueue("a", "fit", user="u"), "one job per clip at a time")
        self.assertIsNotNone(self.s.enqueue("b", "all", user="u"))
        j = self.s.take()
        self.assertEqual((j["clip"], j["stage"], j["user"]), ("b", "all", "u"), "renders go first")
        self.assertEqual(self.s.recover(), 1)
        self.assertEqual(self.s.take()["clip"], "b")

    def test_members_jobs(self):
        """only the members who pressed Start get jobs; jobs from before members existed are dropped on start"""
        self.s.enqueue("old", "fit")                                    # no member
        self.s.enqueue("a", "fit", user="anna"); self.s.enqueue("b", "all", user="ben")
        self.assertIsNone(self.s.take(set()), "nobody pressed Start")
        self.assertEqual(self.s.take({"anna"})["clip"], "a", "Ben's render waits for Ben")
        self.s.recover()
        self.assertEqual(self.s.db.execute("SELECT state FROM jobs WHERE clip='old'").fetchone()[0], "cancelled")
        self.assertEqual(self.s.counts("ben"), {"queued": 1})
        self.assertEqual(self.s.cancel_queued(user="anna"), 1, "Anna's own waiting jobs only (the running one came back)")
        self.assertEqual(self.s.counts("ben"), {"queued": 1})

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
            name = "\u4eba\u751f\u4e00\u4e16_11\u670812\u65e5(1)"
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


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class TestApi(unittest.TestCase):
    """real server on a temporary library with one synthetic clip, used by a logged-in member"""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        src = os.path.join(cls.d, "src", "\u7535\u5f71Film")
        os.makedirs(src)
        make_video(os.path.join(src, "11\u670812\u65e5(1).mp4"), n=30, cut=15)
        cls._data = config.DATA
        config.DATA = os.path.join(cls.d, "data")
        from agent.server import App, Server, make_handler
        c = dict(config.load(), work_root=os.path.join(cls.d, "w"),
                 submit_root=os.path.join(cls.d, "s"), marks_root=os.path.join(cls.d, "m"), autopilot="off", ai_enabled=False)
        cls.app = App(c)
        cls.port = free_port()
        cls.app.server = Server(("127.0.0.1", cls.port), make_handler(cls.app))
        threading.Thread(target=cls.app.server.serve_forever, daemon=True).start()
        cls.cookie = None
        st, d = cls.post(cls, "/api/login", {"name": "Tester"})
        assert st == 200, d
        st, d = cls.post(cls, "/api/use_folder", {"path": os.path.join(cls.d, "src")})
        assert st == 200, d
        cls.ws = cls.app.ws("tester")
        cls.cid = cls.ws.lib.clips[0].id

    @classmethod
    def tearDownClass(cls):
        cls.app.server.shutdown()
        cls.app.store.db.close()
        config.DATA = cls._data
        shutil.rmtree(cls.d, ignore_errors=True)

    @staticmethod
    def _open(req, cookie=True):
        if cookie and TestApi.cookie:
            req.add_header("Cookie", TestApi.cookie)
        r = urllib.request.build_opener(NoRedirect).open(req, timeout=30)
        sc = r.headers.get("Set-Cookie")
        if sc and sc.startswith("ba_session=") and "Max-Age=0" not in sc:
            TestApi.cookie = sc.split(";")[0]                # logged in: the next requests are this member's
        return r

    def get(self, path, headers=None, cookie=True):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), headers=headers or {})
        try:
            with self._open(req, cookie) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def post(self, path, body=None, cookie=True):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=json.dumps(body or {}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with TestApi._open(req, cookie) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_login_needed(self):
        code, _, h = self.get("/", cookie=False)
        self.assertEqual((code, h["Location"]), (302, "/login?next=%2F"), "a page without a login goes to the login page")
        self.assertEqual(self.get("/api/clips", cookie=False)[0], 401)
        self.assertEqual(self.get("/media/clipthumb/%s.jpg" % self.cid, cookie=False)[0], 401)
        self.assertEqual(self.post("/api/start", cookie=False)[0], 401)
        self.assertEqual(self.get("/login", cookie=False)[0], 200)
        self.assertEqual(self.get("/api/health", cookie=False)[0], 200, "the launcher checks the app without a login")
        self.assertEqual(self.post("/api/login", {"name": "../x"}, cookie=False)[0], 400)
        self.assertEqual(self.post("/api/login", {"name": "con"}, cookie=False)[0], 400)
        self.assertIn("Tester", json.loads(self.get("/api/members", cookie=False)[1])["members"])

    def test_members_see_only_their_own(self):
        """another member with another folder: separate clips, ids, results and queue; the first member's clip is
        not found for them"""
        other_src = os.path.join(self.d, "src2")
        os.makedirs(other_src, exist_ok=True)
        make_video(os.path.join(other_src, "11\u670812\u65e5(1).mp4"), n=12)        # same file name as the tester's clip
        mine = self.cookie
        try:
            TestApi.cookie = None
            self.assertEqual(self.post("/api/login", {"name": "Ben"})[0], 200)
            d = json.loads(self.get("/api/clips")[1])
            self.assertEqual((d["clips"], d["source"]["chosen"]), ([], False), "a new member starts with no folder")
            self.assertEqual(self.get("/api/clip/" + self.cid)[0], 404, "the tester's clip is not Ben's")
            self.assertEqual(self.get("/media/video/%s.mp4" % self.cid)[0], 404)
            self.assertEqual(self.post("/api/clip/%s/reset" % self.cid)[0], 404)
            self.assertEqual(self.post("/api/use_folder", {"path": other_src})[0], 200)
            ben = self.app.ws("ben")
            self.assertEqual(len(ben.lib.clips), 1)
            self.assertNotEqual(ben.lib.clips[0].id, self.cid, "same file name, other member: another clip")
            from agent import clips as clipsmod
            fb, ft = clipsmod.folders(ben.c, ben.lib.clips[0]), clipsmod.folders(self.ws.c, self.ws.lib.clips[0])
            for k in ("marks", "work", "submit"):
                self.assertNotEqual(fb[k], ft[k], k)
                self.assertIn(os.path.join("_users", "ben"), fb[k])
            self.assertEqual(self.post("/api/start")[1]["queued"], 1)
            self.assertFalse(self.app.runner.is_paused("ben"))
            self.assertTrue(self.app.runner.is_paused("tester"), "Ben's Start does not start the tester's queue")
            q = json.loads(self.get("/api/clips")[1])["queue"]
            self.assertEqual(q["queued"], 1)
            TestApi.cookie = mine
            q = json.loads(self.get("/api/clips")[1])["queue"]
            self.assertEqual((q["queued"], q["others"]["queued"], q["paused"]), (0, 1, True), "Ben's job is counted, not listed")
            self.assertEqual(self.post("/api/cancel_queued")[1]["cancelled"], 0, "the tester cannot clear Ben's jobs")
        finally:
            TestApi.cookie = mine
            self.app.runner.pause()
            self.app.store.cancel_queued()

    def test_folder_open_and_quit_only_on_this_pc(self):
        h = self.app.server.RequestHandlerClass
        with unittest.mock.patch.object(h, "local", lambda self: False):
            self.assertEqual(self.post("/api/open", {"path": self.ws.c["work_root"]})[0], 403)
            self.assertEqual(self.post("/api/pick_folder")[0], 403)
            self.assertEqual(self.post("/api/quit")[0], 403)
            self.assertFalse(json.loads(self.get("/api/clips")[1])["me"]["local"])

    def test_start_over_routes(self):
        from agent import clips as clipsmod
        cl = self.ws.lib.clips[0]; f = clipsmod.folders(self.ws.c, cl)
        os.makedirs(f["work"], exist_ok=True)
        with open(os.path.join(f["work"], "old.txt"), "w") as fh:
            fh.write("x")
        for stage in ("fit", "all"):
            jid = self.app.store.enqueue(cl.id, stage, user=self.ws.user)
            self.assertEqual(self.app.store.take({self.ws.user})["id"], jid)
            self.app.store.finish(jid, True)
        self.assertIn(cl.id, self.app.store.timing())
        st, d = self.post("/api/clip/%s/reset?dry=1" % self.cid)
        self.assertEqual(st, 200); self.assertTrue(d["dry"])
        self.assertIn(f["work"], [p["path"] for x in d["clips"] for p in x["paths"]])
        self.assertTrue(os.path.exists(f["work"]), "a dry run deletes nothing")
        self.assertIn(cl.id, self.app.store.timing())
        st, d = self.post("/api/reset_all")
        self.assertEqual(st, 200); self.assertTrue(d["ok"])
        self.assertFalse(os.path.exists(f["work"]))
        self.assertTrue(os.path.exists(cl.video))
        self.assertNotIn(cl.id, self.app.store.timing())
        self.assertIsNone(self.app.store.latest_for(cl.id))

    def test_resume_runs_waiting_queue_without_enqueuing_prepare(self):
        self.app.runner.pause(self.ws.user)
        with unittest.mock.patch.object(self.ws, "start", side_effect=AssertionError("unexpected Prepare")):
            self.assertEqual(self.post("/api/resume")[0], 200)
            self.assertFalse(self.app.runner.is_paused(self.ws.user))
        self.app.runner.pause(self.ws.user)

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
        self.assertEqual(d["clips"][0]["name"], "\u7535\u5f71_11\u670812\u65e5(1)")
        self.assertEqual(d["clips"][0]["status"]["label"], "not prepared")
        code, body, _ = self.get("/api/clip/" + self.cid)
        self.assertEqual(code, 200)
        self.assertIsNone(json.loads(body)["review"])

    def test_rebuilt_comparison_has_a_new_media_version(self):
        cl=self.ws.lib.clips[0];f=clipsmod.folders(self.ws.c,cl)
        folder=os.path.join(f['submit'],'comparison');os.makedirs(folder,exist_ok=True)
        path=os.path.join(folder,'version-test.mp4')
        try:
            with open(path,'wb') as stream:stream.write(b'test render')
            first=json.loads(self.get('/api/clip/'+self.cid)[1])['outputs']
            self.assertTrue(first['comparison']);stamp=os.stat(path).st_mtime_ns
            os.utime(path,ns=(stamp,stamp+1000000000))
            second=json.loads(self.get('/api/clip/'+self.cid)[1])['outputs']
            self.assertGreater(second['comparison_version'],first['comparison_version'])
        finally:os.remove(path)

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

    def test_found_objects_listed_and_deleted(self):
        """the Objects step lists what the set step found; a deleted one is remembered with its frame and box"""
        from unittest import mock
        from agent import clips as clipsmod
        import cv2, numpy as np
        cl = self.ws.lib.clips[0]; f = clipsmod.folders(self.ws.c, cl)
        sb = os.path.join(self.d, "sb")
        os.makedirs(f["work"], exist_ok=True); os.makedirs(os.path.join(sb, "output", "_users", "tester", cl.name))
        try:
            with open(os.path.join(f["work"], "regions_000.json"), "w") as fh:
                json.dump({"tag": "_000", "size": [64, 48], "regions": [], "added": []}, fh)
            with open(os.path.join(f["work"], "plate_000.jpg"), "wb") as fh:
                fh.write(cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes())
            env = {"objects": [{"name": "object_0", "type": "chair", "_src": {"cls": "object", "frame": 0, "px": [5, 6, 20, 30], "label": ["office_chair", 0.8]}},
                               {"name": "object_1", "type": "box", "_src": {"cls": "object", "frame": 9, "px": [30, 6, 50, 30]}},
                               {"name": "drawn_0", "type": "box", "_src": {"cls": "drawn", "frame": 0, "px": [1, 1, 9, 9]}}]}
            with open(os.path.join(sb, "output", "_users", "tester", cl.name, "env_auto.json"), "w") as fh:   # the member's own output
                json.dump(env, fh)
            with mock.patch.dict(os.environ, {"BA_SANDBOX": sb}):
                sh = json.loads(self.get("/api/clip/%s/objects" % self.cid)[1])["shots"][0]
                self.assertEqual([(o["what"], o["type"], o["here"]) for o in sh["found"]],
                                 [("office chair", "chair", True), ("object", "box", False)], "own boxes are not listed as found")
                code, d = self.post("/api/clip/%s/objects?rebuild=0" % self.cid, {"shots": {"_000": {
                    "objects": [{"what": "desk", "type": "table", "bbox": [30, 6, 50, 30],
                                 "outline": [[30, 6], [50, 30], [30, 30]]}],
                    "dropped": [{"frame": 0, "bbox": [5, 6, 20, 30]}]}}})
                self.assertEqual((code, d["objects"]), (200, 1))
                sh = json.loads(self.get("/api/clip/%s/objects" % self.cid)[1])["shots"][0]
                self.assertEqual(sh["dropped"], [{"frame": 0, "bbox": [5.0, 6.0, 20.0, 30.0], "why": "deleted"}])
                self.assertEqual([o["what"] for o in sh["objects"]], ["desk"])
                self.assertEqual(sh["objects"][0]["outline"], [[30, 6], [50, 30], [30, 30]])
        finally:
            shutil.rmtree(f["work"], ignore_errors=True); shutil.rmtree(f["marks"], ignore_errors=True); shutil.rmtree(sb, ignore_errors=True)

    def test_take_out_of_line_and_add_back(self):
        try:
            self.app.store.enqueue(self.cid, "fit", user="tester")
            code, d = self.post("/api/clip/%s/hold" % self.cid)
            self.assertEqual((code, d["cancelled"]), (200, 1), "its waiting job leaves the line")
            st = json.loads(self.get("/api/clips")[1])["clips"][0]["status"]
            self.assertEqual((st["held"], st["label"]), (True, "taken out"))
            self.assertEqual(self.post("/api/start")[1]["queued"], 0, "Start leaves a taken-out clip")
            self.app.runner.pause()
            code, d = self.post("/api/clip/%s/unhold" % self.cid)
            self.assertEqual((code, d["queued"]), (200, False), "paused: back on the list, prepared at the next Start")
            st = json.loads(self.get("/api/clips")[1])["clips"][0]["status"]
            self.assertEqual((st["held"], st["label"]), (False, "not prepared"))
        finally:
            self.app.runner.pause()
            self.post("/api/clip/%s/unhold" % self.cid)
            self.app.runner.pause()
            self.post("/api/cancel_queued")

    def test_background_action_requires_saved_fit_and_uses_its_own_stage(self):
        cl=self.ws.lib.clips[0];f=clipsmod.folders(self.ws.c,cl)
        self.assertEqual(self.post('/api/clip/%s/run?stage=background'%self.cid)[0],409)
        os.makedirs(f['work'],exist_ok=True)
        try:
            for fn in ('scene_data.json','camera.json','clip_info.json'):
                with open(os.path.join(f['work'],fn),'w') as stream:json.dump({},stream)
            code,result=self.post('/api/clip/%s/run?stage=background'%self.cid)
            self.assertEqual(code,200);self.assertTrue(result['queued'])
            self.assertEqual(self.app.store.latest_for(self.cid)['stage'],'background')
            info=json.loads(self.get('/api/clip/%s'%self.cid)[1])
            self.assertTrue(info['background']['ready'])
            self.assertEqual(info['status']['label'],'waiting for background render')
            self.assertEqual(self.post('/api/clip/%s/run?stage=invalid'%self.cid)[0],400)
        finally:
            self.app.runner.pause(self.ws.user);self.app.store.cancel_queued(self.cid)
            shutil.rmtree(f['work'],ignore_errors=True)

    def test_prepare_from_details_clears_taken_out_state(self):
        path = "/api/clip/%s" % self.cid
        self.assertEqual(self.post(path + "/hold")[0], 200)
        try:
            self.assertTrue(self.post(path + "/run?stage=fit")[1]["queued"])
            st = json.loads(self.get("/api/clips")[1])["clips"][0]["status"]
            self.assertEqual((st["held"], st["label"]), (False, "waiting to prepare"))
        finally:
            self.app.runner.pause(self.ws.user)
            self.post(path + "/cancel")
            self.post(path + "/unhold")

    def test_remove_from_list_and_put_back(self):
        code, d = self.post("/api/clip/%s/remove" % self.cid)
        self.assertEqual(code, 200)
        try:
            lst = json.loads(self.get("/api/clips")[1])
            self.assertEqual(lst["clips"], [])
            self.assertEqual(len(lst["removed"]), 1)
            self.assertTrue(os.path.exists(lst["removed"][0]["path"]), "the video file stays")
            self.assertEqual(lst["removed"][0]["id"], self.cid)
            self.assertEqual(self.get("/media/clipthumb/%s.jpg" % self.cid)[0], 200, "a removed video's card still has its picture")
            self.assertEqual(self.post("/api/restore", {"path": "C:\\not\\removed.mp4"})[0], 400)
        finally:
            code, d = self.post("/api/restore", {"path": lst["removed"][0]["path"]})
        self.assertEqual((code, d["clips"]), (200, 1))
        self.assertEqual(json.loads(self.get("/api/clips")[1])["clips"][0]["id"], self.cid)

    def test_zip_downloads_under_its_package_name(self):
        import package as pkg
        os.makedirs(self.ws.c["submit_root"], exist_ok=True)
        zp = os.path.join(self.ws.c["submit_root"], "clipX.zip")
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
            with self._open(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_upload(self):
        p = os.path.join(self.d, "up.mp4")
        make_video(p, n=12)
        with open(p, "rb") as fh:
            data = fh.read()
        code, d = self.upload("\u65b0\u7247Test", "clip A.mp4", data)
        self.assertEqual(code, 200, d)
        self.assertTrue(os.path.exists(os.path.join(self.d, "src", "\u65b0\u7247Test", "clip A.mp4")))
        self.assertEqual(d["clip"]["film"], "\u65b0\u7247Test")
        code, d = self.upload("\u65b0\u7247Test", "clip A.mp4", data)
        self.assertEqual(d["file"], "clip A (2).mp4", "never overwrites an existing clip")
        self.assertEqual(self.upload("\u65b0\u7247Test", "notes.txt", b"x")[0], 400)
        self.assertEqual(self.upload("..\\evil", "a.mp4", data)[0], 400)
        self.assertEqual(self.upload("", "a.mp4", data)[0], 400)
        mine = TestApi.cookie
        try:
            TestApi.cookie = None
            self.assertEqual(self.upload("\u65b0\u7247Test", "clip B.mp4", data)[0], 401, "no upload without a login")
        finally:
            TestApi.cookie = mine

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

    def test_review_timeline_roundtrip_and_reject_unknown_owner(self):
        timeline=[{'id':'cup','name':'Cup','shape':'cyl','size_m':[.08,.12,.08],
                   'events':[{'frame':0,'state':'held','person':1,'joint':'left'},
                             {'frame':10,'state':'free'}, {'frame':20,'state':'hidden'}],
                   'keyframes':[{'frame':10,'box':[10,10,20,30],'depth_m':3.}]}]
        body={'people':{'1':{}},'prop_mode':'manual','prop_timeline':timeline}
        code,result=self.post('/api/clip/%s/review'%self.cid,body)
        self.assertEqual(code,200,result); self.assertFalse(result['queued'])
        code,raw,_=self.get('/api/clip/'+self.cid)
        saved=json.loads(raw)['review']
        self.assertEqual(saved['prop_mode'],'manual')
        self.assertEqual(saved['prop_timeline'][0]['events'][1]['state'],'free')
        timeline[0]['events'][0]['person']=99
        self.assertEqual(self.post('/api/clip/%s/review'%self.cid,body)[0],400)


class TestHeldPose(unittest.TestCase):
    """a pose copied between key frames (Same pose in next): the frames between are blended from the fixes, not shaky"""
    def setUp(self):
        from common import coco_people
        self.coco = coco_people
        rng = np.random.default_rng(1)
        self.base = np.c_[np.linspace(100, 160, 17), np.linspace(100, 400, 17), np.ones(17)]
        frames = {}
        for f in range(17):                                       # the detector: the same pose with 6 px of shake
            k = self.base.copy(); k[:, :2] += rng.normal(0, 6, (17, 2)); frames[str(f)] = k.tolist()
        self.pose = {"people": [{"id": 1, "frames": frames}]}

    def test_still_pose_is_held(self):
        chk = {"frames": {"0": [{"id": 1, "kps": self.base.tolist()}], "8": [{"id": 1, "kps": self.base.tolist()}]}}
        fs, k = self.coco(self.pose, chk)[1]
        self.assertTrue(np.allclose(k[fs.index(4)][:, :2], self.base[:, :2]))     # between the two fixes: the fixed pose

    def test_moving_pose_keeps_detector_motion(self):
        moved = self.base.copy(); moved[:, 0] += 80                               # a real move: not blended
        chk = {"frames": {"0": [{"id": 1, "kps": self.base.tolist()}], "8": [{"id": 1, "kps": moved.tolist()}]}}
        fs, k = self.coco(self.pose, chk)[1]
        lin = self.base[:, :2] * 0.5 + moved[:, :2] * 0.5
        self.assertFalse(np.allclose(k[fs.index(4)][:, :2], lin))


class TestSpeedAndSet(unittest.TestCase):
    """Drawn set objects and cache freshness"""

    def test_drawn_objects_found_by_key_frame(self):
        import env_auto as ea
        env = {"shots": {"_000": {"objects": [{"bbox": [1, 2, 3, 4]}]}, "_014": {"objects": [{"bbox": [1, 2, 3, 4]}]},
                         "_020": {"objects": [{"what": "no box"}]}}}
        self.assertEqual(ea.drawn_frames(env, [{"shot": 1, "frames": [0, 99]}]), {1: {"_000": 0, "_014": 14}})
        two = [{"shot": 1, "frames": [0, 49]}, {"shot": 2, "frames": [50, 99]}]
        env2 = {"shots": {"_s2_000": {"objects": [{"bbox": [1, 2, 3, 4]}]}, "_000": {"objects": [{"bbox": [1, 2, 3, 4]}]}}}
        self.assertEqual(ea.drawn_frames(env2, two), {2: {"_s2_000": 50}}, "shot tags, frame within the shot")

    def test_set_measured_again_when_camera_changes(self):
        a = {"lens": 40.0, "frames": 2, "cam": [[0, 0, 1.2, 91.0, 0, 0], [0, 0, 1.2, 91.0, 0, 179.9]], "env_ai": ""}
        same = dict(a, cam=[[0.01, 0, 1.21, 91.2, 0, 0], [0, 0, 1.2, 91.0, 0, -179.9]])
        self.assertTrue(rp.same_set_inputs(a, same))
        self.assertFalse(rp.same_set_inputs(a, dict(a, lens=42.0)), "lens changed")
        self.assertFalse(rp.same_set_inputs(a, dict(a, cam=[[0, 0, 1.2, 91.0, 0, 0], [0, 0, 1.2, 0.0, 0, 0]])), "static camera")
        self.assertFalse(rp.same_set_inputs(a, dict(a, env_ai="x")), "drawn objects changed")


class TestFrameDepth(unittest.TestCase):
    """the set's depth comes from the real frames: people and pillarbox bars cut out, each frame's scale matched"""
    def test_people_filled_from_other_frames(self):
        import scene_depth as sd
        H, W = 120, 200
        class Cam:
            pass
        class World:                                        # a still camera: every frame sees the same view
            cam = Cam()
            def R_cw(self, i):
                return np.eye(3)
        w = World(); w.cam.H, w.cam.W = H, W; w.cam.K = lambda: np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]])
        truth = np.tile(np.linspace(0.2, 2.0, H, dtype=np.float32)[:, None], (1, W)); truth[40:80, 60:100] = 3.0   # a box nearer
        frames, boxes = [], {}
        for j in range(5):                                  # a person walks across the room
            img = np.repeat((truth / 3.0 * 200 + 20).astype(np.uint8)[:, :, None], 3, axis=2)
            img[:, :20] = 0; img[:, -20:] = 0                # pillarbox bars
            x0 = 30 + 30 * j
            img[20:110, x0:x0 + 25] = 250                     # the person (would read as very near)
            frames.append(img); boxes[j] = [[x0, 20, x0 + 25, 110]]
        calls = []
        def model(bgr):                                     # relative depth: a different scale and offset every call
            n = len(calls); calls.append(bgr.shape)
            return (bgr[..., 0].astype(np.float32) - 20) / 200 * 3.0 * (1 + 0.5 * n) + 0.3 * n
        disp, hidden = sd.frame_depth(frames, 0, w, 2, boxes, model, nmax=5)
        self.assertEqual(calls[0][1], W - 40, "the bars are not shown to the model")
        inner = (slice(5, H - 5), slice(25, W - 25))
        self.assertFalse(hidden[inner].any(), "everywhere was free in some frame")
        ref_scale = 1 + 0.5 * 2                              # frame 2 is the reference (k)
        err = np.abs(disp[inner] - (truth[inner] * ref_scale + 0.6))
        self.assertLess(np.percentile(err, 95), 0.05, "depth matches the room, not the person")


class TestTrackGaps(unittest.TestCase):
    """Conservative box holding for static scene exclusion, without changing person observations"""


    def test_hold_boxes_over_missed_frames(self):
        b = rp.hold_boxes({"33": [100, 100, 200, 300], "40": [110, 100, 210, 300]}, 120)
        self.assertEqual(b[33], [100, 100, 200, 300])
        self.assertIn(0, b, "missed for the first 33 frames: the first box is held")
        self.assertLess(b[0][0], 100, "a held box is a little wider")
        self.assertEqual(b[36][0] < 100, True)
        self.assertNotIn(119, b, "more than 45 frames from any detection: not held")


if __name__ == "__main__":
    unittest.main()

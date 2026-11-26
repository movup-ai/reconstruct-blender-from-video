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
from agent import clips as clipsmod, jobs  # noqa: E402
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
        if not names:
            self.skipTest("no videos in the source folder")
        self.assertEqual(len(names), len(set(names)), "clip names must be unique (they become folder / zip names)")
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [cl.id for cl in clipsmod.Library(c).clips], "ids must be stable between scans")


class TestJobStore(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.s = jobs.JobStore(os.path.join(self.d, "j.db"))

    def tearDown(self):
        self.s.db.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def row(self, stage, state, started, ended, clip="a"):
        self.s.db.execute("INSERT INTO jobs(clip, stage, state, priority, created, started, ended) VALUES (?,?,?,0,?,?,?)",
                          (clip, stage, state, started, started, ended))

    def test_stages(self):
        self.assertEqual(jobs.STAGES, ("movement", "background"))
        self.assertIsNotNone(self.s.enqueue("a", "background"), "background stays valid in the store")
        for old in ("fit", "all", "ai", "depth", "nope"):
            with self.assertRaises(ValueError):
                self.s.enqueue("b", old)

    def test_timing_from_start_to_package(self):
        self.row("movement", "done", 100, 400)
        t = self.s.timing(now=5000)["a"]
        self.assertEqual(t["machine"], 300)
        self.assertEqual(t["done"], {"start": 100, "end": 400, "wall": 300, "machine": 300})
        self.assertEqual(t["steps"], {"movement": {"state": "done", "ran": 300, "ended": 400}})
        self.assertIsNone(t["open"])
        self.row("movement", "running", 2000, None)                     # run again after the package: still going
        t = self.s.timing(now=2600)["a"]
        self.assertEqual(t["done"]["wall"], 300, "the last finished run stays")
        self.assertEqual(t["open"], {"start": 2000, "so_far": 600, "machine": 600})
        self.assertEqual(t["last"], {"stage": "movement", "state": "running", "ran": 600})
        self.assertEqual(t["steps"]["movement"]["state"], "running")
        self.assertEqual(t["machine"], 900, "computer time includes the previous package and the current job")

    def test_computer_time_accumulates_until_reset(self):
        self.row("movement", "cancelled", 100, 580, "clip")              # interrupted by Pause: 8 minutes
        self.row("movement", "done", 900, 1020, "clip")                  # the resumed run: 2 minutes
        self.row("movement", "done", 2000, 2180, "clip")                 # a Re-run: 3 minutes
        timing = self.s.timing(now=3000)["clip"]
        self.assertEqual(timing["machine"], 780)
        self.assertEqual(timing["steps"]["movement"]["ran"], 180)
        self.s.cancel_for_reset({"clip"})
        self.s.clear_clip("clip")
        self.assertNotIn("clip", self.s.timing(now=3010))

    def test_dedupe_priority_recover(self):
        self.assertIsNotNone(self.s.enqueue("a", "background"))
        self.assertIsNone(self.s.enqueue("a", "background"), "one job per clip at a time")
        self.assertIsNotNone(self.s.enqueue("b", "movement"))
        j = self.s.take()
        self.assertEqual((j["clip"], j["stage"]), ("b", "movement"), "movement goes first")
        self.assertEqual(self.s.recover(), 1)
        self.assertEqual(self.s.take()["clip"], "b")

    def test_recover_requeues_running_jobs_and_keeps_waiting_ones(self):
        """restart: the running job waits again; waiting jobs (also ones queued by the older multi-member app) are
        kept, never cancelled"""
        self.s.db.execute("INSERT INTO jobs(clip, stage, state, priority, created) VALUES ('old','movement','queued',0,1)")
        self.s.enqueue("a", "movement"); self.s.enqueue("b", "movement")
        self.assertEqual(self.s.take()["clip"], "old", "first queued, first run")
        self.assertEqual(self.s.counts(), {"running": 1, "queued": 2})
        self.assertEqual(self.s.recover(), 1)
        self.assertEqual(self.s.counts(), {"queued": 3}, "nothing is cancelled on start")
        self.assertEqual(self.s.take()["clip"], "old", "the recovered job keeps its place")
        self.assertEqual(self.s.cancel_queued("a"), 1, "one clip's waiting job")
        self.assertEqual(self.s.cancel_queued(), 1, "all the other waiting jobs (the running one is not touched)")
        self.assertEqual(self.s.counts(), {"running": 1, "cancelled": 2})

    def test_force_replaces_queued(self):
        self.s.enqueue("a", "background")
        self.assertIsNotNone(self.s.enqueue("a", "movement", force=True))
        j = self.s.take()
        self.assertEqual(j["stage"], "movement")
        self.assertIsNone(self.s.take())

    def test_finish_and_latest(self):
        jid = self.s.enqueue("a", "movement")
        self.s.take()
        self.s.finish(jid, False, "boom")
        self.assertEqual(self.s.latest()["a"]["state"], "failed")
        self.assertEqual(self.s.latest()["a"]["error"], "boom")

    def test_limit_and_busy_clips(self):
        self.s.enqueue("a", "movement"); self.s.enqueue("b", "movement")
        self.assertEqual(self.s.take(exclude={"a"})["clip"], "b", "a clip whose old process may still run waits")
        self.assertIsNone(self.s.take(limit=1), "one running job fills one slot")
        self.assertEqual(self.s.take(limit=2)["clip"], "a")

    def test_old_stage_jobs_are_cancelled_when_the_queue_opens(self):
        self.s.db.execute("INSERT INTO jobs(clip, stage, state, priority, created) VALUES ('a','fit','queued',2,1)")
        self.s.db.execute("INSERT INTO jobs(clip, stage, state, priority, created) VALUES ('b','movement','queued',0,2)")
        self.s.db.close()
        self.s = jobs.JobStore(os.path.join(self.d, "j.db"))
        self.assertEqual(self.s.db.execute("SELECT state FROM jobs WHERE clip='a'").fetchone()[0], "cancelled")
        self.assertEqual(self.s.take()["clip"], "b")

    def test_paused_job_keeps_its_place_and_loses_the_mark_when_it_runs(self):
        self.s.enqueue("b-first", "movement")                          # waiting before the job that runs
        time.sleep(0.01)
        jid = self.s.enqueue("a", "movement")
        job = self.s.take(exclude={"b-first"})
        self.assertEqual(job["id"], jid)
        time.sleep(0.01)
        self.s.enqueue("c-later", "movement")                          # queued while a ran
        pending = self.s.cancel_for_reset(set(), all_clips=True, return_jobs=True, running_only=True)
        self.assertEqual([p["id"] for p in pending], [jid], "Pause interrupts only the running job")
        self.s.restore_paused(pending)
        self.s.restore_paused(pending)                                  # never queued twice
        waiting = self.s.db.execute("SELECT * FROM jobs WHERE clip='a' AND state='queued'").fetchall()
        self.assertEqual(len(waiting), 1)
        self.assertEqual((waiting[0]["error"], waiting[0]["priority"], waiting[0]["attempts"]), (clipsmod.PAUSED, 0, 0))
        self.assertEqual(self.s.take(limit=3)["clip"], "b-first", "a resumed job does not jump a job queued before it")
        resumed = self.s.take(limit=3)
        self.assertEqual((resumed["clip"], resumed["error"]), ("a", None), "it goes before a job queued after it")
        self.assertEqual(self.s.take(limit=3)["clip"], "c-later")
        self.assertIsNone(self.s.latest_for("a")["error"], "the paused mark is cleared when it runs")

    def test_resume_clears_the_paused_mark(self):
        self.s.db.execute("INSERT INTO jobs(clip, stage, state, priority, created, error) VALUES ('a','movement','queued',0,1,?)",
                          (clipsmod.PAUSED,))
        self.assertEqual(self.s.clear_paused(), 1)
        self.assertIsNone(self.s.latest_for("a")["error"])


class StatusCase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        src = os.path.join(self.d, "src", "FilmA"); os.makedirs(src)
        make_video(os.path.join(src, "1-1.mp4"), n=10)
        self.c = dict(config.load(), source_root=os.path.join(self.d, "src"), work_root=os.path.join(self.d, "w"),
                      submit_root=os.path.join(self.d, "s"), marks_root=os.path.join(self.d, "m"))
        self.data = unittest.mock.patch.object(config, "DATA", os.path.join(self.d, "data")); self.data.start()
        self.cl = clipsmod.Library(self.c).clips[0]
        self.f = clipsmod.folders(self.c, self.cl)

    def tearDown(self):
        self.data.stop()
        shutil.rmtree(self.d, ignore_errors=True)

    def make_zip(self, fp=None):
        os.makedirs(os.path.dirname(self.f["zip"]), exist_ok=True); os.makedirs(self.f["work"], exist_ok=True)
        with open(self.f["zip"], "wb") as fh:
            fh.write(b"PK\x05\x06" + b"\0" * 18)
        clipsmod.record_finish(self.c, self.cl, fp or clipsmod.fingerprint(self.c, self.cl))


class TestStatus(StatusCase):
    def label(self, job=None):
        return clipsmod.status(self.c, self.cl, job)["label"]

    def test_labels(self):
        self.assertEqual(self.label(), "new")
        self.assertEqual(self.label({"stage": "movement", "state": "queued"}), "waiting")
        self.assertEqual(self.label({"stage": "movement", "state": "queued", "error": clipsmod.PAUSED}), "paused")
        st = clipsmod.status(self.c, self.cl, {"stage": "movement", "state": "running"})
        self.assertEqual((st["label"], st["stage"], st["group"]), ("processing", "movement", "busy"))
        self.assertEqual(self.label({"stage": "movement", "state": "failed", "error": "x"}), "failed")
        self.assertEqual(self.label({"stage": "movement", "state": "stopped"}), "cancelled")
        clipsmod.set_skipped(self.cl.id, True, self.c)
        self.assertEqual(self.label(), "skipped")
        self.assertEqual(clipsmod.status(self.c, self.cl)["group"], "skip")
        self.assertEqual(self.label({"stage": "movement", "state": "running"}), "processing")
        clipsmod.set_skipped(self.cl.id, False, self.c)

    def test_done_beats_a_later_cancelled_or_failed_job_but_not_a_running_one(self):
        self.make_zip()
        st = clipsmod.status(self.c, self.cl)
        self.assertEqual((st["label"], st["done"], st["group"]), ("done", True, "done"))
        self.assertEqual(self.label({"stage": "movement", "state": "stopped"}), "done")
        self.assertEqual(self.label({"stage": "movement", "state": "failed"}), "done")
        self.assertEqual(self.label({"stage": "movement", "state": "running"}), "processing")
        self.assertEqual(self.label({"stage": "movement", "state": "queued"}), "waiting")

    def test_outdated_when_stage1_settings_change(self):
        self.make_zip()
        c2 = dict(self.c, person_height=self.c["person_height"] + 0.1)
        st = clipsmod.status(c2, self.cl)
        self.assertEqual((st["label"], st["done"], st["outdated"]), ("outdated", False, True))
        self.assertEqual(clipsmod.status(c2, self.cl, {"stage": "movement", "state": "failed"})["label"], "failed")

    def test_fingerprint_is_video_package_and_stage1_settings_only(self):
        fp = clipsmod.fingerprint(self.c, self.cl)
        for key, value in (("validation_mode", "audit"), ("character_style", "l3"), ("render_engine", "eevee"),
                           ("source_exposure", True), ("parallel", 4), ("blender_exe", "x")):
            self.assertEqual(clipsmod.fingerprint(dict(self.c, **{key: value}), self.cl), fp, key)
        os.makedirs(self.f["marks"], exist_ok=True)
        for name in ("review.json", "checked.json", "env_ai.json", "hints.json", "desc.txt"):
            with open(os.path.join(self.f["marks"], name), "w") as fh:
                fh.write('{"people": {}}')
        self.assertEqual(clipsmod.fingerprint(self.c, self.cl), fp, "review files are not inputs any more")
        self.assertNotEqual(clipsmod.fingerprint(dict(self.c, lens_mm=float(self.c["lens_mm"]) + 5), self.cl), fp)
        with unittest.mock.patch.object(clipsmod.pkg, "package_name", return_value="film-9"):
            self.assertNotEqual(clipsmod.fingerprint(self.c, self.cl), fp, "the delivery name is part of the package")
        os.utime(self.cl.video, (time.time() + 100, time.time() + 100))
        self.assertNotEqual(clipsmod.fingerprint(self.c, self.cl), fp, "a replaced video")

    def test_zip_made_outside_the_app_counts_as_done(self):
        os.makedirs(os.path.dirname(self.f["zip"]), exist_ok=True)
        with open(self.f["zip"], "wb") as fh:
            fh.write(b"zip")
        self.assertEqual(self.label(), "done")


class FakeApp:
    def __init__(self, store, c, ws=None):
        self.store, self.c, self.ws = store, c, ws


class TestRunner(StatusCase):
    def setUp(self):
        super().setUp()
        self.store = jobs.JobStore(os.path.join(self.d, "j.db"))
        self.ws = unittest.mock.Mock(c=self.c, lib=clipsmod.Library(self.c))
        self.app = FakeApp(self.store, dict(self.c, parallel=1), self.ws)
        self.runner = jobs.Runner(self.app)

    def tearDown(self):
        self.store.db.close()
        super().tearDown()

    def test_command_runs_stage_movement(self):
        cmd = self.runner._cmd({"stage": "movement"}, self.cl)
        self.assertEqual(cmd[cmd.index("--stage") + 1], "movement")
        self.assertIn("--force", cmd)
        self.assertEqual(cmd[cmd.index("--name") + 1], self.cl.name)
        self.assertNotIn("--require-confirm", cmd)

    def test_autopilot_queues_movement_for_a_new_clip_after_start(self):
        self.assertEqual(self.runner.autopilot_pass(), 0, "nobody pressed Start")
        self.runner.active = True
        self.assertEqual(self.runner.autopilot_pass(), 1)
        job = self.store.latest_for(self.cl.id)
        self.assertEqual((job["stage"], job["state"]), ("movement", "queued"))
        self.assertEqual(self.runner.autopilot_pass(), 0, "already waiting")

    def test_autopilot_leaves_done_held_failed_and_cancelled_clips(self):
        self.runner.active = True
        for state in ("failed", "stopped"):
            self.store.db.execute("DELETE FROM jobs")
            self.store.db.execute("INSERT INTO jobs(clip, stage, state, priority, created) VALUES (?,?,?,0,1)",
                                  (self.cl.id, "movement", state))
            self.assertEqual(self.runner.autopilot_pass(), 0, state)
        self.store.db.execute("DELETE FROM jobs")
        clipsmod.set_skipped(self.cl.id, True, self.c)
        self.assertEqual(self.runner.autopilot_pass(), 0, "skipped")
        clipsmod.set_skipped(self.cl.id, False, self.c)
        self.make_zip()
        self.assertEqual(self.runner.autopilot_pass(), 0, "its zip is up to date")

    def test_successful_run_without_a_zip_fails(self):
        self.runner.active = True
        self.ws.lib.get = lambda cid: self.cl
        jid = self.store.enqueue(self.cl.id, "movement")
        job = self.store.take()
        self.runner._cmd = lambda *_: [sys.executable, "-c", "print('done')"]
        self.app.c["max_attempts"] = 1
        self.c["max_attempts"] = 1
        self.runner._run(job)
        row = self.store.latest_for(self.cl.id)
        self.assertEqual((row["id"], row["state"]), (jid, "failed"))
        self.assertIn("without a package", row["error"])


class TestConfig(unittest.TestCase):
    def test_rejects_bad_values_and_removed_settings(self):
        for bad in ({"parallel": 0}, {"unknown": 1}, {"work_root": "relative\\path"}, {"autopilot": "full"},
                    {"finish_parallel": 2}, {"deepseek_key": "x" * 30}, {"anthropic_key": "x" * 30}, {"validation_mode": "audit"},
                    {"character_style": "l3"}, {"render_engine": "eevee"}, {"camera_backend": "da3"}, {"sam3d_stride": 99}):
            with self.assertRaises(ValueError, msg=str(bad)):
                config.save(bad)

    def test_old_config_files_load(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "config.json")
            with open(p, "w") as fh:
                json.dump({"autopilot": "full", "finish_parallel": 3, "deepseek_key": "secret", "validation_mode": "audit",
                           "timeout_fit_min": 30, "timeout_render_min": 100, "some_retired_key": [1, 2]}, fh)
            with unittest.mock.patch.object(config, "CFG_PATH", p):
                c = config.load()
            self.assertEqual(c["timeout_movement_min"], 130, "the two old timeouts make the Movement one")
            self.assertEqual(c["parallel"], 1)
            self.assertEqual(config.migrate({"timeout_movement_min": 90, "timeout_fit_min": 5})["timeout_movement_min"], 90)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_no_secrets_or_removed_settings_are_editable(self):
        for k in ("autopilot", "finish_parallel", "deepseek_key", "anthropic_key", "ai_enabled", "validation_mode",
                  "character_style", "render_engine", "source_exposure", "camera_backend", "set_builder", "render_on_confirm"):
            self.assertNotIn(k, config.EDITABLE)
        self.assertEqual(config.SECRETS, ())
        self.assertNotIn("timeout_ai_min", config.APP_DEFAULTS)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class TestApi(unittest.TestCase):
    """real server on a temporary library with one synthetic clip; one workspace, no login.  The settings file
    (config.json), the data folder and the package names point at temporary copies."""

    @classmethod
    def setUpClass(cls):
        import package as pkg
        cls.d = tempfile.mkdtemp()
        src = os.path.join(cls.d, "src", "\u7535\u5f71Film")
        os.makedirs(src)
        make_video(os.path.join(src, "11\u670812\u65e5(1).mp4"), n=30, cut=15)
        cfg_path = os.path.join(cls.d, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as fh:          # use_folder saves source_root here, not in the project
            json.dump({"work_root": os.path.join(cls.d, "w"), "submit_root": os.path.join(cls.d, "s"),
                       "marks_root": os.path.join(cls.d, "m"), "source_root": ""}, fh)
        cls.patches = [unittest.mock.patch.object(config, "DATA", os.path.join(cls.d, "data")),
                       unittest.mock.patch.object(config, "CFG_PATH", cfg_path),
                       unittest.mock.patch.object(pkg, "NAMES_P", os.path.join(cls.d, "names.json"))]
        for p in cls.patches:
            p.start()
        from agent.server import App, Server, make_handler
        cls.app = App(config.load())
        cls.port = free_port()
        cls.app.server = Server(("127.0.0.1", cls.port), make_handler(cls.app))
        threading.Thread(target=cls.app.server.serve_forever, daemon=True).start()
        st, d = cls.post(cls, "/api/use_folder", {"path": os.path.join(cls.d, "src")})
        assert st == 200, d
        cls.ws = cls.app.ws
        cls.cid = cls.ws.lib.clips[0].id

    @classmethod
    def tearDownClass(cls):
        cls.app.server.shutdown()
        cls.app.server.server_close()
        cls.app.store.db.close()
        for p in reversed(cls.patches):
            p.stop()
        shutil.rmtree(cls.d, ignore_errors=True)

    def tearDown(self):
        self.app.runner.pause()
        self.app.store.db.execute("DELETE FROM jobs")
        clipsmod.set_skipped(self.cid, False, self.ws.c)

    @staticmethod
    def _open(req):
        return urllib.request.build_opener(NoRedirect).open(req, timeout=30)

    def get(self, path, headers=None):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), headers=headers or {})
        try:
            with self._open(req) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def post(self, path, body=None):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=json.dumps(body or {}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with TestApi._open(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def status_of(self):
        return json.loads(self.get("/api/clips")[1])["clips"][0]["status"]

    def test_one_workspace_and_no_login(self):
        """no cookie is needed or given: pages, the API and the media answer directly; the login routes are gone"""
        for p in ("/", "/clip/" + self.cid, "/settings", "/api/clips", "/api/clip/" + self.cid, "/api/health",
                  "/media/clipthumb/%s.jpg" % self.cid):
            code, _, h = self.get(p)
            self.assertEqual(code, 200, p)
            self.assertIsNone(h.get("Set-Cookie"), p)
        for p in ("/login", "/logout", "/api/members", "/api/me"):
            self.assertEqual(self.get(p)[0], 404, p)
        for p in ("/api/login", "/api/logout"):
            self.assertEqual(self.post(p, {"name": "Tester"})[0], 404, p)
        d = json.loads(self.get("/api/clips")[1])
        self.assertNotIn("me", d)
        self.assertIs(d["local"], True, "the test runs on this PC")
        self.assertEqual(set(d["queue"]), {"running", "queued", "failed", "parallel", "now", "paused"})
        self.assertEqual(d["source"]["path"], os.path.join(self.d, "src"))
        self.assertEqual(config.load()["source_root"], os.path.join(self.d, "src"), "the source folder is saved in the settings")
        f = clipsmod.folders(self.ws.c, self.ws.lib.clips[0])
        for k in ("marks", "work", "submit"):
            self.assertNotIn("_users", f[k], k)
        self.assertEqual(self.post("/api/start")[1]["queued"], 1, "Start without a login")
        self.assertFalse(self.app.runner.is_paused())

    def test_folder_open_and_quit_only_on_this_pc(self):
        h = self.app.server.RequestHandlerClass
        with unittest.mock.patch.object(h, "local", lambda self: False):
            self.assertEqual(self.post("/api/open", {"path": self.ws.c["work_root"]})[0], 403)
            self.assertEqual(self.post("/api/pick_folder")[0], 403)
            self.assertEqual(self.post("/api/quit")[0], 403)
            self.assertIs(json.loads(self.get("/api/clips")[1])["local"], False)

    def test_start_over_routes(self):
        cl = self.ws.lib.clips[0]; f = clipsmod.folders(self.ws.c, cl)
        os.makedirs(f["work"], exist_ok=True)
        with open(os.path.join(f["work"], "old.txt"), "w") as fh:
            fh.write("x")
        jid = self.app.store.enqueue(cl.id, "movement")
        self.assertEqual(self.app.store.take()["id"], jid)
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

    def test_resume_runs_waiting_queue_without_enqueuing(self):
        self.app.runner.pause()
        with unittest.mock.patch.object(self.ws, "start", side_effect=AssertionError("unexpected Start")):
            self.assertEqual(self.post("/api/resume")[0], 200)
            self.assertFalse(self.app.runner.is_paused())

    def test_start_queues_failed_and_cancelled_clips_and_resumes_paused_work(self):
        for state in ("failed", "stopped"):
            self.app.runner.pause()
            self.app.store.db.execute("DELETE FROM jobs")
            self.app.store.db.execute("INSERT INTO jobs(clip, stage, state, priority, created) VALUES (?,?,?,0,1)",
                                      (self.cid, "movement", state))
            code, d = self.post("/api/start")
            self.assertEqual((code, d["queued"]), (200, 1), state)
            self.assertFalse(self.app.runner.is_paused())
            self.assertEqual(self.status_of()["label"], "waiting")
        self.app.runner.pause()
        self.app.store.db.execute("DELETE FROM jobs")
        self.app.store.db.execute("INSERT INTO jobs(clip, stage, state, priority, created, error) VALUES (?,?,?,0,1,?)",
                                  (self.cid, "movement", "queued", clipsmod.PAUSED))
        self.assertEqual(self.status_of()["label"], "paused")
        self.assertEqual(self.post("/api/start")[1]["queued"], 0, "the paused job keeps its place")
        self.assertFalse(self.app.runner.is_paused())
        self.assertEqual(self.status_of()["label"], "waiting", "after Resume the job is no longer shown as paused")

    def test_new_pages_and_endpoints(self):
        for p in ("/submissions", "/settings", "/help", "/favicon.svg"):
            self.assertEqual(self.get(p)[0], 200, p)
        code, body, _ = self.get("/api/submissions")
        self.assertEqual((code, json.loads(body)), (200, []))
        self.assertEqual(self.post("/api/open", {"path": "C:\\Windows"})[0], 400, "only the app's own folders")
        self.assertEqual(self.get("/download/nothing")[0], 404)
        from agent.server import advice
        self.assertIn("Settings", advice("Blender not found"))

    def test_removed_endpoints_are_gone(self):
        for path in ("/check/" + self.cid, "/api/clip/%s/objects" % self.cid, "/media/plate/%s/_000" % self.cid,
                     "/media/shotframe/%s/_000" % self.cid, "/media/blocks/%s/blocks.jpg" % self.cid,
                     "/media/review/%s/index.html" % self.cid):
            self.assertEqual(self.get(path)[0], 404, path)
        for path in ("/api/clip/%s/review" % self.cid, "/api/clip/%s/marks" % self.cid, "/api/clip/%s/objects" % self.cid,
                     "/api/clip/%s/finish" % self.cid, "/api/finish_reviewed", "/api/render_confirmed",
                     "/api/ai_test", "/api/claude_test"):
            self.assertEqual(self.post(path, {})[0], 404, path)
        for stage in ("fit", "all", "ai", "depth", "background"):
            self.assertEqual(self.post("/api/clip/%s/run?stage=%s" % (self.cid, stage))[0], 400, stage)

    def test_pages_and_health(self):
        for p in ("/", "/clip/" + self.cid, "/static/app.css", "/static/common.js"):
            self.assertEqual(self.get(p)[0], 200, p)
        for p in ("/static/prop_timeline.js", "/static/limb_depth.js"):
            self.assertEqual(self.get(p)[0], 404, p)
        code, body, _ = self.get("/api/health")
        self.assertEqual(code, 200)
        h = json.loads(body)
        self.assertEqual(h["app"], "Blender Agent")
        self.assertFalse([x for x in h["checks"] if "AI" in x["name"] or "Character" in x["name"]])

    def test_clips_and_info(self):
        code, body, _ = self.get("/api/clips")
        d = json.loads(body)
        self.assertEqual(len(d["clips"]), 1)
        self.assertEqual(d["clips"][0]["name"], "\u7535\u5f71_11\u670812\u65e5(1)")
        self.assertEqual(d["clips"][0]["status"]["label"], "new")
        self.assertNotIn("ai_ready", d)
        self.assertNotIn("finish_parallel", d["queue"])
        code, body, _ = self.get("/api/clip/" + self.cid)
        self.assertEqual(code, 200)
        info = json.loads(body)
        for gone in ("review", "suggested", "ai", "env_ai", "regions", "flagged", "prop_warnings", "limb_depth_report", "blocks"):
            self.assertNotIn(gone, info)
        self.assertEqual((info["background"]["state"], info["background"]["download"]), ("none", None))   # Stage 2: not run
        self.assertIn("timing", info)
        self.assertEqual((info["outputs"]["zip"], info["outputs"]["download"]), (False, None))

    def test_rebuilt_comparison_has_a_new_media_version(self):
        cl = self.ws.lib.clips[0]; f = clipsmod.folders(self.ws.c, cl)
        folder = os.path.join(f['submit'], 'comparison'); os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'version-test.mp4')
        try:
            with open(path, 'wb') as stream: stream.write(b'test render')
            first = json.loads(self.get('/api/clip/' + self.cid)[1])['outputs']
            self.assertTrue(first['comparison']); stamp = os.stat(path).st_mtime_ns
            os.utime(path, ns=(stamp, stamp + 1000000000))
            second = json.loads(self.get('/api/clip/' + self.cid)[1])['outputs']
            self.assertGreater(second['comparison_version'], first['comparison_version'])
        finally: os.remove(path)

    def test_done_clip_offers_its_download(self):
        cl = self.ws.lib.clips[0]; f = clipsmod.folders(self.ws.c, cl)
        os.makedirs(os.path.dirname(f["zip"]), exist_ok=True); os.makedirs(f["work"], exist_ok=True)
        with open(f["zip"], "wb") as fh:
            fh.write(b"PK\x05\x06" + b"\0" * 18)
        try:
            with unittest.mock.patch.object(clipsmod.pkg, "package_name", return_value="film-7"):
                clipsmod.record_finish(self.ws.c, cl, clipsmod.fingerprint(self.ws.c, cl))
                info = json.loads(self.get("/api/clip/" + self.cid)[1])
                self.assertEqual(info["status"]["label"], "done")
                self.assertEqual(info["outputs"]["download"], "/download/" + cl.name)
                self.assertEqual(self.post("/api/start")[1]["queued"], 0, "a done clip is not queued again")
        finally:
            os.remove(f["zip"]); shutil.rmtree(f["work"], ignore_errors=True)

    def test_nothing_runs_before_start(self):
        q = json.loads(self.get("/api/clips")[1])["queue"]
        self.assertTrue(q["paused"], "the app starts paused")
        code, d = self.post("/api/start")
        self.assertEqual((code, d["queued"]), (200, 1))
        self.assertEqual(self.app.store.latest_for(self.cid)["stage"], "movement")
        self.assertFalse(json.loads(self.get("/api/clips")[1])["queue"]["paused"])
        self.assertEqual(self.post("/api/start")[1]["queued"], 0, "a queued clip is not queued twice")
        self.post("/api/pause")
        self.assertTrue(json.loads(self.get("/api/clips")[1])["queue"]["paused"])

    def test_skip_keeps_the_clip_listed_and_include_brings_it_back(self):
        self.app.store.enqueue(self.cid, "movement")
        code, d = self.post("/api/clip/%s/skip" % self.cid)
        self.assertEqual((code, d["cancelled"]), (200, 1), "its waiting job leaves the line")
        lst = json.loads(self.get("/api/clips")[1])
        self.assertEqual([c["id"] for c in lst["clips"]], [self.cid], "a skipped video stays on the list")
        self.assertNotIn("removed", lst, "there is no separate removed list any more")
        st = self.status_of()
        self.assertEqual((st["skipped"], st["label"]), (True, "skipped"))
        self.assertEqual(self.post("/api/start")[1]["queued"], 0, "Start leaves a skipped clip out")
        self.app.runner.pause()
        code, d = self.post("/api/clip/%s/include" % self.cid)
        self.assertEqual((code, d["queued"]), (200, False), "paused: included, processed at the next Start")
        st = self.status_of()
        self.assertEqual((st["skipped"], st["label"]), (False, "new"))

    def test_old_take_out_remove_and_restore_actions_are_gone(self):
        path = "/api/clip/%s" % self.cid
        for action in ("hold", "unhold", "remove"):
            self.assertEqual(self.post(path + "/" + action)[0], 404, action)
        self.assertEqual(self.post("/api/restore", {"path": "x"})[0], 404)

    def test_run_from_details_queues_movement_and_includes_a_skipped_clip(self):
        path = "/api/clip/%s" % self.cid
        self.assertEqual(self.post(path + "/skip")[0], 200)
        self.assertTrue(self.post(path + "/run")[1]["queued"])
        st = self.status_of()
        self.assertEqual((st["skipped"], st["label"]), (False, "waiting"))
        self.assertEqual(self.app.store.latest_for(self.cid)["stage"], "movement")
        self.assertFalse(self.app.runner.is_paused(), "an explicit run lets the queue run")

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
        self.assertEqual(self.get("/media/thumb/%s/..%%5Cconfig.json" % self.cid)[0], 404, "no path traversal")

    def test_errors(self):
        self.assertEqual(self.get("/api/clip/nope")[0], 404)
        self.assertEqual(self.post("/api/settings", {"parallel": 99})[0], 400)
        self.assertEqual(self.post("/api/settings", {"autopilot": "full"})[0], 400)

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
        self.assertIsNone(self.app.store.latest_for(d["clip"]["id"]), "paused: an upload waits for Start")
        code, d = self.upload("\u65b0\u7247Test", "clip A.mp4", data)
        self.assertEqual(d["file"], "clip A (2).mp4", "never overwrites an existing clip")
        self.assertEqual(self.upload("\u65b0\u7247Test", "notes.txt", b"x")[0], 400)
        self.assertEqual(self.upload("..\\evil", "a.mp4", data)[0], 400)
        self.assertEqual(self.upload("", "a.mp4", data)[0], 400)
        shutil.rmtree(os.path.join(self.d, "src", "\u65b0\u7247Test"), ignore_errors=True)
        self.ws.lib.rescan()

    def test_settings_have_no_secrets(self):
        code, body, _ = self.get("/api/settings")
        d = json.loads(body)
        self.assertEqual(code, 200)
        for k in ("deepseek_key", "deepseek_key_set", "anthropic_key_set", "autopilot", "finish_parallel", "validation_mode"):
            self.assertNotIn(k, d)
        self.assertIn("parallel", d)
        code, d = self.post("/api/settings", {})
        self.assertEqual(code, 200)
        self.assertNotIn("deepseek_key_set", d)


if __name__ == "__main__":
    unittest.main()

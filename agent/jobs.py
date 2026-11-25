# -*- coding: utf-8 -*-
"""Persistent job queue (SQLite), worker pool with timeouts and retries, and the autopilot.

stages:  fit  - local models: marks, camera, ground, 3D people, set, preview      (no Blender)
         ai   - local vision model fills the review form (ai.json)               (optional)
         all  - Blender build + render + videos + meta + zip  (needs a confirmed review)
A job that was running when the app stopped is queued again on the next start."""
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

from . import config
from .clips import folders, status, read_json, fingerprint, record_finish
from . import review

log = logging.getLogger("agent.jobs")
STAGES = ("fit", "background", "depth", "ai", "all")
PRIORITY = {"all": 0, "background": 1, "depth": 1, "fit": 2, "ai": 3}


class JobStore:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, clip TEXT NOT NULL, stage TEXT NOT NULL, state TEXT NOT NULL,
                priority INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, created REAL, started REAL, ended REAL,
                error TEXT, log TEXT)""")
            self.db.execute("CREATE INDEX IF NOT EXISTS jobs_clip ON jobs(clip, id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, priority, id)")
            try:                                                 # the member a job belongs to (older queues: none)
                self.db.execute("ALTER TABLE jobs ADD COLUMN user TEXT")
            except sqlite3.OperationalError:
                pass

    def recover(self):
        """jobs that were running when the app stopped go back to the queue.  Waiting jobs from before members
        existed belong to nobody: they are cancelled."""
        with self.lock:
            self.db.execute("UPDATE jobs SET state='cancelled', ended=? WHERE user IS NULL AND state IN ('queued','running')",
                            (time.time(),))
            n = self.db.execute("UPDATE jobs SET state='queued', started=NULL WHERE state='running'").rowcount
        return n

    def enqueue(self, clip, stage, force=False, user=None):
        if stage not in STAGES:
            raise ValueError("bad stage %r" % stage)
        with self.lock:
            busy = self.db.execute("SELECT id, stage FROM jobs WHERE clip=? AND state IN ('queued','running')", (clip,)).fetchall()
            if busy:
                if not force or any(row['stage']==stage for row in busy): return None
                self.db.execute("UPDATE jobs SET state='cancelled', ended=? WHERE clip=? AND state='queued'", (time.time(),clip))
            cur = self.db.execute("INSERT INTO jobs(clip, stage, state, priority, created, user) VALUES (?,?,?,?,?,?)",
                                  (clip, stage, "queued", PRIORITY[stage], time.time(), user))
            return cur.lastrowid

    def take(self, users=None, prepare_limit=None, finish_limit=None):
        """the next job to run; users: only jobs of these members (the ones who pressed Start), None: any job"""
        if users is not None and not users:
            return None
        with self.lock:
            ai_busy = self.db.execute("SELECT 1 FROM jobs WHERE state='running' AND stage='ai'").fetchone()
            lanes = ""
            if prepare_limit is not None:
                count = self.db.execute("SELECT COUNT(*) FROM jobs WHERE state='running' AND stage!='all'").fetchone()[0]
                if count >= prepare_limit: lanes += " AND stage='all'"
            if finish_limit is not None:
                count = self.db.execute("SELECT COUNT(*) FROM jobs WHERE state='running' AND stage='all'").fetchone()[0]
                if count >= finish_limit: lanes += " AND stage!='all'"
            # one job per clip at a time: a job queued behind its clip's running one (a forced Finish during Prepare) waits
            mine = " AND user IN (%s)" % ",".join("?" * len(users)) if users is not None else ""
            q = ("SELECT * FROM jobs WHERE state='queued'" + (" AND stage!='ai'" if ai_busy else "") + mine + lanes +
                 " AND clip NOT IN (SELECT clip FROM jobs WHERE state='running') ORDER BY priority, id LIMIT 1")
            row = self.db.execute(q, tuple(sorted(users)) if users is not None else ()).fetchone()   # one AI job at a time
            if not row:
                return None
            self.db.execute("UPDATE jobs SET state='running', started=?, attempts=attempts+1 WHERE id=?", (time.time(), row["id"]))
            return dict(row, state="running", attempts=row["attempts"] + 1)

    def finish(self, jid, ok, error=None, log_path=None):
        with self.lock:
            self.db.execute("UPDATE jobs SET state=?, ended=?, error=?, log=COALESCE(?, log) WHERE id=?",
                            ("done" if ok else "failed", time.time(), (error or "")[:2000] or None, log_path, jid))

    def requeue(self, jid):
        with self.lock:
            self.db.execute("UPDATE jobs SET state='queued', started=NULL WHERE id=?", (jid,))

    def cancel_queued(self, clip=None, user=None):
        """cancel waiting jobs: one clip's, one member's, or (neither) all"""
        with self.lock:
            q, args = "UPDATE jobs SET state='cancelled', ended=? WHERE state='queued'", [time.time()]
            if clip:
                q += " AND clip=?"; args.append(clip)
            if user:
                q += " AND user=?"; args.append(user)
            return self.db.execute(q, args).rowcount

    def clear_clip(self, clip, user):
        """Forget a reset clip's job status and timing history."""
        with self.lock:
            return self.db.execute("DELETE FROM jobs WHERE clip=? AND (user=? OR user IS NULL) AND state!='running'",
                                   (clip, user)).rowcount

    def cancel_for_reset(self, user, clips, all_user=False, return_jobs=False, running_only=False):
        """Cancel owned jobs without leaving them eligible for retry or recovery."""
        with self.lock:
            rows=self.db.execute("SELECT * FROM jobs WHERE " +
                                 ("state='running'" if running_only else "state IN ('queued','running')")).fetchall()
            selected=[dict(r) for r in rows if (r['user']==user and (all_user or r['clip'] in clips)) or
                      (r['user'] is None and r['clip'] in clips)]
            ids=[r['id'] for r in selected]
            for jid in ids:
                self.db.execute("UPDATE jobs SET state='cancelled',ended=? WHERE id=?",(time.time(),jid))
            return selected if return_jobs else ids

    def restore_paused(self, pending):
        """Keep interrupted attempt time, and run that stage before later work on its clip."""
        with self.lock:
            for job in pending:
                if job['state']=='queued':
                    self.db.execute("UPDATE jobs SET state='queued',ended=NULL WHERE id=?",(job['id'],))
                else:
                    self.db.execute("""INSERT INTO jobs(clip,stage,state,priority,attempts,created,user,error)
                                       VALUES (?,?,'queued',-1,?,?,?,'Paused by user')""",
                                    (job['clip'],job['stage'],max(0,job['attempts']-1),time.time(),job['user']))

    def mark_stopped(self, pending):
        with self.lock:
            for job in pending:
                self.db.execute("UPDATE jobs SET state='stopped',error='Cancelled by user' WHERE id=?",(job['id'],))

    def clear_user(self, user):
        with self.lock:
            return self.db.execute('DELETE FROM jobs WHERE user=?',(user,)).rowcount

    def latest(self):
        """{clip: latest job (not cancelled)}"""
        with self.lock:
            rows = self.db.execute("""SELECT j.* FROM jobs j JOIN (SELECT clip, MAX(id) m FROM jobs WHERE state!='cancelled'
                                      GROUP BY clip) x ON j.id = x.m""").fetchall()
        return {r["clip"]: dict(r) for r in rows}

    def timing(self, now=None):
        """{clip: {"done": {...} or None, "open": {...} or None}}: how long a video took from its first job to its
        submittable package.  A run starts when the first job after the last finished package starts and ends when the
        next Finish job ("all") is done.  wall = start -> zip (queue waits and your review included); machine = the
        time jobs actually ran.  "open": a run that has started but has no package yet (so far).
        "last": the clip's latest job that ran: {stage, state, ran} (seconds it ran, so far when still running).
        "steps": per step ("prepare" = fit jobs, "finish" = all / render jobs) its latest job that ran:
        {state, ran, ended} - what the progress table on the Clips page shows.
        "machine": all recorded job durations since reset, including work after an earlier Finish."""
        now = now or time.time()
        with self.lock:
            rows = self.db.execute("SELECT clip, stage, state, started, ended FROM jobs WHERE started IS NOT NULL ORDER BY id").fetchall()
        out = {}
        for r in rows:
            t = out.setdefault(r["clip"], {"done": None, "open": None, "last": None, "steps": {}, "machine":0.0, "_s": None, "_m": 0.0})
            t["last"] = {"stage": r["stage"], "state": r["state"], "ran": round(max(0.0, (r["ended"] or now) - r["started"]))}
            if r["state"] != "cancelled":
                step = "prepare" if r["stage"] == "fit" else ("background" if r['stage']=='background' else ("finish" if r["stage"] in ("all", "render", "blend") else None))
                if step:
                    t["steps"][step] = {"state": r["state"], "ran": t["last"]["ran"], "ended": r["ended"]}
            if t["_s"] is None:
                t["_s"] = r["started"]
            duration=max(0.0, (r["ended"] or now) - r["started"]) if r["state"] != "queued" else 0.0
            t["_m"] += duration
            t['machine'] += duration
            if r["stage"] == "all" and r["state"] == "done" and r["ended"]:
                t["done"] = {"start": t["_s"], "end": r["ended"], "wall": round(r["ended"] - t["_s"]), "machine": round(t["_m"])}
                t["_s"], t["_m"] = None, 0.0
        for t in out.values():
            t['machine']=round(t['machine'])
            if t["_s"] is not None:
                t["open"] = {"start": t["_s"], "so_far": round(now - t["_s"]), "machine": round(t["_m"])}
            del t["_s"], t["_m"]
        return out

    def latest_for(self, clip):
        with self.lock:
            r = self.db.execute("SELECT * FROM jobs WHERE clip=? AND state!='cancelled' ORDER BY id DESC LIMIT 1", (clip,)).fetchone()
        return dict(r) if r else None

    def counts(self, user=None):
        """{state: number of jobs}, of one member or (None) everyone's"""
        with self.lock:
            if user is None:
                return {r[0]: r[1] for r in self.db.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state")}
            return {r[0]: r[1] for r in self.db.execute("SELECT state, COUNT(*) FROM jobs WHERE user=? GROUP BY state", (user,))}

    def running(self):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM jobs WHERE state='running' ORDER BY started")]


def kill_tree(proc):
    if os.name == "nt":
        if proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            if proc.poll() is None: proc.kill()


class Runner:
    """worker pool; the number of busy workers follows the 'parallel' setting live.  One queue for every member (one
    NVIDIA card); each member starts and pauses their own jobs."""
    MAX_WORKERS = 16

    def __init__(self, app):
        self.app = app
        self.store = app.store
        self.cond = threading.Condition()
        self.busy = 0
        self.procs = {}
        self.stop = threading.Event()
        self.active = set()       # members who pressed Start: nothing of a member runs before that (even from last time)
        self.resetting = set()
        self.inflight = {}
        self.cancelled = set()

    @property
    def paused(self):
        """no member's jobs run"""
        return not self.active

    def is_paused(self, user):
        return user not in self.active

    def start(self):
        n = self.store.recover()
        if n:
            log.info("%d job(s) that were running when the app stopped wait in the queue for Start", n)
        for i in range(self.MAX_WORKERS):
            threading.Thread(target=self._worker, name="worker-%d" % i, daemon=True).start()
        threading.Thread(target=self._autopilot, name="autopilot", daemon=True).start()

    def wake(self):
        with self.cond:
            self.cond.notify_all()

    def resume(self, user):
        """this member's waiting jobs may run"""
        with self.cond:
            if user not in self.resetting:
                self.active.add(user)
                self.cond.notify_all()

    def pause(self, user=None):
        """no new job of this member (None: of anyone) starts; running ones finish, queued ones wait"""
        with self.cond:
            if user is None:
                self.active.clear()
            else:
                self.active.discard(user)

    def shutdown(self):
        self.stop.set()
        for p in list(self.procs.values()):
            kill_tree(p)
        self.wake()

    @contextmanager
    def reset_jobs(self, user, clips, all_user=False, timeout=60., mode='reset'):
        """Quiesce a member before deleting files; workers must finish cleanup first."""
        clips=set(clips)
        with self.cond:
            while user in self.resetting:
                self.cond.wait()
            self.resetting.add(user)
            was_active=user in self.active
            self.active.discard(user)
            pending=self.store.cancel_for_reset(user,clips,all_user,return_jobs=True,running_only=mode=='pause')
            ids={job['id'] for job in pending}
            ids.update(jid for jid,job in self.inflight.items() if job.get('user')==user and
                       (all_user or job['clip'] in clips))
            known={job['id'] for job in pending}
            pending.extend(dict(self.inflight[jid]) for jid in ids-known if jid in self.inflight and jid in self.cancelled)
            self.cancelled.update(ids)
            processes=[self.procs[jid] for jid in ids if jid in self.procs]
        try:
            for proc in processes:
                kill_tree(proc)
            deadline=time.monotonic()+timeout
            with self.cond:
                while any(jid in self.inflight for jid in ids):
                    remaining=deadline-time.monotonic()
                    if remaining<=0:
                        raise TimeoutError('A worker did not stop; generated files were kept. Try Cancel again or restart the app.')
                    self.cond.wait(min(remaining,.25))
                if mode=='pause':self.store.restore_paused(pending)
                elif mode=='cancel':self.store.mark_stopped(pending)
            if mode in ('pause','cancel'):
                self._discard_incomplete_caches(pending)
            yield len(ids)
        finally:
            with self.cond:
                self.resetting.discard(user)
                self.cancelled.difference_update(jid for jid in ids if jid not in self.inflight)
                # Reset all leaves the member paused, exactly like selecting a folder.
                if was_active and not all_user:
                    self.active.add(user)
                self.cond.notify_all()

    def _discard_incomplete_caches(self, pending):
        """An interrupted ZIP-based numpy write must not poison the next cached run."""
        from pathlib import Path
        import zipfile
        from .clips import _inside
        seen=set()
        for job in pending:
            if job['state']!='running':continue
            ws=self.app.ws(job.get('user')) if job.get('user') else None
            if ws is None:continue
            cl=ws.lib.by_id.get(job['clip'])
            if cl is None:continue
            paths=folders(ws.c,cl)
            for root in (paths['work'],paths['marks']):
                for path in Path(root).rglob('*.npz'):
                    if path in seen or not _inside(str(path),[root]):continue
                    seen.add(path)
                    try:
                        with zipfile.ZipFile(path) as archive:archive.infolist()
                    except zipfile.BadZipFile:
                        try:path.unlink()
                        except OSError:log.warning('Could not remove incomplete cache %s',path)
                    except OSError:pass

    def interrupt(self, user, clip=None, pause=False):
        """Stop active process trees without deleting completed work or timing records."""
        mode='pause' if pause else 'cancel'
        with self.reset_jobs(user,[] if clip is None else [clip],all_user=clip is None,mode=mode) as stopped:
            result={'ok':True,'interrupted':stopped,'cancelled':0 if pause else stopped,'resumable':pause,
                    'note':'Paused. Resume restarts the interrupted stage using completed caches.' if pause else 'Processing cancelled; completed files are kept.'}
        result['paused']=self.is_paused(user)
        return result

    # ---------------------------------------------------------------- workers
    def _worker(self):
        while not self.stop.is_set():
            with self.cond:
                while not self.stop.is_set() and not self.active:
                    self.cond.wait(5)
                if self.stop.is_set():
                    return
                job = self.store.take(set(self.active)-self.resetting, prepare_limit=self.app.c['parallel'],
                                      finish_limit=self.app.c.get('finish_parallel', 1))
                if job is None:
                    self.cond.wait(3)
                    continue
                self.busy += 1
                self.inflight[job['id']]=job
            try:
                self._run(job)
            except Exception as e:                           # never let a worker die
                log.exception("job %s crashed", job["id"])
                with self.cond:
                    if job['id'] not in self.cancelled:
                        self.store.finish(job["id"], False, "internal error: %s" % e)
            finally:
                with self.cond:
                    self.inflight.pop(job['id'],None)
                    self.cancelled.discard(job['id'])
                    self.busy -= 1
                    self.cond.notify_all()

    def _cmd(self, job, cl):
        py = sys.executable
        if job["stage"] == "ai":
            return [py, "-m", "agent.ai", cl.id]
        cmd = [py, os.path.join(config.PIPELINE, "run_pipeline.py"), cl.video, "--device", "cuda", "--stage", job["stage"],
               "--name", cl.name]                         # the app's name (made unique when two clips would share one)
        if job["stage"] == "all":
            cmd += ["--force", "--require-confirm"]
        return cmd

    def _run(self, job):
        with self.cond:
            if job['id'] in self.cancelled: return
        ws = self.app.ws(job["user"]) if job.get("user") else None
        if ws is None:
            self.store.finish(job["id"], False, "this job belongs to no member")
            return
        c = ws.c
        try:
            cl = ws.lib.get(job["clip"])
        except KeyError:
            self.store.finish(job["id"], False, "clip no longer in the source folder")
            return
        f = folders(c, cl)
        if job["stage"] == "ai" and not self.app.ai_ready():
            self.store.finish(job["id"], False, "AI suggestions are off (no DeepSeek key or switched off)")
            return
        os.makedirs(f["work"], exist_ok=True)
        logp = os.path.join(f["work"], "job_%s.log" % job["stage"])
        timeout = 60 * c.get({"fit": "timeout_fit_min", "background": "timeout_fit_min", "depth": "timeout_fit_min", "ai": "timeout_ai_min", "all": "timeout_render_min"}[job["stage"]], 60)
        # the job works in this member's folders (common.member_cfg reads these)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", BA_USER=ws.user, BA_SOURCE=c["source_root"],
                   BA_GPU_LOCK=os.path.join(config.DATA, 'model_inference.lock'))
        cmd = self._cmd(job, cl)
        fp = fingerprint(c, cl) if job["stage"] == "all" else None     # what this Finish is made from
        log.info("job %d %s %s (attempt %d)", job["id"], job["stage"], cl.name, job["attempts"])
        if job["attempts"] > 1 and os.path.exists(logp):                # a retry: keep the failed attempt's log beside it
            os.replace(logp, logp[:-4] + ".previous.log")
        t0 = time.time()
        with open(logp, "w", encoding="utf-8", errors="replace") as lf:
            lf.write("# %s  %s  attempt %d\n$ %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), job["stage"], job["attempts"], " ".join(cmd)))
            lf.flush()
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with self.cond:
                if job['id'] in self.cancelled: return
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=config.ROOT,
                                        creationflags=flags,start_new_session=os.name!='nt')
                self.procs[job["id"]] = proc
            try:
                rc = proc.wait(timeout=timeout)
                err = None
            except subprocess.TimeoutExpired:
                kill_tree(proc)
                rc, err = -1, "timed out after %d min" % (timeout // 60)
            finally:
                with self.cond:
                    self.procs.pop(job["id"], None)
        with self.cond:
            if job['id'] in self.cancelled: return
        with open(logp, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        ok = rc == 0 and err is None                   # the exit code is the result (run_pipeline: 1 failed, 3 not confirmed)
        if not ok and err is None:
            err = ("the review is not confirmed: press Mark reviewed or Finish now" if rc == 3 else
                   last_error(txt) or "exit code %s" % rc)
        with self.cond:
            if job['id'] in self.cancelled: return
            if self.stop.is_set():
                self.store.requeue(job["id"])                      # app closing: do it again next start
                return
            if not ok and rc != 3 and job["attempts"] < c.get("max_attempts", 2):
                log.warning("job %d failed (%s) - trying again", job["id"], err)
                self.store.requeue(job["id"])
                return
            self.store.finish(job["id"], ok, err, logp)
            if ok and fp:
                record_finish(c, cl, fp)
            log.info("job %d %s %s: %s in %.0fs", job["id"], job["stage"], cl.name, "ok" if ok else "FAILED: " + err, time.time() - t0)
            if ok:
                self.after(ws, cl, job["stage"])

    # ---------------------------------------------------------------- what happens next
    def after(self, ws, cl, stage):
        """chain the next step for autopilot / AI"""
        c = ws.c
        f = folders(c, cl)
        if stage == "fit" and self.app.ai_ready() and not os.path.exists(f["ai"]):
            self.store.enqueue(cl.id, "ai", user=ws.user)         # AI looks once per clip
        elif stage == "ai" and os.path.exists(os.path.join(f["marks"], "env_ai.json")):
            self.store.enqueue(cl.id, "fit", user=ws.user)        # rebuild the set with the AI's corrections
        elif stage in ("fit", "ai") and c["autopilot"] == "full":
            self.auto_confirm(ws, cl)
        self.wake()

    def auto_confirm(self, ws, cl):
        c = ws.c
        f = folders(c, cl)
        if read_json(f["review"]):                            # a person already looked at it: keep their choices
            rv = read_json(f["review"])
            if not rv.get("confirmed"):
                return
        else:
            review.write(f["review"], review.suggested(c, cl, confirmed=True, auto=True))
        self.store.enqueue(cl.id, "all", user=ws.user)

    def _autopilot(self):
        """keeps the queue fed while autopilot is on, for every member who pressed Start"""
        while not self.stop.wait(20):
            if self.app.c["autopilot"] == "off":
                continue
            try:
                with self.cond:
                    latest = self.store.latest()
                    for user in sorted(self.active):
                        ws = self.app.ws(user)
                        if ws is None:
                            continue
                        c = ws.c
                        for cl in ws.lib.clips:
                            j = latest.get(cl.id)
                            if j and j['state'] in ('queued','running','failed','stopped'):
                                continue                           # busy, or failed for good: leave for a person
                            st = status(c, cl, j)
                            if st["held"]:
                                continue                           # taken out of the line on the Clips page
                            if not st["fit"]:
                                self.store.enqueue(cl.id, "fit", user=user)
                            elif c["autopilot"] == "full" and not st["done"]:
                                if self.app.ai_ready() and not st["ai"] and not st["reviewed"]:
                                    self.store.enqueue(cl.id, "ai", user=user)
                                elif not st["reviewed"] or st["confirmed"]:
                                    self.auto_confirm(ws, cl)
                    self.wake()
            except Exception:
                log.exception("autopilot pass failed")


def last_error(txt):
    lines = [l for l in txt.strip().splitlines() if l.strip()]
    for i in range(len(lines) - 1, -1, -1):
        if "Error" in lines[i] or "error" in lines[i] or "FAILED" in lines[i]:
            return lines[i].strip()[:300]
    return lines[-1][:300] if lines else ""

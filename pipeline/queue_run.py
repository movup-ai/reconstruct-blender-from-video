# -*- coding: utf-8 -*-
"""One overlapping job queue for the whole clip pipeline: every clip flows through the stages on its own, as soon as a
slot of the right kind is free, so detection (GPU), fitting (CPU), set building and Blender rendering run at the same
time on different clips instead of batch after batch.

  python pipeline\\queue_run.py [--films a,b] [--clips FILE] [--stages fit,build_env,render] [--force fit,render]
                                [--dry-run] [--gpu 1] [--fit 3] [--blender 2] [--light 2] [--min-free-gb 4]
                                [--wait-gates] [--no-auto-roles] [--no-final]

Stages (STAGES below, in order; add a Stage to extend the pipeline):
  detect     GPU    automark.py on the clip -> marks\\pose.json (+ detected_gpu.txt; roles from the detection when the
                    clip has no review.json yet, like batch_prepare)
  confirm    gate   review.json confirmed (in the review app, or the auto roles)
  fit        fit    run_pipeline.py <video> --stage fit -> Work_auto\\<clip>\\report.json, scene_data.json, preview.jpg
  env_auto   GPU    env_auto.py <clip> -> output\\<clip>\\env_auto.json (set proposal; only for clips without a
                    hand-built set; a failure does not stop the clip)
  set        gate   output\\<clip>\\env_claude.json exists (written by Claude agents)
  build_env  light  build_env.py <clip> -> the set in scene_data.json + env_check.jpg
  render     blender run_pipeline.py <video> --stage render -> SUBMIT\\<clip>.zip
  validate   light  agent.validate check of the package -> Work_auto\\<clip>\\validate.json
A job stage runs when an output is missing or one of its inputs is newer than its oldest output (like batch_render's
zip check), or when --force names it.  A gate that is not met holds that clip only; with --wait-gates the queue keeps
polling held clips (e.g. while Claude agents write env_claude.json) instead of finishing.
Failed jobs are retried once, then logged and the clip stops there.  Before a heavy job starts, free RAM and free
commit (small paging file) must both be above --min-free-gb, and heavy starts are spaced a few seconds apart.
At the end: review_page.py (Work_auto\\REVIEW.html) and a validation of every package in SUBMIT.
Progress: logs\\queue.log (one line per stage per clip, a summary line with counts and ETA every minute).
Clips without a film folder (the 3 approved test clips) are never touched.  BA_SANDBOX=<dir> redirects everything."""
import argparse, ctypes, glob, json, os, re, shutil, subprocess, sys, threading, time, queue as _q

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from agent import config                                    # noqa: E402
from agent.clips import Library, folders, read_json         # noqa: E402

PROTECTED = {"0529bfe27f2b1dab7f3ae164b6d05abd", "b5b6e9a3c5730842395b00d2f4b1841c", "ec349385598e67568611cec8540c4f90"}
PY = sys.executable
SLACK = 1.0                                                  # seconds: files written together count as the same time
DEFAULT_SECS = {"detect": 90, "fit": 45, "env_auto": 90, "build_env": 15, "render": 90, "validate": 5}
NEED_GB = {"detect": 3.0, "fit": 4.0, "env_auto": 5.0, "render": 2.5}     # rough peak RAM of one heavy job
RAMP_S = 30                                                  # a heavy job reaches its peak within this time
OOM = re.compile(r"Insufficient memory|MemoryError|Memory allocation still failed|out of memory|bad_alloc|"
                 r"paging file is too small|0xC0000017|1455", re.I)


# ------------------------------------------------------------------------------------------------ config / context
def load_cfg():
    """the app's config with the sandbox redirection of common.cfg (BA_SANDBOX)"""
    c = config.load()
    sb = os.environ.get("BA_SANDBOX")
    if sb:
        c["work_root"] = os.path.join(sb, "Work_auto"); c["submit_root"] = os.path.join(sb, "SUBMIT")
        c["marks_root"] = os.path.join(sb, "output"); c["_sandbox"] = sb
    return c


class Ctx:
    """one clip's paths"""
    def __init__(self, c, cl):
        self.c, self.cl, self.name = c, cl, cl.name
        self.f = folders(c, cl)
        self.work = self.f["work"]
        self.out = os.path.join(c.get("marks_root") or os.path.join(ROOT, "output"), cl.name)     # output\<clip>

    def w(self, fn):
        return os.path.join(self.work, fn)

    def m(self, fn):
        return os.path.join(self.f["marks"], fn)

    def env_claude(self):
        for p in (os.path.join(self.out, "env_claude.json"), self.m("env_claude.json")):
            if os.path.exists(p):
                return p
        return os.path.join(self.out, "env_claude.json")

    def hints(self):
        return [self.m("hints.json"), os.path.join(self.out, "hints.json")]

    def checked(self):
        chk = sorted(glob.glob(self.m("checked*.json")), key=os.path.getmtime)
        return chk[-1:]

    def review(self):
        return read_json(self.f["review"]) if os.path.exists(self.f["review"]) else None


# ------------------------------------------------------------------------------------------------------------ stages
class Stage:
    """kind "job": cmd (list or callable run in-process) on a pool;  kind "gate": gate(ctx, sim) -> (ok, why).
    outputs / inputs / needs: callables ctx -> paths.  needs = inputs that must exist (else the clip is held)."""
    def __init__(self, name, kind="job", pool="light", heavy=False, cmd=None, outputs=None, inputs=None, needs=None,
                 ok=None, skip=None, gate=None, soft=False, timeout_min=60, post=None):
        self.name, self.kind, self.pool, self.heavy, self.cmd = name, kind, pool, heavy, cmd
        self.outputs = outputs or (lambda x: []); self.inputs = inputs or (lambda x: []); self.needs = needs or (lambda x: [])
        self.ok = ok or (lambda x, rc, txt: rc == 0); self.skip = skip or (lambda x, a: None)
        self.gate, self.soft, self.timeout_min, self.post = gate, soft, timeout_min, post


def ok_pipeline(x, rc, txt):
    return rc == 0 and "Traceback" not in txt and " 0 failed" in txt


def person_reviewed(x):
    rv = x.review() or {}
    return bool(rv.get("confirmed") and not rv.get("auto"))


def detect_cmd(x, a):
    c = x.c
    cmd = [PY, os.path.join(ROOT, "automark.py"), x.cl.video, "--out", c.get("marks_root") or os.path.join(ROOT, "output"),
           "--base", c["source_root"], "--device", "cuda"]
    return cmd + (["--force"] if "detect" in a.force else [])


def detect_post(x, a):
    open(x.m("detected_gpu.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    if not os.path.exists(x.f["review"]) and a.auto_roles:
        auto_roles(x)


def auto_roles(x):
    from batch_prepare import auto_review
    rv = auto_review(x.c, x.cl)
    roles = [p["role"] for p in (rv or {}).get("people", {}).values() if p["role"] != "ignore"]
    log("%-9s %-42s auto roles: %s" % ("confirm", x.name[:42], ",".join(roles) or "nobody"))


def confirm_gate(x, sim, a):
    rv = x.review()
    if rv and rv.get("confirmed"):
        return True, ""
    if rv is None and a.auto_roles and (os.path.exists(x.f["pose"]) or x.f["pose"] in sim):
        if sim is None:
            auto_roles(x)
        else:
            sim[x.f["review"]] = "auto roles"
        return True, "auto roles"
    return False, "roles not confirmed in the review app" if rv else "no review.json (auto roles off)"


def set_gate(x, sim, a):
    return (True, "") if os.path.exists(x.env_claude()) else (False, "no env_claude.json yet (set authoring)")


def env_auto_skip(x, a):
    if not os.path.exists(os.path.join(HERE, "env_auto.py")):
        return "env_auto.py not installed"
    if os.path.exists(x.env_claude()) and "env_auto" not in a.force:
        return "set already authored"
    return None


def validate_job(x, a):
    from agent.validate import check_package, check_zip
    sub, zp = x.f["submit"], x.f["zip"]
    if not os.path.isdir(sub):
        P = ["no package folder"]
    else:
        P, _ = check_package(sub)
        P += check_zip(zp, sub)
    with open(x.w("validate.json"), "w", encoding="utf-8") as fh:
        json.dump({"ok": not P, "problems": P, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, ensure_ascii=False, indent=1)
    return (0 if not P else 1), "\n".join(P)


STAGES = [
    Stage("detect", pool="gpu", heavy=True, timeout_min=60, cmd=detect_cmd, post=detect_post,
          outputs=lambda x: [x.f["pose"]],
          skip=lambda x, a: "reviewed by a person" if "detect" in a.force and person_reviewed(x) and os.path.exists(x.f["pose"]) else None),
    Stage("confirm", kind="gate", gate=confirm_gate),
    Stage("fit", pool="fit", heavy=True, timeout_min=40, ok=ok_pipeline,
          cmd=lambda x, a: [PY, os.path.join(HERE, "run_pipeline.py"), x.cl.video, "--stage", "fit", "--device", "cuda"],
          outputs=lambda x: [x.w("report.json"), x.w("preview.jpg"), x.w("clip_info.json"), x.w("camera.json"), x.f["scene"]],
          needs=lambda x: [x.f["pose"]],
          inputs=lambda x: [x.f["pose"], x.f["review"]] + x.checked() + x.hints()),
    Stage("env_auto", pool="gpu", heavy=True, soft=True, timeout_min=30, skip=env_auto_skip,
          cmd=lambda x, a: [PY, os.path.join(HERE, "env_auto.py"), x.name],
          outputs=lambda x: [os.path.join(x.out, "env_auto.json")],
          needs=lambda x: [x.w("camera.json")],
          inputs=lambda x: [x.f["pose"]] + x.hints()),
    Stage("set", kind="gate", gate=set_gate),
    Stage("build_env", pool="light", timeout_min=10,
          cmd=lambda x, a: [PY, os.path.join(HERE, "build_env.py"), x.name],
          outputs=lambda x: [x.w("env_check.jpg")],
          needs=lambda x: [x.env_claude(), x.f["scene"], x.w("clip_info.json")],
          inputs=lambda x: [x.env_claude(), x.w("report.json")] + x.hints()),
    Stage("render", pool="blender", heavy=True, timeout_min=180, ok=lambda x, rc, txt: ok_pipeline(x, rc, txt) and os.path.exists(x.f["zip"]),
          cmd=lambda x, a: [PY, os.path.join(HERE, "run_pipeline.py"), x.cl.video, "--stage", "render"],
          outputs=lambda x: [x.f["zip"]],
          needs=lambda x: [x.f["scene"], x.env_claude()],
          inputs=lambda x: [x.f["scene"], x.env_claude()]),
    Stage("validate", pool="light", timeout_min=10, cmd=validate_job,
          outputs=lambda x: [x.w("validate.json")], needs=lambda x: [x.f["zip"]], inputs=lambda x: [x.f["zip"]]),
]
BY_NAME = {s.name: s for s in STAGES}


def mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def stale(st, x, a, sim):
    """-> reason string when the stage must run, None when its outputs are up to date.  sim: paths an earlier stage
    of this clip will (re)write in this run -> name of that stage"""
    if st.name in a.force:
        return "forced"
    outs = st.outputs(x)
    for p in outs:
        if p in sim:
            return "after %s" % sim[p]
        if mtime(p) is None:
            return "no %s" % os.path.basename(p)
    oldest = min(mtime(p) for p in outs) if outs else 0
    for p in st.inputs(x):
        if p in sim:
            return "after %s" % sim[p]
        t = mtime(p)
        if t is not None and t > oldest + SLACK:
            return "%s newer than %s" % (os.path.basename(p), os.path.basename(min(outs, key=mtime)))
    return None


def missing_needs(st, x, sim):
    return [p for p in st.needs(x) if p not in sim and mtime(p) is None]


def active_stages(a):
    """job stages chosen with --stages, plus every gate that lies before a chosen job stage"""
    jobs = set(a.stages) if a.stages else {s.name for s in STAGES if s.kind == "job"}
    last = max([i for i, s in enumerate(STAGES) if s.name in jobs] or [-1])
    return [s for i, s in enumerate(STAGES) if s.name in jobs or (s.kind == "gate" and i < last)]


def plan(x, a, stages, start=0, sim=None):
    """walk the stages from `start` without running anything -> ([(stage, why)], held_at (stage, why) or None)"""
    sim = dict(sim or {}); todo = []
    for st in stages[start:]:
        if st.kind == "gate":
            ok, why = st.gate(x, sim, a)
            if not ok:
                return todo, (st.name, why)
            continue
        why_skip = st.skip(x, a)
        if why_skip:
            continue
        why = stale(st, x, a, sim)
        if not why:
            continue
        miss = missing_needs(st, x, sim)
        if miss:
            if st.soft:
                continue
            return todo, (st.name, "needs " + ", ".join(os.path.basename(p) for p in miss))
        todo.append((st, why))
        for p in st.outputs(x):
            sim[p] = st.name
    return todo, None


# ------------------------------------------------------------------------------------------------------- utilities
LOG = None
_loglock = threading.Lock()


def log(s):
    s = time.strftime("%H:%M:%S ") + s
    with _loglock:
        print(s, flush=True)
        if LOG:
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(s + "\n")


class _MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_gb():
    """(free RAM, free commit) in GB; commit = RAM + paging file, what runs out first with a small paging file"""
    try:
        import psutil
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        return vm.available / 1e9, (vm.available + sw.free) / 1e9
    except ImportError:
        pass
    try:
        m = _MS(); m.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1e9, m.ullAvailPageFile / 1e9
    except Exception:
        try:
            out = subprocess.run(["wmic", "OS", "get", "FreePhysicalMemory,FreeVirtualMemory", "/value"],
                                 capture_output=True, text=True).stdout
            v = dict(l.strip().split("=", 1) for l in out.splitlines() if "=" in l)
            return int(v["FreePhysicalMemory"]) / 1e6, int(v["FreeVirtualMemory"]) / 1e6
        except Exception:
            return 99.0, 99.0


def fmt_t(s):
    s = int(s)
    return "%dh%02dm" % (s // 3600, s % 3600 // 60) if s >= 3600 else "%dm%02ds" % (s // 60, s % 60)


def times_path(c):
    return os.path.join(c.get("_sandbox") or ROOT, "logs", "queue_times.json")


# ------------------------------------------------------------------------------------------------------- the queue
class ClipRun:
    def __init__(self, x, order):
        self.x, self.order = x, order
        self.idx = 0; self.state = "ready"; self.why = ""; self.tries = {}; self.ran = []
        self.oom = 0; self.not_before = 0.0      # out-of-memory requeues, earliest restart
        self.todo = []                         # stage names still expected to run (for counts / ETA)


class Runner:
    def __init__(self, c, clips, a):
        self.c, self.a = c, a
        self.stages = active_stages(a)
        self.limits = {"gpu": a.gpu, "fit": a.fit, "blender": a.blender, "light": a.light}
        self.runs = [ClipRun(Ctx(c, cl), i) for i, cl in enumerate(clips)]
        self.waiting = {k: [] for k in self.limits}            # pool -> [(ClipRun, Stage)]
        self.running = {k: 0 for k in self.limits}
        self.results = _q.Queue()
        self.procs = set(); self.plock = threading.Lock()
        self.last_heavy = 0.0; self.mem_note = 0.0
        self.starts = []                                       # (time, GB) of heavy jobs still ramping up
        self.cut = {}                                          # pool -> (slots taken away, until) after out-of-memory
        self.done_n = {s.name: 0 for s in STAGES}; self.fail_n = {s.name: 0 for s in STAGES}
        self.secs = dict(DEFAULT_SECS); self.secs.update(read_json(times_path(c), {}) or {})
        self.measured = {}

    # ---- clip state machine (main thread only)
    def advance(self, r):
        """move a ready clip forward until it needs a job, is held at a gate, or is finished"""
        while r.idx < len(self.stages):
            st = self.stages[r.idx]; x = r.x
            if st.kind == "gate":
                ok, why = st.gate(x, None, self.a)
                if not ok:
                    if r.state != "held" or r.why != why:
                        log("%-9s %-42s held: %s" % (st.name, x.name[:42], why))
                    r.state, r.why = "held", why; return
                r.idx += 1; continue
            why_skip = st.skip(x, self.a)
            why = None if why_skip else stale(st, x, self.a, {})
            if st.name in r.ran and st.name in self.a.force:
                why = None                                     # forced once per run
            if why_skip or not why:
                r.idx += 1; continue
            miss = missing_needs(st, x, {})
            if miss:
                if st.soft:
                    r.idx += 1; continue
                why = "needs " + ", ".join(os.path.basename(p) for p in miss)
                if r.state != "held" or r.why != why:
                    log("%-9s %-42s held: %s" % (st.name, x.name[:42], why))
                r.state, r.why = "held", why; return
            r.state, r.why = "queued", why
            self.waiting[st.pool].append((r, st)); return
        if r.state != "done":
            log("%-9s %-42s %s" % ("done", r.x.name[:42], "finished: " + ",".join(r.ran) if r.ran else "up to date"))
        r.state = "done"

    def dispatch(self):
        for pool, lst in self.waiting.items():
            lst.sort(key=lambda rs: (-self.stages.index(rs[1]), rs[0].order))        # clips further along first
            now = time.time()
            cut, until = self.cut.get(pool, (0, 0))
            limit = max(1, self.limits[pool] - (cut if now < until else 0))
            ready = [rs for rs in lst if rs[0].not_before <= now]
            while ready and self.running[pool] < limit:
                r, st = ready[0]
                if st.heavy:
                    if now - self.last_heavy < self.a.stagger:
                        break
                    self.starts = [(t, g) for t, g in self.starts if now - t < RAMP_S]
                    want = self.a.min_free_gb + self.a.reserve * sum(g for t, g in self.starts)      # + what just-started jobs will still take
                    ram, commit = free_gb()
                    if min(ram, commit) < want:
                        if now - self.mem_note > 60:
                            log("memory    waiting: %.1f GB RAM / %.1f GB commit free, %s %s needs %.1f GB"
                                % (ram, commit, st.name, r.x.name[:30], want))
                            self.mem_note = now
                        break
                    self.last_heavy = now; self.starts.append((now, NEED_GB.get(st.name, 2.0)))
                ready.pop(0); lst.remove((r, st))
                self.running[pool] += 1; r.state = "running"
                n = r.tries.get(st.name, 0) + 1; r.tries[st.name] = n
                log("%-9s %-42s start%s (%s)" % (st.name, r.x.name[:42], " (retry)" if n > 1 else "", r.why))
                threading.Thread(target=self.job, args=(r, st), daemon=True).start()

    def job(self, r, st):
        x = r.x; t0 = time.time()
        os.makedirs(x.work, exist_ok=True)
        lp = x.w("job_%s.log" % st.name)
        try:
            cmd = st.cmd(x, self.a)
            if isinstance(cmd, tuple):                     # an in-process stage already ran: (rc, text)
                rc, txt = cmd
            else:
                with open(lp, "w", encoding="utf-8", errors="replace") as fh:
                    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                                         env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                                         creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if st.pool == "light" else 0)
                    with self.plock:
                        self.procs.add(p)
                    try:
                        rc = p.wait(timeout=st.timeout_min * 60)
                    except subprocess.TimeoutExpired:
                        p.kill(); rc = -9
                    with self.plock:
                        self.procs.discard(p)
                txt = open(lp, encoding="utf-8", errors="replace").read()
                if rc == -9:
                    txt += "\n[queue] timeout after %d min" % st.timeout_min
            ok = st.ok(x, rc, txt)
            if ok and st.post:
                st.post(x, self.a)
            err = "" if ok else (txt.strip().splitlines()[-1][:160] if txt.strip() else "exit %s" % rc)
            oom = not ok and bool(OOM.search(txt[-20000:]))
        except MemoryError as e:
            ok, err, oom = False, "MemoryError", True
        except Exception as e:
            ok, err, oom = False, "%s: %s" % (type(e).__name__, e), False
        self.results.put((r, st, ok, time.time() - t0, err, lp, oom))

    def finish(self, r, st, ok, dt, err, lp, oom=False):
        self.running[st.pool] -= 1
        if not ok and oom and r.oom < self.a.oom_retries:
            # out of memory is the machine's fault, not the clip's: wait, run one fewer of this kind for a while,
            # and try again without using up the clip's retry
            r.oom += 1; r.tries[st.name] -= 1; r.not_before = time.time() + 60
            cut = self.cut.get(st.pool, (0, 0))[0] if time.time() < self.cut.get(st.pool, (0, 0))[1] else 0
            self.cut[st.pool] = (min(cut + 1, self.limits[st.pool] - 1), time.time() + 600)
            log("%-9s %-42s OUT OF MEMORY after %.0fs, again in 60 s (%s slots %d for 10 min)"
                % (st.name, r.x.name[:42], dt, st.pool, max(1, self.limits[st.pool] - self.cut[st.pool][0])))
            r.state = "queued"; self.waiting[st.pool].append((r, st)); return
        if ok:
            self.done_n[st.name] += 1; r.ran.append(st.name)
            self.measured.setdefault(st.name, []).append(dt)
            if st.name in r.todo:
                r.todo.remove(st.name)
            log("%-9s %-42s ok %.0fs" % (st.name, r.x.name[:42], dt))
            r.idx += 1; r.state = "ready"; self.advance(r)
        elif r.tries.get(st.name, 0) < 2:
            log("%-9s %-42s FAILED %.0fs, retrying in 20 s: %s" % (st.name, r.x.name[:42], dt, err))
            r.not_before = time.time() + 20
            r.state = "queued"; self.waiting[st.pool].append((r, st))
        else:
            self.fail_n[st.name] += 1
            log("%-9s %-42s FAILED twice (%s): %s" % (st.name, r.x.name[:42], lp, err))
            if st.soft:
                if st.name in r.todo:
                    r.todo.remove(st.name)
                r.idx += 1; r.state = "ready"; self.advance(r)
            else:
                r.state, r.why = "failed", "%s failed" % st.name; r.todo = []

    def avg(self, s):
        m = self.measured.get(s)
        return sum(m) / len(m) if m else self.secs.get(s, 60)

    def summary(self):
        parts = []
        for st in self.stages:
            if st.kind != "job":
                continue
            q = sum(1 for lst in self.waiting.values() for r, s in lst if s is st)
            run = sum(1 for r in self.runs if r.state == "running" and self.stages[r.idx] is st)
            left = sum(1 for r in self.runs if st.name in r.todo) - q - run
            if q or run or left > 0 or self.done_n[st.name] or self.fail_n[st.name]:
                parts.append("%s %d/%d run/q, %d ok%s%s" % (st.name, run, q, self.done_n[st.name],
                             ", %d failed" % self.fail_n[st.name] if self.fail_n[st.name] else "",
                             ", %d later" % left if left > 0 else ""))
        st_n = {k: sum(1 for r in self.runs if r.state == k) for k in ("done", "held", "failed")}
        load = {}
        for r in self.runs:                                    # remaining work per pool / its slots; pools overlap
            for s in r.todo:
                p = BY_NAME[s].pool; load[p] = load.get(p, 0) + self.avg(s)
        eta = max([v / self.limits[p] for p, v in load.items()] or [0])
        ram, commit = free_gb()
        log("SUMMARY  %s | clips: %d done, %d held, %d failed of %d | free %.1f GB | ETA %s"
            % ("; ".join(parts) or "idle", st_n["done"], st_n["held"], st_n["failed"], len(self.runs), min(ram, commit), fmt_t(eta)))

    def kill_all(self):
        with self.plock:
            for p in list(self.procs):
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                except Exception:
                    p.kill()

    def run(self):
        t0 = time.time()
        for r in self.runs:
            r.todo = [st.name for st, why in plan(r.x, self.a, self.stages)[0]]
        log("start: %d clips, stages %s, slots %s, %d jobs expected"
            % (len(self.runs), ",".join(s.name for s in self.stages), " ".join("%s %d" % kv for kv in self.limits.items()),
               sum(len(r.todo) for r in self.runs)))
        for r in self.runs:
            self.advance(r)
        last_sum, last_gate = time.time(), time.time()
        try:
            while True:
                self.dispatch()
                try:
                    res = self.results.get(timeout=1.0)
                    self.finish(*res)
                    while True:
                        self.finish(*self.results.get_nowait())
                except _q.Empty:
                    pass
                busy = any(self.running.values()) or any(self.waiting.values())
                held = [r for r in self.runs if r.state == "held"]
                if held and (not busy or time.time() - last_gate > self.a.gate_poll):
                    last_gate = time.time()
                    for r in held:                      # still "held": advance() logs only a changed reason
                        self.advance(r)
                    busy = any(self.running.values()) or any(self.waiting.values())
                if time.time() - last_sum > self.a.summary_every:
                    self.summary(); last_sum = time.time()
                if not busy and not (self.a.wait_gates and any(r.state == "held" for r in self.runs)):
                    break
                if not busy:
                    time.sleep(min(10, self.a.gate_poll))
        except KeyboardInterrupt:
            log("interrupted: stopping running jobs"); self.kill_all(); raise
        self.summary()
        hist = dict(self.secs)
        for s, m in self.measured.items():
            hist[s] = round(0.5 * hist.get(s, sum(m) / len(m)) + 0.5 * sum(m) / len(m), 1)
        try:
            os.makedirs(os.path.dirname(times_path(self.c)), exist_ok=True)
            json.dump(hist, open(times_path(self.c), "w", encoding="utf-8"), indent=1)
        except OSError:
            pass
        n_f = sum(r.state == "failed" for r in self.runs)
        log("queue done in %s: %d jobs ok, %d failed clips, %d held"
            % (fmt_t(time.time() - t0), sum(self.done_n.values()), n_f, sum(r.state == "held" for r in self.runs)))
        for r in self.runs:
            if r.state in ("failed", "held"):
                log("  %-6s %-42s %s" % (r.state, r.x.name[:42], r.why))
        return n_f


# -------------------------------------------------------------------------------------------------------- main
def seed_sandbox(c, clips):
    """sandbox: start from copies of the real marks and set folders (what run_pipeline does on a fit)"""
    from run_pipeline import marks_dir
    real = dict(c); real["marks_root"] = os.path.join(ROOT, "output")
    for cl in clips:
        for src, dst in ((marks_dir(cl.video, real), marks_dir(cl.video, c)),
                         (os.path.join(ROOT, "output", cl.name), os.path.join(c["marks_root"], cl.name))):
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)


def final_steps(c):
    log("final: review page ...")
    t0 = time.time()
    rc = subprocess.run([PY, os.path.join(HERE, "review_page.py")], cwd=ROOT, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    log("final: review page %s (%.0fs) %s" % ("ok" if rc.returncode == 0 else "FAILED", time.time() - t0,
                                              (rc.stdout.strip().splitlines() or [""])[0][:150]))
    from agent.validate import check_all
    if os.path.isdir(c["submit_root"]):
        rows = check_all(c["submit_root"])
        bad = [r for r in rows if not r["ok"]]
        log("final: validate %s: %d packages, %d ok" % (c["submit_root"], len(rows), len(rows) - len(bad)))
        for r in bad:
            log("  FAIL %-40s %s" % (r["package"][:40], "; ".join(r["problems"])[:200]))


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--films", default="", help="only films whose folder name contains one of these (comma list)")
    ap.add_argument("--clips", default="", help="a file with one clip name per line: only these")
    ap.add_argument("--stages", default="", help="comma list of job stages to run (default: all): " + ",".join(s.name for s in STAGES if s.kind == "job"))
    ap.add_argument("--force", default="", help="comma list of stages to run even when up to date")
    ap.add_argument("--dry-run", action="store_true", help="print what would run and why, run nothing")
    ap.add_argument("--gpu", type=int, default=1); ap.add_argument("--fit", type=int, default=3)
    ap.add_argument("--blender", type=int, default=2); ap.add_argument("--light", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=4.0, help="free RAM and commit needed before a heavy job starts")
    ap.add_argument("--stagger", type=float, default=8.0, help="seconds between heavy job starts (memory ramps up)")
    ap.add_argument("--reserve", type=float, default=0.5, help="scale of the RAM kept back for heavy jobs started in the "
                    "last %d s (NEED_GB); 0 = check only --min-free-gb" % RAMP_S)
    ap.add_argument("--oom-retries", type=int, default=3, help="extra tries of a job that ran out of memory")
    ap.add_argument("--wait-gates", action="store_true", help="keep running while clips wait at a gate (polls them)")
    ap.add_argument("--gate-poll", type=float, default=30.0)
    ap.add_argument("--summary-every", type=float, default=60.0)
    ap.add_argument("--no-auto-roles", dest="auto_roles", action="store_false", help="wait for the review app instead of auto roles")
    ap.add_argument("--no-final", action="store_true", help="skip review page + validation at the end")
    a = ap.parse_args()
    known = {s.name for s in STAGES}
    a.stages = [s for s in a.stages.split(",") if s]
    a.force = {s for s in a.force.split(",") if s}
    for s in list(a.stages) + list(a.force):
        if s not in known or BY_NAME[s].kind != "job":
            ap.error("unknown job stage %r (stages: %s)" % (s, ", ".join(n for n in known if BY_NAME[n].kind == "job")))
    if not 1 <= a.gpu <= 2:
        ap.error("--gpu must be 1 or 2 (12 GB VRAM)")
    c = load_cfg()
    LOG = os.path.join(c.get("_sandbox") or ROOT, "logs", "queue.log")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    lib = Library(c)
    clips = [cl for cl in lib.clips if cl.film and cl.name not in PROTECTED and not any(p in cl.name for p in PROTECTED)]
    if a.films:
        clips = [cl for cl in clips if any(k in cl.film for k in a.films.split(","))]
    if a.clips:
        want = {l.strip() for l in open(a.clips, encoding="utf-8-sig") if l.strip()}
        clips = [cl for cl in clips if cl.name in want]
    if c.get("_sandbox"):                                   # (also for --dry-run: the sandbox is a test space)
        seed_sandbox(c, clips)
    if a.dry_run:
        LOG = None
        stages = active_stages(a)
        tot, held, quiet, counts = 0, 0, 0, {}
        print("dry run: %d clips, stages %s%s" % (len(clips), ",".join(s.name for s in stages),
                                                  ", sandbox " + c["_sandbox"] if c.get("_sandbox") else ""))
        load = {}
        for cl in clips:
            x = Ctx(c, cl)
            todo, h = plan(x, a, stages)
            if not todo and not h:
                quiet += 1; continue
            tot += bool(todo); held += bool(h)
            for st, why in todo:
                counts[st.name] = counts.get(st.name, 0) + 1
                load[st.pool] = load.get(st.pool, 0) + (read_json(times_path(c), {}) or DEFAULT_SECS).get(st.name, DEFAULT_SECS[st.name])
            print("  %-44s %s%s" % (cl.name[:44], "  ".join("%s (%s)" % (st.name, why) for st, why in todo),
                                    ("  | held at %s: %s" % h) if h else ""))
        lim = {"gpu": a.gpu, "fit": a.fit, "blender": a.blender, "light": a.light}
        eta = max([v / lim[p] for p, v in load.items()] or [0])
        print("%d clips with work, %d held (some also have work before the gate), %d up to date; jobs: %s; estimated %s"
              % (tot, held, quiet,
                 ", ".join("%s %d" % kv for kv in counts.items()) or "none", fmt_t(eta)))
        return 0
    rc = Runner(c, clips, a).run()
    if not a.no_final:
        final_steps(c)
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())

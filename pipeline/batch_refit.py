# -*- coding: utf-8 -*-
"""Fit every clip again with the current code (camera, cuts, people), then rebuild its hand-built set.

  python pipeline\\batch_refit.py [--films a,b] [--workers 6]
Progress: logs\\refit.log (one line per clip: fit / set result, shots, people)."""
import argparse, os, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from agent import config
from agent.clips import Library, folders

LOG = os.path.join(ROOT, "logs", "refit.log")
_lock = threading.Lock()


def log(s):
    s = time.strftime("%H:%M:%S ") + s
    print(s, flush=True)
    with _lock, open(LOG, "a", encoding="utf-8") as fh:
        fh.write(s + "\n")


def one(c, cl):
    f = folders(c, cl); os.makedirs(f["work"], exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8"); t0 = time.time()
    lp = os.path.join(f["work"], "job_fit.log")
    with open(lp, "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run([sys.executable, os.path.join(HERE, "run_pipeline.py"), cl.video, "--stage", "fit"],
                            stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode
    txt = open(lp, encoding="utf-8", errors="replace").read()
    ok = rc == 0 and "Traceback" not in txt and " 0 failed" in txt
    shots = txt.count("[shot ") or 1
    if not ok:
        log("FAILED fit %-40s (see %s)" % (cl.name[:40], lp)); return False
    with open(os.path.join(f["work"], "job_env.log"), "w", encoding="utf-8", errors="replace") as fh:
        rc2 = subprocess.run([sys.executable, os.path.join(HERE, "build_env.py"), cl.name], stdout=fh, stderr=subprocess.STDOUT,
                             cwd=ROOT, env=env).returncode
    log("%s %-40s shots %d  %.0fs" % ("ok    " if rc2 == 0 else "NOSET ", cl.name[:40], shots, time.time() - t0))
    return rc2 == 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--films", default=""); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    c = config.load(); lib = Library(c)
    clips = [cl for cl in lib.clips if cl.film and (not a.films or any(k in cl.film for k in a.films.split(",")))]
    log("start: %d clips, %d at a time" % (len(clips), a.workers))
    with ThreadPoolExecutor(a.workers) as pool:
        res = list(pool.map(lambda cl: one(c, cl), clips))
    log("done: %d ok, %d not ok" % (sum(res), len(res) - sum(res)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Render and package every clip that has a hand-built set (output\\<clip>\\env_claude.json).

  python pipeline\\batch_render.py [--films a,b] [--workers 2] [--force]

Per clip: build_env.py (the set into scene_data.json, in case a fit replaced it), then run_pipeline --stage render
(Blender + videos + meta + zip, same layout as the client sample).  Clips whose zip is newer than their scene are
skipped unless --force.  Progress: logs\\batch_render.log"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from agent import config
from agent.clips import Library, folders

LOG = os.path.join(ROOT, "logs", "batch_render.log")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(s + "\n")


def run(cmd, lp):
    with open(lp, "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode
    return rc, open(lp, encoding="utf-8", errors="replace").read()


def one(c, cl, force):
    f = folders(c, cl)
    envp = os.path.join(ROOT, "output", cl.name, "env_claude.json")
    if not os.path.exists(envp):
        envp = os.path.join(f["marks"], "env_claude.json")
    if not os.path.exists(envp):
        log("skip %-40s no hand-built set yet" % cl.name[:40]); return None
    if not os.path.exists(f["scene"]):
        log("skip %-40s not fitted" % cl.name[:40]); return None
    if not force and os.path.exists(f["zip"]) and os.path.getmtime(f["zip"]) > max(
            os.path.getmtime(f["scene"]), os.path.getmtime(envp)):
        log("skip %-40s zip is up to date" % cl.name[:40]); return True
    t0 = time.time()
    rc, txt = run([sys.executable, os.path.join(HERE, "build_env.py"), cl.name], os.path.join(f["work"], "job_env.log"))
    if rc != 0:
        log("FAILED %-38s set: %s" % (cl.name[:38], txt.strip().splitlines()[-1][:200] if txt.strip() else rc)); return False
    rc, txt = run([sys.executable, os.path.join(HERE, "run_pipeline.py"), cl.video, "--stage", "render"], os.path.join(f["work"], "job_render.log"))
    ok = rc == 0 and "Traceback" not in txt and " 0 failed" in txt and os.path.exists(f["zip"])
    log("%s %-40s %.0fs" % ("ok    " if ok else "FAILED", cl.name[:40], time.time() - t0))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", default=""); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--force", action="store_true")
    ap.add_argument("--clips", default="", help="a file with one clip name per line: only these")
    a = ap.parse_args()
    c = config.load(); lib = Library(c)
    clips = [cl for cl in lib.clips if cl.film]
    if a.films:
        clips = [cl for cl in clips if any(k in cl.film for k in a.films.split(","))]
    if a.clips:
        want = {l.strip() for l in open(a.clips, encoding="utf-8") if l.strip()}
        clips = [cl for cl in clips if cl.name in want]
    log("start: %d clips" % len(clips))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(lambda cl: one(c, cl, a.force), clips))
    log("done: %d ok, %d failed, %d without a set" % (sum(r is True for r in res), sum(r is False for r in res), sum(r is None for r in res)))


if __name__ == "__main__":
    main()

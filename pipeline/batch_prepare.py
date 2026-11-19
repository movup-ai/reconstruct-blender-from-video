# -*- coding: utf-8 -*-
"""Prepare every clip for Claude: local-model detection (GPU), roles from the detection, fit (camera, people, plates).

  python pipeline\\batch_prepare.py [--films a,b] [--workers 3] [--redetect]

Per film: automark on the GPU (one model load for the whole film), then roles are written from the detection for clips
that have no review yet (review.json, auto), then the fits run in parallel while the next film is being detected.
Clips a person already reviewed keep their marks and review.  Progress: logs\\batch_prepare.log"""
import argparse, json, os, subprocess, sys, time, shutil
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from agent import config
from agent.clips import Library, folders, read_json
from agent import review as review_mod

LOG = os.path.join(ROOT, "logs", "batch_prepare.log")
ORDER = ["main", "second", "third", "fourth"]


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(s + "\n")


def auto_review(c, cl):
    """roles from the detection: biggest / longest-seen people first, small or short-lived detections ignored"""
    f = folders(c, cl)
    pose = read_json(f["pose"])
    if not pose:
        return None
    H = float(pose["picture_area"][3]) if pose.get("picture_area") else float(pose["height"])
    N = float(pose["frames"])
    rows = []
    for p in pose["people"]:
        hs = sorted(b[3] - b[1] for b in p["boxes"].values())
        if not hs:
            continue
        h_med = hs[len(hs) // 2] / H
        n = len(p["frames"])
        rows.append((h_med * min(1.0, n / max(1.0, 0.5 * N)), p["id"], n, h_med))
    rows.sort(reverse=True)
    people, rank, extra = {}, 0, 0
    for score, pid, n, h_med in rows:
        if n < max(12, 0.08 * N) or h_med < 0.07:
            role = "ignore"
        elif rank < len(ORDER):
            role = ORDER[rank]; rank += 1
        elif extra < 2:
            role = "extra"; extra += 1
        else:
            role = "ignore"
        people[str(pid)] = {"role": role, "colour": review_mod.ROLE_COLOUR.get(role) or "purple",
                            "height": float(c["person_height"]), "name": ""}
    from dedupe_roles import duplicates                     # the same body tracked twice: keep the bigger / longer track
    for d, k, s in duplicates(pose, {k for k, p in people.items() if p["role"] != "ignore"}):
        people[d]["role"] = "ignore"; people[d]["colour"] = "purple"
    live = sorted([k for k, p in people.items() if p["role"] in ORDER], key=lambda k: ORDER.index(people[k]["role"]))
    for n, k in enumerate(live):
        people[k]["role"] = ORDER[n]; people[k]["colour"] = review_mod.ROLE_COLOUR[ORDER[n]]
    rv = {"clip": cl.name, "video": cl.video, "confirmed": True, "auto": True, "confirmed_by": "local model (auto roles)",
          "people": people, "camera": {"type": "pan", "lens": None, "note": ""},
          "environment": {"ground": True, "depth_set": True, "use_ai": True, "set_colours": "grey", "props": []}, "notes": ""}
    review_mod.write(f["review"], rv)
    return rv


def fit(c, cl):
    f = folders(c, cl)
    os.makedirs(f["work"], exist_ok=True)
    lp = os.path.join(f["work"], "job_fit.log")
    t0 = time.time()
    with open(lp, "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run([sys.executable, os.path.join(HERE, "run_pipeline.py"), cl.video, "--stage", "fit", "--device", "cuda"],
                            stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode
    txt = open(lp, encoding="utf-8", errors="replace").read()
    ok = rc == 0 and "Traceback" not in txt and " 0 failed" in txt
    log("fit %-40s %s %.0fs" % (cl.name[:40], "ok" if ok else "FAILED (see %s)" % lp, time.time() - t0))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--redetect", action="store_true", help="detect again even where marks exist (never for person-reviewed clips)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    c = config.load()
    lib = Library(c)
    films = sorted({cl.film for cl in lib.clips if cl.film})
    if a.films:
        films = [fm for fm in films if any(k in fm for k in a.films.split(","))]
    log("start: %d films, %d clips" % (len(films), sum(cl.film in films for cl in lib.clips)))
    pool = ThreadPoolExecutor(max_workers=a.workers)
    futs = []
    for film in films:
        clips = [cl for cl in lib.clips if cl.film == film]
        for cl in clips:                                       # marks from the old CPU model: detect again on the GPU
            f = folders(c, cl)
            rv = read_json(f["review"]) or {}
            person = rv.get("confirmed") and not rv.get("auto")
            if os.path.exists(f["pose"]) and not person and (a.redetect or not os.path.exists(os.path.join(f["marks"], "detected_gpu.txt"))):
                shutil.move(f["pose"], f["pose"].replace("pose.json", "pose_cpu_old.json"))
                raw = f["pose"].replace("pose.json", "pose_raw.json")
                if os.path.exists(raw):
                    shutil.move(raw, raw.replace("pose_raw.json", "pose_raw_cpu_old.json"))
        t0 = time.time()
        film_dir = os.path.join(c["source_root"], film)            # the whole film folder (some films keep each clip in its own sub-folder)
        rc = subprocess.run([sys.executable, os.path.join(ROOT, "automark.py"), film_dir, "--out", os.path.join(ROOT, "output"),
                             "--base", c["source_root"], "--device", "cuda"], cwd=ROOT,
                            stdout=open(os.path.join(ROOT, "logs", "automark_%d.log" % films.index(film)), "w", encoding="utf-8"),
                            stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode
        log("detect %s: %d clips, exit %d, %.0fs" % (film, len(clips), rc, time.time() - t0))
        for cl in clips:
            f = folders(c, cl)
            if not os.path.exists(f["pose"]):
                log("  no marks for", cl.name); continue
            open(os.path.join(f["marks"], "detected_gpu.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
            rv = read_json(f["review"]) or {}
            if not rv.get("confirmed"):
                rv = auto_review(c, cl)
                roles = [p["role"] for p in rv["people"].values() if p["role"] != "ignore"]
                log("  roles %-40s %s" % (cl.name[:40], ",".join(roles) or "nobody"))
            if os.path.exists(f["scene"]) and os.path.exists(f["preview"]) and os.path.getmtime(f["scene"]) > os.path.getmtime(f["pose"]) and not a.redetect:
                continue                                       # fitted after its current marks: nothing to do
            futs.append(pool.submit(fit, c, cl))
    n_ok = sum(1 for fu in futs if fu.result())
    log("done: %d of %d fits ok" % (n_ok, len(futs)))


if __name__ == "__main__":
    main()

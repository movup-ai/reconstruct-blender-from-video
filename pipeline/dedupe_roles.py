# -*- coding: utf-8 -*-
"""Find people the detector saw twice (two tracks on the same body) in the automatic roles, and mark the smaller /
shorter-lived copy "ignore".  Only auto reviews (review.json with "auto": true) are changed.

  python pipeline\\dedupe_roles.py [--apply] [--films a,b]
Prints, per clip, the duplicate pairs it finds; with --apply writes the reviews and lists the clips to fit again."""
import argparse, os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from agent import config
from agent.clips import Library, folders, read_json
from agent import review as review_mod


def overlap(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def same_skeleton(ka, kb, box):
    """the two tracks' joints lie on one body (a person half hidden right behind another has his own joints:
    overlapping boxes alone do not make a duplicate).  Without enough shared joints the boxes decide."""
    if ka is None or kb is None:
        return True
    a, b = np.asarray(ka, float), np.asarray(kb, float)
    vis = (a[:, 2] > 0.3) & (b[:, 2] > 0.3)
    if vis.sum() < 5:
        return True
    h = max(10.0, box[3] - box[1])
    return float(np.median(np.linalg.norm(a[vis, :2] - b[vis, :2], axis=1))) < 0.1 * h


def duplicates(pose, active):
    """[(drop id, keep id, share of shared frames where they cover the same body)]"""
    T = {str(p["id"]): p for p in pose["people"] if str(p["id"]) in active}
    out = []
    ids = sorted(T, key=lambda i: -len(T[i]["boxes"]))
    for n, i in enumerate(ids):
        for j in ids[n + 1:]:
            common = set(T[i]["boxes"]) & set(T[j]["boxes"])
            if len(common) < 5:
                continue
            same = np.mean([overlap(T[i]["boxes"][f][:4], T[j]["boxes"][f][:4]) > 0.6 and
                            same_skeleton(T[i]["frames"].get(f), T[j]["frames"].get(f), T[i]["boxes"][f])
                            for f in common])
            if same > 0.5:
                out.append((j, i, float(same)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); ap.add_argument("--films", default="")
    a = ap.parse_args()
    c = config.load(); lib = Library(c)
    refit = []
    for cl in lib.clips:
        if not cl.film or (a.films and not any(k in cl.film for k in a.films.split(","))):
            continue
        f = folders(c, cl); rv = read_json(f["review"]); pose = read_json(f["pose"])
        if not rv or not pose or not rv.get("auto"):
            continue
        active = {k for k, p in rv["people"].items() if p.get("role") != "ignore"}
        dups = duplicates(pose, active)
        if not dups:
            continue
        print("%-42s %s" % (cl.name[:42], ", ".join("P%s=P%s (%.0f%%)" % (d, k, s * 100) for d, k, s in dups)))
        if a.apply:
            for d, k, s in dups:
                rv["people"][d]["role"] = "ignore"; rv["people"][d]["colour"] = "purple"
            order = ["main", "second", "third", "fourth"]                  # close the gaps in the role order
            live = [k for k, p in rv["people"].items() if p["role"] in order]
            live.sort(key=lambda k: order.index(rv["people"][k]["role"]))
            for n, k in enumerate(live):
                rv["people"][k]["role"] = order[n]; rv["people"][k]["colour"] = review_mod.ROLE_COLOUR[order[n]]
            review_mod.write(f["review"], rv)
            refit.append(cl.name)
    print("clips with duplicates: %d" % len(refit) if a.apply else "")
    if a.apply:
        json.dump(refit, open(os.path.join(ROOT, "logs", "dedupe_refit.json"), "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

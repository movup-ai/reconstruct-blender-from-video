# -*- coding: utf-8 -*-
"""Review sheets for packaged clips: compare.py (film | render | outlines on film) at 4 frames spread over the clip.

  python pipeline\\review_sheets.py [--films 隐剑] [--out Work_auto\\review_v21] [--zoom]"""
import argparse, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--films", default="隐剑")
    ap.add_argument("--out", default=os.path.join(ROOT, "Work_auto", "review_v21")); ap.add_argument("--zoom", action="store_true")
    ap.add_argument("--only", default=""); ap.add_argument("--n", type=int, default=32, help="frames per clip (8 per sheet)")
    a = ap.parse_args()
    c = cfg(); os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8") as fh:
        names = json.load(fh)
    only = set(a.only.split(",")) if a.only else None
    for clip, pkg in names.items():
        if a.films and a.films not in clip or (only and pkg not in only):
            continue
        with open(os.path.join(c["work_root"], clip, "scene_data.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        N = d["frames"]
        if a.n <= 4:
            groups = [[min(N - 1, int(N * x)) for x in (0.05, 0.35, 0.65, 0.95)]]
        else:                                         # a.n frames evenly over the clip, 8 per sheet
            allf = sorted(set(np.linspace(0, N - 1, a.n).astype(int).tolist()))
            groups = [allf[i:i + 8] for i in range(0, len(allf), 8)]
        for gi, g in enumerate(groups):
            out = os.path.join(a.out, pkg + ("_zoom" if a.zoom else "") + ("" if len(groups) == 1 else "_%d" % (gi + 1)) + ".jpg")
            cmd = [sys.executable, os.path.join(HERE, "compare.py"), clip, "--frames", ",".join(map(str, g)), "--out", out]
            cmd += ["--zoom", "--size", "220"] if a.zoom else ["--width", "360"]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(pkg, gi + 1, "ok" if p.returncode == 0 else p.stderr[-300:])


if __name__ == "__main__":
    main()

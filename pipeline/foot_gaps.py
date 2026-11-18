# -*- coding: utf-8 -*-
"""Feet vs floor for every person of a fitted clip, and a suggested height for the human_master figure.

  python pipeline\\foot_gaps.py <clip> [--step 6]

Per person: the lowest foot's height above the floor every <step> frames (m), only frames where both ankles were
detected count for the suggestion.  The master figure has short legs: a person fitted too small floats.  Suggested
height = current * (1 + 0.8 * median_gap / hip_height), applied in the clip's review.json ("people" -> "height")."""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg
from rig import Rig, JOINTS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("clip"); ap.add_argument("--step", type=int, default=6)
    a = ap.parse_args()
    c = cfg(); work = os.path.join(c["work_root"], a.clip)
    d = json.load(open(os.path.join(work, "scene_data.json"), encoding="utf-8"))
    for k, ch in d["chars"].items():
        rig = Rig(ch["height"], ch.get("style", "master")); gaps = []; row = []
        hip_h = float(rig.J[0][2][2]) + 0.05 * rig.s
        for f in range(len(ch["root"])):
            if ch.get("visible") and not ch["visible"][f]:
                continue
            r = ch["root"][f]
            W = rig.fk(r[:3], r[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f])
            g = min(W["ankle_L"][1][2], W["ankle_R"][1][2]) - rig.sole - r[2]
            gaps.append(g)
            if f % a.step == 0:
                row.append("%d:%+.2f" % (f, g))
        med = float(np.median(gaps)) if gaps else 0.0
        sug = ch["height"] * (1 + 0.8 * med / hip_h)
        print("%s (%s, height %.2f): median gap %+.3f m -> suggested height %.2f" % (k, ch["colour"], ch["height"], med, sug))
        print("   " + " ".join(row))


if __name__ == "__main__":
    main()

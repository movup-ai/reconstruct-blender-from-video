# -*- coding: utf-8 -*-
"""Floating check for a fitted clip: every frame, every person.

  python pipeline\\ground_check.py <package|clip> [...]

Per person and frame: height of the lowest sole above the floor under it (floor = 0, or the person's floor_z /
the scene ground height field when present), and the horizontal speed of the lowest foot while it is on the floor
(foot sliding = "skating", the main cause of a floating look).  Reports % of frames floating (> 3 cm), buried
(< -3 cm), the median sliding speed of planted feet and the worst stretches."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import cfg, jload
from rig import Rig, JOINTS


def feet(ch, f, rig):
    r = ch["root"][f]
    W = rig.fk((r[0], r[1], r[2]), r[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f])
    return np.array([W["ankle_L"][1], W["ankle_R"][1]]) - [0, 0, rig.sole]


def check(clip, out=True):
    c = cfg(); d = jload(os.path.join(c["work_root"], clip, "scene_data.json"))
    fps = float(d["fps"]); res = {}
    for k, ch in d["chars"].items():
        rig = Rig(ch["height"], ch.get("style", "sample"))
        vis = np.array(ch["visible"], bool); N = len(vis)
        F = np.array([feet(ch, f, rig) for f in range(N)])            # (N, 2, 3) soles
        floor = float(ch.get("floor_z", 0.0) or 0.0)
        low = F[:, :, 2].min(1) - floor
        li = F[:, :, 2].argmin(1)
        P = F[np.arange(N), li]                                         # lowest foot
        sp = np.r_[0, np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)] * fps   # m/s
        same = np.r_[False, li[1:] == li[:-1]]
        planted = vis & (np.abs(low) < 0.04) & same
        fl = vis & (low > 0.03); bu = vis & (low < -0.03)
        res[k] = {"colour": ch["colour"], "frames": int(vis.sum()), "float_pct": round(100 * fl.sum() / max(1, vis.sum()), 1),
                  "buried_pct": round(100 * bu.sum() / max(1, vis.sum()), 1), "max_float_cm": round(100 * float(low[vis].max()), 1) if vis.any() else 0,
                  "slide_med_ms": round(float(np.median(sp[planted])), 3) if planted.any() else None,
                  "slide_p90_ms": round(float(np.percentile(sp[planted], 90)), 3) if planted.any() else None}
        if out:
            print("  %-6s %-6s frames %3d  floating %5.1f%% (max %.1f cm)  buried %4.1f%%  planted-foot slide median %s m/s, p90 %s m/s" % (
                k, ch["colour"], res[k]["frames"], res[k]["float_pct"], res[k]["max_float_cm"], res[k]["buried_pct"],
                res[k]["slide_med_ms"], res[k]["slide_p90_ms"]))
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    names = json.load(open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8")); inv = {v: k for k, v in names.items()}
    for a in sys.argv[1:]:
        clip = inv.get(a, a); print("==", a, clip); check(clip)

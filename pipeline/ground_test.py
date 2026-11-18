# -*- coding: utf-8 -*-
"""Try ground_feet.plant on a copy of a clip's scene and print floating / sliding before and after (no files changed).
  python pipeline\\ground_test.py <package|clip>"""
import copy, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import cfg, jload
from rig import Rig
import ground_feet


def measure(d):
    out = {}
    fps = float(d["fps"])
    for k, ch in d["chars"].items():
        rig = Rig(ch["height"], ch.get("style", "sample"))
        vis = np.array(ch["visible"], bool); N = len(vis)
        S = np.array([[w["ankle_L"][1], w["ankle_R"][1]] for w in (ground_feet._fk(ch, rig, f) for f in range(N))]); S[:, :, 2] -= rig.sole
        low = S[:, :, 2].min(1); li = S[:, :, 2].argmin(1)
        P = S[np.arange(N), li]; sp = np.r_[0, np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)] * fps
        same = np.r_[False, li[1:] == li[:-1]]; planted = vis & (np.abs(low) < 0.04) & same
        out[k] = (round(100 * (vis & (low > 0.03)).sum() / max(1, vis.sum()), 1), round(100 * (vis & (low < -0.03)).sum() / max(1, vis.sum()), 1),
                  round(float(np.median(sp[planted])), 3) if planted.any() else None)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names = json.load(open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8")); inv = {v: k for k, v in names.items()}
    clip = inv.get(sys.argv[1], sys.argv[1])
    d = jload(os.path.join(cfg()["work_root"], clip, "scene_data.json"))
    d.pop("feet_planted", None)
    b = measure(d); e = copy.deepcopy(d); ground_feet.plant(e); a = measure(e)
    for k in b:
        print("%-6s floating %5.1f%% -> %5.1f%%   buried %5.1f%% -> %5.1f%%   planted-foot slide %s -> %s m/s" % (k, b[k][0], a[k][0], b[k][1], a[k][1], b[k][2], a[k][2]))


if __name__ == "__main__":
    main()

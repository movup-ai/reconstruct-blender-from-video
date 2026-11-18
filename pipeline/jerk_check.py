# -*- coding: utf-8 -*-
"""Find sudden movements in a fitted clip: every frame, every character, every joint.

  python pipeline\\jerk_check.py <clip> [--deg 6] [--acc 5] [--move 0.04]

A frame is flagged when a joint angle (or the body's facing) changes by more than --deg degrees from the previous
frame, or its change differs from the neighbouring frames' changes by more than --acc degrees (a jerk: sudden start /
stop / reversal), or the body root moves more than --move metres in one frame. Prints the flagged stretches per
character and joint, so they can be looked at with compare.py --zoom --frames ..."""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def stretches(idx):
    out = []
    for i in idx:
        if out and i <= out[-1][1] + 2:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("clip")
    ap.add_argument("--deg", type=float, default=6.0); ap.add_argument("--acc", type=float, default=5.0)
    ap.add_argument("--move", type=float, default=0.04)
    a = ap.parse_args()
    d = json.load(open(os.path.join(cfg()["work_root"], a.clip, "scene_data.json"), encoding="utf-8"))
    chars = d["chars"].items() if isinstance(d["chars"], dict) else enumerate(d["chars"])
    total = 0
    for key, c in chars:
        vis = np.array(c.get("visible") or [1] * len(c["root"]), bool)
        root = np.array(c["root"], float)
        series = {"facing": root[:, 3]} if root.shape[1] > 3 else {}
        for j, r in c["rot"].items():
            r = np.array(r, float)
            for ax, nm in enumerate("xyz"):
                if np.ptp(r[:, ax]) > 0.5:
                    series["%s.%s" % (j, nm)] = r[:, ax]
        found = []
        for nm, s in series.items():
            v = wrap(np.diff(s))                                   # change per frame
            jerk = np.abs(np.diff(v))                              # change of the change
            bad = set(np.where(np.abs(v) > a.deg)[0] + 1) | set(np.where(jerk > a.acc)[0] + 1)
            bad = sorted(i for i in bad if vis[i] and vis[i - 1])
            for s0, s1 in stretches(bad):
                found.append((s0, s1, nm, float(np.abs(v[s0 - 1:s1]).max())))
        step = np.linalg.norm(np.diff(root[:, :3], axis=0), axis=1)
        for s0, s1 in stretches([i + 1 for i in np.where(step > a.move)[0] if vis[i] and vis[i + 1]]):
            found.append((s0, s1, "position", float(step[s0 - 1:s1].max())))
        found.sort()
        print("%s (%s, track %s): %d sudden stretches" % (key, c.get("colour"), c.get("track_id"), len(found)))
        for s0, s1, nm, mx in found:
            unit = "m" if nm == "position" else "deg"
            print("   frames %3d-%-3d %-14s max %.2f %s/frame" % (s0, s1, nm, mx, unit))
        total += len(found)
    print("total", total)


if __name__ == "__main__":
    main()

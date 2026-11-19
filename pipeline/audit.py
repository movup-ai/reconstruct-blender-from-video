# -*- coding: utf-8 -*-
"""Frame-by-frame audit: how far the fitted mannequin's joints are from the joints the local model saw.

  python pipeline\\audit.py <clip name> [--top 12] [--joints head,arms]
For every person and frame: the distance in pixels between each projected mannequin joint and the detected joint
(only joints the model was sure of, > 0.5), divided by the person's height on screen, so it reads as "% of the body".
Prints the worst frames per person, which joints are off, and the average per joint.  The detected joints are only a hint
(hand-set poses in hints.json override them on purpose); a high number is a place to look, not always an error."""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload
from build_env import Cam
from compare import rig_uv

NAMES = ["nose", "eye_L", "eye_R", "ear_L", "ear_R", "shoulder_L", "shoulder_R", "elbow_L", "elbow_R", "wrist_L", "wrist_R",
         "hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"]
GROUPS = {"head": [0, 1, 2, 3, 4], "arms": [5, 6, 7, 8, 9, 10], "legs": [11, 12, 13, 14, 15, 16], "all": list(range(17))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--top", type=int, default=12); ap.add_argument("--joints", default="head,arms")
    a = ap.parse_args()
    c = cfg(); work = os.path.join(c["work_root"], a.name)
    data = jload(os.path.join(work, "scene_data.json")); cam = Cam(data)
    ci = jload(os.path.join(work, "clip_info.json")); y0 = ci["y0"]
    from run_pipeline import marks_dir
    pose = jload(os.path.join(marks_dir(ci["video"], c), "pose.json"))
    tracks = {p["id"]: p for p in pose["people"]}
    use = sorted({j for g in a.joints.split(",") for j in GROUPS[g]})
    for kind, ch in data["chars"].items():
        p = tracks.get(ch["track_id"])
        if p is None:
            continue
        rows = []
        for f in range(data["frames"]):
            k = p["frames"].get(str(f))
            if k is None or not ch["visible"][f]:
                continue
            k = np.array(k, float); uv = rig_uv(ch, cam, f)
            ok = [j for j in use if k[j, 2] > 0.5]
            if len(ok) < 3:
                continue
            ys = k[[j for j in range(17) if k[j, 2] > 0.3], 1]
            h = max(40.0, float(np.ptp(ys)))
            e = {j: float(np.linalg.norm(uv[j] - (k[j, :2] - [0, y0]))) / h * 100 for j in ok}
            rows.append((f, float(np.mean(list(e.values()))), e))
        if not rows:
            print(kind, "no data"); continue
        avg = {}
        for _, _, e in rows:
            for j, v in e.items():
                avg.setdefault(j, []).append(v)
        print("== %s %s  frames %d  mean error %.1f%% of body height" % (kind, ch["label"], len(rows), np.mean([r[1] for r in rows])))
        print("   per joint mean %%:  " + "  ".join("%s %.0f" % (NAMES[j], np.mean(v)) for j, v in sorted(avg.items())))
        for f, m, e in sorted(rows, key=lambda r: -r[1])[:a.top]:
            worst = sorted(e.items(), key=lambda t: -t[1])[:3]
            print("   f%-4d %5.1f%%   worst: %s" % (f, m, ", ".join("%s %.0f%%" % (NAMES[j], v) for j, v in worst)))


if __name__ == "__main__":
    main()

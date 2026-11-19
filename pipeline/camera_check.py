# -*- coding: utf-8 -*-
"""Camera motion check, every frame: the film's pan / tilt (background image shift, people masked out with their
detected boxes) against the render camera's turn / tilt from scene_data (degrees).

  python pipeline\\camera_check.py <package|clip> [--write-hint]

--write-hint: writes the film's measured pan / tilt as the camera path into the clip's hints.json
("camera": {"path": [[frame, yaw, pitch, 0, 0, 0, 0], ...]}, per shot when the clip has several), keeping the
height / tilt keys, so the next fit follows the film's camera move."""
import json, os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import cfg, jload, read_frames
from run_pipeline import marks_dir

W_SMALL = 640


def film_pan(frames, boxes, scale):
    """cumulative (dx, dy) px of the background between consecutive frames"""
    out = [(0.0, 0.0)]; prev = None
    for i, g in enumerate(frames):
        e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 30, 90).astype(np.float32)
        for b in boxes.get(i, []):                                   # people out
            x0, y0, x1, y1 = [int(v * scale) for v in b]
            e[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10] = 0
        e = cv2.GaussianBlur(e, (9, 9), 0)
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, e, cv2.createHanningWindow(e.shape[::-1], cv2.CV_32F))
            out.append((dx, dy) if resp > 0.03 else (0.0, 0.0))
        prev = e
    return np.cumsum(np.array(out), axis=0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8") as fh:
        names = json.load(fh)
    inv = {v: k for k, v in names.items()}
    clip = inv.get(sys.argv[1], sys.argv[1]); c = cfg(); work = os.path.join(c["work_root"], clip)
    ci = jload(os.path.join(work, "clip_info.json")); y0, Hp = ci["y0"], ci["Hpic"]
    frames, _ = read_frames(ci["video"])
    W = frames[0].shape[1]; sc = W_SMALL / float(W)
    G = [cv2.cvtColor(cv2.resize(f[y0:y0 + Hp], (W_SMALL, int(round(Hp * sc))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames]
    pose = jload(os.path.join(marks_dir(ci["video"], c), "pose.json"))
    boxes = {}
    for t in pose["people"]:
        for f, b in t["boxes"].items():
            boxes.setdefault(int(f), []).append([b[0], b[1] - y0, b[2], b[3] - y0])
    sd = jload(os.path.join(work, "scene_data.json"))
    hfov = 2 * np.degrees(np.arctan(sd.get("sensor", 36.0) / 2.0 / sd["lens"]))
    dpp = hfov / W_SMALL
    cam = np.array(sd["cam"], float)
    shots = [s["frames"] for s in sd.get("shots", [])] or [[0, len(G) - 1]]
    paths = {}
    for si, (a, b) in enumerate(shots, 1):
        bx = {i - a: boxes.get(i, []) for i in range(a, b + 1)}
        cf = film_pan(G[a:b + 1], bx, sc)
        fyaw = cf[:, 0] * dpp                                         # content moves right = camera turned left (+yaw)
        fpit = cf[:, 1] * dpp                                         # content moves down = camera tilted up (+pitch)
        ryaw = cam[a:b + 1, 5] - cam[a, 5]; rpit = cam[a:b + 1, 3] - cam[a, 3]
        gap = np.abs(fyaw - ryaw)
        print("shot %d (f%d-%d): film turn %+.1f deg, tilt %+.1f | render turn %+.1f, tilt %+.1f | worst turn gap %.1f deg at f%d" % (
            si, a, b, fyaw[-1], fpit[-1], ryaw[-1], rpit[-1], gap.max(), a + int(gap.argmax())))
        step = max(1, (b - a) // 12)
        print("   frame:" + "".join("%6d" % (a + i) for i in range(0, b - a + 1, step)))
        print("   film :" + "".join("%6.1f" % fyaw[i] for i in range(0, b - a + 1, step)))
        print("   rendr:" + "".join("%6.1f" % ryaw[i] for i in range(0, b - a + 1, step)))
        print("   film tilt :" + "".join("%6.1f" % fpit[i] for i in range(0, b - a + 1, step)))
        print("   rendr tilt:" + "".join("%6.1f" % rpit[i] for i in range(0, b - a + 1, step)))
        k = list(range(0, b - a + 1, 3)) + ([b - a] if (b - a) % 3 else [])
        from scipy.ndimage import gaussian_filter1d
        sy = gaussian_filter1d(fyaw, 2.0, mode="nearest"); sp = gaussian_filter1d(fpit, 2.0, mode="nearest")
        paths[str(si)] = [[i, round(float(sy[i]), 2), round(float(sp[i]), 2), 0, 0, 0, 0] for i in k]
    if "--write-hint" in sys.argv:
        hp = os.path.join(ROOT, "output", clip, "hints.json")
        h = {}
        if os.path.exists(hp):
            with open(hp, encoding="utf-8") as fh:
                h = json.load(fh)
        camh = h.setdefault("camera", {})
        if len(paths) == 1:
            camh["path"] = paths["1"]
        else:
            for si, p in paths.items():
                camh.setdefault("shots", {}).setdefault(si, {})["path"] = p
        with open(hp, "w", encoding="utf-8") as fh:
            json.dump(h, fh, ensure_ascii=False, indent=1)
        print("camera path written to", hp)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Background alignment, every frame: how far the film's background and the render's background have moved
since frame 0 (phase correlation of edge maps, people masked out with their detected boxes), and the gap.

  python pipeline\\bg_align.py <clip> [--fix]

--fix: corrects the camera path in output\\<clip>\\hints.json so the render's background follows the film's:
yaw / pitch keys every 4 frames += the measured gap (pixels -> degrees through the lens).  Run the fit + render
again afterwards and re-run this check."""
import json, os, sys
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import cfg, jload, read_frames
from run_pipeline import marks_dir

W_SMALL = 640


def shifts(G, masks):
    out = [(0.0, 0.0)]; prev = None
    win = None
    for g, m in zip(G, masks):
        e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 30, 90).astype(np.float32)
        e[m] = 0
        e = cv2.GaussianBlur(e, (9, 9), 0)
        if win is None:
            win = cv2.createHanningWindow(e.shape[::-1], cv2.CV_32F)
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, e, win)
            out.append((dx, dy) if resp > 0.02 else (0.0, 0.0))
        prev = e
    return np.cumsum(np.array(out), axis=0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    clip = sys.argv[1]; c = cfg(); work = os.path.join(c["work_root"], clip)
    ci = jload(os.path.join(work, "clip_info.json")); y0, Hp = ci["y0"], ci["Hpic"]
    frames, _ = read_frames(ci["video"]); N = len(frames)
    W = frames[0].shape[1]; sc = W_SMALL / float(W); h = int(round(Hp * sc))
    pose = jload(os.path.join(marks_dir(ci["video"], c), "pose.json"))
    masks = [np.zeros((h, W_SMALL), bool) for _ in range(N)]
    for t in pose["people"]:
        for f, b in t["boxes"].items():
            f = int(f)
            if f < N:
                x0, yy0, x1, yy1 = b[0] * sc - 12, (b[1] - y0) * sc - 12, b[2] * sc + 12, (b[3] - y0) * sc + 12
                masks[f][max(0, int(yy0)):int(yy1), max(0, int(x0)):int(x1)] = True
    F = [cv2.cvtColor(cv2.resize(fr[y0:y0 + Hp], (W_SMALL, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for fr in frames]
    R = []
    for i in range(N):
        im = cv2.imdecode(np.fromfile(os.path.join(work, "render", "wm_%04d.jpg" % (i + 1)), np.uint8), cv2.IMREAD_COLOR)
        R.append(cv2.cvtColor(cv2.resize(im, (W_SMALL, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    cf = shifts(F, masks); cr = shifts(R, masks)
    gap = cf - cr
    sd = jload(os.path.join(work, "scene_data.json"))
    dpp = 2 * np.degrees(np.arctan(sd.get("sensor", 36.0) / 2.0 / sd["lens"])) / W_SMALL
    print("background shift since f0 (px at %d wide; + = content moved right / down)" % W_SMALL)
    print("frame  film_x render_x gap_x | film_y render_y gap_y")
    for f in range(0, N, max(1, N // 16)):
        print("%5d %7.1f %8.1f %6.1f | %6.1f %8.1f %6.1f" % (f, cf[f, 0], cr[f, 0], gap[f, 0], cf[f, 1], cr[f, 1], gap[f, 1]))
    print("worst gap: x %.1f px (%.2f deg) at f%d, y %.1f px at f%d" % (np.abs(gap[:, 0]).max(), np.abs(gap[:, 0]).max() * dpp,
          int(np.abs(gap[:, 0]).argmax()), np.abs(gap[:, 1]).max(), int(np.abs(gap[:, 1]).argmax())))
    if "--fix" in sys.argv:
        hp = os.path.join(ROOT, "output", clip, "hints.json")
        hj = json.load(open(hp, encoding="utf-8"))
        path = sorted(hj["camera"]["path"])
        fr_k = [k[0] for k in path]
        g = gaussian_filter1d(gap, 2.0, axis=0, mode="nearest")
        new = []
        for f in list(range(0, N, 4)) + [N - 1]:
            base = [float(np.interp(f, fr_k, [k[i] for k in path])) for i in range(1, 7)]
            base[0] += g[f, 0] * dpp            # content too far left in the render -> turn the camera left (+yaw)
            base[1] += g[f, 1] * dpp            # content too low in the render -> tilt up (+pitch)
            new.append([f] + [round(v, 3) for v in base])
        hj["camera"]["path"] = new
        json.dump(hj, open(hp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("camera path corrected in", hp)


if __name__ == "__main__":
    main()

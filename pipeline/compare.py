# -*- coding: utf-8 -*-
"""Film frame next to the rendered frame for a few frames of a clip - to see what does not match.

  python pipeline\\compare.py <clip name> [--frames 0,40,80] [--out FILE] [--width 640]
        whole picture: film | render | render outlines drawn on the film
  python pipeline\\compare.py <clip name> --zoom [--frames ...] [--size 360]
        every character close up: film (yellow = detected joints, colour = fitted mannequin) | render
Uses Work_auto\\<clip>\\render\\wm_NNNN.jpg (made by --stage render)."""
import argparse, os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload, imwrite_u, read_frames
from build_env import Cam
from rig import Rig, JOINTS

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
COL = {"red": (60, 62, 235), "blue": (230, 120, 50), "green": (90, 190, 70), "yellow": (60, 210, 235), "purple": (190, 90, 150)}


def rig_uv(ch, cam, f):
    rig = Rig(ch["height"], ch.get("style", "sample"))
    r = ch["root"][f]
    W = rig.fk((r[0], r[1], r[2]), r[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f])
    return cam.project(rig.coco_points(W), f)[0]


def zoom_sheet(a, c, work, ci, frames, frs):
    data = jload(os.path.join(work, "scene_data.json")); cam = Cam(data)
    from run_pipeline import marks_dir
    pose = jload(os.path.join(a.marks or marks_dir(ci["video"], c), "pose.json"))
    y0, Hp = ci["y0"], ci["Hpic"]
    tracks = {p["id"]: p for p in pose["people"]}
    S = a.size; rows = []
    for f in frs:
        rp = os.path.join(work, "render", "wm_%04d.jpg" % (f + 1))
        if not os.path.exists(rp):
            continue
        src = frames[f][y0:y0 + Hp]
        r = cv2.resize(cv2.imdecode(np.fromfile(rp, np.uint8), cv2.IMREAD_COLOR), (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
        tiles = []
        for kind, ch in data["chars"].items():
            p = tracks.get(ch["track_id"])
            if p is None or not ch["visible"][f]:
                continue
            uv = rig_uv(ch, cam, f)
            k = np.array(p["frames"][str(f)], float) if str(f) in p["frames"] else None
            sure = None if k is None else k[k[:, 2] > 0.3, :2] - [0, y0]
            inside = uv[(uv[:, 0] > -20) & (uv[:, 0] < src.shape[1] + 20) & (uv[:, 1] > -20) & (uv[:, 1] < src.shape[0] + 20)]
            pts = np.vstack([inside, sure]) if sure is not None and len(sure) >= 4 else (inside if len(inside) >= 2 else uv)
            cx, cy = (pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2
            half = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 40) * 0.62
            x0, y0c = int(round(cx - half)), int(round(cy - half)); n = int(round(2 * half))
            def crop(img):
                out = np.zeros((n, n, 3), np.uint8)
                xa, ya = max(0, x0), max(0, y0c); xb, yb = min(img.shape[1], x0 + n), min(img.shape[0], y0c + n)
                if xb > xa and yb > ya:
                    out[ya - y0c:yb - y0c, xa - x0:xb - x0] = img[ya:yb, xa:xb]
                return cv2.resize(out, (S, S), interpolation=cv2.INTER_CUBIC)
            sc = S / float(n)
            film = crop(src); plain = film.copy()
            colr = COL.get(ch["colour"], (200, 200, 200))
            P = (uv - [x0, y0c]) * sc
            for i, j in BONES:
                cv2.line(film, tuple(P[i].astype(int)), tuple(P[j].astype(int)), colr, 2, cv2.LINE_AA)
            if k is not None:
                for (x, y, s_) in k:
                    if s_ > 0.3:
                        cv2.circle(film, (int((x - x0) * sc), int((y - y0 - y0c) * sc)), 3, (0, 255, 255), -1)
            cv2.putText(plain, "f%d %s" % (f, kind), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            tiles += [plain, film, crop(r)]
        if tiles:
            rows.append(np.hstack(tiles))
    w = max(x.shape[1] for x in rows)
    rows = [np.hstack([x, np.zeros((x.shape[0], w - x.shape[1], 3), np.uint8)]) for x in rows]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--frames", default=""); ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--zoom", action="store_true")
    ap.add_argument("--size", type=int, default=360); ap.add_argument("--marks", default=None)
    a = ap.parse_args()
    c = cfg(); work = os.path.join(c["work_root"], a.name)
    ci = jload(os.path.join(work, "clip_info.json"))
    frames, _ = read_frames(ci["video"])
    N = len(frames); y0, Hp = ci["y0"], ci["Hpic"]
    frs = [int(x) for x in a.frames.split(",") if x != ""] or list(np.linspace(0, N - 1, 8 if a.zoom else 6).astype(int))
    if a.zoom:
        out = a.out or os.path.join(work, "compare_zoom.jpg")
        imwrite_u(out, zoom_sheet(a, c, work, ci, frames, frs), 90)
        print(out); return
    rows = []
    for f in frs:
        src = frames[f][y0:y0 + Hp]
        rp = os.path.join(work, "render", "wm_%04d.jpg" % (f + 1))
        if not os.path.exists(rp):
            continue
        r = cv2.imdecode(np.fromfile(rp, np.uint8), cv2.IMREAD_COLOR)
        r = cv2.resize(r, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
        ed = cv2.Canny(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), 40, 120)
        mix = src.copy(); mix[ed > 0] = (0, 255, 255)                 # render outlines drawn on the film frame
        s = a.width / float(src.shape[1])
        t = [cv2.resize(x, (a.width, int(round(src.shape[0] * s)))) for x in (src, r, mix)]
        cv2.putText(t[0], "f%d" % f, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        rows.append(np.hstack(t))
    out = a.out or os.path.join(work, "compare.jpg")
    imwrite_u(out, np.vstack(rows), 88)
    print(out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Every frame of a clip: film | render side by side, 40 frames per sheet (8 columns x 5 rows).

  python pipeline\\all_frames.py <package|clip> [--tile 230] [--from F --to T] [--crop x0,y0,x1,y1]
-> Work_auto\\review_all\\<pkg>_all_KK.jpg   (--crop: picture pixels to zoom on, e.g. the area where people are)"""
import argparse, json, os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import cfg, jload, read_frames, imwrite_u


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pkg"); ap.add_argument("--tile", type=int, default=230)
    ap.add_argument("--from", dest="f0", type=int, default=0); ap.add_argument("--to", dest="f1", type=int, default=10 ** 6)
    ap.add_argument("--crop", default=""); ap.add_argument("--cols", type=int, default=8); ap.add_argument("--rows", type=int, default=5)
    a = ap.parse_args()
    names = json.load(open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8")); inv = {v: k for k, v in names.items()}
    clip = inv.get(a.pkg, a.pkg); c = cfg(); work = os.path.join(c["work_root"], clip)
    ci = jload(os.path.join(work, "clip_info.json")); y0, Hp = ci["y0"], ci["Hpic"]
    frames, _ = read_frames(ci["video"])
    out_dir = os.path.join(c["work_root"], "review_all"); os.makedirs(out_dir, exist_ok=True)
    N = min(len(frames), a.f1 + 1); fr = list(range(a.f0, N))
    per = a.cols * a.rows; tiles = []
    for f in fr:
        film = frames[f][y0:y0 + Hp]
        rp = os.path.join(work, "render", "wm_%04d.jpg" % (f + 1))
        r = cv2.imdecode(np.fromfile(rp, np.uint8), cv2.IMREAD_COLOR)
        r = cv2.resize(r, (film.shape[1], film.shape[0]), interpolation=cv2.INTER_AREA)
        if a.crop:
            x0, yy0, x1, yy1 = [int(v) for v in a.crop.split(",")]
            film = film[yy0:yy1, x0:x1]; r = r[yy0:yy1, x0:x1]
        h = int(round(film.shape[0] * a.tile / film.shape[1]))
        t = np.hstack([cv2.resize(film, (a.tile, h), interpolation=cv2.INTER_AREA), cv2.resize(r, (a.tile, h), interpolation=cv2.INTER_AREA)])
        cv2.putText(t, "f%d" % f, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.line(t, (a.tile, 0), (a.tile, h), (0, 0, 0), 2)
        tiles.append(t)
    for k in range(0, len(tiles), per):
        grp = tiles[k:k + per]
        while len(grp) % a.cols:
            grp.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(grp[i:i + a.cols]) for i in range(0, len(grp), a.cols)]
        out = os.path.join(out_dir, "%s_all_%02d.jpg" % (a.pkg, k // per + 1))
        imwrite_u(out, np.vstack(rows), 85)
        print(out, "frames %d-%d" % (fr[k], fr[min(k + per, len(fr)) - 1]))


if __name__ == "__main__":
    main()

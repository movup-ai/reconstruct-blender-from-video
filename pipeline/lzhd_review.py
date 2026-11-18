# -*- coding: utf-8 -*-
"""Review sheet for an LZHD render: per frame, film (yellow = detected joints) | white model (same dots) side by side.

  python pipeline\\lzhd_review.py <video> <clip> [--frames 0,20,40] [--n 8] [--out sheet.jpg] [--crop x0,y0,x1,y1]"""
import argparse, json, os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg
import run_pipeline as rp

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("clip")
    ap.add_argument("--frames", default=""); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=""); ap.add_argument("--crop", default=""); ap.add_argument("--w", type=int, default=760)
    a = ap.parse_args()
    c = cfg(); work = os.path.join(c["work_root"], a.clip)
    ci = json.load(open(os.path.join(work, "clip_info.json"), encoding="utf-8"))
    pose = json.load(open(os.path.join(rp.marks_dir(a.video, c), "pose.json"), encoding="utf-8"))
    N = ci["frames"]; y0 = ci["y0"]
    fr = [int(x) for x in a.frames.split(",")] if a.frames else [int(round(x)) for x in np.linspace(0, N - 1, a.n)]
    cap = cv2.VideoCapture(a.video); rows = []
    for f in fr:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, film = cap.read()
        rp_ = os.path.join(work, "render", "wm_%04d.jpg" % (f + 1))
        rend = cv2.imdecode(np.fromfile(rp_, np.uint8), cv2.IMREAD_COLOR) if os.path.exists(rp_) else None
        if not ok or rend is None:
            continue
        rend = cv2.resize(rend, (film.shape[1], ci["Hpic"]))
        rend = cv2.copyMakeBorder(rend, y0, film.shape[0] - ci["Hpic"] - y0, 0, 0, cv2.BORDER_CONSTANT)
        film = cv2.convertScaleAbs(film, alpha=1.8, beta=8)
        for im in (film, rend):
            for p in pose["people"]:
                k = p["frames"].get(str(f))
                if not k:
                    continue
                for i, j in BONES:
                    if k[i][2] > 0.3 and k[j][2] > 0.3:
                        cv2.line(im, (int(k[i][0]), int(k[i][1])), (int(k[j][0]), int(k[j][1])), (0, 220, 255), 2)
                for x, y, s in k:
                    if s > 0.3:
                        cv2.circle(im, (int(x), int(y)), 7, (0, 255, 255), -1)
                cv2.putText(im, str(p["id"]), (int(k[5][0]), int(k[5][1]) - 20), 0, 1.4, (0, 255, 255), 3)
            cv2.putText(im, "f%d" % f, (20, 60), 0, 2, (0, 0, 255), 4)
        pair = np.hstack([film, rend])
        if a.crop:
            x0, yy0, x1, yy1 = map(int, a.crop.split(","))
            pair = np.hstack([film[yy0:yy1, x0:x1], rend[yy0:yy1, x0:x1]])
        rows.append(cv2.resize(pair, (2 * a.w, int(pair.shape[0] * 2 * a.w / pair.shape[1]))))
    out = a.out or os.path.join(work, "lzhd_review.jpg")
    cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(out)
    print(out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Reference pictures for writing a clip's set by hand: for every shot, its first, middle and last frame with a pixel
grid (labels every 100 px, in the coordinates env_claude.json uses), the camera numbers, and where the people stand.

  python pipeline\\env_frames.py <clip name>        ->  Work_auto\\<clip>\\env_frames.jpg  (+ prints the shot table)"""
import os, sys, json
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload, imwrite_u, read_frames_at


def grid(img, step=100):
    im = img.copy(); H, W = im.shape[:2]
    for x in range(0, W, step):
        cv2.line(im, (x, 0), (x, H - 1), (0, 255, 255), 1)
        cv2.putText(im, str(x), (x + 3, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    for y in range(0, H, step):
        cv2.line(im, (0, y), (W - 1, y), (0, 255, 255), 1)
        cv2.putText(im, str(y), (3, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return im


def main():
    name = sys.argv[1]
    c = cfg(); work = os.path.join(c["work_root"], name)
    ci = jload(os.path.join(work, "clip_info.json")); data = jload(os.path.join(work, "scene_data.json"))
    camj = jload(os.path.join(work, "camera.json"))
    frames = read_frames_at(ci["video"], [v for s in camj["shots"] for v in (s["frames"][0], (s["frames"][0] + s["frames"][1]) // 2, s["frames"][1])])
    y0, Hp = ci["y0"], ci["Hpic"]
    rows = []
    print("clip %s: %d frames, picture %dx%d (pixel coordinates below are inside the picture area)" % (name, data["frames"], ci["W"], Hp))
    for s in camj["shots"]:
        a, b = s["frames"]; tag = "" if len(camj["shots"]) == 1 else "_s%d" % s["shot"]
        g = s.get("ground") or {}
        people = [k for k, ch in data["chars"].items() if (k.startswith("s%d_" % s["shot"]) or not tag) and any(ch["visible"][a:b + 1])]
        print("shot %d%s: frames %d-%d, camera height %.2f m%s, tilt %.1f deg, turn %.1f deg, people %s" % (
            s["shot"], " (key %s)" % s["shot"] if tag else "", a, b, g.get("camera_height", 1.6),
            "" if "camera_height" in g else " (a guess: no full-body people to measure it; check the floor)", g.get("pitch_deg", 0.0),
            s.get("max_rotation_deg", 0), ", ".join(people) or "none"))
        tiles = []
        for f in sorted({a, (a + b) // 2, b}):
            im = grid(frames[f][y0:y0 + Hp])
            cv2.putText(im, "shot %d  frame %d" % (s["shot"], f), (10, Hp - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            tiles.append(im)
        while len(tiles) < 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(tiles))
    sheet = np.vstack(rows)
    sc = min(1.0, 3000.0 / sheet.shape[1])
    if sheet.shape[0] * sc > 9000:
        sc = 9000.0 / sheet.shape[0]
    out = os.path.join(work, "env_frames.jpg")
    imwrite_u(out, cv2.resize(sheet, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), 88)
    print("scale of the sheet: %.3f (divide sheet pixels by this to get picture pixels)" % sc)
    print(out)


if __name__ == "__main__":
    main()

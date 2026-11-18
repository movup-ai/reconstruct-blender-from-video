# -*- coding: utf-8 -*-
"""Rebuild a clip's dot editor (marks check.html) from its CURRENT pose.json, with every frame (or every Nth).
Opened from the app at /check/<clip id>; Save writes marks\\checked.json, which the fit uses (confirmed dots win).
Frames already confirmed in checked.json are shown with the confirmed dots.

  python pipeline\\make_check_page.py <package|clip> [--step 1] [--quality 80]"""
import argparse, base64, json, os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from common import read_frames
from agent import config
from agent.clips import Library, folders

KP_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(16, 14, "R"), (14, 12, "R"), (15, 13, "L"), (13, 11, "L"), (11, 12, "C"), (5, 11, "L"), (6, 12, "R"),
            (5, 6, "C"), (5, 7, "L"), (7, 9, "L"), (6, 8, "R"), (8, 10, "R"), (0, 1, "C"), (0, 2, "C"),
            (1, 3, "L"), (2, 4, "R")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pkg"); ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--quality", type=int, default=80)
    a = ap.parse_args()
    names = json.load(open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8")); inv = {v: k for k, v in names.items()}
    clip = inv.get(a.pkg, a.pkg)
    c = config.load(); cl = next(x for x in Library(c).clips if x.name == clip); f = folders(c, cl)
    pose = json.load(open(f["pose"], encoding="utf-8"))
    chk = {}
    cp = os.path.join(f["marks"], "checked.json")
    if os.path.exists(cp):
        chk = json.load(open(cp, encoding="utf-8")).get("frames", {})
    frames, _ = read_frames(cl.video)
    W = frames[0].shape[1]; _, y0, _, hp = pose["picture_area"]; y1 = y0 + hp
    page_frames = []
    for fi in range(0, len(frames), a.step):
        if str(fi) in chk:
            ppl = chk[str(fi)]
        else:
            ppl = [{"id": p["id"], "kps": [[round(float(x), 1), round(float(y), 1), round(float(s), 3)] for x, y, s in p["frames"][str(fi)]]}
                   for p in pose["people"] if str(fi) in p["frames"]]
        jpg = cv2.imencode(".jpg", frames[fi][y0:y1], [cv2.IMWRITE_JPEG_QUALITY, a.quality])[1].tobytes()
        page_frames.append({"f": fi, "img": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(), "people": ppl})
    tpl = open(os.path.join(ROOT, "check_template.html"), encoding="utf-8").read()
    page = {"clip": clip, "video": cl.video, "picture_area": [0, y0, W, hp], "kp_names": KP_NAMES, "skeleton": SKELETON,
            "frames": page_frames}
    with open(os.path.join(f["marks"], "check.html"), "w", encoding="utf-8") as fh:
        fh.write(tpl.replace("__PAGE_DATA__", json.dumps(page, ensure_ascii=False)))
    print("%s: %d frames -> %s   (app: /check/%s)" % (a.pkg, len(page_frames), os.path.join(f["marks"], "check.html"), cl.id))


if __name__ == "__main__":
    main()

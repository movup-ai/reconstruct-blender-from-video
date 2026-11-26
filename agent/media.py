# -*- coding: utf-8 -*-
"""Thumbnails and crops (cached on disk).  OpenCV cannot open Chinese paths on Windows, so videos are copied to a
temporary ASCII path when needed."""
import os
import threading

import cv2


from .clips import folders, read_json
from common import open_video, close_video  # pipeline (on sys.path through config)

_lock = threading.Lock()


def write_jpg(path, img, q=85):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise IOError("could not encode %s" % path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(buf.tobytes())
    os.replace(tmp, path)


def read_frames_at(video, idxs):
    cap, tmp = open_video(video)
    out = {}
    try:
        for i in sorted(set(idxs)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if ok:
                out[i] = img
    finally:
        close_video(cap, tmp)
    return out


def clip_thumb(c, cl):
    p = os.path.join(c["work_root"], "_thumbs", cl.id + ".jpg")
    if os.path.exists(p):
        return p
    with _lock:
        if os.path.exists(p):
            return p
        cap, tmp = open_video(cl.video)
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.3))
            ok, img = cap.read()
        finally:
            close_video(cap, tmp)
        if not ok:
            return None
        write_jpg(p, cv2.resize(img, (320, int(img.shape[0] * 320 / img.shape[1]))), 80)
    return p


def person_crops(c, cl, size=220):
    """one crop per tracked person (the frame where they are biggest, middle half of the track); cached"""
    f = folders(c, cl)
    pose = read_json(f["pose"])
    if not pose:
        return {}
    td = os.path.join(f["marks"], "thumbs")
    out, want = {}, {}
    for p in pose.get("people", []):
        path = os.path.join(td, "P%d.jpg" % p["id"])
        out[p["id"]] = path
        if not p.get("boxes"):
            continue                                     # nothing to cut a picture from
        if not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(f['pose']):
            fs = sorted(int(x) for x in p["boxes"])
            mid = fs[len(fs) // 4: 3 * len(fs) // 4 + 1] or fs
            best = max(mid, key=lambda x: p["boxes"][str(x)][3] - p["boxes"][str(x)][1])
            want[p["id"]] = (best, p["boxes"][str(best)])
    if want:
        frames = read_frames_at(cl.video, [b for b, _ in want.values()])
        for pid, (fi, box) in want.items():
            img = frames.get(fi)
            if img is None:
                continue
            x0, y0, x1, y1 = box[:4]
            w, h = x1 - x0, y1 - y0
            x0, y0 = int(max(0, x0 - 0.15 * w)), int(max(0, y0 - 0.15 * h))
            x1, y1 = int(min(img.shape[1], x1 + 0.15 * w)), int(min(img.shape[0], y1 + 0.15 * h))
            crop = img[y0:y1, x0:x1]
            if crop.size:
                s = float(size) / max(crop.shape[:2])
                write_jpg(out[pid], cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s)))), 88)
    return {k: v for k, v in out.items() if os.path.exists(v)}

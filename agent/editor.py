# -*- coding: utf-8 -*-
"""Movement editor (/edit/<clip id>): the detected joints of a clip, 30 frames to review, every frame editable,
and a Render button that refits and re-renders the clip with the corrected movement.

Flow: Claude builds the set + .blend -> the local model detects the movement (marks pose.json) -> the user checks
30 frames (any other frame from the frame strip), fixes dots, saves (marks checked.json, confirmed dots win in the
fit) -> Render: fit + set + render + package, with live progress."""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

from .clips import folders, read_json
from . import config

KP_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
            "right_ankle"]
SKELETON = [(16, 14, "R"), (14, 12, "R"), (15, 13, "L"), (13, 11, "L"), (11, 12, "C"), (5, 11, "L"), (6, 12, "R"),
            (5, 6, "C"), (5, 7, "L"), (7, 9, "L"), (6, 8, "R"), (8, 10, "R"), (0, 1, "C"), (0, 2, "C"), (1, 3, "L"), (2, 4, "R")]
N_REVIEW = 30
IMG_W = 1400                              # frames are served at this width (dots stay in full-frame pixels)


def _pkg_name(cl):
    try:
        with open(os.path.join(config.ROOT, "data", "package_names.json"), encoding="utf-8") as fh:
            names = json.load(fh)
        return names.get(cl.name)
    except (OSError, ValueError):
        return None


def review_frames(pose, n=N_REVIEW):
    """n frames over the clip: in each of n equal parts the frame whose detections are least sure (the ones most
    likely to need a fix), first and last frame included"""
    N = int(pose["frames"])
    conf = np.full(N, 2.0)
    for p in pose["people"]:
        for f, k in p["frames"].items():
            k = np.asarray(k, float); f = int(f)
            if f < N:
                conf[f] = min(conf[f], float(k[:, 2].mean()))
    edges = np.linspace(0, N, n + 1).astype(int)
    out = set([0, N - 1])
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            out.add(int(a + np.argmin(conf[a:b])))
    return sorted(out)


def payload(c, cl):
    f = folders(c, cl)
    pose = read_json(f["pose"])
    if not pose:
        raise FileNotFoundError("this clip has no detected movement yet (run the detection first)")
    chk = read_json(os.path.join(f["marks"], "checked.json"), {}) or {}
    frames = {}
    for p in pose["people"]:
        for fr, k in p["frames"].items():
            frames.setdefault(fr, []).append({"id": p["id"], "kps": [[round(float(x), 1), round(float(y), 1), round(float(s), 3)] for x, y, s in k]})
    edited = sorted(int(x) for x in (chk.get("frames") or {}))
    detected = {}
    for fr, ppl in (chk.get("frames") or {}).items():
        detected[str(fr)] = frames.get(str(fr), [])            # the original detection, for "reset frame"
        frames[str(fr)] = ppl
    rv = read_json(f["review"], {}) or {}
    roles = {str(k): {"colour": v.get("colour"), "role": v.get("role")} for k, v in (rv.get("people") or {}).items()}
    _, y0, W, hp = pose["picture_area"]
    return {"clip": cl.name, "id": cl.id, "package": _pkg_name(cl), "frames_total": int(pose["frames"]),
            "fps": pose.get("fps"), "picture_area": [0, y0, W, hp], "img_w": IMG_W, "kp_names": KP_NAMES,
            "skeleton": SKELETON, "review": review_frames(pose), "people": frames, "detected": detected, "edited": edited,
            "checked": sorted(int(x) for x in chk.get("checked_frames", [])), "roles": roles,
            "saved": chk.get("saved"), "render": status(cl.id)}


def save(c, cl, body):
    """merge the posted frames into marks checked.json; posted frames replace the stored ones"""
    f = folders(c, cl)
    if not isinstance(body.get("frames"), dict):
        raise ValueError("frames missing")
    p = os.path.join(f["marks"], "checked.json")
    chk = read_json(p, {}) or {}
    fr = chk.get("frames") or {}
    for k, v in body["frames"].items():
        fr[str(int(k))] = [{"id": int(q["id"]), "kps": [[float(a), float(b), float(s)] for a, b, s in q["kps"]]} for q in v]
    for k in body.get("restore") or []:                       # back to the detection for these frames
        fr.pop(str(int(k)), None)
    chk.update(clip=cl.name, video=cl.video, frames=fr, saved=time.strftime("%Y-%m-%d %H:%M:%S"),
               checked_frames=sorted(set(int(x) for x in chk.get("checked_frames", [])) | set(int(x) for x in body.get("checked", []))))
    os.makedirs(f["marks"], exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(chk, fh, ensure_ascii=False)
    os.replace(tmp, p)
    return {"ok": True, "edited": len(fr), "checked": len(chk["checked_frames"]), "saved": chk["saved"]}


# ------------------------------------------------------------------ frames (decoded once per clip, kept as JPEG)
_FR = {}
_FR_LOCK = threading.Lock()


def _decode(cl, y0, hp, slot):
    import cv2
    cap = cv2.VideoCapture(cl.video)
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        pic = img[y0:y0 + hp]
        h = int(round(pic.shape[0] * IMG_W / pic.shape[1]))
        small = cv2.resize(pic, (IMG_W, h), interpolation=cv2.INTER_AREA)
        slot["jpg"][i] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 86])[1].tobytes()
        i += 1
        slot["ev"].set()
    cap.release()
    slot["done"] = True; slot["ev"].set()


def frame_jpeg(c, cl, n):
    with _FR_LOCK:
        slot = _FR.get(cl.id)
        if slot is None:
            if len(_FR) >= 3:                                     # keep at most three clips in memory
                _FR.pop(next(iter(_FR)))
            pose = read_json(folders(c, cl)["pose"]) or {}
            _, y0, _, hp = pose.get("picture_area", [0, 0, 0, 0])
            slot = {"jpg": {}, "done": False, "ev": threading.Event()}
            _FR[cl.id] = slot
            threading.Thread(target=_decode, args=(cl, int(y0), int(hp) or 10 ** 6, slot), daemon=True).start()
    t0 = time.time()
    while n not in slot["jpg"] and not slot["done"] and time.time() - t0 < 120:
        slot["ev"].wait(1.0); slot["ev"].clear()
    return slot["jpg"].get(n)


# ------------------------------------------------------------------ render job (fit + set + render + package)
_JOBS = {}
_JOB_LOCK = threading.Lock()
STEPS = [("fit", "Fitting the movement to the dots"), ("set", "Building the set"), ("render", "Rendering and packaging")]


def status(cid):
    j = _JOBS.get(cid)
    if not j:
        return {"state": "idle"}
    return {k: j[k] for k in ("state", "step", "step_label", "started", "finished", "error", "log")}


def start_render(c, cl):
    with _JOB_LOCK:
        j = _JOBS.get(cl.id)
        if j and j["state"] == "running":
            return status(cl.id)
        j = {"state": "running", "step": 0, "step_label": STEPS[0][1], "started": time.time(), "finished": None,
             "error": None, "log": []}
        _JOBS[cl.id] = j
    threading.Thread(target=_run, args=(c, cl, j), daemon=True).start()
    return status(cl.id)


def _run(c, cl, j):
    py = sys.executable
    pl = os.path.join(config.ROOT, "pipeline")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cmds = [[py, os.path.join(pl, "run_pipeline.py"), cl.video, "--stage", "fit"],
            [py, os.path.join(pl, "build_env.py"), cl.name],
            [py, os.path.join(pl, "run_pipeline.py"), cl.video, "--stage", "render"]]
    try:
        for i, cmd in enumerate(cmds):
            j["step"], j["step_label"] = i, STEPS[i][1]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", env=env, cwd=config.ROOT)
            for line in p.stdout:
                line = line.rstrip()
                if line:
                    j["log"] = (j["log"] + [line])[-12:]
            p.wait()
            if p.returncode != 0 or (i != 1 and any("1 failed" in l for l in j["log"])):
                raise RuntimeError("%s failed: %s" % (STEPS[i][0], (j["log"] or ["?"])[-1]))
        j["state"] = "done"
    except Exception as e:
        j["state"] = "failed"; j["error"] = str(e)
    j["finished"] = time.time()

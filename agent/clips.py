# -*- coding: utf-8 -*-
"""Clip discovery, per-clip folders and status (status is read from the files, so it survives restarts)."""
import hashlib
import json
import os
import time

from . import config  # noqa: F401  (puts the pipeline on sys.path)
from run_pipeline import clip_name, marks_dir, collect  # pipeline


class Clip:
    __slots__ = ("id", "video", "name", "film", "rel")

    def __init__(self, video, c):
        root = os.path.abspath(c["source_root"])
        self.video = os.path.abspath(video)
        self.rel = os.path.relpath(self.video, root)
        self.id = hashlib.sha1(self.rel.lower().encode("utf-8")).hexdigest()[:10]   # stable across restarts
        self.name = clip_name(self.video, c)
        self.film = self.rel.split(os.sep)[0] if os.sep in self.rel else ""

    def to_dict(self):
        return {"id": self.id, "name": self.name, "film": self.film, "rel": self.rel}


class Library:
    """all clips under source_root; rescanned on demand"""

    def __init__(self, c):
        self.c = c
        self.clips = []
        self.by_id = {}
        self.scanned = 0
        self.rescan()

    def rescan(self):
        root = self.c["source_root"]
        vids = collect([root]) if os.path.isdir(root) else []
        clips = [Clip(v, self.c) for v in vids]
        skip = {s.lower() for s in self.c.get("source_exclude", ["3D"])}        # the client's sample folder is not work
        clips = [cl for cl in clips if cl.film.lower() not in skip]
        names = {}
        for cl in clips:                                    # names must be unique: they are folder / zip names
            names.setdefault(cl.name, []).append(cl)
        for dup in (v for v in names.values() if len(v) > 1):
            for cl in dup:
                cl.name = "%s_%s" % (cl.name, cl.id[:4])
        self.clips = clips
        self.by_id = {cl.id: cl for cl in clips}
        self.scanned = time.time()
        return len(clips)

    def get(self, cid):
        cl = self.by_id.get(cid)
        if cl is None:
            raise KeyError("unknown clip %s" % cid)
        return cl

    def index_of(self, cid):
        for i, cl in enumerate(self.clips):
            if cl.id == cid:
                return i
        return -1


def folders(c, cl):
    md = marks_dir(cl.video, c)
    work = os.path.join(c["work_root"], cl.name)
    sub = os.path.join(c["submit_root"], cl.name)
    return {"marks": md, "work": work, "submit": sub, "zip": sub + ".zip",
            "pose": os.path.join(md, "pose.json"), "review": os.path.join(md, "review.json"),
            "ai": os.path.join(md, "ai.json"), "preview": os.path.join(work, "preview.jpg"),
            "scene": os.path.join(work, "scene_data.json")}


def read_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


LABELS = {  # label -> group used by the start page
    "not started": "new", "waiting to prepare": "busy", "preparing": "busy", "AI checking": "busy",
    "waiting for AI": "busy", "ready for review": "ready", "confirmed": "conf", "auto-confirmed": "conf",
    "waiting to render": "busy", "rendering": "busy", "done": "done", "failed": "fail"}


def status(c, cl, job=None):
    """flags from the files + the clip's latest job -> one label"""
    f = folders(c, cl)
    rv = read_json(f["review"]) if os.path.exists(f["review"]) else None
    s = {"marks": os.path.exists(f["pose"]), "fit": os.path.exists(f["preview"]) and os.path.exists(f["scene"]),
         "ai": os.path.exists(f["ai"]), "reviewed": rv is not None, "confirmed": bool(rv and rv.get("confirmed")),
         "auto": bool(rv and rv.get("auto")), "done": os.path.exists(f["zip"])}
    if s["done"] and rv and rv.get("confirmed"):             # confirmed again after the last render
        try:
            s["done"] = os.path.getmtime(f["zip"]) >= os.path.getmtime(f["review"]) - 1
        except OSError:
            pass
    s["job"] = job and {k: job[k] for k in ("id", "stage", "state", "attempts", "error") if k in job}
    state, stage = (job or {}).get("state"), (job or {}).get("stage")
    s["ai_failed"] = state == "failed" and stage == "ai"
    if state in ("queued", "running"):
        label = {("fit", "queued"): "waiting to prepare", ("fit", "running"): "preparing",
                 ("ai", "queued"): "waiting for AI", ("ai", "running"): "AI checking",
                 ("all", "queued"): "waiting to render", ("all", "running"): "rendering"}[(stage, state)]
    elif state == "failed" and stage != "ai":           # a failed AI suggestion does not block the clip
        label = "failed"
    elif s["done"]:
        label = "done"
    elif s["confirmed"]:
        label = "auto-confirmed" if s["auto"] else "confirmed"
    elif s["fit"]:
        label = "ready for review"
    else:
        label = "not started"
    s["label"] = label
    s["group"] = LABELS[label]
    return s

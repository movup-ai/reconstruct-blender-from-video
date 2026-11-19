# -*- coding: utf-8 -*-
"""Clip discovery, per-clip folders and status (status is read from the files, so it survives restarts)."""
import glob
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
        key = self.rel.lower()                              # stable across restarts; the project's SOURCE keeps its old ids,
        if os.path.normcase(root) != os.path.normcase(os.path.join(config.ROOT, "SOURCE")):   # other folders mix in
            key = os.path.normcase(root) + "|" + key       # their path, so their clips never share jobs with SOURCE's
        self.id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        self.name = clip_name(self.video, c)
        self.film = self.rel.split(os.sep)[0] if os.sep in self.rel else ""

    def to_dict(self):
        return {"id": self.id, "name": self.name, "film": self.film, "rel": self.rel}


class Library:
    """all clips under source_root; rescanned on demand"""

    def __init__(self, c):
        self.c = c
        self.clips = []
        self.clashes = []
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
        self.clashes = []
        for cl in clips:                                    # same name as a clip of another folder that still exists
            other = owner(self.c, cl.name)
            if other and os.path.normcase(os.path.realpath(other)) != os.path.normcase(os.path.realpath(cl.video)) and os.path.exists(other):
                self.clashes.append({"clip": cl.name, "other": other})
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


_FP_CACHE = {}
REVIEW_NOISE = ("saved", "confirmed", "auto", "source", "clip", "video")


def _digest(p):
    """sha1 of a file's bytes, cached by (size, mtime)"""
    try:
        st = os.stat(p)
    except OSError:
        return None
    key = (p, st.st_size, st.st_mtime_ns)
    if key not in _FP_CACHE:
        with open(p, "rb") as fh:
            _FP_CACHE[key] = hashlib.sha1(fh.read()).hexdigest()
    return _FP_CACHE[key]


def fingerprint(c, cl):
    """everything a Finish depends on that a person can change: roles / camera / notes (review.json without its
    save time and confirm flag), joint fixes, objects, hints, the hand-built set, the description, the package name
    and the video itself.  Same fingerprint as the last Finish = nothing to render again."""
    import package as pkg                                # pipeline
    f = folders(c, cl)
    out = os.path.join(config.ROOT, "output", cl.name)
    rv = read_json(f["review"], {}) or {}
    h = hashlib.sha1(json.dumps({k: v for k, v in rv.items() if k not in REVIEW_NOISE}, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    chk = sorted(glob.glob(os.path.join(f["marks"], "checked*.json")), key=os.path.getmtime)
    fixes = (read_json(chk[-1], {}) or {}) if chk else {}          # the joint editor stamps every save: not a change
    h.update(json.dumps({k: v for k, v in fixes.items() if k != "saved"}, sort_keys=True).encode("utf-8"))
    parts = [os.path.join(d, n) for d in (f["marks"], out) for n in ("env_ai.json", "hints.json", "env_claude.json", "desc.txt")]
    for p in parts:
        h.update(("%s|" % (_digest(p) if p else None)).encode("utf-8"))
    try:
        st = os.stat(cl.video)
        h.update(("%d|%d|" % (st.st_size, int(st.st_mtime))).encode("utf-8"))
    except OSError:
        pass
    h.update(str(pkg.package_name(cl.name)).encode("utf-8"))
    return h.hexdigest()


def finished_fp(c, cl):
    """the fingerprint of the last successful Finish (None: never finished from the app)"""
    return (read_json(os.path.join(folders(c, cl)["work"], "finished.json"), {}) or {}).get("fp")


def record_finish(c, cl, fp):
    p = os.path.join(folders(c, cl)["work"], "finished.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"fp": fp, "at": time.strftime("%Y-%m-%d %H:%M:%S")}, fh)


def owner(c, name):
    """the video a clip name was last worked on from (Work_auto\\<name>\\clip_info.json), or None"""
    v = (read_json(os.path.join(c["work_root"], name, "clip_info.json"), {}) or {}).get("video")
    return os.path.normpath(os.path.join(config.ROOT, v)) if v else None    # runs from the command line may store it relative


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
    "waiting for AI": "busy", "ready for review": "ready", "reviewed": "conf", "auto-confirmed": "conf",
    "changed since Finish": "conf", "waiting to finish": "busy", "finishing": "busy", "finished": "done", "failed": "fail"}


def status(c, cl, job=None):
    """flags from the files + the clip's latest job -> one label"""
    f = folders(c, cl)
    rv = read_json(f["review"]) if os.path.exists(f["review"]) else None
    s = {"marks": os.path.exists(f["pose"]), "fit": os.path.exists(f["preview"]) and os.path.exists(f["scene"]),
         "ai": os.path.exists(f["ai"]), "reviewed": rv is not None, "confirmed": bool(rv and rv.get("confirmed")),
         "auto": bool(rv and rv.get("auto")), "done": os.path.exists(f["zip"])}
    if s["done"]:
        last = finished_fp(c, cl)
        if last:                                             # finished from the app: anything changed since?
            s["done"] = last == fingerprint(c, cl)
        elif rv and rv.get("confirmed"):                     # older package: confirmed again after the last render
            try:
                s["done"] = os.path.getmtime(f["zip"]) >= os.path.getmtime(f["review"]) - 1
            except OSError:
                pass
    s["changed"] = os.path.exists(f["zip"]) and not s["done"]
    s["job"] = job and {k: job[k] for k in ("id", "stage", "state", "attempts", "error") if k in job}
    state, stage = (job or {}).get("state"), (job or {}).get("stage")
    s["ai_failed"] = state == "failed" and stage == "ai"
    if state in ("queued", "running"):
        label = {("fit", "queued"): "waiting to prepare", ("fit", "running"): "preparing",
                 ("ai", "queued"): "waiting for AI", ("ai", "running"): "AI checking",
                 ("all", "queued"): "waiting to finish", ("all", "running"): "finishing"}[(stage, state)]
    elif state == "failed" and stage != "ai":           # a failed AI suggestion does not block the clip
        label = "failed"
    elif s["done"]:
        label = "finished"
    elif s["confirmed"]:
        label = "changed since Finish" if s["changed"] else ("auto-confirmed" if s["auto"] else "reviewed")
    elif s["fit"]:
        label = "ready for review"
    else:
        label = "not started"
    s["label"] = label
    s["group"] = LABELS[label]
    return s

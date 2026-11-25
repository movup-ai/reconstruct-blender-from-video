# -*- coding: utf-8 -*-
"""Clip discovery, per-clip folders and status (status is read from the files, so it survives restarts)."""
import glob
import hashlib
import json
import os
import shutil
import threading
import time
import zipfile

from . import config  # noqa: F401  (puts the pipeline on sys.path)
from run_pipeline import clip_name, marks_dir, collect, VIDEO_EXT  # pipeline
import package as pkg  # pipeline


# ---- the client's task zips: the v2.1 package layout, with only the source video filled in (target/original.mp4)
#      and meta/task.json.  Each zip's video is written next to it as <zip name>.mp4, which is then an ordinary clip;
#      the zip name is its delivery name.  The zip itself is never changed.
_UNZIP_LOCK = threading.Lock()


def _unzipped_p(c):
    return os.path.join(config.user_data(c), "unzipped.json")


def _zip_video(names):
    """the source video's entry in a task zip: target/original.*, else the only video in it (None when there is
    none, or when the zip is a finished submission - it has a white model)"""
    low = {n.lower().replace("\\", "/"): n for n in names if not n.endswith("/")}
    if any(k.endswith("white_model/blendervid.mp4") for k in low):
        return None
    for k, n in low.items():
        if k.rsplit("/", 2)[-2:-1] == ["target"] and k.rsplit("/", 1)[-1].startswith("original.") and k.endswith(VIDEO_EXT):
            return n
    vids = [n for k, n in low.items() if k.endswith(VIDEO_EXT)]
    return vids[0] if len(vids) == 1 else None


def _task_info(z, names):
    """the useful part of meta/task.json ({} when the zip has none or it is unreadable)"""
    for n in names:
        if n.lower().replace("\\", "/").endswith("meta/task.json"):
            try:
                t = json.loads(z.read(n).decode("utf-8-sig"))
            except (ValueError, KeyError, OSError, UnicodeDecodeError):
                return {}
            sv = t.get("sourceVideo") or {}
            return {k: v for k, v in (("code", t.get("taskCode")), ("task", t.get("taskId")), ("name", t.get("taskName") or sv.get("name")),
                                      ("project", t.get("projectTitle")), ("duration", sv.get("durationSec"))) if v not in (None, "")}
    return {}


def unpack_zips(root, c, skip=(), write=True):
    """find the task zips under <root> (top folders in <skip> left out, like the client's 3D samples) and write each
    one's video next to it as <zip name><ext>.  A video of that name that the app did not write is left alone.
    write=False only looks (the folder preview).  -> ({normcase(video): {"video", "zip", "task"}}, [problems])"""
    skip = {s.lower() for s in skip}
    found, problems = {}, []
    if not os.path.isdir(root):
        return found, problems
    no_write = [os.path.normcase(os.path.abspath(c[k])) for k in ("submit_root", "work_root") if c.get(k)]
    with _UNZIP_LOCK:
        made = read_json(_unzipped_p(c), {}) or {}
        changed = False
        for r, ds, fs in os.walk(root):
            if os.path.normcase(os.path.abspath(r)) == os.path.normcase(os.path.abspath(root)):
                ds[:] = [d for d in ds if d.lower() not in skip]
            ds[:] = [d for d in ds if not any(os.path.normcase(os.path.abspath(os.path.join(r, d))) == p for p in no_write)]
            for f in sorted(fs):
                if not f.lower().endswith(".zip"):
                    continue
                zp = os.path.join(r, f)
                try:
                    st = os.stat(zp)
                    stamp = [st.st_size, int(st.st_mtime)]
                    rec = next((v for v in made.values() if v.get("zip") == os.path.normcase(zp)), None)
                    if rec and rec.get("stamp") == stamp and os.path.exists(rec["video"]) and os.path.getsize(rec["video"]) == rec.get("size"):
                        found[os.path.normcase(rec["video"])] = {"video": rec["video"], "zip": zp, "task": rec.get("task") or {}}
                        continue
                    with zipfile.ZipFile(zp) as z:
                        names = z.namelist()
                        ent = _zip_video(names)
                        if not ent:
                            continue                    # not a task zip (no video, or a finished submission)
                        info = z.getinfo(ent)
                        dest = os.path.splitext(zp)[0] + os.path.splitext(ent)[1].lower()
                        key = os.path.normcase(dest)
                        task = _task_info(z, names)
                        ours = key in made and made[key].get("zip") == os.path.normcase(zp)
                        if os.path.exists(dest) and not ours and os.path.getsize(dest) != info.file_size:
                            problems.append("%s: %s already exists and is not the video in the zip - left as it is"
                                            % (os.path.relpath(zp, root), os.path.basename(dest)))
                            continue
                        if write and not (os.path.exists(dest) and os.path.getsize(dest) == info.file_size):
                            part = "%s.%d.part" % (dest, threading.get_ident())
                            try:
                                with z.open(info) as src, open(part, "wb") as out:
                                    shutil.copyfileobj(src, out, 1 << 20)
                                os.replace(part, dest)
                            finally:
                                if os.path.exists(part):
                                    os.remove(part)
                        if write:
                            made[key] = {"video": dest, "zip": os.path.normcase(zp), "stamp": stamp, "size": info.file_size, "task": task}
                            changed = True
                        found[key] = {"video": dest, "zip": zp, "task": task}
                except (OSError, zipfile.BadZipFile, RuntimeError) as e:     # RuntimeError: a zip with a password
                    problems.append("%s: cannot read the video (%s)" % (os.path.relpath(zp, root), e))
        gone = [k for k, v in made.items() if not os.path.exists(v.get("zip", "")) or not os.path.exists(v.get("video", ""))]
        for k in gone:
            made.pop(k); changed = True
        if changed and write:
            os.makedirs(os.path.dirname(_unzipped_p(c)), exist_ok=True)
            tmp = _unzipped_p(c) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(made, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, _unzipped_p(c))
    return found, problems


class Clip:
    __slots__ = ("id", "video", "name", "film", "rel", "zip", "task")

    def __init__(self, video, c):
        root = os.path.abspath(c["source_root"])
        self.video = os.path.abspath(video)
        self.rel = os.path.relpath(self.video, root)
        key = self.rel.lower()                              # stable across restarts; the project's SOURCE keeps its old ids,
        if os.path.normcase(root) != os.path.normcase(os.path.join(config.ROOT, "SOURCE")):   # other folders mix in
            key = os.path.normcase(root) + "|" + key       # their path, so their clips never share jobs with SOURCE's
        if c.get("_user"):                                  # a member's clip: never the same id as another member's,
            key = "member:" + c["_user"] + "|" + os.path.normcase(root) + "|" + self.rel.lower()   # even in one folder
        self.id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        self.name = clip_name(self.video, c)
        self.film = self.rel.split(os.sep)[0] if os.sep in self.rel else ""
        self.zip = None                                     # the task zip the video came from (path in source_root)
        self.task = {}                                      # its meta/task.json (code, name, project...)

    def to_dict(self):
        d = {"id": self.id, "name": self.name, "film": self.film, "rel": self.rel}
        if self.zip:
            d["zip"] = self.zip
            d["task"] = self.task
        return d


class Library:
    """all clips under source_root; rescanned on demand.  preview=True (the folder dialog) lists the task zips' videos
    without writing them out"""

    def __init__(self, c, preview=False):
        self.c = c
        self.preview = preview
        self.zip_problems = []
        self.clips = []
        self.clashes = []
        self.removed = []
        self.removed_by_id = {}
        self.by_id = {}
        self.scanned = 0
        self.rescan()

    def rescan(self):
        root = self.c["source_root"]
        skip = {s.lower() for s in self.c.get("source_exclude", ["3D"])}        # the client's sample folder is not work
        zips, self.zip_problems = unpack_zips(root, self.c, skip, write=not self.preview) if root else ({}, [])
        vids = collect([root]) if os.path.isdir(root) else []
        have = {os.path.normcase(v) for v in vids}
        vids += [z["video"] for k, z in zips.items() if k not in have]          # (preview: not written out yet)
        clips = [Clip(v, self.c) for v in vids]
        for cl in clips:
            z = zips.get(os.path.normcase(cl.video))
            if z:
                cl.zip, cl.task = os.path.relpath(z["zip"], os.path.abspath(root)), z["task"]
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
        if not self.preview:                                # a task zip's name is the clip's delivery name
            pkg.register_names({cl.name: os.path.splitext(os.path.basename(cl.zip))[0].lower() for cl in clips if cl.zip}, self.c)
        gone = removed_set(self.c)                         # taken off the list by the person (files stay where they are)
        off = [cl for cl in clips if os.path.normcase(cl.video) in gone]
        self.removed = [{"id": cl.id, "path": cl.video, "name": cl.name, "rel": cl.rel} for cl in off]
        self.removed_by_id = {cl.id: cl for cl in off}                  # for their cards' thumbnails
        clips = [cl for cl in clips if os.path.normcase(cl.video) not in gone]
        self.clips = clips
        self.by_id = {cl.id: cl for cl in clips}
        self.scanned = time.time()
        return len(clips)

    def remove(self, cl):
        """take a video off the list: it is not prepared or finished; the file is not touched"""
        gone = removed_set(self.c); gone.add(os.path.normcase(cl.video)); save_removed(gone, self.c)
        self.rescan()

    def restore(self, path):
        gone = removed_set(self.c); gone.discard(os.path.normcase(os.path.abspath(path))); save_removed(gone, self.c)
        self.rescan()

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


def _held_p(c=None):
    return os.path.join(config.user_data(c or {}), "held.json")


def held_set(c=None):
    """clips taken out of the waiting line on the Clips page (clip ids): Start and autopilot leave them until added back.
    c: a member's settings (their own list)"""
    return set(read_json(_held_p(c), []) or [])


def set_held(cid, on, c=None):
    ids = held_set(c)
    if (cid in ids) == bool(on):
        return
    ids.add(cid) if on else ids.discard(cid)
    os.makedirs(os.path.dirname(_held_p(c)), exist_ok=True)
    with open(_held_p(c), "w", encoding="utf-8") as fh:
        json.dump(sorted(ids), fh, indent=1)


def _removed_p(c=None):
    return os.path.join(config.user_data(c or {}), "removed.json")


def removed_set(c=None):
    """videos taken off the list in the app (normcased full paths); c: a member's settings (their own list)"""
    return set(read_json(_removed_p(c), []) or [])


def save_removed(gone, c=None):
    os.makedirs(os.path.dirname(_removed_p(c)), exist_ok=True)
    with open(_removed_p(c), "w", encoding="utf-8") as fh:
        json.dump(sorted(gone), fh, ensure_ascii=False, indent=1)


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
    from common import output_dir                        # pipeline
    f = folders(c, cl)
    out = os.path.join(output_dir(c), cl.name)
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
    h.update(str(pkg.package_name(cl.name, c)).encode("utf-8"))
    settings = {k: c.get(k, config.DEFAULT_CFG.get(k)) for k in
                ("pose_mode", "camera_backend", "da3_model", "da3_python", "da3_meters_per_unit",
                 "da3_max_views", "foot_contact", "set_adaptive_frames", "set_max_frames",
                 "metric_scene", "metric_max_views", "metric_resolution", "scene_verify", "scene_repair", "scene_repair_limit",
                 "tracker_backend", "sam3_checkpoint", "sam3_python", "sam3_mask_side", "sam3_max_objects", "sam3_probability",
                 "sam_points", "sam3_image_batch", "sam3_image_max_masks",
                 "human_backend", "sam3d_checkpoint", "sam3d_mhr", "sam3d_python", "sam3d_stride", "sam3d_surface_side")}
    h.update(json.dumps({"pipeline_version": 7, **settings}, sort_keys=True).encode("utf-8"))
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


def prepared_paths(c, cl):
    """everything preparing, reviewing and finishing a clip left on disk -> [(path, what)] that exist.
    Not included: the video, its delivery name, its thumbnail, the film's cast notes (shared by its clips)."""
    from common import output_dir                                     # pipeline
    f = folders(c, cl)
    out = [(f["marks"], "detected joints, roles, joint fixes, objects, AI suggestions, description"),
           (os.path.join(output_dir(c), cl.name), "hints and a hand-built set"),
           (f["work"], "fit, set, preview, renders and job logs"),
           (f["submit"], "the finished package"),
           (f["zip"], "the finished zip")]
    return [(p, what) for p, what in out if os.path.exists(p)]


def _inside(p, roots):
    """True when p is strictly inside one of roots (never a root itself)"""
    p = os.path.normcase(os.path.realpath(p))
    for r in roots:
        r = os.path.normcase(os.path.realpath(r))
        if p != r and p.startswith(r.rstrip("\\/") + os.sep):
            return True
    return False


def reset(c, cl):
    """Start over: delete what prepared_paths() lists, so the next Prepare detects the people again from the video.
    Returns {"removed": [paths], "failed": [(path, reason)]}; a file another program holds open is reported, not hidden."""
    import shutil
    from common import output_dir                                     # pipeline
    from run_pipeline import marks_base
    roots = [marks_base(c), output_dir(c), c["work_root"], c["submit_root"]]
    removed, failed = [], []
    for p, _ in prepared_paths(c, cl):
        if not _inside(p, roots):                                     # never delete outside the app's own folders
            failed.append((p, "outside the app's folders")); continue
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            removed.append(p)
        except OSError as e:
            failed.append((p, e.strerror or str(e)))
    return {"removed": removed, "failed": failed}


def read_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


LABELS = {  # label -> group used by the start page
    "not prepared": "new", "waiting to prepare": "busy", "preparing": "busy", "AI checking": "busy",
    "waiting for AI": "busy", "ready for review": "ready", "reviewed": "conf", "auto-confirmed": "conf",
    "changed since Finish": "conf", "waiting to finish": "busy", "finishing": "busy", "finished": "done", "failed": "fail",
    "taken out": "new"}


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
    s["held"] = cl.id in held_set(c)
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
        label = "taken out" if s["held"] else "not prepared"
    s["label"] = label
    s["group"] = LABELS[label]
    return s

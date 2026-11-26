# -*- coding: utf-8 -*-
"""Clip discovery, per-clip folders and status (status is read from the files, so it survives restarts)."""
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
    return os.path.join(config.DATA, "unzipped.json")


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
        if not self.preview:
            migrate_removed(clips, self.c)                 # videos removed by an earlier version: now skipped (still listed)
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


def _skipped_p(c=None):
    return os.path.join(config.DATA, "held.json")       # (file name kept from the earlier "take out")


def skipped_set(c=None):
    """clips skipped on the Clips page (clip ids): they stay on the list, Start and autopilot leave them out until
    they are included again; the video file is never touched"""
    return set(read_json(_skipped_p(c), []) or [])


def set_skipped(cid, on, c=None):
    ids = skipped_set(c)
    if (cid in ids) == bool(on):
        return
    ids.add(cid) if on else ids.discard(cid)
    os.makedirs(os.path.dirname(_skipped_p(c)), exist_ok=True)
    with open(_skipped_p(c), "w", encoding="utf-8") as fh:
        json.dump(sorted(ids), fh, indent=1)


def migrate_removed(clips, c=None):
    """earlier versions also had "Remove", which hid a video from the list (removed.json, by path): those videos
    become skipped clips, still listed, and removed.json goes"""
    p = os.path.join(config.DATA, "removed.json")
    gone = set(read_json(p, []) or [])
    if not gone:
        return
    for cl in clips:
        if os.path.normcase(cl.video) in gone:
            set_skipped(cl.id, True, c)
    try:
        os.remove(p)
    except OSError:
        pass


# the settings Stage 1 (Movement) output depends on; a setting the pipeline no longer has reads as None (stable)
STAGE1_SETTINGS = ("lens_mm", "lens_auto", "person_height", "pose_mode", "pose_engine", "foot_contact",
                   "metric_scene", "metric_max_views", "metric_resolution", "max_people", "track_occlusion_seconds",
                   "tracker_backend", "sam3_checkpoint", "sam3_mask_side", "sam3_max_objects", "sam3_probability",
                   "human_backend", "sam3d_checkpoint", "sam3d_mhr", "sam3d_stride", "sam3d_surface_side",
                   "pose3d_weight", "pose3d_sigma", "colours", "out_fps")


def fingerprint(c, cl):
    """what a Movement run is made from: the video itself (size and time), its package (delivery) name and the
    Stage 1 settings.  Same fingerprint as the last successful run = its zip is up to date."""
    h = hashlib.sha1()
    try:
        st = os.stat(cl.video)
        h.update(("%d|%d|" % (st.st_size, int(st.st_mtime))).encode("utf-8"))
    except OSError:
        h.update(b"missing|")
    h.update(("%s|" % pkg.package_name(cl.name, c)).encode("utf-8"))
    settings = {k: c.get(k, config.DEFAULT_CFG.get(k)) for k in STAGE1_SETTINGS}
    h.update(json.dumps({"pipeline_version": "stage1-movement-1", **settings}, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def finished_fp(c, cl):
    """the fingerprint of the last successful Movement run (None: never made from the app)"""
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
            "pose": os.path.join(md, "pose.json"), "preview": os.path.join(work, "preview.jpg"),
            "scene": os.path.join(work, "scene_data.json")}


def prepared_paths(c, cl):
    """everything processing a clip left on disk -> [(path, what)] that exist.
    Not included: the video and its delivery name."""
    from common import output_dir                                     # pipeline
    f = folders(c, cl)
    out = [(f["marks"], "detected people and joints"),
           (os.path.join(output_dir(c), cl.name), "older per-clip notes"),
           (f["work"], "people, camera, floor, preview, renders and job logs"),
           (os.path.join(c['work_root'],'_thumbs',cl.id+'.jpg'), 'cached thumbnail'),
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
    """Start over: delete what prepared_paths() lists, so the next run detects the people again from the video.
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


PAUSED = "Paused by user"     # the error text of a job put back in the queue by Pause (cleared when it runs / on Resume)
LABELS = {  # label -> group used by the Clips page
    "new": "new", "waiting": "busy", "paused": "busy", "processing": "busy", "done": "done",
    "outdated": "old", "failed": "fail", "cancelled": "new", "skipped": "skip"}


def status(c, cl, job=None):
    """flags from the files + the clip's latest job -> one label.
    A queued or running job is shown first; otherwise a valid zip (made from today's video, delivery name and
    Stage 1 settings) is "done", whatever happened to a later job that is no longer running."""
    f = folders(c, cl)
    has_zip = os.path.exists(f["zip"])
    s = {"marks": os.path.exists(f["pose"]), "fit": os.path.exists(f["preview"]) and os.path.exists(f["scene"]),
         "zip": has_zip, "done": False}
    if has_zip:
        last = finished_fp(c, cl)
        s["done"] = last is None or last == fingerprint(c, cl)    # None: a zip made outside the app is taken as it is
    s["outdated"] = has_zip and not s["done"]
    s["skipped"] = cl.id in skipped_set(c)
    s["job"] = job and {k: job[k] for k in ("id", "stage", "state", "attempts", "error") if k in job}
    state = (job or {}).get("state")
    s["stage"] = (job or {}).get("stage") if state in ("queued", "running") else None
    if state == "running":
        label = "processing"
    elif state == "queued":
        label = "paused" if (job or {}).get("error") == PAUSED else "waiting"
    elif s["skipped"]:                                     # chosen by the person: shown even over an older finished zip
        label = "skipped"
    elif s["done"]:
        label = "done"
    elif state == "failed":
        label = "failed"
    elif state == "stopped":
        label = "cancelled"
    elif s["outdated"]:
        label = "outdated"
    else:
        label = "new"
    s["label"] = label
    s["group"] = LABELS[label]
    return s

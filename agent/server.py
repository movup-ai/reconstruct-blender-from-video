# -*- coding: utf-8 -*-
"""HTTP server (127.0.0.1 only): the web pages, the JSON API and the media files."""
import csv
import glob
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

from . import APP_NAME, __version__, config
from .clips import Library, folders, status, read_json
from .jobs import JobStore, Runner
from . import review, media, ai, brief
import package as pkg  # pipeline (on sys.path through config)

log = logging.getLogger("agent.server")


class HttpError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


class App:
    def __init__(self, c=None):
        self.c = c or config.load()
        config.ensure_dirs(self.c)
        self.lib = Library(self.c)
        self.store = JobStore(os.path.join(config.DATA, "jobs.db"))
        self.runner = Runner(self)
        self._ai = {"t": 0, "v": None}
        self.started = time.time()
        self.server = None

    # ------------------------------------------------------------ helpers
    def ai_ready(self):
        if not self.c.get("ai_enabled"):
            return False
        return ai.status(self.c)["ready"]

    def clip(self, cid):
        try:
            return self.lib.get(cid)
        except KeyError:
            raise HttpError(404, "unknown clip %s" % cid)

    def status_all(self):
        latest = self.store.latest()
        return [(cl, status(self.c, cl, latest.get(cl.id))) for cl in self.lib.clips]

    def queue_summary(self):
        cnt = self.store.counts()
        run = []
        for j in self.store.running():
            cl = self.lib.by_id.get(j["clip"])
            pr = job_progress(self.c, cl, j) if cl else {}
            run.append({"clip": j["clip"], "name": cl.name if cl else j["clip"], "stage": j["stage"],
                        "since": round(time.time() - (j["started"] or time.time())), "progress": pr.get("pct"), "step": pr.get("step")})
        return {"running": len(run), "queued": cnt.get("queued", 0), "failed": cnt.get("failed", 0),
                "parallel": self.c["parallel"], "autopilot": self.c["autopilot"], "now": run}

    def health(self):
        from common import blender_exe
        from scene_depth import MODEL_PATH
        be = blender_exe(self.c)
        try:
            free = shutil.disk_usage(self.c["work_root"]).free
        except OSError:
            free = None
        checks = [
            {"name": "Source folder", "ok": os.path.isdir(self.c["source_root"]), "detail": "%s · %d clips" % (self.c["source_root"], len(self.lib.clips))},
            {"name": "Blender", "ok": bool(be), "detail": be or "not found - install Blender 5.x or set its path in Settings"},
            {"name": "Depth model", "ok": os.path.exists(MODEL_PATH), "detail": "models\\depth_anything_v2_small.onnx" if os.path.exists(MODEL_PATH) else "missing - run Setup.bat"},
            {"name": "Sample character", "ok": os.path.exists(os.path.join(config.ROOT, "assets", "sample_character.json")), "detail": "assets\\sample_character.json"},
            {"name": "Pose models", "ok": bool(glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "rtmlib", "hub", "checkpoints", "*.onnx"))), "detail": "downloaded on first use if missing"},
            {"name": "AI suggestions (optional)", "ok": ai.status(self.c)["ready"], "optional": True, "detail": ai.status(self.c)["detail"]},
            {"name": "Disk space", "ok": free is None or free > 20e9, "detail": "%.0f GB free" % (free / 1e9) if free else "?"},
        ]
        return {"app": APP_NAME, "version": __version__, "uptime_s": round(time.time() - self.started),
                "ok": all(x["ok"] for x in checks if not x.get("optional")), "checks": checks}

    def next_to_review(self, after_id):
        items = self.status_all()
        i = next((k for k, (cl, _) in enumerate(items) if cl.id == after_id), -1)
        for k in list(range(i + 1, len(items))) + list(range(0, max(i, 0))):
            if items[k][1]["label"] == "ready for review":
                return items[k][0].id
        return None

    def clip_info(self, cid):
        cl = self.clip(cid)
        c = self.c
        f = folders(c, cl)
        job = self.store.latest_for(cl.id)
        st = status(c, cl, job)
        idx = self.lib.index_of(cl.id)
        out = {"clip": cl.to_dict(), "status": st, "index": idx, "count": len(self.lib.clips),
               "prev": self.lib.clips[idx - 1].id if idx > 0 else None,
               "next": self.lib.clips[idx + 1].id if 0 <= idx < len(self.lib.clips) - 1 else None,
               "next_review": self.next_to_review(cl.id), "defaults": {"height": c["person_height"], "lens": c["lens_mm"]},
               "ai_ready": self.ai_ready(), "package": pkg.package_name(cl.name)}
        pose = read_json(f["pose"])
        if pose:
            media.person_crops(c, cl)
            rep = read_json(os.path.join(f["work"], "report.json"), {}) or {}
            fitted = {}
            for p in rep.get("people", []):
                fitted.setdefault(str(p["track"]), []).append(p)
            aip = (read_json(f["ai"], {}) or {}).get("people", {})
            flags = person_flags(rep)
            jf = {int(round(fr / 8.0)) for fs in jump_frames(rep).values() for fr in fs}    # joint editor: every 8th frame
            out["flagged"] = min(len(jf), max(3, -(-(pose["frames"] // 8 + 1) // 4))) if jf else 0  # the editor flags its worst quarter
            out["people"] = []
            for p in pose["people"]:
                fs = sorted(int(x) for x in p["frames"])
                fit = fitted.get(str(p["id"])) or []
                out["people"].append({"id": p["id"], "frames": len(fs), "first": fs[0], "last": fs[-1],
                                      "fit_px": fit[0]["median_error_px"] if fit else None,
                                      "shots": sorted({q.get("shot", 1) for q in fit}), "ai": aip.get(str(p["id"])),
                                      "flags": (flags.get(str(p["id"])) or {}).get("flags", []),
                                      "not_person": (flags.get(str(p["id"])) or {}).get("not_person")})
            out.update(frames=pose["frames"], fps=round(pose["fps"], 3), size=[pose["width"], pose["height"]])
            cam = read_json(os.path.join(f["work"], "camera.json"))
            if cam:
                out["camera"] = {k: cam.get(k) for k in ("max_rotation_deg", "parallax_px", "cuts", "warnings", "shots")}
            out["warnings"] = rep.get("warnings", [])
            sd = read_json(f["scene"])
            if sd:
                kinds = {}
                for m in sd.get("meshes", []):
                    k = (re.match(r"[a-z]+", m["name"]) or [m["name"]])[0]
                    kinds[k] = kinds.get(k, 0) + 1
                out["set"] = kinds
                out["blocks"] = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(f["work"], "blocks*.jpg")))]
        rv = read_json(f["review"])
        out["review"] = rv
        out["suggested"] = review.suggested(c, cl) if pose else None
        out["ai"] = {k: v for k, v in (read_json(f["ai"], {}) or {}).items() if k != "raw"} or None
        out["env_ai"] = (read_json(os.path.join(f["marks"], "env_ai.json"), {}) or {}).get("shots")
        regs = {}
        for rp in sorted(glob.glob(os.path.join(f["work"], "regions*.json"))):
            r = read_json(rp) or {}
            regs[r.get("tag", "")] = {"regions": [{k: x.get(k) for k in ("id", "type", "dropped", "height_m", "distance_m")} for x in r.get("regions", [])],
                                      "added": r.get("added", [])}
        out["regions"] = regs
        out["outputs"] = {"comparison": bool(glob.glob(os.path.join(f["submit"], "comparison", "*.mp4"))), "zip": os.path.exists(f["zip"]),
                          "zip_path": f["zip"]}
        if job:
            lp = job.get("log") or os.path.join(f["work"], "job_%s.log" % job["stage"])
            txt = ""
            if os.path.exists(lp):
                with open(lp, encoding="utf-8", errors="replace") as fh:
                    txt = fh.read()[-8000:]
            out["job"] = {"id": job["id"], "stage": job["stage"], "state": job["state"], "attempts": job["attempts"],
                          "error": job.get("error"), "log": txt, "advice": advice(job.get("error") or "", txt) if job["state"] == "failed" else None}
            if job["state"] == "running":
                out["job"].update(job_progress(c, cl, job))
        return out

    # ------------------------------------------------------------ the simple flow: folder -> prepare -> review -> Finish
    def use_folder(self, path, title):
        """make <path> the source folder: package names for its new videos, then every video is prepared"""
        scan = folder_scan(self.c, path, title)
        if scan["unnamed"] and not scan["slug"]:
            raise HttpError(400, "type an English title for this folder (letters and numbers), e.g. the hidden blade")
        upd = {"source_root": scan["path"]}
        if self.c["autopilot"] == "off":                       # preparing starts by itself
            upd["autopilot"] = "prepare"
        self.c = config.save(upd)
        self.lib.c = self.c
        self.lib.rescan()
        new = pkg.assign_names([cl.name for cl in self.lib.clips], scan["title"])
        queued = sum(1 for cl, st in self.status_all() if not st["fit"] and self.store.enqueue(cl.id, "fit"))
        self.runner.wake()
        log.info("source folder %s: %d videos, %d new package names, %d queued", scan["path"], len(self.lib.clips), len(new), queued)
        return {"ok": True, "clips": len(self.lib.clips), "named": len(new), "queued": queued, "clashes": self.lib.clashes}

    def finish(self, cl, force=False):
        """queue the final job (fit, set, Blender, videos, v2.1 zip) for a reviewed clip.
        -> "queued" | "unchanged" (same input as the last Finish) | "busy" | an error"""
        st = status(self.c, cl, self.store.latest_for(cl.id))
        if not st["fit"]:
            raise HttpError(409, "prepare the clip first")
        if not pkg.package_name(cl.name):
            raise HttpError(409, "%s has no v2.1 package name yet: choose its folder again on the Clips page and type a title" % cl.name)
        if not st["confirmed"]:
            raise HttpError(409, "mark the clip reviewed first")
        if st["done"] and not force:
            return "unchanged"
        if st["job"] and st["job"]["state"] in ("queued", "running"):
            return "busy"
        return "queued" if self.store.enqueue(cl.id, "all", force=True) else "busy"

    def finish_reviewed(self):
        out = {"queued": 0, "unchanged": 0, "busy": 0, "no_name": 0}
        for cl, st in self.status_all():
            if not st["confirmed"] or not st["fit"]:
                continue
            if not pkg.package_name(cl.name):
                out["no_name"] += 1
                continue
            out[self.finish(cl)] += 1
        self.runner.wake()
        return out

    def report_rows(self):
        rows = []
        for cl, st in self.status_all():
            f = folders(self.c, cl)
            rep = read_json(os.path.join(f["work"], "report.json"), {}) or {}
            rows.append({"film": cl.film, "clip": cl.name, "status": st["label"], "people": len(rep.get("people", [])),
                         "shots": len(rep.get("shots", []) or [1]) if rep else "", "review": "auto" if st["auto"] else ("person" if st["confirmed"] else ""),
                         "zip": f["zip"] if st["done"] else "", "warnings": " | ".join(rep.get("warnings", [])),
                         "error": (st["job"] or {}).get("error") or ""})
        return rows


# ---------------------------------------------------------------- progress, advice, submissions
# where each step starts, % of the job (measured on this PC: finding people is about half of a first preparation;
# a Finish reuses the people found and the lens, and spends its time on the camera, the set and Blender)
STEPS = {"fit": [("  marks:", 45, "measuring the lens"), ("  camera ...", 58, "solving the camera"),
                 ("  people ...", 75, "fitting 3D people"), ("  set ...", 84, "building the set"), ("  quality:", 96, "checking the result")],
         "all": [("  marks:", 3, "measuring the lens"), ("  camera ...", 6, "solving the camera"),
                 ("  people ...", 25, "fitting 3D people"), ("  set ...", 33, "building the set"), ("  quality:", 50, "checking the result")]}


def job_progress(c, cl, job):
    """{pct, step} from the job's log and the rendered frames"""
    f = folders(c, cl)
    lp = job.get("log") or os.path.join(f["work"], "job_%s.log" % job["stage"])
    try:
        with open(lp, encoding="utf-8", errors="replace") as fh:
            fh.seek(max(0, os.path.getsize(lp) - 60000))
            txt = fh.read()
    except OSError:
        return {"pct": 0, "step": "starting"}
    if job["stage"] == "ai":
        return {"pct": 50 if "environment" not in txt else 85, "step": "DeepSeek is looking at the clip"}
    pct, step = 1, "starting"
    m = re.findall(r"detected (\d+) frames \(frame \d+/(\d+)", txt)      # frames done so far (the frame number jumps:
    if m:                                                                # the detector runs over the clip in passes)
        a, b = int(m[-1][0]), int(m[-1][1])
        pct, step = 2 + 43 * min(a, b) / max(b, 1), "finding people (%d of %d frames)" % (min(a, b), b)
    for key, p, s in STEPS.get(job["stage"], STEPS["fit"]):
        if key in txt:
            pct, step = p, s
    shots = re.findall(r"\[shot (\d+): frames", txt)
    if shots and "  quality:" not in txt:
        step += " (shot %s)" % shots[-1]
    m = re.search(r"blender build \+ render \((\d+) frames\)", txt)
    if m:
        n = int(m.group(1))
        done = len(glob.glob(os.path.join(f["work"], "render", "wm_*.jpg")))
        pct, step = 55 + 35 * min(done, n) / max(n, 1), "rendering frame %d of %d" % (min(done, n), n)
    if "videos + zip" in txt:
        pct, step = 92, "encoding the videos and packing the zip"
    return {"pct": round(min(pct, 99)), "step": step}


def person_flags(rep):
    """report.json -> {track id: {"flags": [...], "not_person": reason or None}}: what the quality check found per
    detected person, and whether it looks like a statue / picture / junk (a suggestion only, never applied)"""
    out = {}
    for r in (rep.get("quality") or {}).get("people", []):
        d = out.setdefault(str(r.get("track")), {"flags": [], "not_person": None})
        for w in r.get("flags", []):
            if "pose never changes" in w:
                d["not_person"] = "the pose never changes, like a statue or a picture"
            elif w not in d["flags"]:
                d["flags"].append(w)
    for j in rep.get("left_out", []) or []:
        d = out.setdefault(str(j.get("track")), {"flags": [], "not_person": None})
        d["not_person"] = "left out of the fit: %s" % j.get("why", "short and unclear")
    return out


def jump_frames(rep, stitched=None):
    """report.json -> {person id in the joint editor: [frames where the fitted person moved suddenly]}.
    The editor numbers people as automark found them; a track the fit joined (pose.json "stitched" {old: new})
    shows there under its old number too."""
    out = {}
    for r in (rep.get("quality") or {}).get("people", []):
        if r.get("jump_frames"):
            out.setdefault(str(r.get("track")), []).extend(r["jump_frames"])
    for old, new in (stitched or {}).items():
        if str(new) in out:
            out[str(old)] = list(out[str(new)])
    return out


def check_page(c, cl):
    """the joint editor for a clip: the CURRENT check_template.html filled with the clip's frames (taken from the
    check.html automark wrote), so editor updates reach clips prepared earlier"""
    f = folders(c, cl)
    p = os.path.join(f["marks"], "check.html")
    if not os.path.exists(p):
        raise HttpError(404, "this clip has no marks yet")
    with open(p, encoding="utf-8") as fh:
        html = fh.read()
    a = html.find("const PAGE = "); b = html.find("\nconst cv = ", a)
    tp = os.path.join(config.ROOT, "check_template.html")
    if a >= 0 and b > a and os.path.exists(tp):
        with open(tp, encoding="utf-8") as fh:
            tpl = fh.read()
        data = html[a + len("const PAGE = "):b].rstrip().rstrip(";")
        if "__PAGE_DATA__" in tpl:
            html = tpl.replace("__PAGE_DATA__", data, 1)
    return html


PICK_LOCK = threading.Lock()
PICK_SCRIPT = r"""
import sys, tkinter as tk
from tkinter import filedialog
r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True); r.update()
p = filedialog.askdirectory(parent=r, initialdir=sys.argv[1], mustexist=True, title="Choose the folder with the source videos")
sys.stdout.write(p or "")
"""


def pick_folder(start):
    """the Windows folder dialog, opened on this PC by the server (a web page cannot hand back a folder path)"""
    if not PICK_LOCK.acquire(blocking=False):
        raise HttpError(409, "the folder window is already open - look for it on the taskbar")
    try:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, "-c", PICK_SCRIPT, start if os.path.isdir(start) else os.path.expanduser("~")],
                           capture_output=True, timeout=1800, env=env)
        p = r.stdout.decode("utf-8", "replace").strip()
        return os.path.normpath(p) if p else None
    finally:
        PICK_LOCK.release()


def title_guess(names):
    """the English title already used for these clips, else ''"""
    for v in names:
        if v:
            return v.rsplit("-", 1)[0].replace("_", " ")
    return ""


def folder_scan(c, path, title=""):
    """what picking <path> as the source folder would do: its videos, their clip and package names"""
    if not path or not os.path.isabs(path) or not os.path.isdir(path):
        raise HttpError(400, "not a folder: %s" % path)
    c2 = dict(c, source_root=os.path.abspath(path))
    lib = Library(c2)
    names = pkg.package_names()
    title = title or title_guess(names.get(cl.name) for cl in lib.clips)
    plan = pkg.propose_names([cl.name for cl in lib.clips], title, names)
    rows = [{"file": cl.rel, "clip": cl.name, "package": plan[cl.name][0], "new_name": plan[cl.name][1],
             "prepared": os.path.exists(folders(c2, cl)["pose"])} for cl in lib.clips]
    return {"path": c2["source_root"], "title": title, "slug": pkg.title_slug(title), "videos": rows,
            "unnamed": sum(1 for r in rows if not r["package"]), "clashes": lib.clashes,
            "current": os.path.normcase(c2["source_root"]) == os.path.normcase(os.path.abspath(c["source_root"]))}


ADVICE = [
    (r"Blender not found", "Install Blender 5.x, or set the Blender path on the Settings page."),
    (r"timed out", "The job took too long. Try again with fewer clips at once (Settings), or check the clip plays correctly."),
    (r"no pose\.json|has not been prepared", "Prepare the clip first (Re-check button)."),
    (r"MemoryError|Unable to allocate", "The PC ran out of memory. Lower 'Clips at once' on the Settings page and retry."),
    (r"No space left|disk full", "The disk is full. Free some space on the drive with the Work and SUBMIT folders."),
    (r"cannot open video|no frames", "The video file could not be read. Check it plays, or re-export it as H.264 MP4."),
    (r"DeepSeek API error 401|wrong or expired key", "The DeepSeek key was rejected. Save a new key on the Settings page."),
    (r"DeepSeek API error 402", "The DeepSeek account has no balance left."),
    (r"ffmpeg failed", "Making the videos failed. Retry once; if it fails again, open the technical log."),
    (r"not confirmed", "Confirm the clip before rendering."),
]


def advice(err, log_txt=""):
    for pat, msg in ADVICE:
        if re.search(pat, err + "\n" + log_txt[-3000:], re.I):
            return msg
    return "Retry the clip. If it fails again, open the technical log below and send it to support."


def object_shots(c, cl):
    """per shot: background picture, picture size and the object list (from DeepSeek or edited by hand)"""
    f = folders(c, cl)
    env = (read_json(os.path.join(f["marks"], "env_ai.json"), {}) or {})
    shots = []
    for rp in sorted(glob.glob(os.path.join(f["work"], "regions*.json"))):
        r = read_json(rp) or {}
        tag = r.get("tag", "")
        if not os.path.exists(os.path.join(f["work"], "plate%s.jpg" % tag)):
            continue
        sh = (env.get("shots") or {}).get(tag, {})
        shots.append({"tag": tag, "size": r.get("size"), "plate": "/media/plate/%s/%s" % (cl.id, tag),
                      "scene": sh.get("scene", ""), "objects": [o for o in sh.get("objects", []) if "bbox" in o],
                      "placed": sum(1 for o in sh.get("objects", []) if "pos" in o), "built": r.get("added", [])})
    return {"shots": shots, "source": env.get("provider", ""), "edited": bool(env.get("edited"))}


def save_objects(c, cl, body):
    import props
    f = folders(c, cl)
    p = os.path.join(f["marks"], "env_ai.json")
    env = read_json(p, {}) or {}
    env.setdefault("provider", "edited by hand")
    env["edited"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sizes = {s["tag"]: s["size"] for s in object_shots(c, cl)["shots"]}
    shots = env.setdefault("shots", {})
    n = 0
    for tag, sh in (body.get("shots") or {}).items():
        if tag not in sizes:
            raise ValueError("unknown shot %s" % tag)
        W, H = sizes[tag]
        objs = []
        for o in sh.get("objects", [])[:60]:
            b = [float(v) for v in o.get("bbox", [])]
            if len(b) != 4:
                raise ValueError("every object needs a box")
            b = [max(0.0, min(W, b[0])), max(0.0, min(H, b[1])), max(0.0, min(W, b[2])), max(0.0, min(H, b[3]))]
            if b[2] - b[0] < 3 or b[3] - b[1] < 3:
                continue
            objs.append({"what": str(o.get("what", ""))[:40], "type": props.norm_type(o.get("type")), "bbox": [round(v, 1) for v in b],
                         "attach": o.get("attach") if o.get("attach") in ("floor", "wall") else None,
                         "facing": o.get("facing") if o.get("facing") in ("camera", "left", "right", "away") else "camera",
                         "size": o.get("size") if isinstance(o.get("size"), list) and len(o.get("size")) == 3 else None,
                         "colour_role": "orange" if o.get("interactive") or o.get("colour_role") == "orange" else None})
        cur = shots.setdefault(tag, {"scene": "", "labels": {}, "drop": []})
        placed = [o for o in cur.get("objects", []) if "pos" in o]      # objects Claude placed in metres stay
        cur["objects"] = placed + objs
        n += len(objs)
    os.makedirs(f["marks"], exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(env, fh, ensure_ascii=False, indent=1)
    return n


def submissions(c):
    root = c["submit_root"]
    out = []
    if not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        sub = os.path.join(root, d)
        if not os.path.isdir(sub) or d.startswith("_"):
            continue
        zp = sub + ".zip"
        size = 0
        for r_, _, fs in os.walk(sub):
            size += sum(os.path.getsize(os.path.join(r_, x)) for x in fs)
        meta = read_json(os.path.join(sub, "meta", "meta.jsonl"))
        if meta is None:
            try:
                with open(os.path.join(sub, "meta", "meta.jsonl"), encoding="utf-8") as fh:
                    meta = json.loads(fh.readline())
            except (OSError, ValueError):
                meta = {}
        out.append({"name": d, "folder": sub, "zip": zp if os.path.exists(zp) else None,
                    "zip_mb": round(os.path.getsize(zp) / 1e6, 1) if os.path.exists(zp) else None, "folder_mb": round(size / 1e6, 1),
                    "made": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(zp if os.path.exists(zp) else sub))),
                    "duration": meta.get("pattern_video_duration"), "resolution": meta.get("source_video_resolution")})
    return out


# ---------------------------------------------------------------- HTTP
def make_handler(app):
    class H(BaseHTTPRequestHandler):
        server_version = "BlenderAgent/" + __version__

        def log_message(self, fmt, *a):
            pass

        # ---- responses
        def send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False)
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def page(self, name):
            with open(os.path.join(config.WEB, name), encoding="utf-8") as fh:
                return self.send(200, fh.read(), "text/html; charset=utf-8")

        def file(self, p, ctype=None):
            if not p or not os.path.isfile(p):
                raise HttpError(404, "file not found")
            ctype = ctype or mimetypes.guess_type(p)[0] or "application/octet-stream"
            size = os.path.getsize(p)
            rng = self.headers.get("Range")
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if m and (m.group(1) or m.group(2)):
                a = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
                b = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
                b = min(b, size - 1)
                if a > b:
                    self.send_response(416); self.send_header("Content-Range", "bytes */%d" % size); self.end_headers(); return
                self.send_response(206)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, size))
            else:
                a, b = 0, size - 1
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(b - a + 1))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(p, "rb") as fh:
                fh.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (ConnectionError, OSError):
                        return
                    left -= len(chunk)

        def body_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20 * 1024 * 1024:
                raise HttpError(413, "request too large")
            raw = self.rfile.read(n).decode("utf-8") if n else ""
            try:
                return json.loads(raw) if raw else {}
            except ValueError:
                raise HttpError(400, "body is not valid JSON")

        def handle_any(self, fn):
            try:
                fn()
            except HttpError as e:
                self.send(e.code, {"error": str(e)})
            except (ValueError, KeyError) as e:
                self.send(400, {"error": str(e)})
            except (ConnectionError, BrokenPipeError):
                pass
            except Exception as e:
                log.error("%s %s failed:\n%s", self.command, self.path, traceback.format_exc())
                try:
                    self.send(500, {"error": "internal error: %s" % e})
                except Exception:
                    pass

        def do_GET(self):
            self.handle_any(self.get)

        def do_HEAD(self):
            self.handle_any(self.get)

        def do_POST(self):
            self.handle_any(self.post)

        # ---- GET
        def get(self):
            u = urlparse(self.path)
            parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            q = parse_qs(u.query)
            c = app.c
            if not parts:
                return self.page("index.html")
            if parts[0] == "clip" and len(parts) == 2:
                app.clip(parts[1])
                return self.page("clip.html")
            if parts[0] == "edit" and len(parts) == 2:                  # movement editor
                app.clip(parts[1])
                return self.page("editor.html")
            if parts[:2] == ["api", "clip"] and len(parts) == 4 and parts[3] == "dots":
                from . import editor
                try:
                    return self.send(200, editor.payload(c, app.clip(parts[2])))
                except FileNotFoundError as e:
                    raise HttpError(404, str(e))
            if parts[:2] == ["api", "clip"] and len(parts) == 4 and parts[3] == "rerender":
                from . import editor
                app.clip(parts[2])
                return self.send(200, editor.status(parts[2]))
            if parts[0] == "frame" and len(parts) == 3 and parts[2].isdigit():
                from . import editor
                jpg = editor.frame_jpeg(c, app.clip(parts[1]), int(parts[2]))
                if jpg is None:
                    raise HttpError(404, "no such frame")
                return self.send(200, jpg, "image/jpeg")
            if parts in (["submissions"], ["settings"], ["help"]):
                return self.page(parts[0] + ".html")
            if parts == ["favicon.ico"] or parts == ["favicon.svg"]:
                return self.file(os.path.join(config.WEB, "favicon.svg"), "image/svg+xml")
            if parts == ["api", "submissions"]:
                return self.send(200, submissions(c))
            if parts[0] == "static" and len(parts) == 2 and re.match(r"^[\w.-]+$", parts[1]):
                return self.file(os.path.join(config.WEB, parts[1]))
            if parts == ["api", "health"]:
                return self.send(200, app.health())
            if parts == ["api", "clips"]:
                names = pkg.package_names()
                return self.send(200, {"clips": [dict(cl.to_dict(), status=st, package=names.get(cl.name)) for cl, st in app.status_all()],
                                       "queue": app.queue_summary(), "ai_ready": app.ai_ready(),
                                       "source": {"path": c["source_root"], "exists": os.path.isdir(c["source_root"]),
                                                  "clashes": app.lib.clashes}})
            if parts == ["api", "folder"]:                              # what picking a folder would do
                return self.send(200, folder_scan(c, (q.get("path") or [""])[0], (q.get("title") or [""])[0]))
            if parts == ["api", "settings"]:
                out = {k: c.get(k) for k in config.EDITABLE if k not in config.SECRETS}
                out["deepseek_key_set"] = bool(ai.api_key(c))
                return self.send(200, out)
            if parts[:2] == ["api", "clip"] and len(parts) == 3:
                return self.send(200, app.clip_info(parts[2]))
            if parts[:2] == ["api", "clip"] and len(parts) == 4 and parts[3] == "objects":
                import props
                return self.send(200, dict(object_shots(c, app.clip(parts[2])),
                                           types={k: {"size": v[0], "on": v[1]} for k, v in props.TYPES.items()}))
            if parts == ["api", "report"]:
                return self.send(200, app.report_rows())
            if parts == ["api", "validate"]:
                from . import validate
                return self.send(200, validate.check_all(c["submit_root"]) if os.path.isdir(c["submit_root"]) else [])
            if parts == ["report.csv"]:
                buf = io.StringIO()
                rows = app.report_rows()
                w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else ["clip"])
                w.writeheader(); w.writerows(rows)
                return self.send(200, "﻿" + buf.getvalue(), "text/csv; charset=utf-8",
                                 {"Content-Disposition": 'attachment; filename="blender_agent_report.csv"'})
            if parts[0] == "check" and len(parts) == 2:             # marks editor, saving back to the app
                cl = app.clip(parts[1]); f = folders(c, cl)
                html = check_page(c, cl)
                chk = sorted(glob.glob(os.path.join(f["marks"], "checked*.json")), key=os.path.getmtime)
                inj = "<script>window.APP_SAVE='/api/clip/%s/marks';" % cl.id
                inj += "window.APP_FLAGGED=" + json.dumps(jump_frames(read_json(os.path.join(f["work"], "report.json"), {}) or {},
                                                                      (read_json(f["pose"], {}) or {}).get("stitched"))) + ";"
                if chk:
                    with open(chk[-1], encoding="utf-8") as fh:
                        inj += "window.APP_CHECKED=" + fh.read() + ";"
                inj += "</script>"
                return self.send(200, html.replace("<head>", "<head>" + inj, 1), "text/html; charset=utf-8")
            if parts[0] == "download" and len(parts) == 2:             # a package zip from the SUBMIT folder
                name = os.path.basename(parts[1])
                zp = os.path.join(c["submit_root"], name + ".zip")
                if not name or not os.path.isfile(zp):
                    raise HttpError(404, "no zip for %s" % name)
                return self.file(zp, "application/zip")
            if parts[0] == "package" and len(parts) == 3 and parts[2] in ("comparison", "white_model", "target"):
                name = os.path.basename(parts[1])
                m = sorted(glob.glob(os.path.join(c["submit_root"], name, parts[2], "*.mp4")))
                return self.file(m[0] if m else None, "video/mp4")
            if parts[0] == "media" and len(parts) >= 3:
                kind = parts[1]; cl = app.clip(parts[2].split(".")[0]); f = folders(c, cl)
                if kind == "video":
                    return self.file(cl.video, "video/mp4")
                if kind == "plate" and len(parts) == 4 and re.match(r"^_[\w]+$", parts[3]):
                    return self.file(os.path.join(f["work"], "plate%s.jpg" % parts[3]), "image/jpeg")
                if kind == "preview":
                    return self.file(f["preview"], "image/jpeg")
                if kind == "clipthumb":
                    return self.file(media.clip_thumb(c, cl), "image/jpeg")
                if kind == "thumb" and len(parts) == 4:
                    return self.file(os.path.join(f["marks"], "thumbs", "P%d.jpg" % int(re.sub(r"\D", "", parts[3]) or 0)), "image/jpeg")
                if kind == "blocks" and len(parts) == 4 and re.match(r"^blocks[\w-]*\.jpg$", parts[3]):
                    return self.file(os.path.join(f["work"], parts[3]), "image/jpeg")
                if kind in ("comparison", "white_model"):
                    m = sorted(glob.glob(os.path.join(f["submit"], kind, "*.mp4")))
                    return self.file(m[0] if m else None, "video/mp4")
            raise HttpError(404, "not found")

        # ---- upload: raw file body, streamed to disk (no size held in memory)
        def upload(self, q):
            VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
            fname = os.path.basename((q.get("name") or [""])[0]).strip()
            film = (q.get("film") or [""])[0].strip()
            if not fname or os.path.splitext(fname)[1].lower() not in VIDEO_EXT:
                raise HttpError(400, "only video files (%s)" % ", ".join(VIDEO_EXT))
            if not film or re.search(r'[\\/:*?"<>|]', film) or film in (".", "..") or len(film) > 80:
                raise HttpError(400, "give the film a plain folder name (no \\ / : * ? \" < > |)")
            fname = re.sub(r'[\\/:*?"<>|]+', "_", fname)
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                raise HttpError(400, "empty file")
            if n > 20 * 1024 ** 3:
                raise HttpError(413, "file larger than 20 GB")
            root = os.path.abspath(app.c["source_root"])
            dest_dir = next((os.path.dirname(cl.video) for cl in app.lib.clips if cl.film == film), os.path.join(root, film))
            if not os.path.abspath(dest_dir).startswith(root):
                raise HttpError(400, "bad folder")
            os.makedirs(dest_dir, exist_ok=True)
            stem, ext = os.path.splitext(fname)
            dest = os.path.join(dest_dir, fname); k = 2
            while os.path.exists(dest):                               # never overwrite a clip
                dest = os.path.join(dest_dir, "%s (%d)%s" % (stem, k, ext)); k += 1
            tmp = dest + ".uploading"
            left = n
            try:
                with open(tmp, "wb") as fh:
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            raise HttpError(400, "upload interrupted")
                        fh.write(chunk); left -= len(chunk)
                os.replace(tmp, dest)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            app.lib.rescan()
            cl = next((x for x in app.lib.clips if os.path.normcase(x.video) == os.path.normcase(os.path.abspath(dest))), None)
            if cl and app.c["autopilot"] != "off":
                app.store.enqueue(cl.id, "fit"); app.runner.wake()
            log.info("uploaded %s (%.1f MB) to %s", os.path.basename(dest), n / 1e6, dest_dir)
            return self.send(200, {"ok": True, "file": os.path.basename(dest), "clip": cl.to_dict() if cl else None})

        # ---- POST
        def post(self):
            u = urlparse(self.path)
            parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            q = parse_qs(u.query)
            if parts == ["api", "upload"]:
                return self.upload(q)
            if parts == ["api", "open"]:
                raw = self.body_json().get("path", "")
                p = config.LOGS if raw == "logs" else os.path.abspath(raw)
                roots = [os.path.abspath(app.c[k]) for k in ("submit_root", "work_root", "source_root")] + [config.ROOT]
                if not any(p == r or p.startswith(r + os.sep) for r in roots) or not os.path.exists(p):
                    raise HttpError(400, "that folder is outside the app's folders")
                if os.name == "nt":
                    if os.path.isfile(p):
                        subprocess.Popen(["explorer", "/select,", p])
                    else:
                        os.startfile(p)
                return self.send(200, {"ok": True})
            if parts == ["api", "settings"]:
                old = app.c["source_root"]
                app.c = config.save(self.body_json())
                app.lib.c = app.c
                if app.c["source_root"] != old:
                    app.lib.rescan()
                app.runner.wake()
                return self.send(200, {"ok": True, "settings": {k: app.c.get(k) for k in config.EDITABLE if k not in config.SECRETS},
                                       "deepseek_key_set": bool(ai.api_key(app.c))})
            if parts == ["api", "ai_test"]:
                return self.send(200, ai.test_key(app.c))
            if parts == ["api", "prepare_all"]:
                n = sum(1 for cl, st in app.status_all() if not st["fit"] and app.store.enqueue(cl.id, "fit"))
                app.runner.wake()
                return self.send(200, {"ok": True, "queued": n})
            if parts in (["api", "finish_reviewed"], ["api", "render_confirmed"]):
                return self.send(200, dict(app.finish_reviewed(), ok=True))
            if parts == ["api", "pick_folder"]:
                return self.send(200, {"path": pick_folder(app.c["source_root"])})
            if parts == ["api", "use_folder"]:
                d = self.body_json()
                return self.send(200, app.use_folder(str(d.get("path", "")), str(d.get("title", ""))))
            if parts == ["api", "retry_failed"]:
                n = 0
                for cl, st in app.status_all():
                    if st["label"] == "failed" and st["job"] and app.store.enqueue(cl.id, st["job"]["stage"]):
                        n += 1
                app.runner.wake()
                return self.send(200, {"ok": True, "queued": n})
            if parts == ["api", "cancel_queued"]:
                return self.send(200, {"ok": True, "cancelled": app.store.cancel_queued()})
            if parts == ["api", "rescan"]:
                before = set(app.lib.by_id)
                n = app.lib.rescan()
                new = [cl for cl in app.lib.clips if cl.id not in before]
                if app.c["autopilot"] != "off":
                    for cl in new:
                        app.store.enqueue(cl.id, "fit")
                    app.runner.wake()
                return self.send(200, {"ok": True, "clips": n, "new": len(new)})
            if parts == ["api", "quit"]:
                self.send(200, {"ok": True})
                threading.Thread(target=app.shutdown, daemon=True).start()
                return
            if parts[:2] == ["api", "clip"] and len(parts) == 4:
                cl = app.clip(parts[2]); f = folders(app.c, cl); action = parts[3]
                if action == "dots":                                     # movement editor: save corrected dots
                    from . import editor
                    return self.send(200, editor.save(app.c, cl, self.body_json()))
                if action == "rerender":                                 # movement editor: fit + set + render
                    from . import editor
                    return self.send(200, editor.start_render(app.c, cl))
                if action == "marks":
                    d = self.body_json()
                    if not isinstance(d.get("frames"), dict):
                        raise HttpError(400, "marks must contain frames")
                    os.makedirs(f["marks"], exist_ok=True)
                    with open(os.path.join(f["marks"], "checked.json"), "w", encoding="utf-8") as fh:
                        json.dump(d, fh, ensure_ascii=False)
                    return self.send(200, {"ok": True})
                if action == "review":
                    d = review.validate(self.body_json())
                    d.setdefault("clip", cl.name); d.setdefault("video", cl.video)
                    review.write(f["review"], d)
                    if d["notes"].strip():                    # the description goes into meta.jsonl
                        with open(os.path.join(f["marks"], "desc.txt"), "w", encoding="utf-8") as fh:
                            fh.write(d["notes"].strip().splitlines()[0][:300])
                    # confirmed: write the hand-off files for Claude.  Rendering starts by itself only when
                    # "render_on_confirm" is on; otherwise Claude builds the set and renders from the brief.
                    queued = False
                    if d["confirmed"] and app.c.get("render_on_confirm", False):
                        queued = bool(app.store.enqueue(cl.id, "all", force=True))
                    hand = None
                    try:
                        brief.write_clip(app.c, cl)
                        hand = brief.write_index(app.c, app.lib, app.store)[1]
                    except Exception:
                        log.exception("could not write the hand-off files for %s", cl.name)
                    app.runner.wake()
                    return self.send(200, {"ok": True, "queued": queued, "next_review": app.next_to_review(cl.id),
                                           "handoff": hand and {k: hand[k] for k in ("clips", "confirmed", "all_confirmed")}})
                if action == "run":
                    stage = (q.get("stage") or ["fit"])[0]
                    if stage == "ai" and not app.ai_ready():
                        raise HttpError(409, "the local AI is not available (Ollama not running or model missing)")
                    if stage == "all" and not (read_json(f["review"]) or {}).get("confirmed"):
                        raise HttpError(409, "confirm the clip first")
                    jid = app.store.enqueue(cl.id, stage, force=True)
                    app.runner.wake()
                    return self.send(200, {"ok": True, "queued": bool(jid)})
                if action == "objects":
                    n = save_objects(app.c, cl, self.body_json())
                    jid = app.store.enqueue(cl.id, "fit", force=True) if (q.get("rebuild") or ["1"])[0] == "1" else None
                    app.runner.wake()
                    return self.send(200, {"ok": True, "objects": n, "queued": bool(jid)})
                if action == "finish":                                   # final job for one reviewed clip
                    r = app.finish(cl, force=(q.get("force") or ["0"])[0] == "1")
                    app.runner.wake()
                    return self.send(200, {"ok": True, "result": r})
                if action == "cancel":
                    return self.send(200, {"ok": True, "cancelled": app.store.cancel_queued(cl.id)})
            raise HttpError(404, "not found")

    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def serve(app, port):
    app.server = Server(("127.0.0.1", port), make_handler(app))
    app.runner.start()
    log.info("%s %s on http://127.0.0.1:%d  (%d clips)", APP_NAME, __version__, port, len(app.lib.clips))
    try:
        app.server.serve_forever(poll_interval=0.5)
    finally:
        app.runner.shutdown()


def _shutdown(app):
    app.runner.shutdown()
    if app.server:
        app.server.shutdown()


App.shutdown = _shutdown

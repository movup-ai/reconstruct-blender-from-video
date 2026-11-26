# -*- coding: utf-8 -*-
"""HTTP server: the web pages, the JSON API and the media files.  One workspace (one source folder, one queue);
there are no users and no login.  Listens on config "host" (127.0.0.1:
this PC only; 0.0.0.0: the other PCs of the network too).

The flow is automatic: Start queues Stage 1 (Movement) for every clip without a valid zip, and each run goes from
the raw video to a checked v2.1 zip with no review step."""
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
from .clips import Library, folders, status, read_json, set_skipped
from .jobs import JobStore, Runner
from . import media
import package as pkg  # pipeline (on sys.path through config)

log = logging.getLogger("agent.server")


class HttpError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


class App:
    """the settings, the job queue and its workers (one NVIDIA card) and the one workspace"""

    def __init__(self, c=None):
        self.c = c or config.load()
        config.ensure_dirs(self.c)
        self.store = JobStore(os.path.join(config.DATA, "jobs.db"))
        self.runner = Runner(self)
        self.ws = Workspace(self)
        self.started = time.time()
        self.server = None

    def settings_changed(self, c):
        """new settings: the workspace follows them (source, work and SUBMIT folders)"""
        self.c = c
        self.ws.refresh()

    def health(self, ws=None):
        from common import blender_exe, human_model_assets
        try:
            from scene_depth import MODEL_PATH                      # the camera's depth model
        except ImportError:
            MODEL_PATH = os.path.join(config.ROOT, "models", "depth_anything_v2_small.onnx")
        be = blender_exe(self.c)
        try:
            free = shutil.disk_usage(self.c["work_root"]).free
        except OSError:
            free = None
        checks = []
        if ws is not None:
            src = ws.c["source_root"]
            checks.append({"name": "Source folder", "ok": bool(src) and os.path.isdir(src),
                           "detail": "%s · %d clips" % (src, len(ws.lib.clips)) if src else "not chosen yet - choose it on the Clips page"})
        checks += [
            {"name": "Blender", "ok": bool(be), "detail": be or "not found - install Blender 5.x or set its path in Settings"},
            {"name": "Depth model", "ok": os.path.exists(MODEL_PATH), "detail": "models\\depth_anything_v2_small.onnx" if os.path.exists(MODEL_PATH) else "missing - run Setup.bat"},
            {"name": "RTMPose joints", "ok": bool(glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "rtmlib", "hub", "checkpoints", "rtmpose*.onnx"))), "detail": "masked joint model; run Setup.bat if missing"},
            {"name": "Disk space", "ok": free is None or free > 20e9, "detail": "%.0f GB free" % (free / 1e9) if free else "?"},
        ]
        checks += [{"name": name, "ok": os.path.isfile(path), "detail": path}
                   for name, path in human_model_assets(self.c)]
        return {"app": APP_NAME, "version": __version__, "uptime_s": round(time.time() - self.started),
                "ok": all(x["ok"] for x in checks if not x.get("optional")), "checks": checks}


class Workspace:
    """the work: the source folder (config source_root), its clips, and the results (output, Work_auto, SUBMIT)"""

    def __init__(self, app):
        self.app = app
        self.store = app.store
        self.runner = app.runner
        self.lib = None
        self.refresh()

    def refresh(self):
        """settings again: the source folder and the result folders"""
        self.c = dict(self.app.c)
        for k in ("work_root", "submit_root"):
            os.makedirs(self.c[k], exist_ok=True)
        if self.lib is None:
            self.lib = Library(self.c)
        else:
            self.lib.c = self.c
            self.lib.rescan()

    def enqueue(self, cl, stage="movement", force=False):
        with self.runner.cond:
            if self.runner.resetting:
                raise HttpError(409, 'Reset is still stopping jobs and clearing files. Wait for it to finish.')
            return self.store.enqueue(cl.id, stage, force=force)

    def clip(self, cid):
        try:
            return self.lib.get(cid)
        except KeyError:
            raise HttpError(404, "unknown clip %s" % cid)

    def status_all(self):
        latest = self.store.latest()
        return [(cl, status(self.c, cl, latest.get(cl.id))) for cl in self.lib.clips]

    def queue_summary(self):
        """the queue: running jobs (with progress), waiting and failed counts, paused or not"""
        counts = self.store.counts()
        run = []
        for j in self.store.running():
            cl = self.lib.by_id.get(j["clip"])
            if cl is None:                                       # a clip of an earlier source folder
                continue
            pr = job_progress(self.c, cl, j)
            run.append({"clip": j["clip"], "name": cl.name, "stage": j["stage"],
                        "since": round(time.time() - (j["started"] or time.time())), "progress": pr.get("pct"), "step": pr.get("step")})
        return {"running": len(run), "queued": counts.get("queued", 0), "failed": counts.get("failed", 0),
                "parallel": self.app.c["parallel"], "now": run, "paused": self.runner.is_paused()}

    def reset(self, cls, dry=False, all_clips=False):
        """Stop writers before clearing outputs/history; Reset all restores the fresh folder list."""
        from . import clips as clipsmod
        if all_clips:
            self.lib.rescan()
            cls = list(self.lib.clips)
        running = {j["clip"] for j in self.store.running()}
        busy = [cl.name for cl in cls if cl.id in running]
        if dry:
            items = [{"clip": cl.name, "paths": [{"path": p, "what": w} for p, w in clipsmod.prepared_paths(self.c, cl)]} for cl in cls]
            return {"ok": True, "dry": True, "busy": busy, "clips": items}
        removed, failed, done = 0, [], []
        try:
            with self.runner.reset_jobs([cl.id for cl in cls], all_clips=all_clips) as stopped:
                for cl in cls:
                    with media._lock:
                        r = clipsmod.reset(self.c, cl)
                    self.store.clear_clip(cl.id)
                    removed += len(r['removed'])
                    failed += [{'clip': cl.name, 'path': p, 'reason': why} for p, why in r['failed']]
                    if not r['failed']:
                        clipsmod.set_skipped(cl.id, False, self.c)   # back to new: skipped clips are included again
                        done.append(cl.name)
                if all_clips:
                    self.store.clear_all()
                    for filename in ('CLAUDE_HANDOFF.json', 'CLAUDE_HANDOFF.md'):    # left by an older version
                        path = os.path.join(self.c['work_root'], filename)
                        if os.path.exists(path):
                            if not clipsmod._inside(path, [self.c['work_root']]):
                                failed.append({'clip': 'folder', 'path': path, 'reason': "outside the app's folders"})
                                continue
                            try:
                                os.remove(path); removed += 1
                            except OSError as error:
                                failed.append({'clip': 'folder', 'path': path, 'reason': str(error)})
                    self.lib.rescan()
        except TimeoutError as error:
            raise HttpError(409, str(error))
        log.info('reset %d clips, %d items deleted, %d jobs cancelled, %d failures', len(done), removed, stopped, len(failed))
        return {'ok': not failed, 'clips': done, 'removed': removed, 'failed': failed, 'busy': [], 'stopped': stopped}

    def clip_info(self, cid):
        cl = self.clip(cid)
        c = self.c
        f = folders(c, cl)
        job = self.store.latest_for(cl.id)
        st = status(c, cl, job)
        idx = self.lib.index_of(cl.id)
        package = pkg.package_name(cl.name, c)
        out = {"clip": cl.to_dict(), "status": st, "index": idx, "count": len(self.lib.clips),
               "prev": self.lib.clips[idx - 1].id if idx > 0 else None,
               "next": self.lib.clips[idx + 1].id if 0 <= idx < len(self.lib.clips) - 1 else None,
               "package": package, "paused": self.runner.is_paused(),
               "timing": self.store.timing().get(cl.id)}
        pose = read_json(f["pose"])
        if pose:
            media.person_crops(c, cl)
            rep = read_json(os.path.join(f["work"], "report.json"), {}) or {}
            fitted = {}
            for p in rep.get("people", []) or []:
                fitted.setdefault(str(p.get("track")), []).append(p)
            out["people"] = []
            for p in pose.get("people", []):
                fs = sorted(int(x) for x in p.get("frames", {}))
                if not fs:
                    continue
                fit = fitted.get(str(p["id"])) or []
                out["people"].append({"id": p["id"], "frames": len(fs), "first": fs[0], "last": fs[-1],
                                      "fit_px": fit[0].get("median_error_px") if fit else None,
                                      "shots": sorted({q.get("shot", 1) for q in fit})})
            out.update(frames=pose.get("frames"), fps=round(pose.get("fps") or 0, 3), size=[pose.get("width"), pose.get("height")])
            cam = read_json(os.path.join(f["work"], "camera.json"))
            if cam:
                out["camera"] = {k: cam.get(k) for k in ("lens_mm", "max_rotation_deg", "parallax_px", "cuts", "warnings", "shots")}
            if rep.get("lens"):
                out.setdefault("camera", {})["lens"] = rep["lens"]
            out["warnings"] = rep.get("warnings", [])
        comparisons = glob.glob(os.path.join(f['submit'], 'comparison', '*.mp4'))
        has_zip = os.path.exists(f["zip"])
        out["outputs"] = {"comparison": bool(comparisons),
                          'comparison_version': max((os.stat(p).st_mtime_ns for p in comparisons), default=None),
                          "zip": has_zip, "zip_path": f["zip"],
                          "download": "/download/%s" % cl.name if has_zip and package else None}
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

    # ------------------------------------------------------------ the flow: folder -> Start -> Movement -> zip
    def use_folder(self, path, title):
        """make <path> the source folder (saved in config.json), and give its videos without a delivery name one when
        a title is given.  Nothing of a new folder starts: the queue pauses and waiting jobs of the previous folder
        are removed; Start runs it."""
        scan = folder_scan(self.c, path, title)
        if title.strip() and scan["unnamed"] and not scan["slug"]:
            raise HttpError(400, "type an English title for this folder (letters and numbers), e.g. the hidden blade")
        if not scan["current"]:                               # another folder: nothing of it runs before Start
            self.runner.pause()
        old = set(self.lib.by_id)
        self.app.settings_changed(config.save({"source_root": scan["path"]}))      # refreshes this workspace
        dropped = sum(self.store.cancel_queued(cid) for cid in old - set(self.lib.by_id))
        new = pkg.assign_names([cl.name for cl in self.lib.clips], scan["title"], self.c) if scan["slug"] else {}
        log.info("source folder %s: %d videos, %d new package names", scan["path"], len(self.lib.clips), len(new))
        return {"ok": True, "clips": len(self.lib.clips), "named": len(new), "dropped": dropped, "clashes": self.lib.clashes}

    def start(self):
        """Start (also Resume): queue Movement for every video without a valid zip - new, failed, cancelled or out
        of date - except skipped ones or those already waiting / running, then let the queue run (jobs put back by
        Pause keep their place)."""
        queued = sum(1 for cl, st in self.status_all()
                     if not st["done"] and not st["skipped"] and not (st["job"] and st["job"]["state"] in ("queued", "running"))
                     and self.enqueue(cl, "movement"))
        self.runner.resume()
        log.info("start: %d video(s) queued", queued)
        return {"ok": True, "queued": queued}

    def run(self, cl):
        """run one clip now (Re-run on its page): Movement again, even when its zip is up to date"""
        jid = self.enqueue(cl, "movement", force=True)
        set_skipped(cl.id, False, self.c)                      # an explicit run includes a skipped clip again
        self.runner.resume()                                   # asked for explicitly: the queue runs
        return jid

    def report_rows(self):
        rows = []
        for cl, st in self.status_all():
            f = folders(self.c, cl)
            rep = read_json(os.path.join(f["work"], "report.json"), {}) or {}
            rows.append({"film": cl.film, "clip": cl.name, "status": st["label"], "people": len(rep.get("people", []) or []),
                         "shots": len(rep.get("shots", []) or [1]) if rep else "",
                         "zip": f["zip"] if st["done"] else "", "warnings": " | ".join(rep.get("warnings", []) or []),
                         "error": (st["job"] or {}).get("error") or ""})
        return rows


# ---------------------------------------------------------------- progress, advice, submissions
# where each step of a Movement run starts, % of the job (read from the pipeline's log lines)
STEPS = [("  marks:", 40, "measuring the lens"), ("  camera ...", 45, "solving the camera"),
         ("  people ...", 55, "fitting 3D people"), ("  set ...", 62, "placing the floor"), ("  floor", 62, "placing the floor"),
         ("  quality:", 66, "checking the result")]


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
    if 'waiting for model inference slot' in txt and 'model inference slot acquired' not in txt.rsplit('waiting for model inference slot', 1)[-1]:
        return {'pct': 1, 'step': 'waiting for the model inference slot; rendering can continue'}
    pct, step = 1, "starting"
    m = re.findall(r"detected (\d+) frames \(frame \d+/(\d+)", txt)      # frames done so far (the frame number jumps:
    if m:                                                                # the detector runs over the clip in passes)
        a, b = int(m[-1][0]), int(m[-1][1])
        pct, step = 2 + 38 * min(a, b) / max(b, 1), "finding people (%d of %d frames)" % (min(a, b), b)
    for key, p, s in STEPS:
        if key in txt and p >= pct:
            pct, step = p, s
    shots = re.findall(r"\[shot (\d+): frames", txt)
    if shots and "  quality:" not in txt:
        step += " (shot %s)" % shots[-1]
    m = re.search(r"blender build \+ render \((\d+) frames\)", txt)
    if m:
        n = int(m.group(1))
        done = len(glob.glob(os.path.join(f["work"], "render", "wm_*.jpg")))
        pct, step = 68 + 24 * min(done, n) / max(n, 1), "rendering frame %d of %d" % (min(done, n), n)
    if "videos + zip" in txt:
        pct, step = 93, "encoding the videos and packing the zip"
    return {"pct": round(min(pct, 99)), "step": step}


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
    lib = Library(c2, preview=True)
    names = pkg.package_names(c)
    zipped = {cl.name: os.path.splitext(os.path.basename(cl.zip))[0].lower() for cl in lib.clips if cl.zip}
    zipped = {k: v for k, v in zipped.items() if pkg.PKG_RE.match(v) and not names.get(k)}   # named by their zip
    title = title or title_guess(names.get(cl.name) for cl in lib.clips)
    plan = pkg.propose_names([cl.name for cl in lib.clips if cl.name not in zipped], title, c=c)
    plan.update({k: (v, False) for k, v in zipped.items()})
    rows = [{"file": cl.rel, "clip": cl.name, "package": plan[cl.name][0], "new_name": plan[cl.name][1],
             "prepared": os.path.exists(folders(c2, cl)["zip"])} for cl in lib.clips]
    return {"path": c2["source_root"], "title": title, "slug": pkg.title_slug(title), "videos": rows,
            "unnamed": sum(1 for r in rows if not r["package"]), "clashes": lib.clashes,
            "current": bool(c["source_root"]) and os.path.normcase(c2["source_root"]) == os.path.normcase(os.path.abspath(c["source_root"]))}


ADVICE = [
    (r"Blender not found", "Install Blender 5.x, or set the Blender path on the Settings page."),
    (r"timed out", "The job took too long. Try again with fewer clips at once (Settings), or check the clip plays correctly."),
    (r"MemoryError|Unable to allocate|out of memory", "The PC ran out of memory. Lower 'Clips at once' on the Settings page and retry."),
    (r"No space left|disk full", "The disk is full. Free some space on the drive with the Work and SUBMIT folders."),
    (r"cannot open video|no frames", "The video file could not be read. Check it plays, or re-export it as H.264 MP4."),
    (r"no delivery name|package name", "Give the video its delivery name: press Name videos… on the Clips page and type the film's English title."),
    (r"package check|v2\.1", "The finished package did not pass the v2.1 check. Re-run the clip; if it fails again, open the technical log."),
    (r"ffmpeg failed", "Making the videos failed. Retry once; if it fails again, open the technical log."),
]


def advice(err, log_txt=""):
    for pat, msg in ADVICE:
        if re.search(pat, err + "\n" + log_txt[-3000:], re.I):
            return msg
    return "Re-run the clip. If it fails again, open the technical log below and send it to support."


def download_name(fn):
    """Content-Disposition for a download saved as <fn> (non-ASCII names in the RFC 5987 form)"""
    from urllib.parse import quote
    ascii_ = fn.encode("ascii", "replace").decode("ascii").replace("?", "_").replace('"', "_")
    return "attachment; filename=\"%s\"; filename*=UTF-8''%s" % (ascii_, quote(fn))


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
        meta = read_json(os.path.join(sub, "meta", "meta.json"))          # v2.1
        if meta:
            meta = {"pattern_video_duration": meta.get("duration"),
                    "source_video_resolution": (meta.get("output_video") or {}).get("resolution")}
        else:                                                               # older packages: meta.jsonl, one line
            try:
                with open(os.path.join(sub, "meta", "meta.jsonl"), encoding="utf-8") as fh:
                    meta = json.loads(fh.readline())
            except (OSError, ValueError):
                meta = {}
        out.append({"name": d, "package": pkg.package_name(d, c), "folder": sub, "zip": zp if os.path.exists(zp) else None,
                    "zip_mb": round(os.path.getsize(zp) / 1e6, 1) if os.path.exists(zp) else None, "folder_mb": round(size / 1e6, 1),
                    "made": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(zp if os.path.exists(zp) else sub))),
                    "duration": meta.get("pattern_video_duration"), "resolution": meta.get("source_video_resolution"),
                    "comparison": bool(glob.glob(os.path.join(sub, "comparison", "*.mp4"))),
                    "mtime": os.path.getmtime(zp if os.path.exists(zp) else sub)})
    out.sort(key=lambda r: -r["mtime"])                       # newest first
    return out


def settings_out(app, ws, local=None):
    """the settings the Settings page shows"""
    out = {k: app.c.get(k) for k in config.EDITABLE if k not in config.SECRETS}
    out["source_root"] = ws.c["source_root"]
    if local is not None:
        out["local"] = local
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

        def file(self, p, ctype=None, filename=None):
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
            if filename:                                       # the name the browser saves it under
                self.send_header("Content-Disposition", download_name(filename))
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
            raw = (getattr(self, "_raw", b"") or b"").decode("utf-8")       # read in do_POST, before any answer
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
            # Read the whole request body before answering: a reply sent while the browser is still sending the body
            # makes Windows reset the connection, so the browser (or a test) sees "connection reset" instead of the
            # answer.  An upload streams its own body; a too-large body closes the connection after the answer.
            self._raw = b""
            if urlparse(self.path).path.rstrip("/") != "/api/upload":
                n = int(self.headers.get("Content-Length") or 0)
                if n > 20 * 1024 * 1024:
                    self.close_connection = True
                    return self.send(413, {"error": "request too large"})
                self._raw = self.rfile.read(n) if n > 0 else b""
            self.handle_any(self.post)

        def drain(self, limit=64 * 1024 * 1024):
            """an upload refused before it was read: read and drop its body, or close the connection when it is huge"""
            n = int(self.headers.get("Content-Length") or 0)
            if n > limit:
                self.close_connection = True
                return
            while n > 0:
                chunk = self.rfile.read(min(1 << 20, n))
                if not chunk:
                    break
                n -= len(chunk)

        # ---- who is asking
        def local(self):
            """the request comes from the PC running the app (only there can it open folders or windows)"""
            ip = self.client_address[0]
            return ip in ("127.0.0.1", "::1") or ip.startswith("::ffff:127.")

        # ---- GET
        def get(self):
            u = urlparse(self.path)
            parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            q = parse_qs(u.query)
            if parts == ["favicon.ico"] or parts == ["favicon.svg"]:
                return self.file(os.path.join(config.WEB, "favicon.svg"), "image/svg+xml")
            if parts[:1] == ["static"] and len(parts) == 2 and re.match(r"^[\w.-]+$", parts[1]):
                return self.file(os.path.join(config.WEB, parts[1]))
            ws = app.ws
            if parts == ["api", "health"]:
                return self.send(200, app.health(ws))
            c = ws.c
            if not parts:
                return self.page("index.html")
            if parts[0] == "clip" and len(parts) == 2:
                ws.clip(parts[1])
                return self.page("clip.html")
            if parts in (["submissions"], ["settings"], ["help"]):
                return self.page(parts[0] + ".html")
            if parts == ["api", "submissions"]:
                tm, ids = app.store.timing(), {cl.name: cl.id for cl in ws.lib.clips}
                return self.send(200, [dict(r, took=(tm.get(ids.get(r["name"])) or {}).get("done")) for r in submissions(c)])
            if parts == ["api", "clips"]:
                if time.time() - ws.lib.scanned > 10:                   # new videos in the folder appear by themselves
                    ws.lib.rescan()
                names, tm = pkg.package_names(c), app.store.timing()
                src = c["source_root"]
                return self.send(200, {"clips": [dict(cl.to_dict(), status=st, package=names.get(cl.name), took=tm.get(cl.id))
                                                 for cl, st in ws.status_all()],
                                       "queue": ws.queue_summary(),
                                       "local": self.local(),
                                       "source": {"path": src, "chosen": bool(src), "exists": bool(src) and os.path.isdir(src),
                                                  "clashes": ws.lib.clashes, "zip_problems": ws.lib.zip_problems}})
            if parts == ["api", "folder"]:                              # what picking a folder would do
                return self.send(200, folder_scan(c, (q.get("path") or [""])[0], (q.get("title") or [""])[0]))
            if parts == ["api", "settings"]:
                out = settings_out(app, ws, self.local())
                from common import has_nvidia
                out["nvidia"] = has_nvidia()
                return self.send(200, out)
            if parts[:2] == ["api", "clip"] and len(parts) == 3:
                return self.send(200, ws.clip_info(parts[2]))
            if parts == ["api", "report"]:
                return self.send(200, ws.report_rows())
            if parts == ["api", "validate"]:
                from . import validate
                return self.send(200, validate.check_all(c["submit_root"]) if os.path.isdir(c["submit_root"]) else [])
            if parts == ["report.csv"]:
                buf = io.StringIO()
                rows = ws.report_rows()
                w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else ["clip"])
                w.writeheader(); w.writerows(rows)
                return self.send(200, "﻿" + buf.getvalue(), "text/csv; charset=utf-8",
                                 {"Content-Disposition": 'attachment; filename="blender_agent_report.csv"'})
            if parts[0] == "download" and len(parts) == 2:             # a package zip from the SUBMIT folder
                name = os.path.basename(parts[1])
                zp = os.path.join(c["submit_root"], name + ".zip")
                if not name or not os.path.isfile(zp):
                    raise HttpError(404, "no zip for %s" % name)
                # v2.1: the delivered zip carries the package name (SUBMIT\<clip>.zip is named after the clip)
                return self.file(zp, "application/zip", (pkg.package_name(name, c) or name) + ".zip")
            if parts[0] == "package" and len(parts) == 3 and parts[2] in ("comparison", "white_model", "target"):
                name = os.path.basename(parts[1])
                m = sorted(glob.glob(os.path.join(c["submit_root"], name, parts[2], "*.mp4")))
                return self.file(m[0] if m else None, "video/mp4")
            if parts[0] == "media" and len(parts) >= 3:
                kind = parts[1]; cl = ws.clip(parts[2].split(".")[0]); f = folders(c, cl)
                if kind == "video":
                    return self.file(cl.video, "video/mp4")
                if kind == "preview":
                    return self.file(f["preview"], "image/jpeg")
                if kind == "clipthumb":
                    return self.file(media.clip_thumb(c, cl), "image/jpeg")
                if kind == "thumb" and len(parts) == 4:
                    return self.file(os.path.join(f["marks"], "thumbs", "P%d.jpg" % int(re.sub(r"\D", "", parts[3]) or 0)), "image/jpeg")
                if kind in ("comparison", "white_model"):
                    m = sorted(glob.glob(os.path.join(f["submit"], kind, "*.mp4")))
                    return self.file(m[0] if m else None, "video/mp4")
            raise HttpError(404, "not found")

        # ---- upload: raw file body, streamed to disk (no size held in memory)
        def upload(self, ws, q):
            try:
                return self._upload(ws, q)
            except HttpError:
                if not getattr(self, "_up_started", False):           # refused before reading: drop the body first
                    self.drain()
                raise

        def _upload(self, ws, q):
            self._up_started = False
            VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
            fname = os.path.basename((q.get("name") or [""])[0]).strip()
            film = (q.get("film") or [""])[0].strip()
            if not ws.c["source_root"] or not os.path.isdir(ws.c["source_root"]):
                raise HttpError(409, "choose your source folder first (Clips page)")
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
            root = os.path.abspath(ws.c["source_root"])
            dest_dir = next((os.path.dirname(cl.video) for cl in ws.lib.clips if cl.film == film), os.path.join(root, film))
            if not os.path.abspath(dest_dir).startswith(root):
                raise HttpError(400, "bad folder")
            os.makedirs(dest_dir, exist_ok=True)
            stem, ext = os.path.splitext(fname)
            dest = os.path.join(dest_dir, fname); k = 2
            while os.path.exists(dest):                               # never overwrite a clip
                dest = os.path.join(dest_dir, "%s (%d)%s" % (stem, k, ext)); k += 1
            tmp = dest + ".uploading"
            left = n
            self._up_started = True
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
            ws.lib.rescan()
            cl = next((x for x in ws.lib.clips if os.path.normcase(x.video) == os.path.normcase(os.path.abspath(dest))), None)
            if cl and not app.runner.is_paused():                   # the queue runs: the new video joins it
                try:
                    ws.enqueue(cl, "movement"); app.runner.wake()
                except HttpError:
                    pass                                             # a Reset is running: Start queues it later
            log.info("uploaded %s (%.1f MB) to %s", os.path.basename(dest), n / 1e6, dest_dir)
            return self.send(200, {"ok": True, "file": os.path.basename(dest), "clip": cl.to_dict() if cl else None})

        # ---- POST
        def post(self):
            u = urlparse(self.path)
            parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            q = parse_qs(u.query)
            ws = app.ws
            if parts == ["api", "upload"]:
                return self.upload(ws, q)
            if parts == ["api", "open"]:                                 # opens Explorer: only on the PC running the app
                if not self.local():
                    raise HttpError(403, "folders open only on the PC running Blender Agent")
                raw = self.body_json().get("path", "")
                p = config.LOGS if raw == "logs" else os.path.abspath(raw)
                roots = [os.path.abspath(ws.c[k]) for k in ("submit_root", "work_root", "source_root") if ws.c[k]] + [config.ROOT]
                if not any(p == r or p.startswith(r + os.sep) for r in roots) or not os.path.exists(p):
                    raise HttpError(400, "that folder is outside the app's folders")
                if os.name == "nt":
                    if os.path.isfile(p):
                        subprocess.Popen(["explorer", "/select,", p])
                    else:
                        os.startfile(p)
                return self.send(200, {"ok": True})
            if parts == ["api", "settings"]:                             # settings (a new source folder goes through use_folder)
                d = self.body_json()
                src = d.pop("source_root", None)
                if src is not None and os.path.normcase(os.path.abspath(src) if src else "") != os.path.normcase(ws.c["source_root"]):
                    ws.use_folder(str(src), "")
                if d:
                    app.settings_changed(config.save(d))
                app.runner.wake()
                return self.send(200, {"ok": True, "settings": settings_out(app, ws)})
            if parts in (["api", "start"], ["api", "prepare_all"]):
                return self.send(200, ws.start())
            if parts == ["api", "pause"]:
                try:
                    return self.send(200, app.runner.interrupt(pause=True))
                except TimeoutError as error:
                    raise HttpError(409, str(error))
            if parts == ['api', 'cancel_session']:
                try:
                    return self.send(200, app.runner.interrupt())
                except TimeoutError as error:
                    raise HttpError(409, str(error))
            if parts == ["api", "resume"]:
                app.runner.resume()
                return self.send(200, {"ok": True})
            if parts == ["api", "pick_folder"]:                          # a folder window on this PC's screen
                if not self.local():
                    raise HttpError(403, "the folder window opens only on the PC running Blender Agent: type the folder's path")
                return self.send(200, {"path": pick_folder(ws.c["source_root"] or os.path.expanduser("~"))})
            if parts == ["api", "use_folder"]:
                d = self.body_json()
                return self.send(200, ws.use_folder(str(d.get("path", "")), str(d.get("title", ""))))
            if parts == ["api", "retry_failed"]:
                n = 0
                for cl, st in ws.status_all():
                    if st["label"] == "failed" and ws.enqueue(cl, "movement"):
                        n += 1
                app.runner.resume()
                return self.send(200, {"ok": True, "queued": n})
            if parts == ["api", "reset_all"]:                            # Start over with every clip on the list
                return self.send(200, ws.reset([], dry=(q.get("dry") or ["0"])[0] == "1", all_clips=True))
            if parts == ["api", "cancel_queued"]:
                return self.send(200, {"ok": True, "cancelled": app.store.cancel_queued()})
            if parts == ["api", "rescan"]:
                before = set(ws.lib.by_id)
                n = ws.lib.rescan()
                new = [cl for cl in ws.lib.clips if cl.id not in before]
                if new and not app.runner.is_paused():                   # the queue runs: new videos join it
                    for cl in new:
                        ws.enqueue(cl, "movement")
                    app.runner.wake()
                return self.send(200, {"ok": True, "clips": n, "new": len(new)})
            if parts == ["api", "quit"]:                                 # stops the app
                if not self.local():
                    raise HttpError(403, "Blender Agent can be stopped only on the PC running it")
                self.send(200, {"ok": True})
                threading.Thread(target=app.shutdown, daemon=True).start()
                return
            if parts[:2] == ["api", "clip"] and len(parts) == 4:
                cl = ws.clip(parts[2]); action = parts[3]
                if action == "run":                                      # Re-run: Movement again for this clip
                    stage = (q.get("stage") or ["movement"])[0]
                    if stage != "movement":
                        raise HttpError(400, "unknown job stage")
                    return self.send(200, {"ok": True, "queued": bool(ws.run(cl))})
                if action == "cancel":
                    try:
                        return self.send(200, app.runner.interrupt(clip=cl.id))
                    except TimeoutError as error:
                        raise HttpError(409, str(error))
                if action == "skip":                                     # ✕ on its card: stays listed, Start leaves it out
                    job = app.store.latest_for(cl.id)
                    if job and job["state"] == "running":
                        raise HttpError(409, "%s is being processed right now: cancel it first, then skip it" % cl.name)
                    n = app.store.cancel_queued(cl.id)
                    set_skipped(cl.id, True, ws.c)                      # the video file and finished results are not touched
                    return self.send(200, {"ok": True, "cancelled": n})
                if action == "include":                                  # Include on a skipped card: back in line
                    set_skipped(cl.id, False, ws.c)
                    st = status(ws.c, cl, app.store.latest_for(cl.id))
                    queued = bool(not st["done"] and not app.runner.is_paused() and ws.enqueue(cl, "movement"))
                    if queued:
                        app.runner.wake()
                    return self.send(200, {"ok": True, "queued": queued})
                if action == "reset":                                    # Start over: delete what processing left
                    return self.send(200, ws.reset([cl], dry=(q.get("dry") or ["0"])[0] == "1"))
            raise HttpError(404, "not found")

    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def serve(app, port, host="127.0.0.1"):
    app.server = Server((host, port), make_handler(app))
    app.runner.start()
    log.info("%s %s on http://%s:%d", APP_NAME, __version__, host, port)
    try:
        app.server.serve_forever(poll_interval=0.5)
    finally:
        app.runner.shutdown()


def _shutdown(app):
    app.runner.shutdown()
    if app.server:
        app.server.shutdown()


App.shutdown = _shutdown

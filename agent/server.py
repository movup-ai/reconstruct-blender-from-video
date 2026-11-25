# -*- coding: utf-8 -*-
"""HTTP server: the web pages, the JSON API and the media files.  Members log in by name (agent/users.py); every
clip page, file and action belongs to the logged-in member's workspace.  Listens on config "host" (127.0.0.1:
this PC only; 0.0.0.0: the other PCs of the network too)."""
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
from .clips import Library, folders, status, read_json, set_held
from .jobs import JobStore, Runner
from .users import Users, COOKIE, token_from_cookie
from . import review, media, ai, brief
import package as pkg  # pipeline (on sys.path through config)
from common import member_cfg, output_dir  # pipeline

log = logging.getLogger("agent.server")


class HttpError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


class App:
    """what every member shares: the settings, the job queue and its workers (one NVIDIA card), the logins"""

    def __init__(self, c=None):
        self.c = c or config.load()
        config.ensure_dirs(self.c)
        self.users = Users(self.c.get("session_days", 30))
        self.spaces = {}
        self._spaces_lock = threading.Lock()
        self.store = JobStore(os.path.join(config.DATA, "jobs.db"))
        self.runner = Runner(self)
        self._ai = {"t": 0, "v": None}
        self.started = time.time()
        self.server = None

    def ai_ready(self):
        if not self.c.get("ai_enabled"):
            return False
        return ai.status(self.c)["ready"]

    def ws(self, user):
        """a member's workspace (their folder, clips and results), or None for an unknown member"""
        with self._spaces_lock:
            w = self.spaces.get(user)
            if w is None:
                if not self.users.get(user):
                    return None
                w = self.spaces[user] = Workspace(self, user)
            return w

    def settings_changed(self, c):
        """new shared settings: every member's folders follow them (work and SUBMIT live under the shared ones)"""
        self.c = c
        with self._spaces_lock:
            spaces = list(self.spaces.values())
        for w in spaces:
            w.refresh()

    def health(self, ws=None):
        from common import blender_exe, human_model_assets
        from scene_depth import MODEL_PATH
        be = blender_exe(self.c)
        try:
            free = shutil.disk_usage(self.c["work_root"]).free
        except OSError:
            free = None
        checks = []
        if ws is not None:
            src = ws.c["source_root"]
            checks.append({"name": "Your source folder", "ok": bool(src) and os.path.isdir(src),
                           "detail": "%s · %d clips" % (src, len(ws.lib.clips)) if src else "not chosen yet - choose it on the Clips page"})
        checks += [
            {"name": "Blender", "ok": bool(be), "detail": be or "not found - install Blender 5.x or set its path in Settings"},
            {"name": "Depth model", "ok": os.path.exists(MODEL_PATH), "detail": "models\\depth_anything_v2_small.onnx" if os.path.exists(MODEL_PATH) else "missing - run Setup.bat"},
            {"name": "Character (%s)" % self.c.get("character_style", "master"), "ok": os.path.exists(os.path.join(config.ROOT, "assets", char_file(self.c))),
             "detail": "assets\\" + char_file(self.c)},
            {"name": "RTMPose joints", "ok": bool(glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "rtmlib", "hub", "checkpoints", "rtmpose*.onnx"))), "detail": "masked joint model; run Setup.bat if missing"},
            {"name": "AI suggestions (optional)", "ok": ai.status(self.c)["ready"], "optional": True, "detail": ai.status(self.c)["detail"]},
            {"name": "Disk space", "ok": free is None or free > 20e9, "detail": "%.0f GB free" % (free / 1e9) if free else "?"},
        ]
        checks += [{"name": name, "ok": os.path.isfile(path), "detail": path}
                   for name, path in human_model_assets(self.c)]
        return {"app": APP_NAME, "version": __version__, "uptime_s": round(time.time() - self.started),
                "ok": all(x["ok"] for x in checks if not x.get("optional")), "checks": checks}


class Workspace:
    """one member's work: their source folder, its clips, and their results (output, Work_auto and SUBMIT
    under _users\\<member>).  A member sees and changes only their own clips."""

    def __init__(self, app, user):
        self.app = app
        self.user = user
        self.store = app.store
        self.runner = app.runner
        self.lib = None
        self.refresh()

    def refresh(self):
        """settings again: the member's folder, the shared ones around it"""
        rec = self.app.users.get(self.user) or {}
        self.c = member_cfg(self.app.c, self.user, rec.get("source_root") or "")
        for k in ("work_root", "submit_root"):
            os.makedirs(self.c[k], exist_ok=True)
        if self.lib is None:
            self.lib = Library(self.c)
        else:
            self.lib.c = self.c
            self.lib.rescan()

    @property
    def name(self):
        return (self.app.users.get(self.user) or {}).get("name", self.user)

    def enqueue(self, cl, stage, force=False):
        with self.runner.cond:
            if self.user in self.runner.resetting:
                raise HttpError(409,'Reset is still stopping jobs and clearing files. Wait for it to finish.')
            return self.store.enqueue(cl.id, stage, force=force, user=self.user)

    def clip(self, cid):
        try:
            return self.lib.get(cid)
        except KeyError:
            raise HttpError(404, "unknown clip %s" % cid)

    def status_all(self):
        latest = self.store.latest()
        return [(cl, status(self.c, cl, latest.get(cl.id))) for cl in self.lib.clips]

    def queue_summary(self):
        """this member's queue; the machine is shared, so other members' jobs are counted (not named)"""
        mine, every = self.store.counts(self.user), self.store.counts()
        run, others = [], 0
        for j in self.store.running():
            cl = self.lib.by_id.get(j["clip"]) if j.get("user") == self.user else None
            if cl is None:
                others += 1
                continue
            pr = job_progress(self.c, cl, j)
            run.append({"clip": j["clip"], "name": cl.name, "stage": j["stage"],
                        "since": round(time.time() - (j["started"] or time.time())), "progress": pr.get("pct"), "step": pr.get("step")})
        return {"running": len(run), "queued": mine.get("queued", 0), "failed": mine.get("failed", 0),
                "others": {"running": others, "queued": every.get("queued", 0) - mine.get("queued", 0)},
                "parallel": self.app.c["parallel"], "finish_parallel": self.app.c.get('finish_parallel',1),
                "autopilot": self.app.c["autopilot"], "now": run,
                "paused": self.runner.is_paused(self.user)}

    def reset(self, cls, dry=False, all_clips=False):
        """Stop writers before clearing outputs/history; Reset all restores the fresh folder list."""
        from . import clips as clipsmod
        if all_clips:
            self.lib.rescan()
            cls=list(self.lib.clips)+list(self.lib.removed_by_id.values())
        running = {j["clip"] for j in self.store.running()}
        busy = [cl.name for cl in cls if cl.id in running]
        if dry:
            items = [{"clip": cl.name, "paths": [{"path": p, "what": w} for p, w in clipsmod.prepared_paths(self.c, cl)]} for cl in cls]
            return {"ok": True, "dry": True, "busy": busy, "clips": items}
        removed, failed, done = 0, [], []
        try:
            with self.runner.reset_jobs(self.user,[cl.id for cl in cls],all_user=all_clips) as stopped:
                restored=set()
                for cl in cls:
                    with media._lock:
                        r = clipsmod.reset(self.c, cl)
                    self.store.clear_clip(cl.id, self.user)
                    removed += len(r['removed'])
                    failed += [{'clip':cl.name,'path':p,'reason':why} for p,why in r['failed']]
                    if not r['failed']:
                        clipsmod.set_held(cl.id,False,self.c)
                        restored.add(os.path.normcase(os.path.abspath(cl.video)))
                        done.append(cl.name)
                if all_clips:
                    self.store.clear_user(self.user)
                    clipsmod.save_removed(clipsmod.removed_set(self.c)-restored,self.c)
                    for filename in ('CLAUDE_HANDOFF.json','CLAUDE_HANDOFF.md'):
                        path=os.path.join(self.c['work_root'],filename)
                        if os.path.exists(path):
                            if not clipsmod._inside(path,[self.c['work_root']]):
                                failed.append({'clip':'folder','path':path,'reason':"outside the app's folders"})
                                continue
                            try:
                                os.remove(path);removed+=1
                            except OSError as error:
                                failed.append({'clip':'folder','path':path,'reason':str(error)})
                    self.lib.rescan()
        except TimeoutError as error:
            raise HttpError(409,str(error))
        log.info('%s: reset %d clips, %d items deleted, %d jobs cancelled, %d failures',self.user,len(done),removed,stopped,len(failed))
        return {'ok':not failed,'clips':done,'removed':removed,'failed':failed,'busy':[], 'stopped':stopped}

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
               "ai_ready": self.app.ai_ready(), "package": pkg.package_name(cl.name, c), "paused": self.runner.is_paused(self.user)}
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
            verification = read_json(os.path.join(f["work"], "scene_verification.json"), {}) or {}
            out["scene_verification"] = {"depth_checked": verification.get("depth_checked", False),
                "issues": [i for i in verification.get("issues", []) if i.get("status") == "unresolved"][:30],
                "repairs": verification.get("repairs", [])}
            sd = read_json(f["scene"])
            if sd:
                out['prop_warnings'] = sd.get('prop_timeline_warnings', [])
                out['geometry_protection'] = sd.get('geometry_protection', [])
                kinds = {}
                for m in sd.get("meshes", []):
                    k = (re.match(r"[a-z]+", m["name"]) or [m["name"]])[0]
                    kinds[k] = kinds.get(k, 0) + 1
                out["set"] = kinds
                out["blocks"] = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(f["work"], "blocks*.jpg")))]
        rv = read_json(f["review"])
        out["review"] = rv
        background_preview = os.path.join(f['work'],'env_check.jpg')
        out['background'] = {'ready':all(os.path.isfile(os.path.join(f['work'],fn)) for fn in
                                      ('scene_data.json','camera.json','clip_info.json')),
                             'preview':os.path.getmtime(background_preview) if os.path.isfile(background_preview) else None,
                             'report':read_json(os.path.join(f['work'],'report_background.json'))}
        out['background']['alignment'] = read_json(os.path.join(f['work'],'object_alignment.json'))
        out['background']['timings'] = {name:read_json(os.path.join(f['work'],name+'_profile.json'))
                                        for name in ('environment','verification')}
        out["suggested"] = review.suggested(c, cl) if pose else None
        out["ai"] = {k: v for k, v in (read_json(f["ai"], {}) or {}).items() if k != "raw"} or None
        out["env_ai"] = (read_json(os.path.join(f["marks"], "env_ai.json"), {}) or {}).get("shots")
        regs = {}
        for rp in sorted(glob.glob(os.path.join(f["work"], "regions*.json"))):
            r = read_json(rp) or {}
            regs[r.get("tag", "")] = {"regions": [{k: x.get(k) for k in ("id", "type", "dropped", "height_m", "distance_m")} for x in r.get("regions", [])],
                                      "added": r.get("added", [])}
        out["regions"] = regs
        comparisons=glob.glob(os.path.join(f['submit'],'comparison','*.mp4'))
        out["outputs"] = {"comparison": bool(comparisons), 'comparison_version':max((os.stat(p).st_mtime_ns for p in comparisons),default=None), "zip": os.path.exists(f["zip"]),
                          "zip_path": f["zip"],
                          'frame_review':os.path.isfile(os.path.join(f['submit'],'model_3d','reconstruction','review','index.html'))}
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
        """make <path> this member's source folder, and give its videos without a delivery name one when a title is
        given.  Nothing of a new folder starts: the member's queue pauses, their waiting jobs of the previous folder
        are removed; they press Start."""
        scan = folder_scan(self.c, path, title)
        if title.strip() and scan["unnamed"] and not scan["slug"]:
            raise HttpError(400, "type an English title for this folder (letters and numbers), e.g. the hidden blade")
        if not scan["current"]:                               # another folder: nothing of it runs before Start
            self.runner.pause(self.user)
        old = set(self.lib.by_id)
        self.app.users.set_source(self.user, scan["path"])
        self.refresh()
        dropped = sum(self.store.cancel_queued(cid) for cid in old - set(self.lib.by_id))
        new = pkg.assign_names([cl.name for cl in self.lib.clips], scan["title"], self.c) if scan["slug"] else {}
        log.info("%s: source folder %s: %d videos, %d new package names", self.user, scan["path"], len(self.lib.clips), len(new))
        return {"ok": True, "clips": len(self.lib.clips), "named": len(new), "dropped": dropped, "clashes": self.lib.clashes}

    def start(self):
        """Start: queue every video of this member that is not prepared yet (except those taken out of the line),
        and let their queue run"""
        queued = sum(1 for cl, st in self.status_all()
                     if not st["fit"] and not st["held"] and not (st["job"] and st["job"]["state"] in ("queued", "running"))
                     and self.enqueue(cl, "fit"))
        self.runner.resume(self.user)
        log.info("%s: start: %d video(s) queued to prepare", self.user, queued)
        return {"ok": True, "queued": queued}

    def finish(self, cl, force=False):
        """queue the final job (fit, set, Blender, videos, v2.1 zip) for a reviewed clip.
        -> "queued" | "unchanged" (same input as the last Finish) | "busy" | an error"""
        st = status(self.c, cl, self.store.latest_for(cl.id))
        if not st["fit"]:
            raise HttpError(409, "prepare the clip first")
        if not pkg.package_name(cl.name, self.c):
            raise HttpError(409, "%s has no delivery name yet: press Name videos… on the Clips page and type the film's English title" % cl.name)
        if not st["confirmed"]:
            raise HttpError(409, "mark the clip reviewed first")
        if st["done"] and not force:
            return "unchanged"
        if st["job"] and st["job"]["state"] in ("queued", "running"):
            return "busy"
        return "queued" if self.enqueue(cl, "all", force=True) else "busy"

    def background(self, cl):
        work = folders(self.c, cl)['work']
        if any(not os.path.isfile(os.path.join(work,fn)) for fn in ('scene_data.json','camera.json','clip_info.json')):
            raise HttpError(409, 'Prepare the clip first; background rendering reuses its saved people and camera.')
        return self.enqueue(cl,'background',force=True)

    def finish_reviewed(self):
        out = {"queued": 0, "unchanged": 0, "busy": 0, "no_name": 0}
        for cl, st in self.status_all():
            if not st["confirmed"] or not st["fit"]:
                continue
            if not pkg.package_name(cl.name, self.c):
                out["no_name"] += 1
                continue
            out[self.finish(cl)] += 1
        if out["queued"]:
            self.runner.resume(self.user)                       # an explicit request: the queue runs
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
    if 'waiting for model inference slot' in txt and 'model inference slot acquired' not in txt.rsplit('waiting for model inference slot',1)[-1]:
        return {'pct':1,'step':'waiting for the model inference slot; rendering can continue'}
    if job['stage']=='background':
        pct,step=5,'loading saved scene'
        for key,p,s in [('annotated object segmentation:',40,'segmenting drawn objects'),
                        ('annotated background ready:',75,'building background prototypes'),
                        ('scene verification:',90,'checking background placement'),
                        ('background ready:',100,'background preview ready')]:
            if key in txt:pct,step=p,s
        return {'pct':pct,'step':step}
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
        pose = read_json(f.get("pose") or os.path.join(f["marks"], "pose.json"))
        if pose:                       # the people of every frame from the CURRENT pose.json: joined tracks (one person =
            try:                       # one id) and filled gaps, as the fit sees them; check.html keeps the pictures
                page = json.loads(data)
                for fr in page["frames"]:
                    k = str(fr["f"])
                    fr["people"] = [{"id": p["id"], "kps": p["frames"][k]} for p in pose["people"] if k in p["frames"]]
                data = json.dumps(page, ensure_ascii=False)
            except (ValueError, KeyError, TypeError):
                pass
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
    lib = Library(c2, preview=True)
    names = pkg.package_names(c)
    zipped = {cl.name: os.path.splitext(os.path.basename(cl.zip))[0].lower() for cl in lib.clips if cl.zip}
    zipped = {k: v for k, v in zipped.items() if pkg.PKG_RE.match(v) and not names.get(k)}   # named by their zip
    title = title or title_guess(names.get(cl.name) for cl in lib.clips)
    plan = pkg.propose_names([cl.name for cl in lib.clips if cl.name not in zipped], title, c=c)
    plan.update({k: (v, False) for k, v in zipped.items()})
    rows = [{"file": cl.rel, "clip": cl.name, "package": plan[cl.name][0], "new_name": plan[cl.name][1],
             "prepared": os.path.exists(folders(c2, cl)["pose"])} for cl in lib.clips]
    return {"path": c2["source_root"], "title": title, "slug": pkg.title_slug(title), "videos": rows,
            "unnamed": sum(1 for r in rows if not r["package"]), "clashes": lib.clashes,
            "current": bool(c["source_root"]) and os.path.normcase(c2["source_root"]) == os.path.normcase(os.path.abspath(c["source_root"]))}


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


PLATE_TAG = re.compile(r"(?:_s(\d+))?_(\d{3})$")


def found_objects(c, cl):
    """the objects the set step found by itself (output/<clip>/env_auto.json) -> [{shot, frame, what, type, bbox}]:
    bbox in the pixels of the background pictures (the video frame without its black bars)"""
    env = read_json(os.path.join(output_dir(c), cl.name, "env_auto.json"), {}) or {}
    per = env.get("shots") if isinstance(env.get("shots"), dict) else {"1": env}
    out = []
    for sid, sh in per.items():
        for o in (sh or {}).get("objects", []):
            s = o.get("_src") or {}
            if s.get("cls") == "drawn" or len(s.get("px") or []) != 4 or o.get("off"):
                continue                                              # boxes drawn on this page are listed as such
            lab = s.get("label")
            out.append({"shot": int(sid), "frame": int(s.get("frame", 0)),
                        "type": "other" if o.get("type") == "mesh" else (o.get("type") or "box"),   # mesh: its measured shape
                        "what": (lab[0] if lab else s.get("cls", "")).replace("_", " "), "bbox": [float(v) for v in s["px"]]})
    return out


def object_shots(c, cl):
    """per background picture: the picture, its size, the objects drawn here (or by DeepSeek), the objects the set step
    found by itself on it ("found") and the found ones the person deleted or replaced ("dropped": [{frame, bbox}])"""
    f = folders(c, cl)
    env = (read_json(os.path.join(f["marks"], "env_ai.json"), {}) or {})
    cam = read_json(os.path.join(f["work"], "camera.json"), {}) or {}
    starts = {int(s["shot"]): int(s["frames"][0]) for s in (cam.get("shots") or []) if s.get("frames")}
    shots = []
    for rp in sorted(glob.glob(os.path.join(f["work"], "regions*.json"))):
        r = read_json(rp) or {}
        tag = r.get("tag", "")
        if not os.path.exists(os.path.join(f["work"], "plate%s.jpg" % tag)):
            continue
        sh = (env.get("shots") or {}).get(tag, {})
        m = PLATE_TAG.match(tag)
        sid = int(m.group(1)) if m and m.group(1) else 1
        frame = starts.get(sid, 0) + (int(m.group(2)) if m else 0)
        shots.append({"tag": tag, "size": r.get("size"), "plate": "/media/shotframe/%s/%s" % (cl.id, tag), "shot": sid, "frame": frame,
                      "scene": sh.get("scene", ""), "objects": [o for o in sh.get("objects", []) if "bbox" in o],
                      "dropped": [d for d in sh.get("dropped", []) if len(d.get("bbox") or []) == 4],
                      "found": [], "placed": sum(1 for o in sh.get("objects", []) if "pos" in o), "built": r.get("added", [])})
    for o in found_objects(c, cl):                     # each found object on the picture of its own frame, else the nearest
        cand = [s for s in shots if s["shot"] == o["shot"]] or shots
        if cand:
            s = min(cand, key=lambda s: abs(s["frame"] - o["frame"]))
            s["found"].append(dict(o, here=s["frame"] == o["frame"]))
    return {"shots": shots, "source": env.get("provider", ""), "edited": bool(env.get("edited"))}


def distance_level(v):
    """an annotation's distance level: 1 (nearest) .. 9 (farthest), or None (auto: as measured)"""
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= 9 else None


CARD_ROLES = ("bg", "fg", "near", "far_i", "lamp", "special")   # pipeline/colour_card.py ROLES: the set's colour meanings


def card_role(o):
    """an annotation's colour-card meaning, picked on the clip page (None: decided from what it is and where)"""
    r = o.get("role")
    if r in CARD_ROLES:
        return r
    return "near" if o.get("interactive") or o.get("colour_role") == "orange" else None   # older notes


def save_objects(c, cl, body):
    import props
    f = folders(c, cl)
    import dependencies
    dependencies.adopt_existing(cl.video, c, f['work'], f['marks'], os.path.join(output_dir(c), cl.name))
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
            size = o.get('size')
            if size is not None:
                from prop_timeline import finite_vector
                size = finite_vector(size, 3, 'Object dimensions')
                if min(size) <= 0 or max(size) > 100: raise ValueError('Object dimensions must be between 0 and 100 metres')
            objs.append({"what": str(o.get("what", ""))[:200], "type": props.norm_type(o.get("type")), "bbox": [round(v, 1) for v in b],
                         "attach": o.get("attach") if o.get("attach") in ("floor", "wall") else None,
                         "facing": o.get("facing") if o.get("facing") in ("camera", "left", "right", "away") else "camera",
                         "size": size,
                         "role": card_role(o),
                         "level": distance_level(o.get("level"))})
            outline = o.get("outline")
            if outline is not None:
                from prop_timeline import finite_vector
                if not isinstance(outline, list) or not 3 <= len(outline) <= 1024:
                    raise ValueError("An object outline must have 3–1024 points")
                points = [finite_vector(point, 2, "Outline point") for point in outline]
                objs[-1]["outline"] = [[round(max(b[i], min(b[i+2], v)), 2) for i, v in enumerate(point)]
                                       for point in points]
        cur = shots.setdefault(tag, {"scene": "", "labels": {}, "drop": []})
        if "dropped" in sh:                                           # found objects deleted or replaced on this page
            cur["dropped"] = [{"frame": int(d["frame"]), "bbox": [round(float(v), 1) for v in d["bbox"]],
                               "why": "changed" if d.get("why") == "changed" else "deleted"}
                              for d in sh["dropped"][:200] if len(d.get("bbox") or []) == 4 and "frame" in d]
        placed = [o for o in cur.get("objects", []) if "pos" in o]      # objects Claude placed in metres stay
        cur["objects"] = placed + objs
        n += len(objs)
    os.makedirs(f["marks"], exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(env, fh, ensure_ascii=False, indent=1)
    return n


# role -> the colour the joint editor draws a person in (the review page's role colours, brighter on video)
DRAW_HEX = {"red": "#ff4d4d", "blue": "#4d8dff", "green": "#3ddc6b", "yellow": "#ffd23f", "purple": "#c07bff", "pink": "#ff7ac8",
            "orange": "#ff9b3d", "cyan": "#4fe0ff", "brown": "#c8925a", "white": "#f5f5f5", "grey": "#9a9a9a"}
ROLE_NAME = {"main": "Main", "second": "Second", "third": "Third", "fourth": "Fourth", "extra": "Other", "ignore": "Not a person"}


def editor_colours(rv, stitched=None):
    """review -> {joint editor person id: {hex, role}}.  The editor numbers people as automark found them; a track the
    fit joined (pose.json "stitched" {old: new}) takes the colour of the track it joined."""
    ppl = (rv or {}).get("people") or {}
    out = {}
    for pid, p in ppl.items():
        role = p.get("role", "extra")
        out[str(pid)] = {"hex": DRAW_HEX["grey"] if role == "ignore" else DRAW_HEX.get(p.get("colour") or "", DRAW_HEX["purple"]),
                         "role": ROLE_NAME.get(role, "")}
    for old, new in (stitched or {}).items():
        if str(new) in out:
            out[str(old)] = out[str(new)]
    return out


def download_name(fn):
    """Content-Disposition for a download saved as <fn> (non-ASCII names in the RFC 5987 form)"""
    from urllib.parse import quote
    ascii_ = fn.encode("ascii", "replace").decode("ascii").replace("?", "_").replace('"', "_")
    return "attachment; filename=\"%s\"; filename*=UTF-8''%s" % (ascii_, quote(fn))


def char_file(c):
    """the character definition the fit and Blender use for config character_style"""
    return {"l3": "l3/character.json", "base21": "base_character_v21.json", "sample": "sample_character.json"}.get(c.get("character_style", "master"), "human_master.json")


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

        def member(self):
            """the logged-in member's workspace, or None"""
            user = app.users.user_for(token_from_cookie(self.headers.get("Cookie")))
            return app.ws(user) if user else None

        def need_member(self, parts):
            """the member's workspace; without a login a page goes to the login page and anything else gets 401"""
            ws = self.member()
            if ws is None:
                if parts and parts[0] in ("api", "media", "download", "package", "report.csv"):
                    raise HttpError(401, "log in first")
                from urllib.parse import quote
                self.send(302, b"", "text/plain", {"Location": "/login?next=" + quote(self.path, safe="")})
                return None
            return ws

        # ---- GET
        def get(self):
            u = urlparse(self.path)
            parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            q = parse_qs(u.query)
            # ---- open to everyone: the login page and what it needs, and the health check of the launcher
            if parts == ["login"]:
                return self.page("login.html")
            if parts == ["favicon.ico"] or parts == ["favicon.svg"]:
                return self.file(os.path.join(config.WEB, "favicon.svg"), "image/svg+xml")
            if parts[:1] == ["static"] and len(parts) == 2 and re.match(r"^[\w.-]+$", parts[1]):
                return self.file(os.path.join(config.WEB, parts[1]))
            if parts == ["api", "members"]:
                return self.send(200, {"members": app.users.names()})
            if parts == ["api", "health"]:
                return self.send(200, app.health(self.member()))
            ws = self.need_member(parts)
            if ws is None:
                return
            c = ws.c
            if not parts:
                return self.page("index.html")
            if parts == ["api", "me"]:
                return self.send(200, {"name": ws.name, "key": ws.user, "local": self.local(), "source": c["source_root"]})
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
                                       "queue": ws.queue_summary(), "ai_ready": app.ai_ready(),
                                       "me": {"name": ws.name, "local": self.local()},
                                       "source": {"path": src, "chosen": bool(src), "exists": bool(src) and os.path.isdir(src),
                                                  "clashes": ws.lib.clashes, "zip_problems": ws.lib.zip_problems},
                                       "removed": ws.lib.removed})
            if parts == ["api", "folder"]:                              # what picking a folder would do
                return self.send(200, folder_scan(c, (q.get("path") or [""])[0], (q.get("title") or [""])[0]))
            if parts == ["api", "settings"]:
                out = {k: app.c.get(k) for k in config.EDITABLE if k not in config.SECRETS}
                out["source_root"] = c["source_root"]                   # the member's own folder
                out["deepseek_key_set"] = bool(ai.api_key(app.c))
                import claude_set  # pipeline
                out["anthropic_key_set"] = bool(claude_set.api_key(app.c))
                from common import has_nvidia
                out["nvidia"] = has_nvidia()
                out["local"] = self.local()
                return self.send(200, out)
            if parts[:2] == ["api", "clip"] and len(parts) == 3:
                return self.send(200, ws.clip_info(parts[2]))
            if parts[:2] == ["api", "clip"] and len(parts) == 4 and parts[3] == "objects":
                import props
                return self.send(200, dict(object_shots(c, ws.clip(parts[2])),
                                           types={k: {"size": v[0], "on": v[1]} for k, v in props.TYPES.items()}))
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
            if parts[0] == "check" and len(parts) == 2:             # marks editor, saving back to the app
                cl = ws.clip(parts[1]); f = folders(c, cl)
                html = check_page(c, cl)
                chk = sorted(glob.glob(os.path.join(f["marks"], "checked*.json")), key=os.path.getmtime)
                inj = "<script>window.APP_SAVE='/api/clip/%s/marks';" % cl.id
                stitched = (read_json(f["pose"], {}) or {}).get("stitched")
                inj += "window.APP_FLAGGED=" + json.dumps(jump_frames(read_json(os.path.join(f["work"], "report.json"), {}) or {}, stitched)) + ";"
                inj += "window.APP_COLOURS=" + json.dumps(editor_colours(read_json(f["review"]) or review.suggested(c, cl), stitched), ensure_ascii=False) + ";"
                inj += "window.APP_CLIP_URL='/clip/%s';" % cl.id
                inj += "window.APP_JOINED=" + json.dumps({str(k): v for k, v in (stitched or {}).items()}) + ";"
                if chk:
                    with open(chk[-1], encoding="utf-8") as fh:
                        inj += "window.APP_CHECKED=" + fh.read() + ";"
                inj += "</script>"
                return self.send(200, html.replace("<head>", "<head>" + inj, 1), "text/html; charset=utf-8")
            if parts[0] == "download" and len(parts) == 2:             # a package zip from the member's SUBMIT folder
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
            if parts == ["media", "clipthumb", parts[-1]] and parts[-1].split(".")[0] in ws.lib.removed_by_id:   # a removed video's card
                return self.file(media.clip_thumb(c, ws.lib.removed_by_id[parts[-1].split(".")[0]]), "image/jpeg")
            if parts[0] == "media" and len(parts) >= 3:
                kind = parts[1]; cl = ws.clip(parts[2].split(".")[0]); f = folders(c, cl)
                if kind == 'review':
                    root=os.path.join(f['submit'],'model_3d','reconstruction','review')
                    if len(parts)==4 and parts[3] in ('index.html','review.json'):
                        return self.file(os.path.join(root,parts[3]))
                    if len(parts)==5 and parts[3] in ('source','model') and re.fullmatch(r'\d{6}\.jpg',parts[4]):
                        return self.file(os.path.join(root,parts[3],parts[4]),'image/jpeg')
                    raise HttpError(404,'not found')
                if kind == "video":
                    return self.file(cl.video, "video/mp4")
                if kind == "plate" and len(parts) == 4 and re.match(r"^_[\w]+$", parts[3]):
                    return self.file(os.path.join(f["work"], "plate%s.jpg" % parts[3]), "image/jpeg")
                if kind == "shotframe" and len(parts) == 4 and re.match(r"^_[\w]+$", parts[3]):   # the original key frame
                    sh = next((s for s in object_shots(c, cl)["shots"] if s["tag"] == parts[3]), None)
                    p = media.shot_frame(c, cl, parts[3], sh["frame"]) if sh else None
                    return self.file(p or os.path.join(f["work"], "plate%s.jpg" % parts[3]), "image/jpeg")
                if kind == "preview":
                    return self.file(f["preview"], "image/jpeg")
                if kind == 'background':
                    return self.file(os.path.join(f['work'],'env_check.jpg'),'image/jpeg')
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
            if cl and app.c["autopilot"] != "off":
                ws.enqueue(cl, "fit"); app.runner.wake()
            log.info("%s uploaded %s (%.1f MB) to %s", ws.user, os.path.basename(dest), n / 1e6, dest_dir)
            return self.send(200, {"ok": True, "file": os.path.basename(dest), "clip": cl.to_dict() if cl else None})

        # ---- POST
        def post(self):
            u = urlparse(self.path)
            parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            q = parse_qs(u.query)
            if parts == ["api", "login"]:                                # a name is enough: it tells members apart
                key, token = app.users.login(self.body_json().get("name", ""))
                age = int(app.c.get("session_days", 30)) * 86400
                log.info("%s logged in from %s", key, self.client_address[0])
                return self.send(200, {"ok": True, "name": app.users.get(key)["name"]},
                                 headers={"Set-Cookie": "%s=%s; Path=/; Max-Age=%d; HttpOnly; SameSite=Lax" % (COOKIE, token, age)})
            if parts == ["api", "logout"]:
                app.users.logout(token_from_cookie(self.headers.get("Cookie")))
                return self.send(200, {"ok": True}, headers={"Set-Cookie": "%s=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax" % COOKIE})
            ws = self.member()
            if ws is None:
                if parts == ["api", "upload"]:
                    self.drain()
                raise HttpError(401, "log in first")
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
            if parts == ["api", "settings"]:                             # shared settings; the source folder is the member's
                d = self.body_json()
                src = d.pop("source_root", None)
                if src is not None and os.path.normcase(os.path.abspath(src) if src else "") != os.path.normcase(ws.c["source_root"]):
                    ws.use_folder(str(src), "")
                if d:
                    app.settings_changed(config.save(d))
                app.runner.wake()
                out = {k: app.c.get(k) for k in config.EDITABLE if k not in config.SECRETS}
                out["source_root"] = ws.c["source_root"]
                import claude_set  # pipeline
                return self.send(200, {"ok": True, "settings": out, "deepseek_key_set": bool(ai.api_key(app.c)),
                                       "anthropic_key_set": bool(claude_set.api_key(app.c))})
            if parts == ["api", "ai_test"]:
                return self.send(200, ai.test_key(app.c))
            if parts == ["api", "claude_test"]:                          # Settings -> Background set -> Test
                import claude_set  # pipeline
                return self.send(200, claude_set.test_key(app.c))
            if parts in (["api", "start"], ["api", "prepare_all"]):
                return self.send(200, ws.start())
            if parts == ["api", "restore"]:                              # put a removed video back on the list
                path = str(self.body_json().get("path", ""))
                if not any(os.path.normcase(r["path"]) == os.path.normcase(path) for r in ws.lib.removed):
                    raise HttpError(400, "that video is not on the removed list")
                ws.lib.restore(path)
                return self.send(200, {"ok": True, "clips": len(ws.lib.clips)})
            if parts == ["api", "pause"]:
                app.runner.pause(ws.user)
                return self.send(200, {"ok": True})
            if parts == ["api", "resume"]:
                app.runner.resume(ws.user)
                return self.send(200, {"ok": True})
            if parts in (["api", "finish_reviewed"], ["api", "render_confirmed"]):
                return self.send(200, dict(ws.finish_reviewed(), ok=True))
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
                    if st["label"] == "failed" and st["job"] and ws.enqueue(cl, st["job"]["stage"]):
                        n += 1
                app.runner.resume(ws.user)
                return self.send(200, {"ok": True, "queued": n})
            if parts == ["api", "reset_all"]:                            # Start over with every clip on the member's list
                return self.send(200, ws.reset([], dry=(q.get("dry") or ["0"])[0] == "1",all_clips=True))
            if parts == ["api", "cancel_queued"]:
                return self.send(200, {"ok": True, "cancelled": app.store.cancel_queued(user=ws.user)})
            if parts == ["api", "rescan"]:
                before = set(ws.lib.by_id)
                n = ws.lib.rescan()
                new = [cl for cl in ws.lib.clips if cl.id not in before]
                if app.c["autopilot"] != "off":
                    for cl in new:
                        ws.enqueue(cl, "fit")
                    app.runner.wake()
                return self.send(200, {"ok": True, "clips": n, "new": len(new)})
            if parts == ["api", "quit"]:                                 # stops the app for every member
                if not self.local():
                    raise HttpError(403, "Blender Agent can be stopped only on the PC running it")
                self.send(200, {"ok": True})
                threading.Thread(target=app.shutdown, daemon=True).start()
                return
            if parts[:2] == ["api", "clip"] and len(parts) == 4:
                cl = ws.clip(parts[2]); f = folders(ws.c, cl); action = parts[3]
                if action == "marks":
                    d = self.body_json()
                    if not isinstance(d.get("frames"), dict):
                        raise HttpError(400, "marks must contain frames")
                    os.makedirs(f["marks"], exist_ok=True)
                    p = os.path.join(f["marks"], "checked.json"); tmp = p + ".tmp"      # whole file or nothing:
                    with open(tmp, "w", encoding="utf-8") as fh:                         # a half-written file would
                        json.dump(d, fh, ensure_ascii=False)                               # lose every joint fix
                    os.replace(tmp, p)
                    return self.send(200, {"ok": True})
                if action == "review":
                    d = review.validate(self.body_json())
                    import dependencies
                    dependencies.adopt_existing(cl.video, ws.c, f['work'], f['marks'], os.path.join(output_dir(ws.c), cl.name))
                    from prop_timeline import validate as validate_timeline
                    pose = read_json(f['pose'], {}) or {}
                    d['prop_timeline'] = validate_timeline(d['prop_timeline'], pose.get('frames'))
                    d.setdefault("clip", cl.name); d.setdefault("video", cl.video)
                    review.write(f["review"], d)
                    if d["notes"].strip():                    # the description goes into meta.jsonl
                        with open(os.path.join(f["marks"], "desc.txt"), "w", encoding="utf-8") as fh:
                            fh.write(d["notes"].strip().splitlines()[0][:300])
                    # confirmed: write the hand-off files for Claude.  Rendering starts by itself only when
                    # "render_on_confirm" is on; otherwise Claude builds the set and renders from the brief.
                    queued = False
                    if d["confirmed"] and app.c.get("render_on_confirm", False):
                        queued = bool(ws.enqueue(cl, "all", force=True))
                    hand = None
                    try:
                        brief.write_clip(ws.c, cl)
                        hand = brief.write_index(ws.c, ws.lib, app.store)[1]
                    except Exception:
                        log.exception("could not write the hand-off files for %s", cl.name)
                    app.runner.wake()
                    return self.send(200, {"ok": True, "queued": queued, "next_review": ws.next_to_review(cl.id),
                                           "handoff": hand and {k: hand[k] for k in ("clips", "confirmed", "all_confirmed")}})
                if action == "run":
                    stage = (q.get("stage") or ["fit"])[0]
                    if stage not in ('fit','background','ai','all'):
                        raise HttpError(400,'unknown job stage')
                    if stage == "ai" and not app.ai_ready():
                        raise HttpError(409, "the local AI is not available (Ollama not running or model missing)")
                    if stage == "all" and not (read_json(f["review"]) or {}).get("confirmed"):
                        raise HttpError(409, "confirm the clip first")
                    jid = ws.background(cl) if stage=='background' else ws.enqueue(cl, stage, force=True)
                    if stage == "fit":
                        set_held(cl.id, False, ws.c)                       # explicit Prepare puts the clip back in line
                    app.runner.resume(ws.user)                           # the person asked for it: their queue runs
                    return self.send(200, {"ok": True, "queued": bool(jid)})
                if action == "objects":
                    n = save_objects(ws.c, cl, self.body_json())
                    jid = ws.background(cl) if (q.get("rebuild") or ["1"])[0] == "1" else None
                    if jid:
                        app.runner.resume(ws.user)
                    return self.send(200, {"ok": True, "objects": n, "queued": bool(jid)})
                if action == "finish":                                   # final job for one reviewed clip
                    r = ws.finish(cl, force=(q.get("force") or ["0"])[0] == "1")
                    if r == "queued":
                        app.runner.resume(ws.user)
                    return self.send(200, {"ok": True, "result": r})
                if action == "remove":                                   # off the list; the file stays in the folder
                    job = app.store.latest_for(cl.id)
                    if job and job["state"] == "running":
                        raise HttpError(409, "%s is being processed right now: remove it when that job is done" % cl.name)
                    app.store.cancel_queued(cl.id)
                    ws.lib.remove(cl)
                    return self.send(200, {"ok": True, "removed": cl.name})
                if action == "cancel":
                    return self.send(200, {"ok": True, "cancelled": app.store.cancel_queued(cl.id)})
                if action == "hold":                                     # out of the waiting line (Take out on its card)
                    job = app.store.latest_for(cl.id)
                    if job and job["state"] == "running":
                        raise HttpError(409, "%s is being processed right now" % cl.name)
                    n = app.store.cancel_queued(cl.id)
                    if not status(ws.c, cl, job)["fit"]:
                        set_held(cl.id, True, ws.c)                      # Start and autopilot leave it until it is added back
                    return self.send(200, {"ok": True, "cancelled": n})
                if action == "unhold":                                   # back in line (Add back on its card)
                    set_held(cl.id, False, ws.c)
                    st = status(ws.c, cl, app.store.latest_for(cl.id))
                    queued = bool(not st["fit"] and not app.runner.is_paused(ws.user) and ws.enqueue(cl, "fit"))
                    if queued:
                        app.runner.wake()
                    return self.send(200, {"ok": True, "queued": queued})
                if action == "reset":                                    # Start over: delete what preparing left
                    return self.send(200, ws.reset([cl], dry=(q.get("dry") or ["0"])[0] == "1"))
            raise HttpError(404, "not found")

    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def serve(app, port, host="127.0.0.1"):
    app.server = Server((host, port), make_handler(app))
    app.runner.start()
    log.info("%s %s on http://%s:%d  (%d members)", APP_NAME, __version__, host, port, len(app.users.names()))
    try:
        app.server.serve_forever(poll_interval=0.5)
    finally:
        app.runner.shutdown()


def _shutdown(app):
    app.runner.shutdown()
    if app.server:
        app.server.shutdown()


App.shutdown = _shutdown

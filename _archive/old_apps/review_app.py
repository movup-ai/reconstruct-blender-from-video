# -*- coding: utf-8 -*-
"""
Key-point review app (local only, http://127.0.0.1:8765).

For every clip: the local models' result (marks, camera, 3D people, grey set) is shown on the clip's frames with an
editable form - people roles / colours / heights, camera type and lens, environment and props, notes - plus the
frame-by-frame marks editor.  "Confirm & continue" saves review.json and queues the Blender build + render + package
for that clip; nothing is rendered until a clip is confirmed.

  .venv\\Scripts\\python review_app.py            (start_review.bat does this and opens the browser)
"""
import os, sys, json, glob, threading, queue, itertools, subprocess, time, shutil, tempfile, mimetypes, re, traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pipeline"))
from common import cfg, jload, jsave
from run_pipeline import clip_name, marks_dir, collect

C = cfg()
PORT = int(os.environ.get("REVIEW_PORT", "8765"))
PY = sys.executable
ROLES = [("main", "主角色", "red"), ("second", "第二角色", "blue"), ("third", "第三角色", "green"),
         ("fourth", "第四角色", "yellow"), ("extra", "其他人物", "purple"), ("ignore", "不建模 / ignore", None)]
COLOURS = ["red", "blue", "green", "yellow", "purple", "pink", "orange", "cyan"]
PROP_KINDS = [("orange", "周围可交互物体 orange"), ("pink", "远处可交互物体 pink"), ("purple", "其他特殊目标 purple"),
              ("brown", "前景 brown"), ("white", "光源 white"), ("grey", "背景 grey")]

# ----------------------------------------------------------------------------- clips
def clips():
    vids = collect([C["source_root"]])
    return [{"id": i, "video": v, "name": clip_name(v, C), "film": os.path.relpath(v, C["source_root"]).split(os.sep)[0]}
            for i, v in enumerate(vids)]
CLIPS = clips()

def paths(cid):
    cl = CLIPS[cid]; md = marks_dir(cl["video"], C); work = os.path.join(C["work_root"], cl["name"])
    return cl, md, work

def status(cid):
    cl, md, work = paths(cid)
    rv = os.path.join(md, "review.json")
    s = {"marks": os.path.exists(os.path.join(md, "pose.json")), "fit": os.path.exists(os.path.join(work, "preview.jpg")),
         "reviewed": os.path.exists(rv), "confirmed": False,
         "done": os.path.exists(os.path.join(C["submit_root"], cl["name"] + ".zip")),
         "job": JOBS.get(cid, {}).get("state")}
    if s["reviewed"]:
        try: s["confirmed"] = bool(jload(rv).get("confirmed"))
        except Exception: pass
    busy = s["job"] in ("queued", "running")
    s["label"] = (("rendering" if s["job"] == "running" else "waiting to render") if busy and JOBS[cid]["stage"] == "all" else
                  ("preparing" if s["job"] == "running" else "waiting to prepare") if busy else
                  "failed" if s["job"] == "failed" else
                  "done" if s["done"] else "confirmed" if s["confirmed"] else "ready for review" if s["fit"] else "not started")
    return s

# ----------------------------------------------------------------------------- jobs: a pool, several clips at once
SET_P = os.path.join(HERE, "review_settings.json")
SETTINGS = {"parallel": 3}
if os.path.exists(SET_P):
    try: SETTINGS.update(jload(SET_P))
    except Exception: pass
MAX_WORKERS = 8
JOBS = {}; JQ = queue.PriorityQueue(); SEQ = itertools.count(); RUNNING = [0]; LOCK = threading.Condition()
def enqueue(cid, stage):
    if JOBS.get(cid, {}).get("state") in ("queued", "running"):
        return False
    JOBS[cid] = {"state": "queued", "stage": stage, "t": time.time()}
    JQ.put((0 if stage == "all" else 1, next(SEQ), cid, stage))        # confirmed renders go first
    return True

def queue_summary():
    st = [j["state"] for j in JOBS.values()]
    return {"running": st.count("running"), "queued": st.count("queued"), "parallel": SETTINGS["parallel"],
            "running_clips": [CLIPS[c]["name"] + " (" + ("render" if j["stage"] == "all" else "local models") + ")"
                              for c, j in JOBS.items() if j["state"] == "running"]}

def worker():
    while True:
        _, _, cid, stage = JQ.get()
        with LOCK:
            while RUNNING[0] >= SETTINGS["parallel"]:
                LOCK.wait()
            RUNNING[0] += 1
        try:
            run_job(cid, stage)
        except Exception as e:
            JOBS[cid].update(state="failed", error=str(e))
        finally:
            with LOCK:
                RUNNING[0] -= 1; LOCK.notify_all()

def run_job(cid, stage):
    if True:
        cl, md, work = paths(cid)
        os.makedirs(work, exist_ok=True)
        logp = os.path.join(work, "job_log.txt")
        JOBS[cid].update(state="running", started=time.time(), log=logp)
        cmd = [PY, os.path.join(HERE, "pipeline", "run_pipeline.py"), cl["video"], "--stage", stage, "--device", "cpu"]
        if stage in ("all", "render"):
            cmd += ["--force", "--require-confirm"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        with open(logp, "w", encoding="utf-8", errors="ignore") as lf:
            lf.write("$ " + " ".join(cmd) + "\n"); lf.flush()
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        txt = open(logp, encoding="utf-8", errors="ignore").read()
        ok = r.returncode == 0 and "Traceback" not in txt and " 0 failed" in txt
        JOBS[cid].update(state="done" if ok else "failed", ended=time.time())
for _ in range(MAX_WORKERS):
    threading.Thread(target=worker, daemon=True).start()

# ----------------------------------------------------------------------------- media helpers
def open_video(path):
    cap = cv2.VideoCapture(path)
    if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
        return cap, None
    d = tempfile.mkdtemp(prefix="review_"); t = os.path.join(d, "v.mp4"); shutil.copyfile(path, t)
    return cv2.VideoCapture(t), d

def make_thumbs(cid):
    """one crop per tracked person (middle of its track), cached in <marks>/thumbs/"""
    cl, md, work = paths(cid)
    td = os.path.join(md, "thumbs"); pose = jload(os.path.join(md, "pose.json"))
    want = [p for p in pose["people"] if not os.path.exists(os.path.join(td, "P%d.jpg" % p["id"]))]
    if not want:
        return
    os.makedirs(td, exist_ok=True)
    cap, tmp = open_video(cl["video"])
    for p in want:
        fs = sorted(int(f) for f in p["boxes"])
        best = max(fs[len(fs) // 4: 3 * len(fs) // 4 + 1] or fs, key=lambda f: p["boxes"][str(f)][3] - p["boxes"][str(f)][1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, best); ok, img = cap.read()
        if not ok:
            continue
        x0, y0, x1, y1 = p["boxes"][str(best)][:4]
        w, h = x1 - x0, y1 - y0; m = 0.15
        x0, y0 = int(max(0, x0 - w * m)), int(max(0, y0 - h * m)); x1, y1 = int(min(img.shape[1], x1 + w * m)), int(min(img.shape[0], y1 + h * m))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        s = 220.0 / max(crop.shape[:2]); crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        with open(os.path.join(td, "P%d.jpg" % p["id"]), "wb") as fh:
            fh.write(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
    cap.release()
    if tmp: shutil.rmtree(tmp, ignore_errors=True)

def clip_thumb(cid):
    """small first-frame picture of a clip for the start page (cached)"""
    cl, md, work = paths(cid)
    p = os.path.join(C["work_root"], "_thumbs", "%d.jpg" % cid)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        cap, tmp = open_video(cl["video"]); n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.3)); ok, img = cap.read(); cap.release()
        if tmp: shutil.rmtree(tmp, ignore_errors=True)
        if not ok: return None
        img = cv2.resize(img, (240, int(img.shape[0] * 240 / img.shape[1])))
        with open(p, "wb") as fh: fh.write(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    return p

def clip_info(cid):
    cl, md, work = paths(cid)
    out = {"id": cid, "queue": queue_summary(), "next_review": next_to_review(cid), "name": cl["name"], "film": cl["film"], "video": cl["video"], "status": status(cid),
           "n": len(CLIPS), "roles": ROLES, "colours": COLOURS, "prop_kinds": PROP_KINDS,
           "default_height": C["person_height"], "default_lens": C["lens_mm"]}
    pp = os.path.join(md, "pose.json")
    if os.path.exists(pp):
        pose = jload(pp)
        make_thumbs(cid)
        rep = jload(os.path.join(work, "report.json")) if os.path.exists(os.path.join(work, "report.json")) else {}
        fitted = {}
        for p in rep.get("people", []):
            fitted.setdefault(str(p["track"]), []).append(p)
        ppl = []
        for p in pose["people"]:
            fs = sorted(int(f) for f in p["frames"])
            hs = [p["boxes"][str(f)][3] - p["boxes"][str(f)][1] for f in fs]
            ppl.append({"id": p["id"], "frames": len(fs), "first": fs[0], "last": fs[-1], "height_px": round(float(np.median(hs))),
                        "fit": (fitted.get(str(p["id"])) or [None])[0],
                        "shots": sorted({q.get("shot", 1) for q in fitted.get(str(p["id"]), [])})})
        out.update(frames=pose["frames"], fps=round(pose["fps"], 3), size=[pose["width"], pose["height"]], people=ppl)
        out["camera"] = jload(os.path.join(work, "camera.json")) if os.path.exists(os.path.join(work, "camera.json")) else None
        if out["camera"]:
            out["camera"] = {k: out["camera"][k] for k in ("max_rotation_deg", "parallax_px", "cuts", "warnings") if k in out["camera"]}
        out["report"] = {k: rep.get(k) for k in ("ground", "warnings", "people", "camera_rotation_deg")} if rep else None
        sd = os.path.join(work, "scene_data.json")
        if os.path.exists(sd):
            d = jload(sd)
            kinds = {}
            for m in d.get("meshes", []):
                k = re.match(r"[a-z]+", m["name"]).group(0) if re.match(r"[a-z]+", m["name"]) else m["name"]
                kinds[k] = kinds.get(k, 0) + 1
            out["set"] = [{"name": "%s" % k, "colour": "", "faces": n} for k, n in kinds.items()]
            out["shots"] = d.get("shots")
            out["blocks"] = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(work, "blocks*.jpg")))]
    rv = os.path.join(md, "review.json")
    out["review"] = jload(rv) if os.path.exists(rv) else None
    sub = os.path.join(C["submit_root"], cl["name"])
    out["outputs"] = {k: os.path.exists(p) for k, p in (("comparison", os.path.join(sub, "comparison")), ("zip", sub + ".zip"))}
    j = JOBS.get(cid)
    if j:
        log = ""
        if j.get("log") and os.path.exists(j["log"]):
            log = open(j["log"], encoding="utf-8", errors="ignore").read()[-6000:]
        out["job"] = {"state": j["state"], "stage": j["stage"], "log": log}
    return out

def next_to_review(after):
    for k in list(range(after + 1, len(CLIPS))) + list(range(0, after)):
        st = status(k)
        if st["label"] in ("ready for review",):
            return k
    return None

# ----------------------------------------------------------------------------- http
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def file(self, p, ctype=None):
        if not p or not os.path.exists(p):
            return self.send(404, "not found", "text/plain")
        ctype = ctype or mimetypes.guess_type(p)[0] or "application/octet-stream"
        size = os.path.getsize(p); rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-"); a = int(a or 0); b = int(b) if b else size - 1; b = min(b, size - 1)
            self.send_response(206); self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, size))
        else:
            a, b = 0, size - 1; self.send_response(200)
        self.send_header("Content-Type", ctype); self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1)); self.send_header("Cache-Control", "no-store"); self.end_headers()
        with open(p, "rb") as fh:
            fh.seek(a); left = b - a + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk: break
                try: self.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError): return
                left -= len(chunk)

    def body(self):
        n = int(self.headers.get("Content-Length", 0)); return self.rfile.read(n).decode("utf-8") if n else ""

    def do_GET(self):
        try:
            u = urlparse(self.path); parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            if not parts:
                return self.send(200, open(os.path.join(HERE, "review_index.html"), encoding="utf-8").read(), "text/html; charset=utf-8")
            if parts[0] == "clip" and len(parts) == 2:
                return self.send(200, open(os.path.join(HERE, "review_clip.html"), encoding="utf-8").read(), "text/html; charset=utf-8")
            if parts[:2] == ["api", "clips"]:
                return self.send(200, {"clips": [dict(c, video=None, status=status(c["id"])) for c in CLIPS], "queue": queue_summary()})
            cid = int(parts[-1].split(".")[0]) if parts[-1].split(".")[0].isdigit() else None
            if parts[:2] == ["api", "clip"]:
                return self.send(200, clip_info(cid))
            if parts[0] == "check":                      # marks editor, saving straight back to the app
                cl, md, work = paths(cid)
                p = os.path.join(md, "check.html")
                if not os.path.exists(p): return self.send(404, "no marks yet", "text/plain")
                html = open(p, encoding="utf-8").read()
                chk = sorted(glob.glob(os.path.join(md, "checked*.json")), key=os.path.getmtime)
                inj = "<script>window.APP_SAVE='/api/marks/%d';%s</script>" % (cid, ("window.APP_CHECKED=" + open(chk[-1], encoding="utf-8").read() + ";") if chk else "")
                return self.send(200, html.replace("<head>", "<head>" + inj, 1), "text/html; charset=utf-8")
            if parts[0] == "media":
                cl, md, work = paths(int(parts[2].split(".")[0]))
                kind = parts[1]
                if kind == "video": return self.file(cl["video"], "video/mp4")
                if kind == "clipthumb": return self.file(clip_thumb(int(parts[2].split(".")[0])), "image/jpeg")
                if kind == "blocks": return self.file(os.path.join(work, os.path.basename(parts[3])), "image/jpeg")
                if kind == "preview": return self.file(os.path.join(work, "preview.jpg"), "image/jpeg")
                if kind == "thumb": return self.file(os.path.join(md, "thumbs", "P%s.jpg" % parts[3].split(".")[0]), "image/jpeg")
                if kind in ("comparison", "white_model"):
                    sub = os.path.join(C["submit_root"], cl["name"], kind)
                    m = sorted(glob.glob(os.path.join(sub, "*.mp4")))
                    return self.file(m[0] if m else None, "video/mp4")
            return self.send(404, "not found", "text/plain")
        except Exception as e:
            traceback.print_exc(); return self.send(500, {"error": str(e)})

    def do_POST(self):
        try:
            u = urlparse(self.path); parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            if parts[:2] == ["api", "settings"]:
                d = json.loads(self.body() or "{}")
                if "parallel" in d:
                    SETTINGS["parallel"] = int(min(MAX_WORKERS, max(1, int(d["parallel"]))))
                    jsave(SET_P, SETTINGS)
                    with LOCK: LOCK.notify_all()
                return self.send(200, {"ok": True, "settings": SETTINGS})
            if parts[:2] == ["api", "prepare_all"]:
                n = sum(enqueue(c["id"], "fit") for c in CLIPS if status(c["id"])["label"] in ("not started", "failed") and not status(c["id"])["fit"])
                return self.send(200, {"ok": True, "queued": n})
            if parts[:2] == ["api", "render_confirmed"]:
                n = sum(enqueue(c["id"], "all") for c in CLIPS if status(c["id"])["confirmed"] and not status(c["id"])["done"])
                return self.send(200, {"ok": True, "queued": n})
            cid = int(parts[-1]); cl, md, work = paths(cid)
            if parts[:2] == ["api", "marks"]:
                d = json.loads(self.body()); os.makedirs(md, exist_ok=True)
                jsave(os.path.join(md, "checked.json"), d); return self.send(200, {"ok": True})
            if parts[:2] == ["api", "review"]:
                d = json.loads(self.body()); d["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
                os.makedirs(md, exist_ok=True); jsave(os.path.join(md, "review.json"), d)
                queued = enqueue(cid, "all") if d.get("confirmed") else False
                return self.send(200, {"ok": True, "queued": queued})
            if parts[:2] == ["api", "run"]:
                stage = parse_qs(u.query).get("stage", ["fit"])[0]
                if stage not in ("fit", "all"): return self.send(400, {"error": "bad stage"})
                return self.send(200, {"ok": True, "queued": enqueue(cid, stage)})
            return self.send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc(); return self.send(500, {"error": str(e)})

if __name__ == "__main__":
    print("%d clips.  Review app: http://127.0.0.1:%d" % (len(CLIPS), PORT), flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

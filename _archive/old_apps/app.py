# -*- coding: utf-8 -*-
"""
白模 Studio - the white-model app.  Double-click start_app.bat; it opens in the browser at http://127.0.0.1:8765
Folder of clips -> scan keypoints -> check / fix them -> build submission zips -> AI edits by typed command.
Runs only on this PC (127.0.0.1); nothing is shared on the network.
"""
import os, sys, re, json, glob, time, threading, subprocess, shutil, collections, base64, mimetypes, urllib.parse, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pipeline"))
from common import cfg, CFG_PATH, jload, jsave, blender_exe
from run_pipeline import clip_name, VIDEO_EXT
import edit_ops

PORT = 8765
OUT = os.path.join(HERE, "output")
CACHE = os.path.join(HERE, ".cache")
SETTING_KEYS = ["source_root", "submit_root", "work_root", "blender_exe", "ai_provider", "anthropic_key", "deepseek_key",
                "anthropic_model", "deepseek_model", "ollama_model", "ollama_url", "character_style", "person_height", "lens_mm"]
SECRET = ("anthropic_key", "deepseek_key")
APP_DEFAULTS = {"ai_provider": "local", "ollama_model": "qwen3-vl:8b", "ollama_url": "http://127.0.0.1:11434",
                "anthropic_model": "claude-sonnet-5-5", "deepseek_model": "deepseek-flash"}

def ollama_exe():
    p = shutil.which("ollama")
    if p: return p
    for cand in (os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
                 r"C:\Program Files\Ollama\ollama.exe"):
        if os.path.exists(cand): return cand
    return None
LOCK = threading.Lock()

def conf():
    c = dict(APP_DEFAULTS); c.update(cfg()); return c

def save_conf(upd):
    c = jload(CFG_PATH) if os.path.exists(CFG_PATH) else {}
    c.update(upd)
    with open(CFG_PATH, "w", encoding="utf-8") as fh:
        json.dump(c, fh, ensure_ascii=False, indent=2)

def detect_device():
    try:
        import onnxruntime as o
        return "cuda" if "CUDAExecutionProvider" in o.get_available_providers() else "cpu"
    except Exception:
        return "cpu"
DEVICE = detect_device()

# ------------------------------------------------------------------ clips
CLIPS = {}            # id -> video path

def scan_folder(root, skip=()):
    vids = []
    skip = [os.path.abspath(x).lower() for x in skip if x]
    if root and os.path.isdir(root):
        for r, ds, fs in os.walk(root):
            ds[:] = sorted(x for x in ds if os.path.abspath(os.path.join(r, x)).lower() not in skip and not x.startswith(".")
                           and not os.path.exists(os.path.join(r, x, "meta", "meta.jsonl")))      # finished submission folders
            vids += [os.path.join(r, f) for f in sorted(fs) if f.lower().endswith(VIDEO_EXT)]
    out = collections.OrderedDict(); seen = collections.Counter()
    for v in vids:
        n = clip_name(v); seen[n] += 1
        out[n if seen[n] == 1 else "%s_%d" % (n, seen[n])] = v
    return out

def marks_index():
    idx = {}
    for p in glob.glob(os.path.join(OUT, "**", "pose.json"), recursive=True):
        idx.setdefault(os.path.basename(os.path.dirname(p)), os.path.dirname(p))
    return idx

_CACHE = {}
def cached(path, fn):
    k = (path, os.path.getmtime(path))
    if k not in _CACHE:
        _CACHE[k] = fn(path)
    return _CACHE[k]

def clip_status(cid, video, c, midx):
    stem = os.path.splitext(os.path.basename(video))[0]
    d = {"id": cid, "video": video, "file": os.path.basename(video),
         "film": os.path.basename(os.path.dirname(video)), "marks": False}
    md = midx.get(stem)
    if md:
        try:
            d.update(marks=True, **cached(os.path.join(md, "pose.json"), lambda p: (lambda j: {
                "frames": j["frames"], "fps": round(j["fps"], 2), "seconds": round(j["frames"] / j["fps"], 2),
                "people": len(j["people"]), "review": len(j.get("review_frames", []))})(jload(p))))
            ck = os.path.join(md, "checked.json")
            if os.path.exists(ck):
                d["checked"] = cached(ck, lambda p: len(jload(p).get("checked_frames", [])))
        except Exception as e:
            d["marks_error"] = str(e)
    w = os.path.join(c["work_root"], cid); sub = os.path.join(c["submit_root"], cid)
    d["paths"] = {"work": w, "sub": sub, "zip": os.path.join(c["submit_root"], cid + ".zip"), "preview": os.path.join(w, "preview.jpg"),
                  "wm": os.path.join(sub, "white_model", cid + ".mp4"), "cmp": os.path.join(sub, "comparison", cid + ".mp4"),
                  "blend": os.path.join(sub, "model_3d", cid + ".blend")}
    sd = os.path.join(w, "scene_data.json"); zp = os.path.join(c["submit_root"], cid + ".zip")
    d["built"] = os.path.exists(sd); d["preview"] = os.path.exists(os.path.join(w, "preview.jpg"))
    d["zip"] = os.path.exists(zp)
    d["edited"] = d["built"] and d["zip"] and os.path.getmtime(sd) > os.path.getmtime(zp) + 1
    rp = os.path.join(w, "report.json")
    if os.path.exists(rp):
        try: d["warnings"] = cached(rp, lambda p: jload(p).get("warnings", []))
        except Exception: pass
    return d

# ------------------------------------------------------------------ background jobs (one at a time)
class Job:
    def __init__(self, title, cmds, total, kind):
        self.title, self.cmds, self.total, self.kind = title, cmds, total, kind
        self.log = collections.deque(maxlen=400); self.done_n = 0; self.current = ""
        self.running = True; self.ok = None; self.proc = None; self.cancelled = False; self.started = time.time()
        self.fails = 0
        threading.Thread(target=self.run, daemon=True).start()
    def run(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = 0x08000000 if os.name == "nt" else 0
        try:
            for cmd in self.cmds:
                if self.cancelled: break
                self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                             creationflags=flags, text=True, encoding="utf-8", errors="replace")
                for line in self.proc.stdout:
                    line = line.rstrip()
                    if not line: continue
                    self.log.append(line)
                    m = re.match(r"=== (\S+)", line)
                    if m:
                        if self.current: self.done_n += 1
                        self.current = m.group(1)
                    m = re.match(r"(\S.*): \d+ frames \d+x\d+", line)          # automark: a new clip
                    if m and self.kind == "scan":
                        if self.current: self.done_n += 1
                        self.current = m.group(1)
                    if "Traceback" in line or "FAILED" in line: self.fails += 1
                self.proc.wait()
            self.ok = (not self.cancelled) and self.fails == 0
        except Exception as e:
            self.log.append("ERROR: %s" % e); self.ok = False
        self.done_n = self.total if self.ok else self.done_n
        self.running = False
    def cancel(self):
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
            else:
                self.proc.kill()
    def state(self):
        return {"title": self.title, "kind": self.kind, "running": self.running, "ok": self.ok, "done": self.done_n,
                "total": self.total, "current": self.current, "log": list(self.log)[-60:], "cancelled": self.cancelled,
                "seconds": round(time.time() - self.started)}
JOB = None
QUEUE = []            # clips confirmed by the user, built one after another in the background

def queue_worker():
    while True:
        time.sleep(2)
        try:
            with LOCK:
                busy = JOB is not None and JOB.running
            if busy or not QUEUE:
                continue
            cid = QUEUE[0]
            if cid not in CLIPS:
                QUEUE.pop(0); continue
            cmd = [sys.executable, os.path.join(HERE, "pipeline", "run_pipeline.py"), CLIPS[cid], "--device", DEVICE, "--stage", "all", "--force"]
            start_job("Building %s (confirmed)%s" % (cid, " · %d more waiting" % (len(QUEUE) - 1) if len(QUEUE) > 1 else ""), [cmd], 1, "build")
            QUEUE.pop(0)
        except Exception as e:
            print("queue:", e)

def start_job(title, cmds, total, kind):
    global JOB
    with LOCK:
        if JOB and JOB.running:
            raise RuntimeError("Another job is running - wait for it or press Cancel")
        JOB = Job(title, cmds, total, kind)

# ------------------------------------------------------------------ frames, stills
def picture_area(cid):
    c = conf(); ci = os.path.join(c["work_root"], cid, "clip_info.json")
    if os.path.exists(ci):
        d = jload(ci); return d["y0"], d["Hpic"]
    md = marks_index().get(os.path.splitext(os.path.basename(CLIPS[cid]))[0])
    if md:
        pa = jload(os.path.join(md, "pose.json"))["picture_area"]; return pa[1], pa[3]
    return 0, None

def video_frame(cid, f):
    import cv2
    v = CLIPS[cid]
    cap = cv2.VideoCapture(v)
    if not cap.isOpened():
        os.makedirs(CACHE, exist_ok=True)
        t = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9_]", "_", cid) + os.path.splitext(v)[1])
        if not os.path.exists(t): shutil.copyfile(v, t)
        cap = cv2.VideoCapture(t)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read(); cap.release()
    if not ok: return None
    y0, h = picture_area(cid)
    return img[y0:y0 + h] if h else img

def jpg_bytes(img, width=None, q=85):
    import cv2
    if width and img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()

def wm_frame_path(cid, f):
    w = os.path.join(conf()["work_root"], cid)
    st = os.path.join(w, "still.json")
    if os.path.exists(st):
        s = jload(st)
        if s.get("frame") == f and os.path.exists(os.path.join(w, "still.jpg")) and s.get("mtime", 0) >= os.path.getmtime(os.path.join(w, "scene_data.json")) - 1:
            return os.path.join(w, "still.jpg")
    p = os.path.join(w, "render", "wm_%04d.jpg" % (f + 1))
    if os.path.exists(p) and not clip_status(cid, CLIPS[cid], conf(), {}).get("edited"):
        return p
    return None

def render_still(cid, f):
    c = conf(); w = os.path.join(c["work_root"], cid); dp = os.path.join(w, "scene_data.json")
    be = blender_exe(c)
    if not be: raise RuntimeError("Blender not found - set its path in Settings")
    if not os.path.exists(dp): raise RuntimeError("Build this clip first")
    data = jload(dp); data["work_dir"] = w; data["submit_dir"] = os.path.join(c["submit_root"], cid); jsave(dp, data)
    flags = 0x08000000 if os.name == "nt" else 0
    p = subprocess.run([be, "-b", "-P", os.path.join(HERE, "pipeline", "blender_build.py"), "--", dp, "still", str(f + 1)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, creationflags=flags)
    if not os.path.exists(os.path.join(w, "still.jpg")) or p.returncode != 0:
        raise RuntimeError("Blender still failed:\n" + (p.stdout or "")[-1500:])
    jsave(os.path.join(w, "still.json"), {"frame": f, "mtime": time.time()})
    return os.path.join(w, "still.jpg")

def data_url(path_or_bytes, width=1280):
    import cv2, numpy as np
    if isinstance(path_or_bytes, str):
        img = cv2.imdecode(np.fromfile(path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = path_or_bytes
    if img is None: return None
    return "data:image/jpeg;base64," + base64.b64encode(jpg_bytes(img, width, 82)).decode()

# ------------------------------------------------------------------ check page
def check_page(cid):
    md = marks_index().get(os.path.splitext(os.path.basename(CLIPS[cid]))[0])
    if not md: return None
    src = open(os.path.join(md, "check.html"), encoding="utf-8").read()
    a = src.index("const PAGE = ") + len("const PAGE = "); b = src.index("\nconst cv", a)
    page = src[a:b].rstrip().rstrip(";")
    tpl = open(os.path.join(HERE, "check_template.html"), encoding="utf-8").read()
    ck = os.path.join(md, "checked.json")
    inj = "<script>window.APP_SAVE=%s;window.APP_CHECKED=%s;</script>" % (
        json.dumps("/api/checked/" + urllib.parse.quote(cid)), open(ck, encoding="utf-8").read() if os.path.exists(ck) else "null")
    return tpl.replace("</head>", inj + "\n</head>", 1).replace("__PAGE_DATA__", page)

# ------------------------------------------------------------------ folder picker (runs on this PC)
def browse(kind, title, start):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    try:
        if kind == "file":
            p = filedialog.askopenfilename(title=title, initialdir=start or None, filetypes=[("Blender", "blender.exe"), ("All", "*.*")])
        else:
            p = filedialog.askdirectory(title=title, initialdir=start or None, mustexist=False)
    finally:
        root.destroy()
    return os.path.normpath(p) if p else ""

# ------------------------------------------------------------------ HTTP
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)): body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str): body = body.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)
    def body(self):
        n = int(self.headers.get("Content-Length") or 0); raw = self.rfile.read(n) if n else b""
        return raw
    def jbody(self):
        raw = self.body(); return json.loads(raw.decode("utf-8")) if raw else {}
    def err(self, e, code=400): self.send(code, {"error": str(e)})

    def send_file(self, path):
        if not path or not os.path.isfile(path): return self.err("not found", 404)
        size = os.path.getsize(path); ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range"); start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1): start = int(m.group(1))
                if m.group(2): end = min(int(m.group(2)), size - 1)
                elif not m.group(1): start = max(0, size - int(m.group(2) or 0))
        n = end - start + 1
        self.send_response(206 if rng else 200); self.send_header("Content-Type", ctype); self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(n)); self.send_header("Cache-Control", "no-store")
        if rng: self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start); left = n
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk: break
                try: self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError): break
                left -= len(chunk)

    def allowed(self, p):
        c = conf(); p = os.path.abspath(p)
        roots = [c["work_root"], c["submit_root"], OUT, c.get("source_root") or ""]
        return any(r and os.path.commonpath([p.lower(), os.path.abspath(r).lower()]) == os.path.abspath(r).lower() for r in roots
                   if os.path.splitdrive(p)[0].lower() == os.path.splitdrive(os.path.abspath(r))[0].lower())

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query); path = urllib.parse.unquote(u.path)
        try:
            if path in ("/", "/index.html"):
                return self.send(200, open(os.path.join(HERE, "app.html"), encoding="utf-8").read(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self.send(200, api_state())
            if path.startswith("/check/"):
                cid = path[len("/check/"):]
                if cid not in CLIPS: return self.err("unknown clip", 404)
                html = check_page(cid)
                return self.send(200, html or "<p style='font:16px sans-serif'>Scan keypoints first.</p>", "text/html; charset=utf-8")
            if path == "/api/media":
                p = q.get("p", [""])[0]
                if not self.allowed(p): return self.err("not allowed", 403)
                return self.send_file(p)
            if path.startswith("/api/frame/"):
                cid = path[len("/api/frame/"):]; f = int(q.get("f", ["0"])[0]); kind = q.get("kind", ["orig"])[0]
                if kind == "orig":
                    img = video_frame(cid, f)
                    if img is None: return self.err("no frame", 404)
                    return self.send(200, jpg_bytes(img, 1600), "image/jpeg")
                p = wm_frame_path(cid, f)
                if not p: return self.send(204, b"", "image/jpeg")
                return self.send_file(p)
            return self.err("not found", 404)
        except Exception as e:
            return self.err(e, 500)

    def do_POST(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        try:
            if path == "/api/settings":
                d = self.jbody(); upd = {}
                for k in SETTING_KEYS:
                    if k in d:
                        if k in SECRET and not d[k]: continue                  # empty = keep the saved key
                        v = d[k]
                        if k in ("person_height", "lens_mm"): v = float(v)
                        upd[k] = v
                if d.get("clear_keys"):
                    for k in SECRET: upd[k] = ""
                save_conf(upd); return self.send(200, api_state())
            if path == "/api/browse":
                d = self.jbody()
                return self.send(200, {"path": browse(d.get("kind", "folder"), d.get("title", "Choose"), d.get("start", ""))})
            if path == "/api/open":
                p = self.jbody().get("path", "")
                if not self.allowed(p) and p != conf().get("source_root"): return self.err("not allowed", 403)
                if os.name == "nt":
                    if os.path.isfile(p): subprocess.Popen(["explorer", "/select,", p])
                    else: os.startfile(p)
                return self.send(200, {"ok": True})
            if path == "/api/scan":
                d = self.jbody(); c = conf(); ids = [i for i in d.get("ids", []) if i in CLIPS]
                if not ids: return self.err("Select clips first")
                cmd = [sys.executable, os.path.join(HERE, "automark.py")] + [CLIPS[i] for i in ids] + \
                      ["--out", OUT, "--device", DEVICE, "--base", c["source_root"]] + (["--force"] if d.get("force") else [])
                start_job("Scanning keypoints (%d clips)" % len(ids), [cmd], len(ids), "scan")
                return self.send(200, {"ok": True})
            if path == "/api/build":
                d = self.jbody(); ids = [i for i in d.get("ids", []) if i in CLIPS]
                if not ids: return self.err("Select clips first")
                stage = d.get("stage", "all")
                cmd = [sys.executable, os.path.join(HERE, "pipeline", "run_pipeline.py")] + [CLIPS[i] for i in ids] + \
                      ["--device", DEVICE, "--stage", stage] + (["--force"] if d.get("force") else [])
                title = {"all": "Building zips", "fit": "Making 3D previews", "render": "Rendering video + zip"}.get(stage, "Building")
                start_job("%s (%d clips)" % (title, len(ids)), [cmd], len(ids), "build")
                return self.send(200, {"ok": True})
            if path == "/api/pull_model":
                exe = ollama_exe()
                if not exe: return self.err("Ollama is not installed - get it from ollama.com (free), then try again")
                m = self.jbody().get("model") or conf().get("ollama_model", "qwen3-vl:8b")
                save_conf({"ollama_model": m}); _AI["t"] = 0
                start_job("Downloading local AI model %s (one time, several GB)" % m, [[exe, "pull", m]], 1, "pull")
                return self.send(200, {"ok": True})
            if path == "/api/queue":
                d = self.jbody(); cid = d.get("id")
                if cid not in CLIPS: return self.err("unknown clip")
                if d.get("remove"):
                    if cid in QUEUE: QUEUE.remove(cid)
                elif cid not in QUEUE:
                    QUEUE.append(cid)
                return self.send(200, {"ok": True, "queue": QUEUE})
            if path == "/api/cancel":
                if JOB and JOB.running: JOB.cancel()
                return self.send(200, {"ok": True})
            if path.startswith("/api/checked/"):
                cid = path[len("/api/checked/"):]
                md = marks_index().get(os.path.splitext(os.path.basename(CLIPS[cid]))[0])
                if not md: return self.err("no marks")
                raw = self.body(); json.loads(raw.decode("utf-8"))
                with open(os.path.join(md, "checked.json"), "wb") as fh: fh.write(raw)
                return self.send(200, {"ok": True})
            if path.startswith("/api/still/"):
                cid = path[len("/api/still/"):]; f = int(self.jbody().get("frame", 0))
                render_still(cid, f); return self.send(200, {"ok": True})
            if path.startswith("/api/ai/"):
                return self.send(200, api_ai(path[len("/api/ai/"):], self.jbody()))
            if path.startswith("/api/undo/"):
                return self.send(200, api_undo(path[len("/api/undo/"):]))
            return self.err("not found", 404)
        except Exception as e:
            return self.err(e, 500)

def api_state():
    global CLIPS
    c = conf()
    CLIPS = scan_folder(c.get("source_root"), (c["submit_root"], c["work_root"], OUT, HERE))
    midx = marks_index()
    st = {k: c.get(k) for k in SETTING_KEYS if k not in SECRET}
    st.update({k + "_set": bool((c.get(k) or "").strip()) for k in SECRET})
    run, have, names = ai_status(c)
    return {"settings": st, "device": DEVICE, "blender": blender_exe(c), "clips": [clip_status(i, v, c, midx) for i, v in CLIPS.items()],
            "job": JOB.state() if JOB else None, "queue": list(QUEUE),
            "ollama": {"installed": bool(ollama_exe()), "running": run, "model_ready": have, "models": names}}

_AI = {"t": 0, "v": (False, False, [])}
def ai_status(c):
    if time.time() - _AI["t"] > 8:                       # cheap: checked at most every 8 s
        _AI["v"] = edit_ops.ollama_status(c); _AI["t"] = time.time()
    return _AI["v"]

def edits_dir(cid):
    d = os.path.join(conf()["work_root"], cid, "edits"); os.makedirs(d, exist_ok=True); return d

def api_ai(cid, d):
    c = conf(); w = os.path.join(c["work_root"], cid); dp = os.path.join(w, "scene_data.json")
    if not os.path.exists(dp): raise RuntimeError("Build this clip first (3D preview or zip)")
    data = jload(dp); f = int(d.get("frame", 0))
    imgs = []
    wp = wm_frame_path(cid, f)
    if wp: imgs.append(("White-model render, frame %d:" % (f + 1), data_url(wp)))
    o = video_frame(cid, f)
    if o is not None: imgs.append(("Original film frame %d:" % (f + 1), data_url(o)))
    for i, u in enumerate(d.get("images", [])[:4]):
        imgs.append(("User screenshot %d:" % (i + 1), u))
    hp = os.path.join(w, "ai_history.json"); hist = jload(hp) if os.path.exists(hp) else []
    prompt = "(The user is looking at frame %d.) %s" % (f + 1, d.get("prompt", ""))
    res = edit_ops.ask_ai(c, prompt, imgs, data, hist)
    new, msgs = edit_ops.apply_ops(data, res.get("ops", []))
    applied = any(m.startswith("✓") for m in msgs)
    if applied:
        n = len(glob.glob(os.path.join(edits_dir(cid), "scene_*.json")))
        shutil.copyfile(dp, os.path.join(edits_dir(cid), "scene_%03d.json" % n))
        jsave(dp, new)
    hist += [{"role": "user", "text": d.get("prompt", "")}, {"role": "assistant", "text": json.dumps(res, ensure_ascii=False)}]
    jsave(hp, hist[-20:])
    return {"reply": res.get("reply", ""), "ops": msgs, "applied": applied}

def api_undo(cid):
    bs = sorted(glob.glob(os.path.join(edits_dir(cid), "scene_*.json")))
    if not bs: raise RuntimeError("Nothing to undo")
    dp = os.path.join(conf()["work_root"], cid, "scene_data.json")
    shutil.copyfile(bs[-1], dp); os.remove(bs[-1])
    os.utime(dp, None)
    return {"ok": True, "left": len(bs) - 1}

def main():
    c = conf()
    os.makedirs(c["work_root"], exist_ok=True); os.makedirs(c["submit_root"], exist_ok=True)
    threading.Thread(target=queue_worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = "http://127.0.0.1:%d/" % PORT
    print("白模 Studio is running:", url)
    print("Keep this window open while you use the app. Close it to stop.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()

if __name__ == "__main__":
    main()

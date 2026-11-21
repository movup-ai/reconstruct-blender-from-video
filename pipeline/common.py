# -*- coding: utf-8 -*-
"""shared settings and helpers for the automatic white-model pipeline"""
import json, os, sys, shutil, tempfile, glob, math
import numpy as np
import cv2

AGENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(AGENT, "config.json")
DEFAULT_CFG = {
    "blender_exe": "",                       # empty = search C:\\Program Files\\Blender Foundation\\*\\blender.exe
    "source_root": os.path.join(AGENT, "SOURCE"),         # clips are read from here (film folders or loose videos)
    "source_exclude": ["3D"],                # folders under source_root that are not clips (the client's samples)
    "submit_root": os.path.join(AGENT, "SUBMIT"),
    "work_root": os.path.join(AGENT, "Work_auto"),
    "lens_mm": 34.9,                         # 35 mm-equivalent focal length used for every clip (clip 1 solved to 34.9)
    "person_height": 1.72,
    "character_style": "sample",             # "sample" = client's L3 sample (no neck/hands/feet), "classic" = earlier mannequin
    "out_width": 3840,                       # white model / comparison width, like the client sample (3840 x source aspect)
    "out_fps": 24,
    "video_bitrate": "30M",
    "depth_model_url": "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
    "colours": ["red", "blue", "green", "yellow", "purple", "purple"],   # v2.1: 5th character on = purple
    "max_people": 6,
    "pose3d": True,                          # 3D pose lifter (MotionBERT, pose3d.py) as a prior in the fit; false = 2D marks only
    "pose3d_weight": 1.0,                    # strength of that prior (bone directions, sigma pose3d_sigma)
    "pose3d_sigma": 0.2,
    "pose_mode": "performance",              # automark models: "performance" (YOLOX-x + RTMPose-x) or "balanced" (-m models:
                                             # several times faster on a processor, joints less exact)
    "pose_engine": "tensorrt",               # automark GPU engine: "tensorrt" (fp16, falls back to CUDA) or "cuda"
    "detect_stride": "auto",                 # automark: 1 = every frame; 2 / "auto" = every 2nd frame, rest interpolated when safe
    "detect_stride_cpu": "track",            # on the processor: "track" = person detector on every 4th frame, RTMPose follows
                                             # the people in between (~4x faster); "auto" = as detect_stride
    "sam_points": 16,                        # automatic set (env_auto): SAM point grid per side (32 = more small pieces, ~3x slower)
}

def cfg():
    c = dict(DEFAULT_CFG)
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH, encoding="utf-8") as fh:
            c.update(json.load(fh))
    else:
        with open(CFG_PATH, "w", encoding="utf-8") as fh:
            json.dump(c, fh, ensure_ascii=False, indent=2)
    sb = os.environ.get("BA_SANDBOX")                  # test runs: everything written goes under this folder
    if sb:
        c["work_root"] = os.path.join(sb, "Work_auto"); c["submit_root"] = os.path.join(sb, "SUBMIT")
        c["marks_root"] = os.path.join(sb, "output"); c["_sandbox"] = sb
    return c

def output_dir():
    """the folder holding output\\<clip> (sets, hints): the sandbox's when BA_SANDBOX is set"""
    sb = os.environ.get("BA_SANDBOX")
    return os.path.join(sb, "output") if sb else os.path.join(AGENT, "output")

def log(*a):
    print(*a, flush=True)

def gpu_dlls():
    """make the pip-installed CUDA / cuDNN libraries loadable: cuDNN opens its sub-libraries by name while a model
    runs, so their folders must be on the DLL search path, not just preloaded"""
    import onnxruntime as ort, site
    for sp in site.getsitepackages() + [os.path.dirname(os.path.dirname(ort.__file__))]:
        # nvidia\*\bin: CUDA / cuDNN wheels; tensorrt_libs: TensorRT runtime wheel (tensorrt-cu12-libs, nvinfer_10.dll)
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")) + glob.glob(os.path.join(sp, "tensorrt_libs")):
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                try: os.add_dll_directory(d)
                except (OSError, AttributeError): pass
    try: ort.preload_dlls()
    except Exception: pass
    return ort

_DEVICE = {}

def has_nvidia():
    """True when this PC has an NVIDIA card with its driver and onnxruntime was installed with CUDA"""
    if "nv" not in _DEVICE:
        ok = False
        try:
            sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
            if os.path.exists(os.path.join(sysdir, "nvcuda.dll")) or shutil.which("nvidia-smi"):
                import onnxruntime as ort
                ok = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            ok = False
        _DEVICE["nv"] = ok
    return _DEVICE["nv"]

def pick_device(want="cuda"):
    """the device the models really run on: "cuda" only with an NVIDIA card, otherwise "cpu" (slower, same results).
    BA_DEVICE=cpu forces the processor."""
    if os.environ.get("BA_DEVICE", "").lower() == "cpu" or str(want).lower() == "cpu":
        return "cpu"
    if has_nvidia():
        return "cuda"
    if "told" not in _DEVICE:
        _DEVICE["told"] = True
        log("  no NVIDIA card on this PC - the models run on the processor (slower)")
    return "cpu"

TRT_CACHE = os.path.join(AGENT, "models", "trt_cache")

def gpu_session(model, engine="tensorrt", warm=(), profile=None, exclude=None):
    """onnxruntime GPU session. engine "tensorrt": TensorRT fp16 (engines + timing cache kept in models\\trt_cache, so
    they are built once - the first build takes a few minutes per model), CUDA for anything TensorRT cannot run;
    if TensorRT fails to load / build, the plain CUDA session is used.  profile = (input name, min, opt, max shape)
    for a dynamic batch, so the engine is not rebuilt for every new batch size.  warm = inputs run once (builds the
    engine now and checks it works).  returns (session, "tensorrt" | "cuda")"""
    import onnxruntime as ort
    cuda = ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"})
    def _warm(s):
        n = s.get_inputs()[0].name
        for x in warm:
            s.run(None, {n: x})
    if engine == "tensorrt" and "TensorrtExecutionProvider" in ort.get_available_providers():
        os.makedirs(TRT_CACHE, exist_ok=True)
        o = {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": TRT_CACHE,
             "trt_timing_cache_enable": True, "trt_timing_cache_path": TRT_CACHE,
             "trt_max_workspace_size": 4 << 30, "trt_builder_optimization_level": 3}
        if exclude:                                  # op types left to CUDA (e.g. YOLOX's TopK in its NMS: K > 3840)
            o["trt_op_types_to_exclude"] = exclude
        if profile:
            nm, mn, op, mx = profile
            shp = lambda s: "%s:%s" % (nm, "x".join(str(int(v)) for v in s))
            o.update(trt_profile_min_shapes=shp(mn), trt_profile_opt_shapes=shp(op), trt_profile_max_shapes=shp(mx))
        try:
            s = ort.InferenceSession(model, providers=[("TensorrtExecutionProvider", o), cuda])
            if "TensorrtExecutionProvider" in s.get_providers():
                _warm(s)
                return s, "tensorrt"
            log("  TensorRT not used for %s (providers %s) - using CUDA" % (os.path.basename(model), s.get_providers()))
        except Exception as e:
            log("  TensorRT failed for %s (%s) - using CUDA" % (os.path.basename(model), str(e).splitlines()[0][:300]))
    s = ort.InferenceSession(model, providers=[cuda])
    _warm(s)
    return s, "cuda"

def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)

def jsave(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, separators=(",", ":"))

def read_frames(path):
    """all frames (BGR) + fps; works with Chinese paths on Windows"""
    cap = cv2.VideoCapture(path); tmp = None
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0:
        tmp = tempfile.mkdtemp(prefix="wm_"); t = os.path.join(tmp, "clip" + os.path.splitext(path)[1].lower())
        shutil.copyfile(path, t); cap = cv2.VideoCapture(t)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return out, fps

def read_frames_at(path, idxs):
    """only the frames asked for {index: BGR} (much less memory than read_frames); works with Chinese paths"""
    cap = cv2.VideoCapture(path); tmp = None
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0:
        tmp = tempfile.mkdtemp(prefix="wm_"); t = os.path.join(tmp, "clip" + os.path.splitext(path)[1].lower())
        shutil.copyfile(path, t); cap = cv2.VideoCapture(t)
    want = sorted(set(int(i) for i in idxs)); out = {}; i = 0
    while want and i <= want[-1]:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = f
        i += 1
    cap.release()
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return out

def imwrite_u(path, img, q=90):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())

def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"

def blender_exe(c):
    if c.get("blender_exe") and os.path.exists(c["blender_exe"]):
        return c["blender_exe"]
    cands = sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender*\blender.exe"), reverse=True)
    cands += [shutil.which("blender") or ""]
    for b in cands:
        if b and os.path.exists(b):
            return b
    return None

class Camera:
    """pinhole camera of the picture area. pixel (u,v) in the PICTURE area, v down."""
    def __init__(self, W, Hpic, lens_mm, sensor=36.0):
        self.W, self.H = W, Hpic
        self.f = lens_mm / sensor * W
        self.lens, self.sensor = lens_mm, sensor
    def norm(self, x, y):              # pixel -> normalised camera coords (x right, y down)
        return np.array([(x - self.W / 2) / self.f, (y - self.H / 2) / self.f])
    def pix(self, xn):
        xn = np.atleast_2d(xn); return np.c_[xn[:, 0] * self.f + self.W / 2, xn[:, 1] * self.f + self.H / 2]
    def K(self):
        return np.array([[self.f, 0, self.W / 2], [0, self.f, self.H / 2], [0, 0, 1.0]])

def still_spans(fx, keys, gap=16, share=0.05):
    """pairs of neighbouring fixed frames (at most `gap` apart) where the person barely moved: every joint seen in both
    moved less than `share` of the body height (a pose copied to the next key frame and nudged a little)"""
    out = []
    for a, b in zip(keys, keys[1:]):
        if b - a > gap:
            continue
        both = (fx[a][:, 2] > 0) & (fx[b][:, 2] > 0)
        if both.sum() < 5:
            continue
        ys = fx[a][both, 1]; h = max(ys.max() - ys.min(), 1.0)
        if np.linalg.norm(fx[a][both, :2] - fx[b][both, :2], axis=1).max() < share * h:
            out.append((a, b))
    return out


def coco_people(pose, checked=None):
    """people from pose.json; marks fixed in checked.json are applied, and the fix is blended into nearby frames.
    returns {id: (frames sorted list, kps array [n,17,3])}  (full-frame pixels)"""
    out = {}
    fixes = {}
    if checked:
        for f, ppl in checked.get("frames", {}).items():
            for p in ppl:
                fixes.setdefault(p["id"], {})[int(f)] = np.array(p["kps"], float)
    ids = set(p["id"] for p in pose["people"]) | set(fixes)
    for pid in ids:
        src = next((p for p in pose["people"] if p["id"] == pid), None)
        fr = {int(f): np.array(v, float) for f, v in (src["frames"].items() if src else [])}
        fx = fixes.get(pid, {})
        # frames the checker deleted this person from: drop the auto marks there
        if checked:
            for f, ppl in checked.get("frames", {}).items():
                if int(f) in fr and not any(p["id"] == pid for p in ppl):
                    fr.pop(int(f), None)
        if fx:
            keys = sorted(fx)
            deltas = {}
            for f in keys:
                if f in fr:
                    d = fx[f][:, :2] - fr[f][:, :2]
                    d[(fx[f][:, 2] <= 0) | (fr[f][:, 2] <= 0)] = 0
                    deltas[f] = d
            held = still_spans(fx, keys)
            for f in list(fr):
                if f in fx:
                    fr[f] = fx[f].copy(); fr[f][fx[f][:, 2] > 0, 2] = np.maximum(fr[f][fx[f][:, 2] > 0, 2], 0.95)
                    continue
                span = next(((a, b) for a, b in held if a < f < b), None)
                if span:                         # the person holds still between two fixes: blend the fixes, not the detector's shaky dots
                    a, b = span; t = (f - a) / (b - a)
                    both = (fx[a][:, 2] > 0) & (fx[b][:, 2] > 0)
                    fr[f][both, :2] = fx[a][both, :2] * (1 - t) + fx[b][both, :2] * t
                    fr[f][both, 2] = np.maximum(fr[f][both, 2], 0.9)
                    continue
                if deltas:
                    ks = sorted(deltas); i = np.searchsorted(ks, f)
                    a, b = ks[max(0, i - 1)], ks[min(i, len(ks) - 1)]
                    if abs(f - a) <= 12 or abs(f - b) <= 12:
                        t = 0 if a == b else (f - a) / (b - a)
                        fr[f][:, :2] += deltas[a] * (1 - t) + deltas[b] * t
            for f in keys:                       # people added by the checker in frames the detector missed
                if f not in fr:
                    fr[f] = fx[f].copy()
        if fr:
            fs = sorted(fr); out[pid] = (fs, np.array([fr[f] for f in fs]))
    return out

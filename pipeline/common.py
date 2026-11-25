# -*- coding: utf-8 -*-
"""shared settings and helpers for the automatic white-model pipeline"""
import json, os, shutil, tempfile, glob
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
    "colours": ["red", "blue", "green", "yellow", "cyan", "magenta"],   # colour card: 5th on = a stable colour of its own (purple = special object)
    "max_people": 6,
    "pose3d_weight": 1.0,                    # strength of SAM 3D Body bone-direction constraints
    "pose3d_sigma": 0.2,
    "pose_mode": "performance",              # RTMPose-x; balanced/lightweight use smaller joint models
    "pose_engine": "tensorrt",               # automark GPU engine: "tensorrt" (fp16, falls back to CUDA) or "cuda"
    "sam_points": 32,                        # automatic set (env_auto): SAM point grid per side (16 = fewer small pieces, ~3x faster)
    "pose_backend": "rtmpose",               # joints measured inside SAM 3 masks
    "camera_backend": "flow",               # flow or optional da3 multi-view geometry
    "da3_model": os.path.join(AGENT, "models", "da3"),
    "da3_python": os.path.join(AGENT, "downloads", "da3-venv", "Scripts", "python.exe"),
    "da3_max_views": 12,
    "da3_meters_per_unit": None,             # None = infer scale from visible floor contacts
    "set_adaptive_frames": True,
    "set_max_frames": 5,
    "foot_contact": True,
    "track_occlusion_seconds": 2.0,          # retain identities across bounded detection gaps
    "tracker_backend": "sam3",              # persistent video masks and identities
    "sam3_checkpoint": os.path.join(AGENT, "models", "sam3", "sam3.pt"),
    "sam3_python": os.path.join(AGENT, "downloads", "da3-venv", "Scripts", "python.exe"),
    "sam3_mask_side": 640,
    "sam3_max_objects": 16,
    "sam3_probability": 0.35,
    "sam3_timeout_min": 40,
    "sam3_image_batch": 16,                  # independent point prompts per image decoder call
    "sam3_image_max_masks": 256,             # bounded background proposals before geometric fitting
    "human_backend": "sam3d_body",           # metric body reconstruction
    "sam3d_checkpoint": os.path.join(AGENT, "models", "sam3d_body", "model.ckpt"),
    "sam3d_mhr": os.path.join(AGENT, "models", "sam3d_body", "assets", "mhr_model.pt"),
    "sam3d_python": os.path.join(AGENT, "downloads", "da3-venv", "Scripts", "python.exe"),
    "sam3d_stride": 2,
    "sam3d_surface_side": 160,
    "sam3d_timeout_min": 40,
    "metric_scene": True,                   # shared MoGe geometry and gravity evidence
    "metric_max_views": 3,
    "metric_resolution": 384,
    "scene_verify": True,
    "scene_repair": True,                   # measured planar patches accepted only after verification
    "scene_repair_limit": 4,
    "protect_people": True,                # correct or suppress background geometry covering source-visible people
}

RETIRED_SETTINGS = ("rtmo_model", "detect_stride", "pose3d")
FIXED_BACKENDS = {"tracker_backend": "sam3", "human_backend": "sam3d_body", "pose_backend": "rtmpose"}


def normalize_backends(c):
    """Migrate saved legacy selections to the supported human reconstruction flow."""
    c = dict(c)
    for key in RETIRED_SETTINGS:
        c.pop(key, None)
    c.update(FIXED_BACKENDS)
    return c


def human_model_assets(c):
    """Files required by the fixed SAM 3 / SAM 3D Body processing flow."""
    path = lambda key: c.get(key) or DEFAULT_CFG[key]
    return [
        ("SAM 3 checkpoint", path("sam3_checkpoint")),
        ("SAM 3 Python", path("sam3_python")),
        ("SAM 3 source", os.path.join(AGENT, "third_party", "sam3", "sam3", "model_builder.py")),
        ("SAM 3D Body checkpoint", path("sam3d_checkpoint")),
        ("SAM 3D Body config", os.path.join(os.path.dirname(path("sam3d_checkpoint")), "model_config.yaml")),
        ("SAM 3D Body MHR", path("sam3d_mhr")),
        ("SAM 3D Body Python", path("sam3d_python")),
        ("SAM 3D Body source", os.path.join(AGENT, "third_party", "sam-3d-body", "sam_3d_body", "__init__.py")),
    ]


def cfg():
    c = dict(DEFAULT_CFG)
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH, encoding="utf-8") as fh:
            c.update(json.load(fh))
    else:
        with open(CFG_PATH, "w", encoding="utf-8") as fh:
            json.dump(c, fh, ensure_ascii=False, indent=2)
    c = normalize_backends(c)
    if os.environ.get('BA_FREEZE_PEOPLE') == '1': c['_freeze_people'] = True
    sb = os.environ.get("BA_SANDBOX")                  # test runs: everything written goes under this folder
    if sb:
        c["work_root"] = os.path.join(sb, "Work_auto"); c["submit_root"] = os.path.join(sb, "SUBMIT")
        c["marks_root"] = os.path.join(sb, "output"); c["_sandbox"] = sb
    if os.environ.get("BA_USER"):                      # a job the app runs for one member (see member_cfg)
        c = member_cfg(c, os.environ["BA_USER"], os.environ.get("BA_SOURCE", ""))
    return c

MEMBERS = "_users"                                     # members' own folders inside output, Work_auto and SUBMIT

def member_cfg(c, user, source):
    """the settings one member works with: their own source folder, and their own result folders
    (output\\_users\\<user>, <work_root>\\_users\\<user>, <submit_root>\\_users\\<user>), so members never see or
    overwrite each other's clips.  The app passes the member to its jobs as BA_USER / BA_SOURCE."""
    if c.get("_user"):
        raise ValueError("these settings already belong to member %s" % c["_user"])
    c = dict(c, _user=user, source_root=source or "")
    c["work_root"] = os.path.join(c["work_root"], MEMBERS, user)
    c["submit_root"] = os.path.join(c["submit_root"], MEMBERS, user)
    c["marks_root"] = os.path.join(c.get("marks_root") or os.path.join(AGENT, "output"), MEMBERS, user)
    return c

def output_dir(c=None):
    """the folder holding output\\<clip> (sets, hints): the sandbox's when BA_SANDBOX is set, and the member's own
    (output\\_users\\<user>) for a member's settings (c from member_cfg) or a job run for a member (BA_USER)"""
    sb = os.environ.get("BA_SANDBOX")
    base = os.path.join(sb, "output") if sb else os.path.join(AGENT, "output")
    user = (c or {}).get("_user") or os.environ.get("BA_USER")
    return os.path.join(base, MEMBERS, user) if user else base

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

NO_GPU = ("the NVIDIA card is not available (no card, no driver, or onnxruntime-gpu / the NVIDIA build of PyTorch is "
          "not installed). Blender Agent runs its models on the NVIDIA card only - run Setup.bat.")

def pick_device(want="cuda"):
    """the device the models run on: always "cuda".  GPU only - there is no processor path, so a PC without a
    working NVIDIA card stops here with a clear message instead of running slowly."""
    if str(want).lower() != "cuda":
        raise SystemExit("ERROR: device %r is not supported - the models run on the NVIDIA card only." % want)
    if not has_nvidia():
        raise SystemExit("ERROR: " + NO_GPU)
    return "cuda"

def torch_device():
    """ Return cuda for scene models; stop when PyTorch cannot use the NVIDIA card."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(NO_GPU)
    return "cuda"

TRT_CACHE = os.path.join(AGENT, "models", "trt_cache")

def gpu_session(model, engine="tensorrt", warm=(), profile=None):
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
    with open(p, encoding="utf-8-sig") as fh:                 # (a file saved by a Windows editor may start with a BOM)
        return json.load(fh)

def jsave(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, separators=(",", ":"))

def open_video(path):
    """(cv2.VideoCapture, temp folder or None): OpenCV cannot open a path with Chinese characters on Windows, so such a
    video is copied to a temporary ASCII path first.  Close with close_video."""
    cap = cv2.VideoCapture(path)
    if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
        return cap, None
    cap.release()
    tmp = tempfile.mkdtemp(prefix="wm_")
    t = os.path.join(tmp, "clip" + os.path.splitext(path)[1].lower())
    shutil.copyfile(path, t)
    cap = cv2.VideoCapture(t)
    if not cap.isOpened():
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("cannot open video: %s" % path)
    return cap, tmp

def close_video(cap, tmp):
    cap.release()
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)

def read_frames(path):
    """all frames (BGR) + fps; works with Chinese paths on Windows"""
    cap, tmp = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    close_video(cap, tmp)
    return out, fps

def read_frames_at(path, idxs):
    """only the frames asked for {index: BGR} (much less memory than read_frames), read in order (exact frames);
    works with Chinese paths"""
    cap, tmp = open_video(path)
    want = sorted(set(int(i) for i in idxs)); out = {}; i = 0
    while want and i <= want[-1]:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = f
        i += 1
    close_video(cap, tmp)
    return out

def box_iou(a, b):
    """overlap of two picture boxes [x0, y0, x1, y1] (intersection over union)"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0

def imwrite_u(path, img, q=90):
    """write a picture (also to a path with Chinese characters: encoded in memory)"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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

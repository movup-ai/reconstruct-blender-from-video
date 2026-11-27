# -*- coding: utf-8 -*-
"""The clip's lens from the picture (MoGe), instead of one fixed lens for every clip.

  clip_lens(frames, y0, Hp, shots, work, c) -> (lens_mm, info) or (None, why)
  python pipeline\\lens_auto.py <video>           print the lens MoGe sees in a video

MoGe (microsoft/MoGe, third_party\\MoGe; MoGe-2, moge_model.py) gives the horizontal field of view of a
picture.  Three frames of every shot are measured (first quarter, middle, last quarter; picture area only, long side
720 px) and the clip gets the median as its 35 mm-equivalent lens (36 mm wide sensor, like the rest of the pipeline).
One lens per clip: the Blender camera and every tool here use one.  Shots that disagree are reported.

The model runs in its own python process (torch stays out of the fitting process) and the answer is kept in
Work_auto\\<clip>\\lens_auto.json, so a refit does not measure again.  About 15 s to load + 13 s a frame on a
processor.  config.json "lens_auto": false keeps the fixed "lens_mm" (the only way to set the lens by hand)."""
import os, sys, json, math, subprocess, tempfile, shutil
import numpy as np
from profiling import stage

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LONG_SIDE = 720
LENS_MIN, LENS_MAX = 12.0, 300.0
VERSION = 1
sys.path.insert(0, HERE)
import moge_model


def available(c=None):
    return moge_model.missing() is None


def lens_from_fov(fov_x_deg):
    """horizontal field of view (deg) -> focal length on a 36 mm wide sensor (mm)"""
    return 18.0 / math.tan(math.radians(fov_x_deg) / 2.0)


def sample_frames(shots):
    """frame numbers measured: three per shot (a quarter in, the middle, three quarters in)"""
    out = []
    for s0, s1 in shots:
        n = s1 - s0
        out.append(sorted({s0 + int(n * q) for q in (0.25, 0.5, 0.75)}))
    return out


def summarise(per_shot):
    """[[lens mm of each measured frame] per shot] -> (clip lens, info); the median of the shots' medians"""
    shot_l = [float(np.median(v)) for v in per_shot if len(v)]
    if not shot_l:
        return None, {"why": "no frame could be measured"}
    lens = float(np.clip(np.median(shot_l), LENS_MIN, LENS_MAX))
    info = {"lens_mm": round(lens, 1), "shots_mm": [round(v, 1) for v in shot_l],
            "frames_mm": [[round(float(x), 1) for x in v] for v in per_shot]}
    if max(shot_l) / min(shot_l) > 1.25:
        info["warning"] = "the shots have different lenses (%s mm): one lens (%.0f mm) is used for the whole clip" % (
            ", ".join("%.0f" % v for v in shot_l), lens)
    return lens, info


def _worker(src, dst):
    import cv2, torch
    from common import torch_device
    dev = torch_device()                                      # the NVIDIA card only
    m = moge_model.load(dev)
    z = np.load(src); out = []
    for i in range(int(z["n"])):
        rgb = cv2.cvtColor(z["f%d" % i], cv2.COLOR_BGR2RGB)
        x = torch.tensor(rgb, dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0
        with torch.inference_mode():
            o = m.infer(x, use_fp16=True)
        K = o["intrinsics"].float().cpu().numpy()                 # normalised: fx in picture widths
        out.append(float(2 * math.degrees(math.atan(0.5 / K[0, 0]))))
    with open(dst, "w") as fh:
        json.dump(out, fh)


def measure(pictures, c=None):
    """BGR pictures (picture area) -> horizontal field of view of each (deg), in a separate process"""
    import cv2
    d = tempfile.mkdtemp(prefix="lens_")
    try:
        arrs = {"n": np.array(len(pictures))}
        for i, p in enumerate(pictures):
            s = min(1.0, LONG_SIDE / float(max(p.shape[:2])))
            arrs["f%d" % i] = cv2.resize(p, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else p
        src, dst = os.path.join(d, "in.npz"), os.path.join(d, "out.json")
        np.savez(src, **arrs)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", src, dst],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError("lens worker failed: " + (r.stderr or r.stdout).strip()[-400:])
        with open(dst) as fh:
            return json.load(fh)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@stage("lens")
def clip_lens(frames, y0, Hp, shots, work=None, c=None):
    """frames: the clip's BGR frames; y0, Hp: picture area rows; shots: [(first, end), ...]"""
    why = moge_model.missing()
    if why:
        return None, {"why": why}
    idx = sample_frames(shots)
    key = {"v": VERSION, "moge": "v2", "frames": idx, "size": [int(frames[0].shape[1]), int(Hp)], "y0": int(y0)}
    cp = os.path.join(work, "lens_auto.json") if work else None
    if cp and os.path.exists(cp):
        try:
            with open(cp, encoding="utf-8") as fh:
                old = json.load(fh)
            if old.get("key") == key:
                return old["lens_mm"], dict(old["info"], cached=True)
        except (OSError, ValueError, KeyError):
            pass
    flat = [f for v in idx for f in v]
    fov = measure([frames[f][y0:y0 + Hp] for f in flat], c)
    it = iter(fov)
    lens, info = summarise([[lens_from_fov(next(it)) for _ in v] for v in idx])
    if lens is not None:
        info["model"] = moge_model.NAME
    if lens is not None and cp:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump({"key": key, "lens_mm": lens, "info": info}, fh)
    return lens, info


if __name__ == "__main__":
    if len(sys.argv) in (4, 6) and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3], *([sys.argv[4], int(sys.argv[5])] if len(sys.argv) == 6 else []))
    elif len(sys.argv) == 2:
        sys.path.insert(0, HERE)
        from common import read_frames
        fr, _ = read_frames(sys.argv[1])
        print(clip_lens(fr, 0, fr[0].shape[0], [(0, len(fr))]))
    else:
        print(__doc__)

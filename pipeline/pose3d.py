# -*- coding: utf-8 -*-
"""3D pose lifting: COCO-17 2D tracks -> H36M-17 3D joints per frame (MotionBERT, GPU).

The 2D marks alone cannot tell whether a limb points toward or away from the camera; a lifter trained on motion
capture can.  fit_people.py uses the lifted bone directions as a prior (config "pose3d": true).

  lift_people(people, W, H, cache_path, tag) -> {pid: (frames (T,), X (T,17,3), conf (T,17))}
      X: root-relative 3D joints in the axes of a camera looking straight at the person (x right, y down, z away
      from the camera), MotionBERT's normalised units (only directions are used).  conf: how sure the 2D input of
      that joint was (0 = filled in / outside the picture).
  python pose3d.py <clip name>     lift every person of a clip (cache in Work_auto\\<clip>\\pose3d.npz) and print stats

Model: MotionBERT (Walter0807/MotionBERT, Apache-2.0), checkpoint FT_MB_release_MB_ft_h36m (pose3d, H36M fine-tuned)
from huggingface.co/walterzhu/MotionBERT.  Code in third_party\\MotionBERT-main, weights in models\\motionbert."""
import os, sys, hashlib, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MB_DIR = os.path.join(ROOT, "third_party", "MotionBERT-main")
MB_WEIGHTS = os.path.join(ROOT, "models", "motionbert", "FT_MB_release_MB_ft_h36m.bin")
CLIP = 243                                     # MotionBERT window (frames)
VERSION = 1                                    # bump when the input preparation changes (invalidates caches)

# H36M-17: 0 pelvis 1 RHip 2 RKnee 3 RAnkle 4 LHip 5 LKnee 6 LAnkle 7 spine 8 thorax 9 nose 10 head-top
#          11 LShoulder 12 LElbow 13 LWrist 14 RShoulder 15 RElbow 16 RWrist
H36M_FROM_COCO = {1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15, 9: 0, 11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10}

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from functools import partial
        if MB_DIR not in sys.path:
            sys.path.insert(0, MB_DIR)
        from lib.model.DSTformer import DSTformer
        m = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5, num_heads=8, mlp_ratio=2,
                      norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), maxlen=CLIP, num_joints=17)
        try:
            ck = torch.load(MB_WEIGHTS, map_location="cpu", weights_only=True)
        except Exception:                                      # older pickles: the official checkpoint file
            ck = torch.load(MB_WEIGHTS, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", "", 1): v for k, v in ck["model_pos"].items()}
        m.load_state_dict(sd, strict=True)
        from common import torch_device
        dev = torch_device()                                   # the NVIDIA card only
        _MODEL = (m.to(dev).eval(), dev, torch)
    return _MODEL


def coco_to_h36m(K, W, H, thr=0.3):
    """K (T,17,3) COCO px + score -> (T,17,3) H36M px + conf.  Joints below thr or outside the picture get conf 0
    and a position filled in over time (or from the body when never seen)."""
    K = np.asarray(K, float).copy()
    T = len(K)
    out = np.zeros((T, 17, 3))
    m = 6.0                                                   # px margin: marks pinned to the frame edge are guesses
    ins = (K[..., 0] > m) & (K[..., 0] < W - m) & (K[..., 1] > m) & (K[..., 1] < H - m)
    c = np.where((K[..., 2] >= thr) & ins, np.clip(K[..., 2], 0, 1), 0.0)
    # fill each COCO joint over time where it is unsure
    t = np.arange(T)
    for j in range(17):
        ok = c[:, j] > 0
        if ok.sum() >= 1 and not ok.all():
            for d in range(2):
                K[~ok, j, d] = np.interp(t[~ok], t[ok], K[ok, j, d])
    good = c > 0
    # joints never seen: from the body (standing legs, arms hanging)
    tor = np.zeros(T)
    for f in range(T):
        sh = [i for i in (5, 6) if good[:, i].any()]
        hp = [i for i in (11, 12) if good[:, i].any()]
        tor[f] = np.linalg.norm(K[f, sh, :2].mean(0) - K[f, hp, :2].mean(0)) if sh and hp else 0.0
    tl = float(np.median(tor[tor > 0])) if (tor > 0).any() else 0.25 * H
    for a, b in ((11, 12), (12, 11), (5, 6), (6, 5)):         # a missing side from the other one
        if not good[:, a].any() and good[:, b].any():
            K[:, a, :2] = K[:, b, :2]
    if not good[:, [11, 12]].any() and good[:, [5, 6]].any():  # hips below the picture: under the shoulders
        sw = np.linalg.norm(K[:, 5, :2] - K[:, 6, :2], axis=1, keepdims=True)
        sw = np.maximum(sw, 0.05 * H)
        mid = K[:, [5, 6], :2].mean(1)
        for j, sj in ((11, 5), (12, 6)):
            K[:, j, :2] = mid + (K[:, sj, :2] - mid) * 0.6 + np.c_[np.zeros(len(K)), 1.45 * sw[:, 0]]
        tl = float(np.median(1.45 * sw))
    for side in ((11, 13, 15), (12, 14, 16), (5, 7, 9), (6, 8, 10)):
        for k in range(1, 3):
            j, p = side[k], side[k - 1]
            if not good[:, j].any():
                K[:, j, :2] = K[:, p, :2] + np.array([0, 0.85 * tl if side[0] in (11, 12) else 0.45 * tl])
    for j in range(5):                                        # face: above the shoulders
        if not good[:, j].any():
            K[:, j, :2] = K[:, [5, 6], :2].mean(1) - np.array([0, 0.35 * tl])
    for h, cc in H36M_FROM_COCO.items():
        out[:, h, :2] = K[:, cc, :2]; out[:, h, 2] = c[:, cc]
    hip = K[:, [11, 12], :2].mean(1); sh = K[:, [5, 6], :2].mean(1)
    out[:, 0, :2] = hip; out[:, 0, 2] = np.minimum(c[:, 11], c[:, 12])
    out[:, 8, :2] = sh; out[:, 8, 2] = np.minimum(c[:, 5], c[:, 6])
    out[:, 7, :2] = (hip + sh) / 2; out[:, 7, 2] = np.minimum(out[:, 0, 2], out[:, 8, 2])
    # head top: above the eyes / ears, away from the neck
    fc = [i for i in (1, 2, 3, 4) if True]
    eyes = K[:, fc, :2].mean(1)
    up = eyes - sh; n = np.linalg.norm(up, axis=1, keepdims=True); up = up / np.maximum(n, 1e-6)
    out[:, 10, :2] = eyes + up * 0.45 * np.maximum(n, 0.25 * tl)
    out[:, 10, 2] = c[:, fc].max(1) * np.minimum(1.0, c[:, 0] + 0.5)
    return out


def _normalise(X, W, H):
    """px -> MotionBERT input: centred on the (smoothed) body each frame, one scale per track (body height ~1.2)"""
    from scipy.ndimage import gaussian_filter1d
    X = X.copy()
    ok = X[..., 2] > 0
    cen = np.zeros((len(X), 2))
    for f in range(len(X)):
        v = X[f, [0, 7, 8], :2]
        cen[f] = v.mean(0)
    if len(X) > 3:
        cen = gaussian_filter1d(cen, 2.0, axis=0, mode="nearest")
    tor = np.linalg.norm(X[:, 8, :2] - X[:, 0, :2], axis=1)
    tok = ok[:, 0] & ok[:, 8]
    tl = float(np.median(tor[tok])) if tok.sum() >= 3 else float(np.median(tor))
    s = max(tl * 3.4, 8.0) / 1.2                              # body height ~3.4 torso lengths -> 1.2 units
    X[..., :2] = (X[..., :2] - cen[:, None]) / s
    return X.astype(np.float32), cen, s


def _run(Xn):
    """MotionBERT over (T,17,3) in 243-frame windows (overlapping, blended) with flip averaging -> (T,17,3)"""
    m, dev, torch = _model()
    T = len(Xn)
    if T <= CLIP:
        starts = [0]
    else:
        starts = list(range(0, T - CLIP, CLIP // 2)) + [T - CLIP]
    acc = np.zeros((T, 17, 3)); wsum = np.zeros((T, 1, 1))
    L = [4, 5, 6, 11, 12, 13]; R = [1, 2, 3, 14, 15, 16]
    with torch.no_grad():
        for st in starts:
            seg = Xn[st:st + CLIP]
            x = torch.from_numpy(seg[None]).to(dev)
            xf = x.clone(); xf[..., 0] *= -1; xf[:, :, L + R] = xf[:, :, R + L]
            y = m(torch.cat([x, xf], 0))
            yf = y[1:2].clone(); yf[..., 0] *= -1; yf[:, :, L + R] = yf[:, :, R + L]
            y = ((y[0:1] + yf) / 2)[0].float().cpu().numpy()
            y = y - y[:, :1]                                  # root relative
            n = len(seg)
            w = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)).astype(float)   # blend overlapping windows
            w = np.minimum(w / max(1.0, min(40.0, n / 4)), 1.0)[:, None, None] if len(starts) > 1 else np.ones((n, 1, 1))
            acc[st:st + n] += y * w; wsum[st:st + n] += w
    return acc / np.maximum(wsum, 1e-6)


def lift_track(fs, K, W, H):
    """one person: frames fs (sorted), K (n,17,3) COCO px -> frames (T,) dense, X (T,17,3), conf (T,17)"""
    fs = np.asarray(fs, int); K = np.asarray(K, float)
    allf = np.arange(fs[0], fs[-1] + 1)
    D = np.zeros((len(allf), 17, 3))
    idx = fs - fs[0]
    D[idx] = K
    if len(allf) > len(fs):                                   # frames without marks: in between, conf 0
        miss = np.setdiff1d(np.arange(len(allf)), idx)
        for j in range(17):
            for d in range(2):
                D[miss, j, d] = np.interp(miss, idx, K[:, j, d])
    Hm = coco_to_h36m(D, W, H)
    Xn, _, _ = _normalise(Hm, W, H)
    Y = _run(Xn)
    return allf, Y.astype(np.float32), Hm[..., 2].astype(np.float32)


def _digest(fs, K, W, H):
    h = hashlib.sha1()
    h.update(np.asarray(fs, np.int64).tobytes()); h.update(np.round(np.asarray(K, float), 2).tobytes())
    h.update(np.array([W, H, VERSION], float).tobytes())
    return h.hexdigest()[:16]


def available():
    import importlib.util                     # (without importing torch here: it is loaded in the worker process)
    return importlib.util.find_spec("torch") is not None and os.path.exists(MB_WEIGHTS) and os.path.isdir(MB_DIR)


def _lift_subprocess(people, W, H):
    """lift in a separate python process: torch / CUDA (~2-3 GB of committed memory) never enters the fitting
    process, which matters when several clips are fitted side by side"""
    import subprocess, tempfile
    d = tempfile.mkdtemp(prefix="pose3d_")
    try:
        src, dst = os.path.join(d, "in.npz"), os.path.join(d, "out.npz")
        arrs = {"WH": np.array([W, H], float)}
        for i, (p, (fs, K)) in enumerate(people.items()):
            arrs["f%d" % i] = np.asarray(fs, int); arrs["K%d" % i] = np.asarray(K, float)
        np.savez(src, **arrs)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", src, dst],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError("pose3d worker failed: " + (r.stderr or r.stdout).strip()[-400:])
        with np.load(dst) as z:
            return {p: (np.array(z["f%d" % i]), np.array(z["X%d" % i]), np.array(z["c%d" % i])) for i, p in enumerate(people)}
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def _worker(src, dst):
    z = np.load(src); W, H = z["WH"]
    out = {}
    i = 0
    while "f%d" % i in z.files:
        f, X, c = lift_track(z["f%d" % i], z["K%d" % i], W, H)
        out.update({"f%d" % i: f, "X%d" % i: X, "c%d" % i: c}); i += 1
    np.savez(dst, **out)


subprocess_mode = True                         # False: lift inside this process (keeps the model loaded)


def lift_people(people, W, H, cache_path=None, tag=""):
    """people {pid: (frames, K (n,17,3) px in the picture area)} -> {pid: (frames, X, conf)}; cached by input"""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as z:      # closed again: Windows cannot replace an open file
                cache = {k: np.array(z[k]) for k in z.files}
        except Exception:
            cache = {}
    out = {}; new = 0; t0 = time.time(); todo = {}
    for pid, (fs, K) in people.items():
        if len(fs) < 4:
            continue
        key = "%s_p%s" % (tag or "s", pid)
        dg = _digest(fs, K, W, H)
        if key + "_X" in cache and str(cache.get(key + "_d")) == dg:
            out[pid] = (cache[key + "_f"], cache[key + "_X"], cache[key + "_c"]); continue
        todo[pid] = (key, dg, fs, K)
    if todo:
        res = _lift_subprocess({p: v[2:] for p, v in todo.items()}, W, H) if subprocess_mode else \
            {p: lift_track(fs, K, W, H) for p, (_, _, fs, K) in todo.items()}
        for pid, (key, dg, _, _) in todo.items():
            f, X, c = res[pid]
            out[pid] = (f, X, c); new += 1
            cache.update({key + "_f": f, key + "_X": X, key + "_c": c, key + "_d": np.array(dg)})
    if new and cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp.npz"
        np.savez_compressed(tmp, **cache)
        os.replace(tmp, cache_path)
    if new:
        print("    pose3d: lifted %d people in %.1fs" % (new, time.time() - t0), flush=True)
    return out


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--worker":
    _worker(sys.argv[2], sys.argv[3])
elif __name__ == "__main__":
    sys.path.insert(0, HERE)
    from common import cfg, jload, coco_people
    import run_pipeline as rp
    name = sys.argv[1]
    c = cfg()
    work = os.path.join(c["work_root"], name)
    ci = jload(os.path.join(work, "clip_info.json"))
    pose = jload(os.path.join(rp.marks_dir(ci["video"], c), "pose.json"))
    _, y0, _, Hp = pose["picture_area"]
    ppl = coco_people(pose)
    ppl = {p: (fs, K - np.array([0, y0, 0])) for p, (fs, K) in ppl.items()}
    t0 = time.time()
    res = lift_people(ppl, ci["W"], Hp, os.path.join(work, "pose3d_test.npz"))
    for p, (f, X, cf) in res.items():
        print(p, len(f), "frames  conf>0: %.0f%%" % (100 * (cf > 0).mean()), " spine dir f0:", np.round(X[0, 8] - X[0, 0], 3))
    print("%.1fs" % (time.time() - t0))

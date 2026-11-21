# -*- coding: utf-8 -*-
"""Automatic set proposal from real 3D measurements (run this before writing env_claude.json by hand).

  python pipeline\\env_auto.py <clip> [--out FILE] [--no-mid] [--points 32]

For the first and middle frame of every shot:
  MoGe-2 (microsoft/MoGe, models\\moge2\\model.pt): metric point map + normals, with the pipeline's own field of view
     (so the set lines up with the render camera); its free field-of-view estimate is reported too.
  SAM 2.1 (facebookresearch/sam2, models\\sam2\\sam2.1_hiera_large.pt): the people's masks (from their boxes in
     pose.json) and automatic masks of everything else.
  Ground plane: RANSAC over upward-facing points (people removed, their feet as anchors) -> the camera's height /
     tilt / roll over that ground, compared with Work_auto\\<clip>\\camera.json (the fit's camera).  Walls give the true
     vertical -> the ground's slope (reported).
  Every sizeable mask -> 3D points in the shot's world (floor z = 0, camera above the origin looking +y, metres; scaled
     to the fit's camera height so it lines up with the people) -> class from geometry and colour (floor, sloped
     ground, wall, building, table top, vegetation, rock, furniture, vehicle, far) -> fitted object: oriented box
     (pos / size / yaw), a wall with its real top outline or a sloped ground patch ("patch" with "pts3").  Touching
     pieces of the same thing are merged.
Writes output\\<clip>\\env_auto.json (the env_claude.json format; "_auto" holds the measurements) and
Work_auto\\<clip>\\env_auto.jpg: per frame  masks + labels over the film | the set over the film | the set alone.
The set in env_auto.jpg is drawn with the MoGe camera tilt; build_env's env_check uses the fit's camera, so when the
printout says the tilt disagrees, put the suggested camera into hints.json and re-run the fit (ENV_AUTHORING.md)."""
import argparse, json, math, os, sys, time, warnings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload, imwrite_u, read_frames_at, output_dir, AGENT

TP = os.path.join(AGENT, "third_party")
MODELS = os.path.join(AGENT, "models")
D = np.diag([1.0, -1.0, -1.0])                       # OpenCV camera (x right, y down, z forward) <-> Blender camera
CLS_COL = {"floor": (120, 120, 120), "slope": (40, 160, 220), "wall": (220, 120, 40), "building": (200, 80, 200),
           "top": (40, 220, 220), "vegetation": (60, 200, 60), "rock": (60, 120, 200), "object": (0, 160, 255),
           "vehicle": (255, 0, 160), "far": (200, 200, 120), "sky": (255, 255, 255), "terrain": (90, 160, 140), "pole": (160, 60, 255)}


def setup():
    dl = os.path.join(AGENT, "downloads")
    os.environ.setdefault("HF_HOME", os.path.join(dl, "hf")); os.environ.setdefault("TORCH_HOME", os.path.join(dl, "torch_home"))
    for p in (os.path.join(TP, "MoGe"), os.path.join(TP, "sam2")):
        if p not in sys.path:
            sys.path.insert(0, p)


class Models:
    def __init__(self, points_per_side=32, device=None):
        setup()
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")    # no NVIDIA card: the processor (slower)
        self.torch, self.dev = torch, device
        from moge.model.v2 import MoGeModel
        self.moge = MoGeModel.from_pretrained(os.path.join(MODELS, "moge2", "model.pt")).to(device).eval()
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", os.path.join(MODELS, "sam2", "sam2.1_hiera_large.pt"), device=device)
        self.pps = points_per_side
        self.pred = SAM2ImagePredictor(sam)

    def amp(self):
        """half precision on the GPU; full precision on the processor"""
        import contextlib
        return self.torch.autocast("cuda", dtype=self.torch.bfloat16) if self.dev == "cuda" else contextlib.nullcontext()

    def moge_run(self, rgb, fov_x=None):
        t = self.torch.tensor(rgb, dtype=self.torch.float32, device=self.dev).permute(2, 0, 1) / 255.0
        with self.torch.inference_mode():
            out = self.moge.infer(t, fov_x=fov_x, use_fp16=(self.dev == "cuda"))   # half precision only on the GPU
        return {k: v.float().cpu().numpy() for k, v in out.items() if hasattr(v, "cpu")}

    def sam_auto(self, rgb, n=None, iou_t=0.7, stab_t=0.86, nms_t=0.6, min_area=0.002):
        """automatic masks, done on SAM's 256x256 low-res logits (the library generator spends 10-30 s on full-res
        post-processing; this takes about a second): point grid -> 3 masks each -> quality / stability filter ->
        mask-IoU NMS -> survivors upsampled to the picture"""
        torch = self.torch; F = torch.nn.functional
        n = n or self.pps
        H, W = rgb.shape[:2]
        with torch.inference_mode(), self.amp():
            self.pred.set_image(rgb)
            g = (np.arange(n) + 0.5) / n
            pts = np.array([(x * W, y * H) for y in g for x in g], np.float32)
            L, S = [], []
            for i in range(0, len(pts), 64):
                pc = pts[i:i + 64, None, :]; pl = np.ones((len(pc), 1), np.int32)
                _, uc, lab, _ = self.pred._prep_prompts(pc, pl, None, None, True)
                _, iou, low = self.pred._predict(uc, lab, None, None, multimask_output=True, return_logits=True)
                low = low.float().flatten(0, 1); iou = iou.float().flatten()
                stab = (low > 1).sum((1, 2)).float() / (low > -1).sum((1, 2)).clamp(min=1).float()
                area = (low > 0).float().mean((1, 2))
                k = (iou > iou_t) & (stab > stab_t) & (area > min_area) & (area < 0.98)
                L.append(low[k] > 0); S.append(iou[k] + 0.2 * stab[k])
            M = torch.cat(L); sc = torch.cat(S)
            if not len(M):
                return []
            order = torch.argsort(sc, descending=True); M = M[order]
            Mf = M.flatten(1).half()
            inter = (Mf @ Mf.T).float(); ar = Mf.sum(1).float()
            iouM = (inter / (ar[:, None] + ar[None] - inter).clamp(min=1)).cpu().numpy()
            keep = []
            for i in range(len(M)):
                if all(iouM[i, j] <= nms_t for j in keep):
                    keep.append(i)
            Mk = M[keep].float()[:, None]
            up = F.interpolate(Mk, (H, W), mode="bilinear", align_corners=False)[:, 0] > 0.5
            up = up.cpu().numpy()
        return [{"segmentation": m, "area": int(m.sum())} for m in up]

    def sam_boxes(self, rgb, boxes):
        if not len(boxes):
            return np.zeros(rgb.shape[:2], bool)
        with self.torch.inference_mode(), self.amp():
            self.pred.set_image(rgb)
            m, _, _ = self.pred.predict(box=np.asarray(boxes, np.float32), multimask_output=False)
        m = np.asarray(m).reshape(len(boxes), -1, *rgb.shape[:2])[:, 0]
        return m.astype(bool).any(0)


# ----------------------------------------------------------------------------------------------- geometry helpers
def cam_mats(c):
    x, y, z, ex, ey, ez = c
    a, b, g = [math.radians(v) for v in (ex, ey, ez)]
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx, np.array([x, y, z], float)


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def fit_ground(P, N, ok, up0, feet, rng_max=60.0, iters=500, seed=0):
    """RANSAC plane n.X = c (n up, camera at the origin above it: c < 0) over upward-facing points.
    P, N: (M, 3) camera points / normals; feet: (F, 3) points under the people's ankles (strong anchors)."""
    rs = np.random.default_rng(seed)
    r = np.linalg.norm(P, axis=1)
    cand = ok & (N @ up0 > math.cos(math.radians(30))) & (r < rng_max)
    if cand.sum() < 60:
        cand = ok & (r < rng_max)
    idx = np.nonzero(cand)[0]
    if len(idx) < 30:
        return None
    Q, rq = P[idx], r[idx]
    F = feet if feet is not None and len(feet) else np.zeros((0, 3))
    rf = np.linalg.norm(F, axis=1) if len(F) else np.zeros(0)
    best, bs = None, -1
    for it in range(iters):
        if len(F) and it % 3 == 0:
            pts = np.vstack([F[rs.integers(len(F))], Q[rs.choice(len(Q), 2, replace=False)]])
        else:
            pts = Q[rs.choice(len(Q), 3, replace=False)]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n = unit(n)
        if n @ up0 < 0:
            n = -n
        if n @ up0 < math.cos(math.radians(18)):
            continue
        c = n @ pts[0]
        if c >= -0.2:                                            # camera must be above it
            continue
        inl = np.abs(Q @ n - c) < 0.03 + 0.02 * rq
        sc = inl.sum() * (1.0 + 0.002 * (-c))                    # tie-break: the lower plane (floor, not a table top)
        if len(F):
            sc += 0.06 * len(Q) * (np.abs(F @ n - c) < 0.06 + 0.03 * rf).sum()
        if sc > bs:
            bs, best = sc, (n, c)
    if best is None:
        return None
    n, c = best
    for _ in range(3):                                           # least squares on the inliers (near ones weigh more)
        inl = np.abs(Q @ n - c) < 0.03 + 0.015 * rq
        if inl.sum() < 20:
            break
        S = Q[inl]; w = 1.0 / (1.0 + rq[inl])
        m = (S * w[:, None]).sum(0) / w.sum()
        _, _, vt = np.linalg.svd((S - m) * np.sqrt(w)[:, None], full_matrices=False)
        n2 = vt[2] if vt[2] @ up0 > 0 else -vt[2]
        if n2 @ up0 < math.cos(math.radians(20)):
            break
        n, c = n2, n2 @ m
    inl = np.abs(Q @ n - c) < 0.03 + 0.015 * rq
    return {"n": n, "c": float(c), "h": float(-c), "inliers": int(inl.sum()), "frac": float(inl.mean()),
            "feet_ok": int((np.abs(F @ n - c) < 0.08 + 0.03 * rf).sum()) if len(F) else 0, "feet": int(len(F))}


def ground_frame(n, h):
    """rows = world axes (x right, y forward along the ground, z = n) in OpenCV camera coordinates"""
    f = np.array([0.0, 0.0, 1.0])
    y = unit(f - (f @ n) * n); x = np.cross(y, n)
    return np.vstack([x, y, n])


def plane_fit(P, trim=2):
    m = np.median(P, 0)
    for _ in range(trim + 1):
        _, s, vt = np.linalg.svd(P - m, full_matrices=False)
        n = vt[2]; d = (P - m) @ n
        keep = np.abs(d) <= np.percentile(np.abs(d), 80) + 1e-6
        if keep.sum() < 10:
            break
        m = P[keep].mean(0)
    d = (P - m) @ n
    rms = float(np.sqrt(np.mean(np.minimum(d * d, np.percentile(d * d, 90)))))
    return n, m, rms


def yaw_front(u, centre, C):
    """yaw (deg) for an object whose width runs along horizontal u, with its front (-y) towards the camera C"""
    yaw = math.degrees(math.atan2(u[1], u[0]))
    a = math.radians(yaw)
    if np.array([math.sin(a), -math.cos(a)]) @ (C[:2] - centre[:2]) < 0:
        yaw += 180.0
    return (yaw + 180.0) % 360.0 - 180.0


def rnd(v, k=3):
    return [round(float(x), k) for x in v]


# ----------------------------------------------------------------------------------------------- one frame
class Frame:
    pass


def people_boxes(pose, f, y0, k, W, H):
    out, feet = [], []
    for p in pose.get("people", []):
        fr = p.get("frames", {}); bx = p.get("boxes", {})
        key = None
        for d in (0, 1, -1, 2, -2, 3, -3, 5, -5):
            if str(f + d) in fr or str(f + d) in bx:
                key = str(f + d); break
        if key is None:
            continue
        xs, ys = [], []
        if key in bx:
            b = bx[key]; xs += [b[0], b[2]]; ys += [b[1], b[3]]
        if key in fr:
            kp = np.array(fr[key], float)
            g = kp[kp[:, 2] > 0.2]
            if len(g):
                xs += [g[:, 0].min(), g[:, 0].max()]; ys += [g[:, 1].min(), g[:, 1].max()]
            for j in (15, 16):
                if kp[j, 2] > 0.3:
                    feet.append(((kp[j, 0]) * k, (kp[j, 1] - y0) * k))
        if not xs:
            continue
        x0, x1, ya, yb = min(xs), max(xs), min(ys) - y0, max(ys) - y0
        mw, mh = 0.08 * (x1 - x0), 0.05 * (yb - ya)
        out.append([max(0, (x0 - mw) * k), max(0, (ya - mh) * k), min(W - 1, (x1 + mw) * k), min(H - 1, (yb + mh * 1.5) * k)])
    return out, feet


def side_bars(img, dark=16, rows=0.97):
    """columns of the black bars at the left and right of a 4:3 film inside a wider frame (pillarbox) -> bool mask"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    darkcol = (g < dark).mean(0) >= rows                          # a column that is black from top to bottom
    H, W = g.shape; m = np.zeros((H, W), bool)
    l = int(np.argmin(darkcol)) if not darkcol.all() else W      # contiguous from each edge only
    r = int(np.argmin(darkcol[::-1])) if not darkcol.all() else W
    if l >= 0.02 * W:
        m[:, :l + 2] = True                                      # +2: the soft edge of the bar
    if r >= 0.02 * W:
        m[:, W - r - 2:] = True
    return m


def analyse(models, bgr, f, pose, ci, fov_x, free_fov=False):
    """MoGe + SAM at work resolution for one frame -> Frame (camera-space points, normals, masks)"""
    t0 = time.time()
    H0, W0 = bgr.shape[:2]
    k = min(1.0, 1024.0 / W0)
    Wk, Hk = int(round(W0 * k)), int(round(H0 * k))
    img = cv2.resize(bgr, (Wk, Hk), interpolation=cv2.INTER_AREA) if k < 1 else bgr.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fr = Frame(); fr.k, fr.W, fr.H, fr.img, fr.f = k, Wk, Hk, img, f
    fr.bars = side_bars(img)                                     # black pillarbox bars: not part of the room
    o = models.moge_run(rgb, fov_x=fov_x)
    fr.P = o["points"]; fr.N = o.get("normal"); fr.valid = o["mask"].astype(bool) & np.isfinite(fr.P).all(-1)
    fr.valid &= ~fr.bars
    fr.P = np.where(fr.valid[..., None], fr.P, 0.0)
    if fr.N is None:                                             # normals from the point map
        gx = np.gradient(fr.P, axis=1); gy = np.gradient(fr.P, axis=0)
        fr.N = np.cross(gx, gy)
    fr.N = fr.N / np.maximum(np.linalg.norm(fr.N, axis=-1, keepdims=True), 1e-9)
    flip = (fr.N * fr.P).sum(-1) > 0                             # every visible surface faces the camera
    fr.N[flip] *= -1
    fr.fov_free = None
    if free_fov:
        o2 = models.moge_run(rgb, fov_x=None)
        K = o2.get("intrinsics")
        if K is not None:
            fr.fov_free = float(2 * math.degrees(math.atan(0.5 / K[0, 0])))
    boxes, feet = people_boxes(pose, f, ci["y0"], k, Wk, Hk)
    fr.boxes = boxes
    pm = models.sam_boxes(rgb, boxes) if boxes else np.zeros((Hk, Wk), bool)
    box_m = np.zeros((Hk, Wk), bool)
    for b in boxes:
        box_m[int(b[1]):int(b[3]) + 1, int(b[0]):int(b[2]) + 1] = True
    if pm.sum() < 0.25 * box_m.sum():                            # SAM missed: the boxes themselves
        pm = box_m
    fr.person = cv2.dilate(pm.astype(np.uint8), np.ones((int(9 * k * 1.5) | 1,) * 2, np.uint8)) > 0
    fr.feet = []
    for u, v in feet:                                            # the floor just under each ankle
        uu, vv = int(min(Wk - 1, max(0, u))), int(min(Hk - 1, max(0, v + 0.012 * Hk)))
        if fr.valid[vv, uu]:
            fr.feet.append(fr.P[vv, uu])
    fr.feet = np.array(fr.feet).reshape(-1, 3)
    t1 = time.time()
    masks = models.sam_auto(rgb)
    fr.t_moge, fr.t_sam = t1 - t0, time.time() - t1
    masks = sorted(masks, key=lambda m: -m["area"])
    lab = np.full((Hk, Wk), -1, np.int32)
    for i, m in enumerate(masks):                                # big first, smaller ones on top
        lab[m["segmentation"]] = i
    lab[fr.bars] = -1                                            # nothing is built from the bars
    fr.lab = lab; fr.n_masks = len(masks)
    return fr


def to_world(fr, Rw, Cl, s, gplane=(0.0, 0.0, 0.0)):
    """camera points -> shot world: X = Cl + s * Rw @ Xcv  (Rw: world-from-OpenCV-camera).
    gplane (a, b, c): the ground is z = a x + b y + c (level world on a slope); regions measure heights above it"""
    fr.Rw, fr.Cl, fr.s, fr.gp = Rw, Cl, s, gplane
    fr.X = Cl + s * (fr.P @ Rw.T)
    fr.NW = fr.N @ Rw.T
    fr.rng = np.linalg.norm(fr.P, axis=-1) * s


def pix_ray(fr, fpix, u, v):
    d = np.array([(u - fr.W / 2) / fpix, (v - fr.H / 2) / fpix, 1.0])
    return unit(fr.Rw @ d)


# ----------------------------------------------------------------------------------------------- regions
def colour_stats(img, m):
    px = img[m]
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    green = float(np.mean((hsv[:, 0] > 28) & (hsv[:, 0] < 95) & (hsv[:, 1] > 40) & (hsv[:, 2] > 25)))
    return px.mean(0), float(hsv[:, 2].mean()), float(hsv[:, 1].mean()), green


def region(fr, m, indoor):
    """measurements of one mask (bool, work res) -> dict, or None when too small / mostly sky / people"""
    area = m.sum()
    if area < 0.003 * m.size:
        return None
    pf = (m & fr.person).sum() / area
    if pf > 0.35:
        return None
    mm = m & ~fr.person
    sky = 1.0 - (mm & fr.valid).sum() / max(1, mm.sum())
    er = cv2.erode(mm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    use = er & fr.valid
    if use.sum() < 40:
        use = mm & fr.valid
    R = dict(mask=m, area=float(area / m.size), sky=float(sky))
    if sky > 0.6 or use.sum() < 30:
        R["cls"] = "sky"; return R
    ys, xs = np.nonzero(use)
    step = max(1, int(len(xs) / 4000))
    ys, xs = ys[::step], xs[::step]
    P = fr.X[ys, xs]; N = fr.NW[ys, xs]; rg = fr.rng[ys, xs]
    keep = np.isfinite(P).all(1)
    P, N, rg, ys, xs = P[keep], N[keep], rg[keep], ys[keep], xs[keep]
    rmed = float(np.median(rg))
    near = rg < 2.5 * rmed + 1.0                               # drop flying far edge pixels
    P, N, rg = P[near], N[near], rg[near]
    R["yx"] = (ys[near], xs[near]); R["rg_all"] = rg
    Pabs = P
    a_, b_, c_ = fr.gp
    if a_ or b_ or c_:                                           # heights above the sloped ground
        P = P.copy(); P[:, 2] -= a_ * P[:, 0] + b_ * P[:, 1] + c_
    col, val, sat, green = colour_stats(fr.img, use)
    R.update(P=P, Pabs=Pabs, N=N, rng=rmed, col=col, val=val, sat=sat, green=green, bbox_px=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    nz = N[:, 2]
    R["up_frac"] = float(np.mean(nz > 0.85)); R["vert_frac"] = float(np.mean(np.abs(nz) < 0.35))
    R["nvar"] = float(1.0 - np.linalg.norm(N.mean(0)))                # normal spread: leaves high, walls low
    n, c, rms = plane_fit(P)
    ext = np.percentile(P, 95, 0) - np.percentile(P, 5, 0)
    R["plane"] = (n, c, rms); R["ext"] = ext
    R["planar"] = rms / max(0.05, float(np.sort(ext)[1])) < 0.06 + 0.004 * rmed
    z5, z50, z95 = np.percentile(P[:, 2], [4, 50, 96])
    R["z"] = (float(z5), float(z50), float(z95))
    # texture (leaves, rough stone) from the picture
    g = cv2.cvtColor(fr.img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    R["tex"] = float(np.abs(lap[use]).mean())
    R["cls"] = classify(R, indoor)
    if os.environ.get("ENV_AUTO_DEBUG") and R["area"] > 0.02:
        print("    region f%d %s area %.3f rng %.1f z %s ext %s planar %s nz_med %.2f up %.2f vert %.2f green %.2f tex %.1f nvar %.2f bbox %s" % (
            fr.f, R["cls"], R["area"], R["rng"], rnd(R["z"], 2), rnd(R["ext"], 1), R["planar"], float(np.median(N[:, 2])), R["up_frac"],
            R["vert_frac"], R["green"], R["tex"], R["nvar"], [int(v / fr.k) for v in R["bbox_px"]]))
    return R


def classify(R, indoor):
    n = R["plane"][0]; nzp = abs(n[2]); z5, z50, z95 = R["z"]; r = R["rng"]
    tol = 0.15 + 0.025 * r                                      # how close to the floor counts as "on the floor"
    if r > (35 if indoor else 60):
        return "far"
    if R["up_frac"] > 0.55 and abs(z50) < tol:
        return "floor"
    if R["planar"] and nzp > 0.9 and abs(z50) < tol * 1.5:
        return "floor"
    if R["up_frac"] > 0.45 and z50 > tol and R["planar"] and nzp > 0.9:
        return "top"
    if R["planar"] and nzp > 0.85 and float(np.median(R["N"][:, 2])) < -0.5 and z50 > 1.6:
        return "ceiling"
    hx = max(R["ext"][0], R["ext"][1])
    if R["ext"][2] > 1.8 and hx < 0.3 * R["ext"][2] + 0.3 and z95 > 1.8:
        return "pole"                                            # tree trunk, post, pillar
    if R["planar"] and 0.45 < nzp <= 0.985 and z5 < tol * 2:
        return "slope"
    if not indoor and R["area"] > 0.05 and float(np.median(R["N"][:, 2])) > 0.55 and z5 < max(tol * 3, 0.15 * r):
        return "terrain"                                         # hillside, field, plant cover: the ground's shape
    if (R["green"] > 0.45 and not (R["planar"] and nzp < 0.3 and R["green"] < 0.6)) or (R["nvar"] > 0.3 and not R["planar"] and R["tex"] > 10):
        return "vegetation"
    if R["planar"] and nzp < 0.35 and (R["ext"][2] > 1.0 or max(R["ext"][0], R["ext"][1]) > 1.5):
        return "building" if (not indoor and z95 > 4.0 and r > 8) else "wall"
    if not R["planar"] and R["up_frac"] > 0.3 and z50 < tol * 2 and R["ext"][2] < 1.5 + 0.05 * r:
        return "terrain"
    h = z95 - max(0.0, z5)
    L = max(R["ext"][0], R["ext"][1])
    if False and not indoor and 1.0 < h < 2.4 and 3.2 < L < 7 and R["sat"] > 25 and R["tex"] < 18:
        return "vehicle"
    if not indoor and R["tex"] > 14 and h < 3:
        return "rock"
    return "object"


def name_regions(regs, fr, namer, indoor, min_p=0.3, max_area=0.2):
    """CLIP names each piece (namer.py) and corrects what shape and colour got wrong: a green-painted wall is a wall,
    not vegetation; a desk-shaped blob becomes a desk.  Sets R["label"] = (word, probability) and R["ptype"] (props.py)"""
    if namer is None:
        return regs
    # huge pieces (back wall, floor) are left to the geometry: their crop shows the whole room and CLIP names the room
    todo = [R for R in regs if R["cls"] not in ("sky", "floor", "terrain") and 0.004 <= R["area"] <= max_area and (R["mask"] & ~fr.person).any()]
    names = namer.name(fr.img, [R["mask"] & ~fr.person for R in todo], indoor=indoor)
    from namer import WORDS
    for R, best in zip(todo, names):
        word, p = best[0]; R["label"] = (word, round(p, 2))
        if p < min_p:
            continue
        wc, wt = WORDS[word]
        vertical = R["planar"] and abs(R["plane"][0][2]) < 0.35
        was = R["cls"]
        if wc == "person" and p < 0.6:                          # unsure: a window or a coat, not a missed person
            continue
        if wc == "person":                                       # a person the pose detector missed: not part of the set
            R["cls"] = "sky"
        elif was == "vegetation" and wc != "vegetation":
            R["cls"] = "wall" if (wc == "wall" or vertical) and R["area"] > 0.02 else ("floor" if wc == "floor" else "object")
        elif wc == "wall" and was == "object" and vertical:
            R["cls"] = "wall"
        elif wc == "vegetation" and was in ("object", "pole") and not indoor:
            R["cls"] = "vegetation"
        if wt and R["cls"] in ("object", "top", "vegetation"):
            R["ptype"] = wt
        if R["cls"] != was:
            print("    %s -> %s (CLIP: %s %.2f)" % (was, R["cls"], word, p))
    return regs


def split_planes(fr, R, indoor, max_planes=4):
    """a big mask that is not one surface (SAM often takes back wall + ceiling + floor of a room as one piece) ->
    one region per plane (RANSAC on points + normals), each split again into its connected pieces"""
    P, N, rg = R["Pabs"], R["N"], R["rg_all"]
    if len(P) < 400:
        return [R]
    rs = np.random.default_rng(1)
    rem = np.arange(len(P)); out = []
    for _ in range(max_planes):
        if len(rem) < 150:
            break
        best = None
        for it in range(120):
            i = rem[rs.integers(len(rem))]; n = N[i]; d = n @ P[i]
            ok = (np.abs(P[rem] @ n - d) < 0.04 + 0.02 * rg[rem]) & (N[rem] @ n > math.cos(math.radians(22)))
            if best is None or ok.sum() > best[0].sum():
                best = (ok, n)
        inl = rem[best[0]]
        if len(inl) < max(150, 0.12 * len(P)):
            break
        out.append(inl); rem = rem[~best[0]]
    if len(out) < 2 or sum(len(o) for o in out) < 0.6 * len(P):
        return [R]
    ys, xs = R["yx"]
    step = max(1, int(round(math.sqrt(R["area"] * fr.W * fr.H / max(1, len(P))))))
    k = np.ones((2 * step + 3, 2 * step + 3), np.uint8)
    res = []
    for inl in out:
        m = np.zeros((fr.H, fr.W), np.uint8); m[ys[inl], xs[inl]] = 1
        m = cv2.morphologyEx(cv2.dilate(m, k), cv2.MORPH_CLOSE, k) > 0
        m &= R["mask"]
        n_, lab = cv2.connectedComponents(m.astype(np.uint8))
        for c in range(1, n_):
            mc = lab == c
            if mc.sum() < 0.004 * m.size:
                continue
            r2 = region(fr, mc, indoor)
            if r2 is not None and r2["cls"] != "sky":
                res.append(r2)
    return res or [R]


def onto_fit_floor(R, C, gp):
    """MoGe's ground and the fit's floor can disagree (tilt / scale).  Keep the pixels (scale the region about the
    camera) and move a thing that stands on MoGe's ground so that its foot is on the fit's floor (z = 0), where the
    people stand.  Things not touching the ground keep the global scale.  Heights afterwards: above z = 0."""
    if R["cls"] in ("sky", "far", "ceiling") or "Pabs" not in R:
        return R
    Ph, Pa = R["P"], R["Pabs"]
    r = R["rng"]; tol = 0.15 + 0.025 * r
    k = 1.0
    if True:                                                     # MoGe's ground under its lowest part -> the fit floor
        low = Ph[:, 2] <= np.percentile(Ph[:, 2], 8)
        B = Pa[low].mean(0)
        Bg = np.array([B[0], B[1], gp[0] * B[0] + gp[1] * B[1] + gp[2]])     # MoGe's ground under its foot
        d = Bg - C; rb = np.linalg.norm(d); d = d / max(rb, 1e-9)
        if d[2] < -0.03:
            k = float(np.clip((-C[2] / d[2]) / rb, 0.5, 2.0))
    P2 = C + k * (Pa - C)
    if os.environ.get("ENV_AUTO_DEBUG"):
        print("    floor rescale %s area %.3f k %.2f gp %s" % (R["cls"], R["area"], k, rnd(gp, 3)))
    if k == 1.0 and not any(gp):
        return R
    R = dict(R, P=P2, Pabs=P2, kscale=k)
    R["plane"] = plane_fit(P2); R["ext"] = np.percentile(P2, 95, 0) - np.percentile(P2, 5, 0)
    R["z"] = tuple(float(v) for v in np.percentile(P2[:, 2], [4, 50, 96]))
    if R["cls"] not in ("wall", "building", "pole", "vegetation", "rock", "object", "vehicle", "top", "terrain", "slope"):
        return R
    if R["z"][0] < 2 * tol and k != 1.0:                         # stands on the floor now
        R["z"] = (min(R["z"][0], 0.0),) + R["z"][1:]
    return R


def adjacency(regs, H, W):
    sm = [cv2.resize(r["mask"].astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST) for r in regs]
    dl = [cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0 for m in sm]
    n = len(regs); A = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            if (dl[i] & (sm[j] > 0)).any():
                A[i, j] = A[j, i] = True
    return A


def merge_regions(regs, H, W):
    """pieces of one thing -> one region (walls in the same plane, touching vegetation / rocks / objects)"""
    if not regs:
        return regs
    A = adjacency(regs, H, W)
    par = list(range(len(regs)))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(regs)):
        for j in range(i + 1, len(regs)):
            a, b = regs[i], regs[j]
            if not A[i, j]:
                continue
            ca, cb = a["cls"], b["cls"]
            same = ca == cb or {ca, cb} <= {"wall", "building"} or {ca, cb} <= {"floor", "terrain"} or {ca, cb} <= {"rock", "object"}
            if not same:                                         # a small piece lying on a big wall / slope / floor (stones,
                big_, sm_ = (a, b) if a["area"] >= b["area"] else (b, a)            # patches, stains): part of it
                if big_["cls"] in ("wall", "building", "slope", "floor", "terrain") and sm_["area"] < 0.25 * big_["area"] \
                        and sm_["cls"] in ("object", "rock", "far", "wall", "slope", "top", "terrain", "floor"):
                    n_, m_ = big_["plane"][0], big_["plane"][1]
                    dd = np.median(np.abs((sm_["P"] - m_) @ n_))
                    if dd < 0.25 + 0.02 * sm_["rng"] and big_["planar"]:
                        par[find(j)] = find(i)
                continue
            if ca == "pole" and cb == "pole":                    # separate trunks / posts stay separate
                continue
            if ca in ("wall", "building", "slope", "top"):
                na, nb = a["plane"][0], b["plane"][0]
                if abs(na @ nb) < math.cos(math.radians(14)):
                    continue
                off = abs((b["plane"][1] - a["plane"][1]) @ na)
                if off > 0.3 + 0.03 * min(a["rng"], b["rng"]):
                    continue
            elif ca in ("vegetation", "rock", "object", "vehicle", "far"):
                lo_a, hi_a = np.percentile(a["P"], 5, 0), np.percentile(a["P"], 95, 0)
                lo_b, hi_b = np.percentile(b["P"], 5, 0), np.percentile(b["P"], 95, 0)
                gap = np.maximum(0, np.maximum(lo_a - hi_b, lo_b - hi_a))[:2].max()
                if gap > 0.4 + 0.04 * min(a["rng"], b["rng"]):
                    continue
            par[find(j)] = find(i)
    groups = {}
    for i in range(len(regs)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(regs[g[0]]); continue
        rs = [regs[i] for i in g]
        big = max(rs, key=lambda r: r["area"])
        m = np.zeros_like(big["mask"])
        for r in rs:
            m |= r["mask"]
        R = dict(big); R["mask"] = m; R["area"] = float(sum(r["area"] for r in rs))
        R["P"] = np.vstack([r["P"] for r in rs]); R["N"] = np.vstack([r["N"] for r in rs]); R["Pabs"] = np.vstack([r["Pabs"] for r in rs])
        R["yx"] = (np.concatenate([r["yx"][0] for r in rs]), np.concatenate([r["yx"][1] for r in rs]))
        R["rg_all"] = np.concatenate([r["rg_all"] for r in rs])
        R["bbox_px"] = [min(r["bbox_px"][0] for r in rs), min(r["bbox_px"][1] for r in rs), max(r["bbox_px"][2] for r in rs), max(r["bbox_px"][3] for r in rs)]
        R["plane"] = plane_fit(R["P"]); R["ext"] = np.percentile(R["P"], 95, 0) - np.percentile(R["P"], 5, 0)
        z = np.percentile(R["P"][:, 2], [4, 50, 96]); R["z"] = tuple(float(v) for v in z)
        R["green"] = float(np.average([r["green"] for r in rs], weights=[r["area"] for r in rs]))
        R["merged"] = len(rs)
        out.append(R)
    return out


# ----------------------------------------------------------------------------------------------- objects
def colour_for(cls, R, indoor):
    v = R.get("val", 128)
    if cls == "vegetation":
        return "tree"
    if cls in ("rock",):
        return "rock"
    if cls == "far":
        return "far"
    if cls in ("slope", "terrain"):
        return "tree" if R.get("green", 0) > 0.35 else "ground"
    if cls in ("wall", "building"):
        if not indoor and R.get("tex", 0) > 16:
            return "rock"
        return "light" if v > 170 else ("dark" if v < 60 else "set")
    return "light" if v > 185 else ("dark" if v < 55 else "set")


def oriented_box(P, C, min_depth_frac=0.0, min_depth=0.0):
    """footprint of points -> (centre xy, width, depth, yaw, axis); the hidden back is added away from the camera"""
    xy = P[:, :2]; m = np.median(xy, 0)
    cov = np.cov((xy - m).T) if len(xy) > 3 else np.eye(2)
    w_, v_ = np.linalg.eigh(cov)
    ax = v_[:, 1]; bx = v_[:, 0]
    a = (xy - m) @ ax; b = (xy - m) @ bx
    a0, a1 = np.percentile(a, [3, 97]); b0, b1 = np.percentile(b, [3, 97])
    W = max(0.1, a1 - a0); Dp = max(0.05, b1 - b0)
    Dn = max(Dp, min_depth_frac * W, min_depth)
    to_cam = (C[:2] - m) @ bx                                    # extend the depth on the side away from the camera
    if to_cam > 0:
        bc = b1 - Dn / 2
    else:
        bc = b0 + Dn / 2
    cen = m + ax * (a0 + a1) / 2 + bx * bc
    return cen, W, Dn, ax


def make_object(R, C, indoor, fr, fpix, name):
    """region -> env object dict (or None)"""
    cls = R["cls"]; P = R["P"]; z5, z50, z95 = R["z"]; r = R["rng"]
    tol = 0.15 + 0.025 * r
    col = colour_for(cls, R, indoor)
    src = {"cls": cls, "frame": int(fr.f), "px": [int(v / fr.k) for v in R["bbox_px"]], "range": round(r, 1), "area": round(R["area"], 3)}
    if R.get("label"):
        src["label"] = list(R["label"])                          # what CLIP sees (namer.py), for the review page
    pt = R.get("ptype")
    if cls in ("sky", "floor", "ceiling"):
        return None
    if cls in ("wall", "building") or (cls == "far" and R["planar"] and abs(R["plane"][0][2]) < 0.35):
        n = R["plane"][0].copy(); n[2] = 0; n = unit(n); u = np.array([-n[1], n[0], 0.0])
        cen = R["plane"][1]; off = float(n @ cen)
        s = P @ u; z = P[:, 2]
        s0, s1 = np.percentile(s, [1.5, 98.5])
        if s1 - s0 < 0.3:
            return None
        zb = 0.0 if (z5 < max(0.6, tol * 2) or cls == "far") else float(z5)
        th = 0.12 if indoor else (0.4 if col == "rock" else 0.3)
        nb = int(np.clip((s1 - s0) / 0.25, 6, 28)); edges = np.linspace(s0, s1, nb + 1)
        top = []
        for i in range(nb):
            q = z[(s >= edges[i]) & (s <= edges[i + 1])]
            top.append(np.percentile(q, 95) if len(q) >= 8 else np.nan)
        top = np.array(top, float)
        if np.isnan(top).all():
            return None
        okb = ~np.isnan(top); top = np.interp(np.arange(nb), np.nonzero(okb)[0], top[okb])
        from scipy.ndimage import median_filter
        top = median_filter(top, size=min(5, nb), mode="nearest")
        top = np.maximum(top, zb + 0.2)
        hmax = float(top.max())
        if cls == "building" and np.ptp(top) < 0.25 * (hmax - zb):
            cen2 = n * off + u * (s0 + s1) / 2
            away = n if (cen2[:2] - C[:2]) @ n[:2] > 0 else -n
            dep = float(np.clip(0.6 * (s1 - s0), 4, 15))
            pos = cen2 + away * dep / 2
            return {"name": name, "type": "building", "pos": rnd([pos[0], pos[1], zb], 2), "size": rnd([s1 - s0, hmax - zb, dep], 2),
                    "yaw": round(yaw_front(u, pos, C), 1), "colour": "far" if r > 40 else "set", "_src": src}
        if np.ptp(top) < max(0.2, 0.12 * (hmax - zb)):        # flat top: a plain wall box
            cen2 = n * off + u * (s0 + s1) / 2
            away = n if (cen2[:2] - C[:2]) @ n[:2] > 0 else -n
            pos = cen2 + away * th / 2
            return {"name": name, "type": "wall" if cls != "far" else "box", "pos": rnd([pos[0], pos[1], zb], 2),
                    "size": rnd([s1 - s0, float(np.median(top)) - zb, th], 2), "yaw": round(yaw_front(u, pos, C), 1),
                    "colour": col, "_src": src}
        # outline with the real (sloped / stepped) top
        prof = np.c_[(edges[:-1] + edges[1:]) / 2, top].astype(np.float32)
        prof[0, 0], prof[-1, 0] = s0, s1
        simp = cv2.approxPolyDP(prof.reshape(-1, 1, 2), max(0.08, 0.04 * (hmax - zb)), False).reshape(-1, 2)
        poly = [(s0, zb), (s1, zb)] + [tuple(p) for p in simp[::-1]]
        base = n * off
        gp = fr.gp
        pts3 = []
        for a, b in poly:                                        # heights above the (sloped) ground -> world z
            q = base + u * a
            pts3.append(rnd(q + np.array([0, 0, b + gp[0] * q[0] + gp[1] * q[1] + gp[2]]), 3))
        return {"name": name, "type": "patch", "pts3": pts3, "thick": th, "colour": col, "_src": src}
    if cls == "slope":
        n, cen, _ = plane_fit(R["Pabs"])
        if n[2] < 0:
            n = -n
        cnts, _ = cv2.findContours(R["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        ap = cv2.approxPolyDP(cnt, 0.012 * cv2.arcLength(cnt, True), True).reshape(-1, 2)
        if len(ap) < 3:
            return None
        rmax = float(np.percentile(np.linalg.norm(R["Pabs"] - C, axis=1), 98)) * 1.2
        pts = []
        for u_, v_ in ap:
            d = pix_ray(fr, fpix, u_, v_)
            den = d @ n
            t = ((cen - C) @ n) / den if abs(den) > 1e-6 else -1
            if not (0 < t < rmax):
                t = rmax
            q = C + d * t
            pts.append(rnd(q, 3))
        tilt = math.degrees(math.acos(min(1.0, abs(n[2]))))
        dirn = math.degrees(math.atan2(-n[0], -n[1]))            # uphill direction, 0 = +y (away), 90 = +x
        src.update(tilt_deg=round(tilt, 1), uphill_deg=round(dirn, 0))
        return {"name": name, "type": "patch", "pts3": pts, "thick": 0.3, "colour": col, "_src": src}
    if cls == "top":
        ztop = float(np.median(P[:, 2]))
        cen, W, Dn, ax = oriented_box(P, C)
        u = np.array([ax[0], ax[1], 0.0])
        kind = "table" if 0.55 < ztop < 0.85 and W * Dn < 4 else ("counter" if 0.85 <= ztop < 1.15 else ("bed" if ztop < 0.6 and W > 1.4 and Dn > 1.0 else "box"))
        if pt in ("desk", "table", "counter", "bed", "sink", "stove"):
            kind = pt
        return {"name": name, "type": kind, "pos": rnd([cen[0], cen[1], 0.0], 2), "size": rnd([W, ztop, Dn], 2),
                "yaw": round(yaw_front(u, cen, C), 1), "colour": col, "_src": src}
    if cls == "pole":
        cen = np.median(P[:, :2], 0); hx = float(max(R["ext"][0], R["ext"][1]))
        zt = float(z95)
        if indoor or R.get("tex", 0) < 6:
            return {"name": name, "type": "pillar", "pos": rnd([cen[0], cen[1], 0.0], 2), "size": rnd([max(0.15, hx), zt, max(0.15, hx)], 2),
                    "yaw": 0.0, "colour": col, "_src": src}
        return {"name": name, "type": "tree", "pos": rnd([cen[0], cen[1], 0.0], 2), "size": rnd([max(2.0, 0.45 * zt), zt, max(2.0, 0.45 * zt)], 2),
                "yaw": 0.0, "colour": "tree", "_src": src}
    if cls == "terrain":                                         # uneven ground: bumps for the ground heightfield
        return {"_terrain": True, "P": P, "_src": src}
    # blobs: vegetation, rock, vehicle, object, far
    on_floor = z5 < max(0.5, tol * 2)
    zb = 0.0 if on_floor else float(z5)
    h = max(0.1, z95 - zb)
    mdf = {"vegetation": 0.7, "rock": 0.7, "vehicle": 0.4, "far": 0.5}.get(cls, 0.35)
    cen, W, Dn, ax = oriented_box(P, C, mdf, 0.0)
    u = np.array([ax[0], ax[1], 0.0])
    yaw = round(yaw_front(u, cen, C), 1)
    if max(W, h) < 0.25 or (not on_floor and R["area"] < 0.02 and cls != "vegetation"):   # specks, small floating bits
        return None
    if cls == "vegetation" and indoor and pt in (None, "plant_pot", "bush") and h < 2.5:
        return {"name": name, "type": "plant_pot", "pos": rnd([cen[0], cen[1], zb], 2), "size": rnd([W, h, Dn], 2), "yaw": yaw,
                "colour": "tree", "_src": src}
    if cls == "vegetation":
        if h > 3.0 or not on_floor:
            kind = "pine" if W < 0.45 * (z95) else "tree"
            if not on_floor:
                h = z95; zb = 0.0
            return {"name": name, "type": kind, "pos": rnd([cen[0], cen[1], 0.0], 2), "size": rnd([W, h, max(Dn, 0.8 * W)], 2),
                    "yaw": yaw, "colour": "tree", "_src": src}
        return {"name": name, "type": "bush", "pos": rnd([cen[0], cen[1], zb], 2), "size": rnd([W, h, Dn], 2), "yaw": yaw,
                "colour": "tree", "_src": src}
    if cls == "rock":
        return {"name": name, "type": "rock", "pos": rnd([cen[0], cen[1], zb], 2), "size": rnd([W, h, Dn], 2), "yaw": yaw,
                "colour": "rock", "_src": src}
    if cls == "vehicle":
        return {"name": name, "type": "car", "pos": rnd([cen[0], cen[1], 0.0], 2), "size": rnd([W, h, max(Dn, 1.6)], 2), "yaw": yaw,
                "colour": col, "_src": src}
    if cls == "far":                                             # distant hills / mountains / tree lines: stand on the ground
        if z5 > 0.5 * z95 and z5 > 15:                           # high up and thin: clouds, a far skyline sliver
            return None
        return {"name": name, "type": "box", "pos": rnd([cen[0], cen[1], 0.0], 1), "size": rnd([W, max(z95, 1.0), Dn], 1), "yaw": yaw,
                "colour": "far", "_src": src}
    # furniture / anything else standing there
    if on_floor:
        kind = "cabinet" if h > 0.6 else "box"
        if h > 1.5 and W < 1.6:
            kind = "bookshelf" if Dn < 0.7 else "cabinet"
    else:
        kind = "wall_cabinet" if indoor and zb > 1.2 else "box"
    import props
    if pt and pt in props.TYPES:                                 # CLIP's name wins over the guess from the size
        kind = pt
    o = {"name": name, "type": kind, "pos": rnd([cen[0], cen[1], zb], 2), "size": rnd([W, h, Dn], 2), "yaw": yaw, "colour": col, "_src": src}
    if not on_floor:
        o["abs_z"] = True
    return o


BODY = (5, 6, 11, 12, 13, 14)                                    # shoulders, hips, knees (not the ankles: the floor holds them)


def inside(o, P, pad=0.03):
    """which world points are inside an object (box: its turned footprint and height; patch: its outline as a slab)"""
    if "pts3" in o:
        Q = np.array(o["pts3"], float)
        lo, hi = Q.min(0) - [pad, pad, 0.12], Q.max(0) + [pad, pad, 0.12]
        return np.all((P >= lo) & (P <= hi), axis=1)
    if "pos" not in o or "size" not in o:
        return np.zeros(len(P), bool)
    x, y, z = o["pos"]; w, h, d = o["size"]; t = math.radians(float(o.get("yaw", 0.0)))
    dx_, dy_ = P[:, 0] - x, P[:, 1] - y
    lx = dx_ * math.cos(t) + dy_ * math.sin(t); ly = -dx_ * math.sin(t) + dy_ * math.cos(t)
    return (np.abs(lx) <= w / 2 + pad) & (np.abs(ly) <= d / 2 + pad) & (P[:, 2] >= z - pad) & (P[:, 2] <= z + h + pad)


def off_people(objs, data, frames, dx=0.0):
    """drop objects that a fitted person is inside (2+ of shoulders / hips / knees in any measured frame): a mask that
    took in a person or a big blob in front of them, never a real piece of the set"""
    try:
        import build_env as BE
        cam = BE.Cam(data)
        pts = [np.array([p["P"][j] for j in BODY]) - [dx, 0.0, 0.0] for f in frames for p in BE.people_at(data, cam, f, 0.0)]
    except Exception as e:
        print("  people test skipped (%s)" % e); return objs, []
    keep, gone = [], []
    for o in objs:
        if o["_src"]["cls"] not in ("floor", "terrain", "ceiling") and any(inside(o, P).sum() >= 2 for P in pts):
            gone.append(o["_src"]["cls"]); continue
        keep.append(o)
    if gone:
        print("  dropped %d object(s) that people are inside: %s" % (len(gone), ", ".join(gone)))
    return keep, gone


def px_iou(a, b):
    """overlap of two picture boxes [x0, y0, x1, y1] (intersection over union)"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def user_objects(env, tag, fr, C, f0):
    """the objects drawn on the clip page (Objects step, env_ai.json): each box at the distance MoGe measures inside it
    (the near side of what is there), with the type picked; build_env sizes it from the box (from_bbox "dist")"""
    out = []
    for i, o in enumerate(((env.get("shots") or {}).get(tag) or {}).get("objects", [])):
        b = o.get("bbox")
        if not b or len(b) != 4:
            continue
        x0, y0, x1, y1 = [int(round(v * fr.k)) for v in b]
        m = np.zeros((fr.H, fr.W), bool); m[max(0, y0):max(0, y1) + 1, max(0, x0):max(0, x1) + 1] = True
        m &= fr.valid & ~fr.person & ~fr.bars
        if m.sum() < 20:
            continue
        hd = np.hypot(fr.X[m][:, 0] - C[0], fr.X[m][:, 1] - C[1])
        d = float(np.percentile(hd, 30))                          # its front, not what shows through behind it
        out.append({"name": "drawn_%d" % i, "type": o.get("type") or "box", "bbox": [float(v) for v in b], "frame": int(f0),
                    "dist": round(max(0.3, d), 2), "colour": "orange" if o.get("colour_role") == "orange" else "set",
                    "_src": {"cls": "drawn", "frame": int(f0), "px": [int(v) for v in b], "what": o.get("what", ""), "area": 0.0}})
    return out


def obj_box(o):
    """rough world footprint (x0, x1, y0, y1, z1) of an object for duplicate tests"""
    if "pts3" in o:
        P = np.array(o["pts3"]); return P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(), P[:, 2].max()
    x, y, z = o["pos"]; w, h, d = o["size"]; r = math.hypot(w, d) / 2
    return x - r, x + r, y - r, y + r, z + h


def overlaps(a, b):
    A, B = obj_box(a), obj_box(b)
    ix = max(0, min(A[1], B[1]) - max(A[0], B[0])); iy = max(0, min(A[3], B[3]) - max(A[2], B[2]))
    sa = (A[1] - A[0]) * (A[3] - A[2]); sb = (B[1] - B[0]) * (B[3] - B[2])
    return ix * iy / max(1e-6, min(sa, sb))


def heightfield(gpts, indoor, max_n=40):
    """ground points (floor, sloped / rough ground, low plant cover) -> Gaussian bumps [[x, y, dz]] whose SUM
    (build_env Ground: sigma = falloff / 2) follows the measured ground; None when the ground is flat"""
    if indoor:
        return None
    P = np.vstack(gpts)
    P = P[np.hypot(P[:, 0], P[:, 1]) < 120]
    if len(P) < 300:
        return None
    rs = np.random.default_rng(0)
    if len(P) > 8000:
        P = P[rs.choice(len(P), 8000, replace=False)]
    rms_flat = float(np.sqrt(np.mean(P[:, 2] ** 2)))
    if rms_flat < 0.15:
        return None
    lo, hi = np.percentile(P[:, :2], 1, 0), np.percentile(P[:, :2], 99, 0)
    sp = max(1.5, math.sqrt(float(np.prod(np.maximum(hi - lo, 1.0))) / max_n))
    for _ in range(6):
        gx = np.arange(lo[0], hi[0] + sp, sp); gy = np.arange(lo[1], hi[1] + sp, sp)
        G = np.array([(x, y) for x in gx for y in gy])
        d2 = ((P[:, None, :2] - G[None]) ** 2).sum(-1)
        G = G[(d2 < sp * sp).sum(0) >= 12]
        if len(G) <= max_n:
            break
        sp *= 1.2
    if len(G) < 2:
        return None
    sig = sp
    A = np.exp(-((P[:, None, :2] - G[None]) ** 2).sum(-1) / (2 * sig * sig))
    lam = 0.02 * len(P)
    dz = np.linalg.solve(A.T @ A + lam * np.eye(len(G)), A.T @ P[:, 2])
    rms_fit = float(np.sqrt(np.mean((A @ dz - P[:, 2]) ** 2)))
    if rms_fit > 0.8 * rms_flat:
        return None
    hs = [rnd([g[0], g[1], z], 2) for g, z in zip(G, dz) if abs(z) > 0.03]
    return {"heights": hs, "falloff": round(2 * sig, 2), "rms_flat": round(rms_flat, 2), "rms_fit": round(rms_fit, 2), "n": len(hs)}


# ----------------------------------------------------------------------------------------------- gravity / slope
def gravity_slope(regs, n_ground, Rw):
    """vertical structures (walls) are upright: their normals are horizontal.  With a level camera roll this fixes the
    true vertical; the ground's slope is the angle between it and the ground normal (world coordinates)."""
    A = np.zeros((3, 3)); wsum = 0.0
    for R in regs:
        if R["cls"] in ("wall", "building") and R["planar"]:
            n = unit(R["plane"][0]); w = R["area"]
            A += w * np.outer(n, n); wsum += w
    if wsum < 0.05:
        return None
    ex = unit(Rw @ np.array([1.0, 0, 0]))                       # camera right is level (roll 0)
    A = A / wsum + 0.5 * np.outer(ex, ex)
    ev, V = np.linalg.eigh(A)
    if ev[1] < 0.08:                                            # not determined (one wall facing sideways)
        return None
    g = V[:, 0]
    if g[2] < 0:
        g = -g
    ang = math.degrees(math.acos(min(1.0, abs(g @ n_ground))))
    if ang > 30:
        return None
    t = n_ground - (n_ground @ g) * g                           # ground normal leans downhill
    dirn = math.degrees(math.atan2(-t[0], -t[1])) if np.linalg.norm(t) > 1e-6 else 0.0
    return {"slope_deg": round(ang, 1), "uphill_dir_deg": round(dirn, 0), "wall_weight": round(wsum, 2), "_g": g}


# ----------------------------------------------------------------------------------------------- sheet
class AutoCam:
    """the env_check camera, with per-frame rotations / positions from env_auto (pixel frame = the picture area)"""
    def __init__(self, W, H, f):
        self.W, self.H, self.f = W, H, f; self.R = {}; self.C = {}
    def project(self, P, fr):
        R, C = self.R[fr], self.C[fr]
        Xc = (np.atleast_2d(P) - C) @ R
        z = -Xc[:, 2]; zz = np.maximum(z, 1e-6)
        return np.c_[self.W / 2 + self.f * Xc[:, 0] / zz, self.H / 2 - self.f * Xc[:, 1] / zz], z
    def ray(self, u, v, fr):
        R, C = self.R[fr], self.C[fr]
        d = R @ np.array([(u - self.W / 2) / self.f, -(v - self.H / 2) / self.f, -1.0])
        return C, d / np.linalg.norm(d)


def mask_panel(src, fr, regs, objs):
    im = src.copy(); lay = im.copy(); H, W = im.shape[:2]
    for R in regs:
        m = cv2.resize(R["mask"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        lay[m] = CLS_COL.get(R["cls"], (200, 200, 200))
    pm = cv2.resize(fr.person.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    lay[pm] = (0, 0, 255)
    im = cv2.addWeighted(lay, 0.45, im, 0.55, 0)
    sc = max(1.0, W / 900.0)
    for R in regs:
        x0, y0, x1, y1 = [int(v / fr.k) for v in R["bbox_px"]]
        cv2.rectangle(im, (x0, y0), (x1, y1), CLS_COL.get(R["cls"], (200, 200, 200)), max(1, int(2 * sc)))
    for o in objs:
        s = o.get("_src") or {}
        if s.get("frame") != fr.f:
            continue
        x0, y0, x1, y1 = s["px"]
        t = "%s %s" % (o.get("name", ""), o.get("type", ""))
        cv2.putText(im, t, (x0 + 4, y0 + int(22 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, (0, 0, 0), int(4 * sc), cv2.LINE_AA)
        cv2.putText(im, t, (x0 + 4, y0 + int(22 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, (255, 255, 255), max(1, int(1.5 * sc)), cv2.LINE_AA)
    return im


# ----------------------------------------------------------------------------------------------- main
GROUP = {"wall": "wall", "building": "wall", "far": "wall", "rock": "object", "object": "object", "vehicle": "object", "top": "object"}
APPROVED = {"0529bfe27f2b1dab7f3ae164b6d05abd", "b5b6e9a3c5730842395b00d2f4b1841c", "ec349385598e67568611cec8540c4f90"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="clip names (Work_auto folder names)")
    ap.add_argument("--all", action="store_true", help="every clip in logs\\clip_list.txt that has a fitted Work_auto folder")
    ap.add_argument("--skip-done", action="store_true", help="skip clips that already have env_auto.json")
    ap.add_argument("--out", default=None, help="json to write (one clip only; default output\\<clip>\\env_auto.json)")
    ap.add_argument("--no-mid", action="store_true")
    ap.add_argument("--points", type=int, default=32, help="SAM points per side")
    ap.add_argument("--max-objects", type=int, default=30)
    ap.add_argument("--frame", default="fit", choices=["fit", "moge"], help="fit: build in the fit camera (lines up with env_check; MoGe ground becomes a slope); moge: MoGe-levelled camera (needs its camera hint)")
    ap.add_argument("--level", action="store_true", help="level the world with the walls (true vertical) and make the ground a slope; off by default: leaning retaining walls fool it")
    a = ap.parse_args()
    c = cfg()
    names = list(a.names)
    if a.all:
        with open(os.path.join(AGENT, "logs", "clip_list.txt"), encoding="utf-8") as fh:
            clip_lines = fh.read().splitlines()
        for ln in clip_lines:
            if "|" in ln and not ln.startswith("FILM"):
                nm = ln.split("|")[0].strip()
                if nm and nm not in names:
                    names.append(nm)
    names = [n for n in names if n not in APPROVED and os.path.exists(os.path.join(c["work_root"], n, "scene_data.json"))
             and os.path.exists(os.path.join(c["work_root"], n, "camera.json"))]
    if a.skip_done:
        names = [n for n in names if not os.path.exists(os.path.join(output_dir(), n, "env_auto.json"))]
    if not names:
        print("no clips to do (need Work_auto\\<clip>\\scene_data.json + camera.json)"); return
    t_load = time.time()
    models = Models(a.points)
    import namer as NM
    models.namer = NM.Namer() if NM.available() else None
    print("object names: %s" % ("CLIP (models/clip)" if models.namer else "from shape and colour only (CLIP not installed)"))
    print("models loaded on %s in %.1f s" % (models.dev, time.time() - t_load))
    bad = []
    for n in names:
        try:
            for tries in range(4):                               # the GPU is shared: on out-of-memory wait and retry
                try:
                    run_clip(n, models, a, c); break
                except models.torch.OutOfMemoryError:
                    if tries == 3:
                        raise
                    print("GPU out of memory - retrying in 20 s"); models.torch.cuda.empty_cache(); time.sleep(20)
        except Exception as e:                                   # one bad clip never stops a batch
            import traceback; traceback.print_exc()
            print("FAILED %s: %s" % (n, e)); bad.append(n)
    print("done %d clip(s)%s" % (len(names) - len(bad), (", failed: " + ", ".join(bad)) if bad else ""))
    if bad:
        sys.exit(1)


def run_clip(name, models, a, c):
    T0 = time.time()
    work = os.path.join(c["work_root"], name)
    print("== %s" % name)
    ci = jload(os.path.join(work, "clip_info.json")); data = jload(os.path.join(work, "scene_data.json"))
    camj = jload(os.path.join(work, "camera.json"))
    from run_pipeline import marks_dir
    pp = os.path.join(marks_dir(ci["video"], c), "pose.json")
    pose = jload(pp) if os.path.exists(pp) else {"people": []}
    W, Hp = ci["W"], ci["Hpic"]
    user_env = jload(os.path.join(marks_dir(ci["video"], c), "env_ai.json")) if os.path.exists(os.path.join(marks_dir(ci["video"], c), "env_ai.json")) else {}
    f_full = data["lens"] / data["sensor"] * W
    fov_x = 2 * math.degrees(math.atan(W / 2 / f_full))
    shots = camj.get("shots") or [{"shot": 1, "frames": [0, data["frames"] - 1], "ground": {}}]
    want = {}
    for s in shots:
        a0, a1 = s["frames"]
        want[s["shot"]] = [a0] + ([] if a.no_mid or a1 - a0 < 24 else [(a0 + a1) // 2])
    frames = read_frames_at(ci["video"], [f for v in want.values() for f in v])
    acam = AutoCam(W, Hp, f_full)
    out_shots, report, sheet_rows = {}, {}, []
    for s in shots:
        sid = s["shot"]; a0 = s["frames"][0]; dx = 200.0 * (sid - 1)
        g = s.get("ground") or {}
        Rp0, Cp0 = cam_mats(data["cam"][a0])
        h_c = float(Cp0[2]); p_c = float(90.0 - data["cam"][a0][3])
        objs, regs_by_frame, frs, terr, gpts = [], {}, {}, [], []
        rep = {"frames": want[sid], "fit_camera": {"height": round(h_c, 3), "pitch_down_deg": round(p_c, 2), "source": g.get("source")}}
        indoor = None
        # 1) measure every frame, fit its ground (prior: the fit camera's up)
        fits = {}
        for f in want[sid]:
            t1 = time.time()
            fr = analyse(models, frames[f][ci["y0"]:ci["y0"] + Hp], f, pose, ci, fov_x, free_fov=(f == a0))
            fr.t_start = t1
            Rp, Cp = cam_mats(data["cam"][f])
            up0 = unit(D @ (Rp.T @ np.array([0, 0, 1.0])))
            sub = (slice(None, None, 3), slice(None, None, 3))
            Ps, Ns = fr.P[sub].reshape(-1, 3), fr.N[sub].reshape(-1, 3)
            oks = (fr.valid & ~fr.person)[sub].reshape(-1)
            gfit = fit_ground(Ps, Ns, oks, up0, fr.feet)
            Cl_f = (Cp - Cp0) + np.array([0.0, 0.0, h_c])
            if gfit is not None:                                # sane? (MoGe is metric: height within x0.5..x2, tilt within 15 deg)
                hr = gfit["h"] / max(0.2, float(Cl_f[2]))
                ang = math.degrees(math.acos(min(1.0, float(gfit["n"] @ up0))))
                gfit["sane"] = bool(ang < 15 and gfit["frac"] > 0.15 and gfit["h"] > 0.45 and (0.5 < hr < 2.0 or gfit["feet_ok"] >= 3))   # feet on it: right plane, MoGe scale off
                gfit["score"] = gfit["inliers"] * (1 + gfit["feet_ok"]) * (1.0 if gfit["sane"] else 0.01)
                print("  f%d ground: h %.2f (fit cam %.2f) tilt from fit up %.1f deg, inliers %.2f, feet %d/%d%s" % (
                    f, gfit["h"], Cl_f[2], ang, gfit["frac"], gfit["feet_ok"], gfit["feet"], "" if gfit["sane"] else "  REJECTED"))
            frs[f] = fr; fits[f] = (gfit, Rp, Cp, up0, Cl_f)
        # 2) the shot's level: the frame that sees the most ground; carried to the others with the tracked turn
        cands = [f for f in want[sid] if fits[f][0] is not None and fits[f][0]["sane"]]
        if cands:
            fb = max(cands, key=lambda f: fits[f][0]["score"])
            gfit, Rp_b, _, _, Cl_b = fits[fb]
            Rb = ground_frame(gfit["n"], gfit["h"]) @ D          # world-from-Blender-camera at frame fb
            R0 = Rb @ (Rp0.T @ Rp_b).T
            fwd = R0 @ np.array([0, 0, -1.0]); yw = math.atan2(fwd[0], fwd[1])     # frame a0 looks along +y
            Rz = np.array([[math.cos(yw), -math.sin(yw), 0], [math.sin(yw), math.cos(yw), 0], [0, 0, 1]])
            R0 = Rz @ R0
            s0 = float(Cl_b[2]) / gfit["h"]
            n0 = D @ (R0.T @ np.array([0, 0, 1.0]))          # ground normal in frame a0's camera
            p_m = math.degrees(math.asin(max(-1, min(1, -n0[2]))))
            roll = math.degrees(math.atan2(n0[0], -n0[1]))
            h_m = gfit["h"] * h_c / float(Cl_b[2])               # MoGe's camera height over the ground (metres)
            rep["moge_camera"] = {"height": round(h_m, 3), "pitch_down_deg": round(p_m, 2), "roll_deg": round(roll, 2), "from_frame": fb,
                                  "ground_inliers": round(gfit["frac"], 2), "feet_on_plane": "%d/%d" % (gfit["feet_ok"], gfit["feet"]),
                                  "fov_free_deg": round(frs[a0].fov_free, 1) if frs[a0].fov_free else None}
            dp = p_m - p_c; hr = h_m / h_c
            rep["disagreement"] = {"pitch_deg": round(dp, 2), "height_ratio": round(hr, 3)}
            strong = gfit["frac"] > 0.6 or gfit["feet_ok"] >= 3          # MoGe ground well seen (weak fits mislead: hills, banks)
            rep["moge_ground_strong"] = bool(strong)
            if strong and (abs(dp) > 2.5 or abs(hr - 1) > 0.25):
                src_ = str(g.get("source") or "")
                hh = h_m if ("people" not in src_ or abs(hr - 1) > 0.35) else h_c
                rep["suggest_hint"] = {"height": round(hh, 2), "pitch_down_deg": round(p_m, 1)}
        else:                                                    # no ground seen: the fit's camera; scale from feet or 1
            R0 = Rp0.copy(); s0 = 1.0
            fr0 = frs[a0]; up0 = fits[a0][3]
            if len(fr0.feet):                                    # feet that agree with the fit camera height (+-35%)
                fh = -(fr0.feet @ up0); fh = fh[(fh > 0.65 * h_c) & (fh < 1.5 * h_c)]
                if len(fh):
                    s0 = h_c / float(np.median(fh))
            rep["moge_camera"] = None; rep["ground"] = "not seen - fit camera kept"
            print("shot %d: no usable ground in MoGe - using the fit's camera" % sid)
        gplane0 = (0.0, 0.0, 0.0)
        feet_scale, feet_mid = {}, None
        if a.frame == "fit":
            # MoGe's scale from the fitted people: their ankles' distance from the camera (fit) / MoGe's (same pixels)
            import build_env as BE
            bcam = BE.Cam(data)
            for f in want[sid]:
                try:
                    ppl = BE.people_at(data, bcam, f, 0.0)
                except Exception:
                    ppl = []
                Fk = np.array([p["P"][j] for p in ppl for j in (15, 16)]).reshape(-1, 3)
                if f == a0 and len(Fk):
                    feet_mid = Fk[:, :2].mean(0) - np.array([dx, 0.0])
                if len(Fk) >= 2 and len(frs[f].feet) >= 2:
                    rf = np.median(np.linalg.norm(Fk - fits[f][2], axis=1)); rm = np.median(np.linalg.norm(frs[f].feet, axis=1))
                    if rm > 1e-3:
                        feet_scale[f] = float(rf / rm)
            rep["feet_scale"] = {str(k_): round(v_, 3) for k_, v_ in feet_scale.items()}
        if a.frame == "fit" and not cands and a0 in feet_scale:
            s0 = feet_scale[a0]
        if a.frame == "fit" and cands:
            # build in the fit's own camera (what env_check and the render use): pixels line up whatever the tilt;
            # MoGe's ground becomes the set's ground there (a slope when the two disagree)
            gfit, Rp_b, _, up_b, Cl_b = fits[fb]
            R0 = Rp0.copy()
            s_up = float(Cl_b[2]) / (gfit["h"] / max(0.3, float(gfit["n"] @ up_b)))
            sb = feet_scale.get(fb, s_up)
            if not (0.6 * s_up < sb < 1.7 * s_up):
                sb = s_up
            s0 = feet_scale[a0] if (a0 in feet_scale and 0.6 * sb < feet_scale[a0] < 1.7 * sb) else sb
            Rw_b = (R0 @ (Rp0.T @ Rp_b)) @ D
            nw = Rw_b @ gfit["n"]; X0 = Cl_b + sb * (Rw_b @ (gfit["c"] * gfit["n"]))
            a_, b_ = -nw[0] / nw[2], -nw[1] / nw[2]; c_ = X0[2] - a_ * X0[0] - b_ * X0[1]
            if math.hypot(a_, b_) < math.tan(math.radians(2.0)):
                a_ = b_ = 0.0
            if feet_mid is not None:                             # the people stand at z = 0 (the fit): keep them there
                c_ -= a_ * feet_mid[0] + b_ * feet_mid[1] + c_
            if abs(c_) < 0.05:
                c_ = 0.0
            gplane0 = (float(a_), float(b_), float(c_))
            rep["ground_in_fit_world"] = {"slope_deg": round(math.degrees(math.atan(math.hypot(a_, b_))), 1),
                                          "slope_dir_deg": round(math.degrees(math.atan2(a_, b_)), 0) if (a_ or b_) else 0.0, "z": round(c_, 3)}
        def place_all(R0_, gplane):
            nonlocal indoor
            for f in want[sid]:
                fr = frs[f]; gfit, Rp, Cp, up0, Cl = fits[f]
                Rw = (R0_ @ (Rp0.T @ Rp)) @ D
                sc = s0
                if f != a0 and gfit is not None and gfit["sane"]:
                    sc = float(Cl[2]) / (gfit["h"] / max(0.3, float(gfit["n"] @ up0)))
                if f in feet_scale and 0.6 * s0 < feet_scale[f] < 1.7 * s0:
                    sc = feet_scale[f]
                to_world(fr, Rw, Cl, sc, gplane)
                acam.R[f] = Rw @ D; acam.C[f] = Cl + np.array([dx, 0, 0])
                if indoor is None:                               # indoor / outdoor: sky and far depth
                    skyf = 1.0 - fr.valid[~fr.bars].mean()      # the bars are not sky
                    farq = np.percentile(fr.rng[fr.valid], 90) if fr.valid.any() else 0
                    indoor = bool(skyf < 0.02 and farq < 18)
                    rep["indoor"] = indoor; rep["sky_frac"] = round(float(skyf), 3)
                regs = []
                for i in range(fr.n_masks):
                    R = region(fr, fr.lab == i, indoor)
                    if R is None or R["cls"] == "sky":
                        continue
                    if R["area"] > 0.04 and R["cls"] not in ("vegetation", "pole", "floor", "terrain") and \
                            (not R["planar"] or R["cls"] in ("object", "far", "building", "top")):
                        regs += split_planes(fr, R, indoor)
                    else:
                        regs.append(R)
                regs = merge_regions(regs, fr.H, fr.W)
                regs_by_frame[f] = name_regions(merge_regions(regs, fr.H, fr.W), fr, getattr(models, "namer", None), indoor)
        place_all(R0, gplane0)
        # walls are upright: when they say the ground the people walk on is sloped, level the world (z = up) and
        # make the ground a slope instead (build_env Ground slope_deg / slope_dir_deg)
        gs = gravity_slope(regs_by_frame[a0], np.array([0, 0, 1.0]), acam.R[a0] @ D)
        rep["gravity_slope"] = {k_: v_ for k_, v_ in gs.items() if k_ != "_g"} if gs else None
        slope = None
        if a.frame == "moge" and a.level and gs and 3.0 <= gs["slope_deg"] <= 25.0 and gs["wall_weight"] >= 0.15 and not indoor:
            g = gs["_g"]; ax = np.cross(g, [0, 0, 1.0]); sa = np.linalg.norm(ax); ca = float(g[2])
            ax = ax / max(sa, 1e-12); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rg = np.eye(3) + sa * K + (1 - ca) * K @ K                # rotation taking g to +z
            R0 = Rg @ R0
            fwd = R0 @ np.array([0, 0, -1.0]); yw = math.atan2(fwd[0], fwd[1])
            Rz = np.array([[math.cos(yw), -math.sin(yw), 0], [math.sin(yw), math.cos(yw), 0], [0, 0, 1]])
            R0 = Rz @ R0; Rt = Rz @ Rg
            # the old ground plane (z = 0) seen from the levelled camera (rotation about the camera C)
            Cc = np.array([0, 0, h_c]); n2 = Rt @ np.array([0, 0, 1.0]); p0 = Cc + Rt @ (-Cc)
            a_, b_ = -n2[0] / n2[2], -n2[1] / n2[2]; c_ = p0[2] - a_ * p0[0] - b_ * p0[1]
            slope = {"slope_deg": round(math.degrees(math.atan(math.hypot(a_, b_))), 1),
                     "slope_dir_deg": round(math.degrees(math.atan2(a_, b_)), 0), "z": round(c_, 3),
                     "origin": rnd([-c_ * a_ / (a_ * a_ + b_ * b_ + 1e-12), -c_ * b_ / (a_ * a_ + b_ * b_ + 1e-12)], 3)}   # z = 0 there
            n0 = D @ (R0.T @ np.array([0, 0, 1.0]))
            p_g = math.degrees(math.asin(max(-1, min(1, -n0[2]))))
            rep["levelled"] = dict(slope, camera_pitch_down_deg=round(p_g, 2))
            if abs(p_g - p_c) > 2.5:
                rep["suggest_hint"] = {"height": round(h_c - c_, 2), "pitch_down_deg": round(p_g, 1), "note": "pitch over level (gravity); ground is sloped"}
            else:
                rep.pop("suggest_hint", None)
            place_all(R0, (a_, b_, c_))
        for f in want[sid]:
            fr = frs[f]; gfit, Rp, Cp, up0, Cl = fits[f]; t1 = fr.t_start
            fpix = f_full * fr.k
            regs = regs_by_frame[f]
            Cw = Cl.copy()
            k = 0
            for R in sorted(regs, key=lambda r: -r["area"]):
                ex = R.get("ext", np.zeros(3))
                if R["cls"] in ("floor", "terrain", "slope") or (R["cls"] == "vegetation" and R["area"] > 0.06 and ex[2] < 0.3 * max(ex[0], ex[1])
                                                                 and R["z"][0] < 1.0 + 0.03 * R["rng"]):
                    gpts.append(R["P"])                          # ground shape (heightfield)
                    if R["cls"] == "vegetation":
                        R["cls"] = "terrain"
                if a.frame == "fit":
                    R = onto_fit_floor(R, Cw, fr.gp)
                gp_keep = fr.gp
                if a.frame == "fit":
                    fr.gp = (0.0, 0.0, 0.0)                      # objects are now in the fit world (flat floor z = 0)
                o = make_object(R, Cw, indoor, fr, fpix, "%s%d_%d" % (R["cls"][:4], f, k))
                fr.gp = gp_keep
                k += 1
                if o is None:
                    continue
                if o.get("_terrain"):
                    terr.append(o["P"]); continue
                if f != a0 and R["cls"] in ("wall", "building"):  # the same wall seen again after a pan: one longer wall
                    merged = False
                    for qi, q in enumerate(objs):
                        Rq = q.get("_R")
                        if Rq is None or Rq["cls"] not in ("wall", "building"):
                            continue
                        nq, nr = Rq["plane"][0], R["plane"][0]
                        if abs(nq @ nr) < math.cos(math.radians(15)) or abs((R["plane"][1] - Rq["plane"][1]) @ nq) > 0.6 + 0.04 * R["rng"]:
                            continue
                        Rm = dict(Rq); Rm["P"] = np.vstack([Rq["P"], R["P"]]); Rm["Pabs"] = np.vstack([Rq["Pabs"], R["Pabs"]])
                        Rm["N"] = np.vstack([Rq["N"], R["N"]]); Rm["area"] = Rq["area"] + R["area"]
                        Rm["plane"] = plane_fit(Rm["P"]); Rm["ext"] = np.percentile(Rm["P"], 95, 0) - np.percentile(Rm["P"], 5, 0)
                        Rm["z"] = tuple(float(v) for v in np.percentile(Rm["P"][:, 2], [4, 50, 96]))
                        frq = frs[q["_src"]["frame"]]; gk = frq.gp
                        if a.frame == "fit":
                            frq.gp = (0.0, 0.0, 0.0)
                        o2 = make_object(Rm, Cw, indoor, frq, f_full * frq.k, q["name"])
                        frq.gp = gk
                        if o2 is not None:
                            o2["_R"] = Rm; o2["_src"]["area"] = round(Rm["area"], 3); objs[qi] = o2; merged = True
                        break
                    if merged:
                        continue
                if f != a0 and any(overlaps(o, q) > 0.35 and (GROUP.get(o["_src"]["cls"], o["_src"]["cls"]) == GROUP.get(q["_src"]["cls"], q["_src"]["cls"])) for q in objs):
                    continue
                o["_R"] = R
                objs.append(o)
            rep.setdefault("time_s", {})[str(f)] = {"moge": round(fr.t_moge, 2), "sam": round(fr.t_sam, 2), "total": round(time.time() - t1, 2), "masks": fr.n_masks}
            print("shot %d f%d: %d masks -> %d regions (%s), %.1f s" % (sid, f, fr.n_masks, len(regs),
                  ", ".join("%s %d" % (k_, sum(1 for r in regs if r["cls"] == k_)) for k_ in sorted({r["cls"] for r in regs})), time.time() - t1))
        # drop small leftovers inside bigger objects, keep the biggest ones
        objs.sort(key=lambda o: -o["_src"]["area"])
        keep = []
        for o in objs:
            if any(overlaps(o, q) > 0.8 and o["_src"]["area"] < 0.3 * q["_src"]["area"] and q.get("type") != "patch" for q in keep):
                continue
            keep.append(o)
        keep, rep["dropped_on_people"] = off_people(keep, data, want[sid], dx)
        keep = keep[:a.max_objects]
        mine = user_objects(user_env, "_s%d" % sid if len(shots) > 1 else "", frs[a0], fits[a0][4], a0)
        if mine:                                                 # the boxes drawn on the clip page win over what is found there
            keep = [o for o in keep if not any(px_iou(o["_src"]["px"], u["bbox"]) > 0.3 for u in mine)] + mine
            print("  %d object(s) drawn on the clip page" % len(mine))
        for o in keep:
            o.pop("_R", None)
        for i, o in enumerate(keep):
            o["name"] = "%s_%d" % (o["_src"]["cls"], i)
        ground = {"size": 30 if indoor else 300, "colour": "ground"}
        if slope:
            ground.update(slope_deg=slope["slope_deg"], slope_dir_deg=slope["slope_dir_deg"], origin=slope["origin"])
        elif any(gplane0) and a.frame == "fit-slope":            # (unused) MoGe's ground in the fit world as the set's ground
            ga, gb, gc = gplane0
            if gc:
                ground["z"] = round(gc, 3)
            if ga or gb:
                ground.update(slope_deg=round(math.degrees(math.atan(math.hypot(ga, gb))), 2),
                              slope_dir_deg=round(math.degrees(math.atan2(ga, gb)), 1), origin=[0, 0])
            for o in keep:                                       # "pos" z: heights above the ground -> build_env's terms
                if "pos" in o:
                    if o.get("abs_z"):
                        o["pos"][2] = round(o["pos"][2] + ga * o["pos"][0] + gb * o["pos"][1] + gc, 2)
                    else:
                        o["pos"][2] = round(o["pos"][2] + gc, 2)
        hf = heightfield(gpts, indoor) if gpts else None
        if hf:                                                   # rolling / sloping ground (build_env Ground "heights")
            ground["heights"], ground["falloff"] = hf["heights"], hf["falloff"]
            rep["heightfield"] = {k_: hf[k_] for k_ in ("rms_flat", "rms_fit", "n")}
            for o in keep:
                if o["_src"]["cls"] == "slope":
                    o["off"] = True                             # the heightfield already has it
        out_shots[str(sid)] = {"ground": ground, "objects": keep}
        report[str(sid)] = rep
        print("shot %d: fit camera h %.2f pitch %.1f | MoGe %s | %s | slope %s | %d objects" % (
            sid, h_c, p_c, json.dumps(rep.get("moge_camera")), json.dumps(rep.get("suggest_hint", "camera agrees" if rep.get("moge_ground_strong", True) else "MoGe ground weak - no hint")),
            json.dumps(rep.get("gravity_slope")), len(keep)))
        # sheet rows
        import build_env
        meshes = build_env.build(out_shots[str(sid)], acam, a0, dx, "_s%d" % sid)
        for f in want[sid]:
            src = frames[f][ci["y0"]:ci["y0"] + Hp]
            pal = data["palette"]
            ov = build_env.overlay(meshes, acam, src, f, pal, 0.6)
            solid = build_env.overlay(meshes, acam, np.full_like(src, 225), f, pal, 1.0)
            mp = mask_panel(src, frs[f], regs_by_frame[f], keep)
            for im in (mp, ov, solid):
                cv2.putText(im, "shot %d f%d" % (sid, f), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            sheet_rows.append(np.hstack([mp, ov, solid]))
    sky = [205, 207, 210]
    if len(shots) == 1:
        env = dict(out_shots["1"]); env["sky"] = sky
    else:
        env = {"sky": sky, "shots": out_shots}
    env["_auto"] = {"tool": "env_auto.py (MoGe-2 vitl-normal + SAM 2.1 hiera-large)", "fov_x_deg": round(fov_x, 2), "shots": report,
                    "time_s": round(time.time() - T0, 1)}
    outp = (a.out if a.out and len(a.names) <= 1 else None) or os.path.join(output_dir(), name, "env_auto.json")
    if os.path.basename(outp).lower() == "env_claude.json":
        raise SystemExit("env_auto never writes env_claude.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as fh:
        fh.write(dumps_env(env))
    sheet = np.vstack(sheet_rows)
    scl = min(1.0, 2700.0 / sheet.shape[1])
    if sheet.shape[0] * scl > 9000:
        scl = 9000.0 / sheet.shape[0]
    jp = os.path.join(work, "env_auto.jpg")
    imwrite_u(jp, cv2.resize(sheet, None, fx=scl, fy=scl, interpolation=cv2.INTER_AREA), 86)
    print("%s\n%s\ntotal %.1f s" % (outp, jp, time.time() - T0))


def dumps_env(env):
    """one object per line (easy to edit by hand)"""
    def objs(L, ind):
        return "[\n" + ",\n".join(ind + json.dumps(o, ensure_ascii=False) for o in L) + "\n" + ind[:-1] + "]"
    def shot(se, ind):
        parts = []
        for k, v in se.items():
            parts.append('%s"%s": %s' % (ind, k, objs(v, ind + " ") if k == "objects" else json.dumps(v, ensure_ascii=False)))
        return "{\n" + ",\n".join(parts) + "\n" + ind[:-1] + "}"
    if "shots" in env:
        top = []
        for k, v in env.items():
            if k == "shots":
                top.append(' "shots": {\n' + ",\n".join('  "%s": %s' % (sk, shot(sv, "   ")) for sk, sv in v.items()) + "\n }")
            else:
                top.append(' "%s": %s' % (k, json.dumps(v, ensure_ascii=False)))
        return "{\n" + ",\n".join(top) + "\n}\n"
    return shot(env, " ") + "\n"


if __name__ == "__main__":
    main()

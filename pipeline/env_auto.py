# -*- coding: utf-8 -*-
"""Automatic set proposal from real 3D measurements (run this before writing env_claude.json by hand).

  python pipeline\\env_auto.py <clip> [--out FILE] [--no-mid] [--points 32]

For the first and middle frame of every shot:
  MoGe-2 (microsoft/MoGe, models\\moge2\\model.pt): metric point map + normals, with the pipeline's own field of view
     (so the set lines up with the render camera); its free field-of-view estimate is reported too.
  SAM 3 (models\\sam3\\sam3.pt): the people's masks (from their boxes in
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
    for p in (os.path.join(TP, "MoGe"),):
        if p not in sys.path:
            sys.path.insert(0, p)


class Models:
    def __init__(self, points_per_side=32, device=None, config=None, annotations_only=False):
        setup()
        import torch
        from common import torch_device
        device = torch_device()                              # the NVIDIA card only
        self.torch, self.dev = torch, device
        self.annotations_only = annotations_only
        self.moge = None
        if not annotations_only:
            from moge.model.v2 import MoGeModel
            self.moge = MoGeModel.from_pretrained(os.path.join(MODELS, "moge2", "model.pt")).to(device).eval()
        self.pps = points_per_side
        self.config = dict(config or cfg())
        self.scene_cache = None
        self.seg = None                                      # what each piece is: Mask2Former ADE20K (semseg.py)
        try:
            import semseg
            if not annotations_only and semseg.available():
                self.seg = semseg.Segmenter(device)
        except Exception as e:
            print("segmentation model not used (%s) - pieces are named by CLIP" % str(e)[:200])

    def moge_run(self, rgb, fov_x=None):
        if self.moge is None:
            from moge.model.v2 import MoGeModel
            self.moge = MoGeModel.from_pretrained(os.path.join(MODELS, 'moge2', 'model.pt')).eval()
        self.moge.to(self.dev)
        t = self.torch.tensor(rgb, dtype=self.torch.float32, device=self.dev).permute(2, 0, 1) / 255.0
        with self.torch.inference_mode():
            out = self.moge.infer(t, fov_x=fov_x, use_fp16=True)                   # half precision on the GPU
        return {k: v.float().cpu().numpy() for k, v in out.items() if hasattr(v, "cpu")}

    def _release_scene_gpu(self):
        # SAM 3 runs in a different Python/CUDA runtime. Release resident geometry and semantic weights first.
        if self.moge is not None: self.moge.to("cpu")
        if self.seg is not None:
            self.seg.model.to("cpu")
        self.torch.cuda.empty_cache()

    def _sam_scene(self, rgb, automatic=False, boxes=(), **options):
        from sam3_scene import segment
        return segment(rgb, self.config, automatic=automatic, boxes=boxes,
                       cache_dir=self.scene_cache, before_inference=self._release_scene_gpu, **options)

    def sam_auto(self, rgb, n=None, iou_t=.7, stab_t=.86, nms_t=.6, min_area=.002):
        if getattr(self, 'annotations_only', False): return []
        masks = self._sam_scene(rgb, automatic=True, points=n or self.pps,
                                iou=iou_t, stability=stab_t, nms=nms_t, min_area=min_area)['masks']
        return [{"segmentation": m, "area": int(m.sum())} for m in masks]

    def sam_box_mask(self, rgb, box):
        return self._sam_scene(rgb, boxes=[box])['box_masks'][0]

    def sam_boxes(self, rgb, boxes):
        if not len(boxes):
            return np.zeros(rgb.shape[:2], bool)
        return self._sam_scene(rgb, boxes=boxes)['box_masks'].any(0)

    def semantic(self, rgb):
        if self.seg is None:
            return None
        self.seg.model.to(self.dev)
        return self.seg(rgb)


# ----------------------------------------------------------------------------------------------- geometry helpers
from build_env import cam_mats  # noqa: E402  (one definition of the camera key -> rotation, position)
from common import box_iou as px_iou  # noqa: E402

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
    if pose.get("_held"):                                       # carried objects go with the people
        import held_objects
        for b in held_objects.boxes_at(pose["_held"], f):
            out.append([max(0, b[0] * k), max(0, (b[1] - y0) * k), min(W - 1, b[2] * k), min(H - 1, (b[3] - y0) * k)])
    if pose.get('_prop_timeline'):
        import prop_timeline
        for b in prop_timeline.boxes_at(pose['_prop_timeline'], {}, f):
            out.append([max(0,b[0]*k), max(0,(b[1]-y0)*k), min(W-1,b[2]*k), min(H-1,(b[3]-y0)*k)])
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


def analyse(models, bgr, f, pose, ci, fov_x, free_fov=False, shared=None):
    """MoGe + SAM at work resolution for one frame -> Frame (camera-space points, normals, masks)"""
    t0 = time.time()
    H0, W0 = bgr.shape[:2]
    k = min(1.0, 1024.0 / W0)
    Wk, Hk = int(round(W0 * k)), int(round(H0 * k))
    img = cv2.resize(bgr, (Wk, Hk), interpolation=cv2.INTER_AREA) if k < 1 else bgr.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fr = Frame(); fr.k, fr.W, fr.H, fr.img, fr.f = k, Wk, Hk, img, f
    fr.rgb = rgb                                                 # (the drawn boxes' outlines: SAM with the box as prompt)
    fr.bars = side_bars(img)                                     # black pillarbox bars: not part of the room
    fr.shared_metric = shared is not None
    if shared is None:
        o = models.moge_run(rgb, fov_x=fov_x)
    else:
        from metric_scene import resample_depth
        o = {"points": resample_depth(shared["depth"], shared["K"], (Hk,Wk)),
             "mask": cv2.resize(shared["valid"].astype(np.uint8),(Wk,Hk),interpolation=cv2.INTER_NEAREST)>0}
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
    persistent_masks = pose.get('_sam3_bundle')
    if persistent_masks is not None:
        from sam3_tracking import union_at
        pm = union_at(persistent_masks,f,Wk,Hk)
        import prop_timeline, held_objects
        prop_boxes = prop_timeline.boxes_at(pose.get('_prop_timeline', []), {}, f)
        if pose.get('_held'): prop_boxes += held_objects.boxes_at(pose['_held'], f)
        for b in prop_boxes:
            x0,y1,x1,y2 = [int(v) for v in (b[0]*k,(b[1]-ci['y0'])*k,b[2]*k,(b[3]-ci['y0'])*k)]
            pm[max(0,y1):min(Hk,y2+1),max(0,x0):min(Wk,x1+1)] = True
    else:
        pm = models.sam_boxes(rgb, boxes) if boxes else np.zeros((Hk, Wk), bool)
    box_m = np.zeros((Hk, Wk), bool)
    for b in boxes:
        box_m[int(b[1]):int(b[3]) + 1, int(b[0]):int(b[2]) + 1] = True
    if persistent_masks is None and pm.sum() < 0.25 * box_m.sum():
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
    fr.segm = getattr(models, "seg", None)
    if shared is not None and fr.segm is not None and list(shared["names"]) == fr.segm.names:
        fr.sem = cv2.resize(shared["labels"].astype(np.int16),(Wk,Hk),interpolation=cv2.INTER_NEAREST)
    else:
        fr.sem = models.semantic(rgb)      # class of every pixel (ADE20K)
    masks = sorted(masks, key=lambda m: -m["area"])
    lab = np.full((Hk, Wk), -1, np.int32)
    for i, m in enumerate(masks):                                # big first, smaller ones on top
        lab[m["segmentation"]] = i
    lab[fr.bars] = -1                                            # nothing is built from the bars
    fr.soft = soft_parts(fr) if fr.sem is not None else []
    for m, _ in fr.soft:                                         # the plants / curtains... are their own pieces: SAM's
        lab[m] = -1                                              # regions (a wall with a plant in front) lose them
    fr.lab = lab; fr.n_masks = len(masks)
    return fr


def split_by_class(fr, m, min_share=0.2, min_area=0.004):
    """one SAM region -> one mask per segmentation class that holds at least 20 % of it (SAM often takes a back
    wall and the curtain in front of it as one piece; the segmentation tells them apart).  Pieces too small to be
    built stay with the biggest class."""
    if getattr(fr, "sem", None) is None or not m.any():
        return [m]
    v = fr.sem[m]
    c = np.bincount(v.astype(np.int64), minlength=len(fr.segm.names))
    big = [k for k in np.argsort(-c) if c[k] >= min_share * len(v) and c[k] >= min_area * m.size]
    if len(big) < 2:
        return [m]
    out = [m & (fr.sem == k) for k in big]
    rest = m & ~np.isin(fr.sem, big)
    out[0] = out[0] | rest                                       # the leftover classes go with the biggest
    return out


SOFT_GROUPS = {"plant": "plant", "pot": "plant", "flower": "plant", "vase": "plant"}


def soft_parts(fr, min_area=0.003):
    """the soft / irregular things the segmentation finds (semseg ROLE "relief", and trees indoors) as connected pieces,
    straight from its map: a plant and its pot, flowers in a vase are one piece -> [(mask, class name)]"""
    import semseg
    names = fr.segm.names
    group = {}
    for k, n in enumerate(names):
        if semseg.role(n) in ("relief", "tree"):
            group.setdefault(SOFT_GROUPS.get(n, n), []).append(k)
    free = fr.valid & ~fr.person & ~fr.bars
    out = []
    ker = np.ones((5, 5), np.uint8)
    for gname, ks in group.items():
        m = np.isin(fr.sem, ks) & free
        if m.sum() < min_area * m.size:
            continue
        m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, ker)        # leaves with gaps: one plant
        n, cc, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_area * m.size:
                piece = (cc == i) & free
                name, _ = fr.segm.label(fr.sem, piece)
                out.append((piece, name or gname))
    return out


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


def sem_decide(R, name, share, indoor):
    """what a piece becomes from its segmentation class (semseg.ROLE) and its measured shape -> (cls, ptype, relief).
    The geometry still decides where it is unsure: a "floor" piece must lie low, a "wall" piece must stand upright."""
    import semseg
    cls = R["cls"]; z5, z50, z95 = R["z"]; tol = 0.15 + 0.025 * R["rng"]
    vertical = R["planar"] and abs(R["plane"][0][2]) < 0.35
    if name is None or share < 0.45:
        return cls, None, False
    r = semseg.role(name)
    big = R["area"] > 0.06 and cls in ("wall", "building", "floor", "terrain", "slope", "ceiling")
    if big and r not in ("person", "sky", "floor", "ground", "ceiling", "wall", "building", "far"):
        return cls, None, False                                  # a large surface stays one (a bookcase wall is a wall)
    if r == "person":
        return "sky", None, False                                 # a person the pose detector missed: not the set
    if r == "sky":
        return "sky", None, False
    if r in ("floor", "ground"):
        if cls in ("floor", "terrain", "slope"):
            return cls, None, False
        return ("floor" if abs(z50) < tol * 2 else cls), None, False
    if r == "ceiling":
        return ("ceiling" if z50 > 1.6 else cls), None, False
    if r in ("wall", "building"):
        if cls in ("wall", "building", "far"):
            return cls, None, False
        if vertical or R["area"] > 0.02:
            return ("building" if r == "building" and not indoor else "wall"), None, False
        if share >= 0.8 and cls in ("vegetation", "pole", "object", "rock"):
            return "sky", None, False                            # a scrap of a wall the big wall piece already has
        return cls, None, False
    if r == "far":
        return ("far" if cls not in ("floor", "terrain", "slope") else cls), None, False
    if r == "rock":
        return "rock", None, False
    if r == "tree":
        return ("vegetation", None, False) if not indoor else ("object", None, True)
    if r == "relief":
        return "object", None, True
    if r == "pillar":
        return "pole", "pillar", False
    if r == "car":
        return "vehicle", "car", False
    if cls == "top" and r in ("table", "desk", "counter", "bed", "sink", "stove"):
        return "top", r, False
    return "object", r, False


def sem_regions(regs, fr, indoor):
    """name every piece from the segmentation (fr.sem): R["label"] = (class, share), and its class / props type /
    measured shape (R["relief"]) from sem_decide"""
    for R in regs:
        if R.get("soft"):                                        # cut out by its class already (soft_parts)
            from prototypes import vegetation_kind
            prototype = vegetation_kind(R['label'][0], indoor)
            print('    %s: %s' % (R['label'][0], (prototype+' prototype') if prototype else 'measured shape'))
            continue
        m = R["mask"] & ~fr.person
        name, share = fr.segm.label(fr.sem, m) if m.any() else (None, 0.0)
        R["label"] = (name, round(share, 2))
        was = R["cls"]
        cls, pt, rel = sem_decide(R, name, share, indoor)
        R["cls"] = cls
        if pt:
            R["ptype"] = pt
        if rel:
            R["relief"] = True
        if cls != was or rel:
            print("    %s -> %s%s (%s %.2f)" % (was, cls, " measured shape" if rel else "", name, share))
    return regs


RELIEF_THICK = {"plant": 0.6, "flower": 0.6, "tree": 0.6, "palm": 0.6, "curtain": 0.08, "blind": 0.05, "sculpture": 0.5,
                "vase": 0.9, "pot": 0.9, "bag": 0.5, "basket": 0.8, "animal": 0.5, "towel": 0.1, "blanket": 0.15, "flag": 0.05}


def relief_object(fr, m, C, name, src, colour, label=None, kscale=1.0):
    """a piece built from its measured surface (relief.py) -> env object {"type": "mesh", "v", "f"} or None"""
    import relief
    ok = m & fr.valid & ~fr.person & ~fr.bars
    if ok.sum() < 30:
        return None
    P = C + kscale * (fr.X[ok] - C)
    view = P.mean(0) - C; side = unit(np.array([-view[1], view[0], 0.0]))
    a = P @ side
    width = float(np.percentile(a, 95) - np.percentile(a, 5))      # across the view, measured
    frac = RELIEF_THICK.get((label or "").strip(), 0.4)
    thick = float(np.clip(frac * width, 0.03, 1.2)) if frac >= 0.2 else float(np.clip(frac, 0.02, 0.3))
    mesh = relief.relief_mesh(fr.X, ok, C, thick, scale=kscale)
    if mesh is None:
        return None
    V, F = mesh
    src = dict(src, measured=True)
    return {"name": name, "type": "mesh", "v": V, "f": F, "colour": colour, "what": label or "", "_src": src}


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


def same_label_join(a, b):
    """two touching wall pieces the segmentation gives the same class (curtain folds, a door and its frame, wall
    strips SAM cut apart), close in depth: one piece whatever their surfaces' angle"""
    if a["cls"] not in ("wall", "building") or b["cls"] not in ("wall", "building"):
        return False
    la, lb = a.get("label") or (None, 0), b.get("label") or (None, 0)
    if not la[0] or la[0] != lb[0] or min(la[1], lb[1]) < 0.45:
        return False
    if abs(a["rng"] - b["rng"]) > 0.15 * min(a["rng"], b["rng"]) + 0.25:   # one in front of the other: two things
        return False
    lo_a, hi_a = np.percentile(a["P"], 5, 0), np.percentile(a["P"], 95, 0)
    lo_b, hi_b = np.percentile(b["P"], 5, 0), np.percentile(b["P"], 95, 0)
    gap = np.maximum(0, np.maximum(lo_a - hi_b, lo_b - hi_a))[:2].max()
    return gap < 0.25 + 0.03 * min(a["rng"], b["rng"])


def merge_regions(regs, H, W, join=None):
    """pieces of one thing -> one region (walls in the same plane, touching vegetation / rocks / objects).
    join(a, b): a different rule for which touching pieces are one (same_label_join after naming)"""
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
            if join is not None:
                if join(a, b):
                    par[find(j)] = find(i)
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
    from prototypes import vegetation_kind, measured_object
    prototype = vegetation_kind((R.get('label') or ('',))[0], indoor)
    if cls == 'vegetation' and prototype in (None, 'pine'):
        height = z95 - z5
        low, high = np.percentile(P[:, :2], [5, 95], axis=0)
        if height >= 2 and (high - low).max() >= 2.5 * height:
            prototype = 'forest'
        elif prototype is None:
            prototype = 'bush' if height < 2 else 'pine'
    if prototype:
        return measured_object(prototype, P, name, src)
    pt = R.get("ptype")
    if cls in ("sky", "floor", "ceiling"):
        return None
    if R.get("relief"):                                          # plants, curtains, statues...: their measured shape
        lab = (R.get("label") or (None,))[0]
        col_r = "tree" if lab in ("plant", "flower", "tree", "palm") else colour_for("object", R, indoor)
        o = relief_object(fr, R["mask"], C, name, src, col_r, lab, R.get("kscale", 1.0))
        if o is not None:
            return o
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
    if "pts3" in o or "v" in o:
        Q = np.array(o.get("pts3") or o["v"], float)
        lo, hi = Q.min(0) - [pad, pad, 0.12], Q.max(0) + [pad, pad, 0.12]
        return np.all((P >= lo) & (P <= hi), axis=1)
    if "pos" not in o or "size" not in o:
        return np.zeros(len(P), bool)
    x, y, z = o["pos"]; w, h, d = o["size"]; t = math.radians(float(o.get("yaw", 0.0)))
    dx_, dy_ = P[:, 0] - x, P[:, 1] - y
    lx = dx_ * math.cos(t) + dy_ * math.sin(t); ly = -dx_ * math.sin(t) + dy_ * math.cos(t)
    return (np.abs(lx) <= w / 2 + pad) & (np.abs(ly) <= d / 2 + pad) & (P[:, 2] >= z - pad) & (P[:, 2] <= z + h + pad)


def body_scale(ppl, bcam, fr, Cfit, f, joints=(5, 6, 11, 12)):
    """the depth model's scale from the fitted people's shoulders and hips: their distance in the fit / the depth
    model's distance at their pixels (+ about 0.1 m: the joint is inside the body, the depth is its front) -> ratio or
    None (fewer than 3 joints seen)"""
    rs = []
    for p in ppl:
        for j in joints:
            Pw = np.asarray(p["P"][j], float)
            uv, z = bcam.project(Pw, f)
            if z[0] <= 0.3:
                continue
            u, v = int(uv[0][0] * fr.k), int(uv[0][1] * fr.k)
            if not (2 <= u < fr.W - 2 and 2 <= v < fr.H - 2):
                continue
            win = fr.valid[v - 2:v + 3, u - 2:u + 3]
            if win.sum() < 5:
                continue
            rm = float(np.median(np.linalg.norm(fr.P[v - 2:v + 3, u - 2:u + 3][win], axis=-1))) + 0.1
            rf = float(np.linalg.norm(Pw - Cfit))
            if rm > 0.2:
                rs.append(rf / rm)
    return float(np.median(rs)) if len(rs) >= 3 else None


def wall_plane(o):
    """a wall object's (unit horizontal normal, point on it) or None"""
    if "pts3" in o:
        P = np.array(o["pts3"], float)
        a, b = P[0], P[1]
        t = np.array([b[0] - a[0], b[1] - a[1], 0.0])
        if np.linalg.norm(t) < 1e-6:
            return None
        n = unit(np.array([-t[1], t[0], 0.0]))
        return n, P.mean(0)
    if "pos" in o and "yaw" in o:
        y = math.radians(float(o["yaw"]))
        return np.array([-math.sin(y), math.cos(y), 0.0]), np.array(o["pos"], float)
    return None


def behind_people(o, pts, pad=0.25, most=1.5):
    """a big wall the people walk into (they come through a doorway in it, or the measured wall is a little short):
    moved back along its normal, away from the camera side the people are on, until it is behind them (at most 1.5 m)"""
    pl = wall_plane(o)
    if pl is None or not pts:
        return o
    n, c = pl
    P = np.vstack(pts)
    if not inside(o, P).any():
        return o
    d = (P - c) @ n
    side = -1.0 if np.median(d) < 0 else 1.0                     # the people's side of the wall
    need = float(np.max(-side * d)) + pad                        # how far the wall must go back past the deepest point
    if need <= 0 or need > most:
        return o
    sh = -side * n * need
    o = dict(o)
    if "pts3" in o:
        o["pts3"] = [rnd(np.array(q) + sh, 3) for q in o["pts3"]]
    else:
        o["pos"] = rnd(np.array(o["pos"]) + sh, 2)
    print("  wall %s moved %.2f m back: people walk where it was measured" % (o.get("name", ""), need))
    return o


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
        if o["_src"]["cls"] in ("wall", "building") and o["_src"].get("area", 0) >= 0.15:
            keep.append(behind_people(o, pts)); continue         # a big wall: kept, moved back if people walk in it
        if o["_src"]["cls"] not in ("floor", "terrain", "ceiling") and any(inside(o, P).sum() >= 2 for P in pts):
            gone.append(o["_src"]["cls"]); continue
        keep.append(o)
    if gone:
        print("  dropped %d object(s) that people are inside: %s" % (len(gone), ", ".join(gone)))
    return keep, gone


def px_inside(a, b):
    """how much of picture box a lies inside box b (0..1)"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ar = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / ar if ar > 0 else 0.0


def drawn_frames(env, shots):
    """{shot: {env_ai tag: absolute frame}} for the key frames the Objects step drew boxes on.  The tags are the
    regions' tags: "<shot tag>_<key frame within the shot, 3 digits>" ("_000", "_s2_014")"""
    import re
    out = {}
    one = len(shots) == 1
    for tag, sh in ((env or {}).get("shots") or {}).items():
        if not any(o.get("bbox") for o in (sh or {}).get("objects", [])):
            continue
        m = re.fullmatch(r"(?:_s(\d+))?_(\d+)", tag)
        if not m or (m.group(1) is None) != one:
            continue
        sid = 1 if one else int(m.group(1))
        s = next((x for x in shots if x["shot"] == sid), None)
        if s is None:
            continue
        f = s["frames"][0] + int(m.group(2))
        if f <= s["frames"][1]:
            out.setdefault(sid, {})[tag] = f
    return out


def user_objects(env, tag, fr, C, f0, first=0, indoor=True, fit=True, models=None):
    """the objects annotated on the clip page (Objects step, env_ai.json), every one of them, at the frame it was
    drawn on (that frame's camera and depth, so it lines up with the film there).  The description ("what") says
    what it is (describe.py):
      held by a person ("bag in his hand", "backpack")  -> not a set piece: held_objects.py puts it on the hand / back;
      a regular thing it names (door, pillar, table, car...) or the type picked -> that library model, at the
        distance MoGe measures inside its outline (the near side), sized from the box (build_env from_bbox "dist");
      anything else (a plant, curtain, statue, or a thing the description does not name) -> its measured surface
        (relief.py) inside its outline.
    The outline is SAM's cut of the box (the thing itself, not the wall behind it); colour words set the shade."""
    import props, describe
    out = []
    for i, o in enumerate(((env.get("shots") or {}).get(tag) or {}).get("objects", [])):
        b = o.get("bbox")
        if not b or len(b) != 4:
            continue
        d = describe.parse(o.get("what"), o.get("type"))
        if o.get('size') and d['mode'] != 'held': d['mode'] = 'model'
        what = (o.get("what") or o.get("type") or "").strip()
        if d["mode"] == "held":
            print("  annotation %r: held by a person - placed on the hand / back (held_objects)" % what)
            continue
        x0, y0, x1, y1 = [int(round(v * fr.k)) for v in b]
        m = np.zeros((fr.H, fr.W), bool); m[max(0, y0):max(0, y1) + 1, max(0, x0):max(0, x1) + 1] = True
        outline = o.get("outline")
        if outline:
            mask = np.zeros(m.shape, np.uint8)
            cv2.fillPoly(mask, [np.rint(np.asarray(outline) * fr.k).astype(np.int32)], 1)
            m &= mask.astype(bool)
        boxm = m & fr.valid & ~fr.bars
        m = boxm & ~fr.person
        if m.sum() < 20:                                          # hidden behind a person here: no depth of its own ->
            kind = d.get("kind") or props.norm_type(o.get("type"))   # a model standing on the floor under its box
            print("  annotation %r: hidden by a person in this frame - %s model on the floor under its box" % (what, kind))
            ob = {"name": "drawn_%d" % (first + i), "type": kind, "bbox": [float(v) for v in b], "frame": int(f0),
                  "z": 0.0, "colour": "set", "what": what, "_level": None,
                  "_src": {"cls": "drawn", "frame": int(f0), "px": [int(v) for v in b], "what": what, "area": 0.0,
                           "label": [d["label"], 1.0], "built": "model"}}
            role = o.get("role") or ("near" if o.get("colour_role") == "orange" else d.get("role"))
            if role:
                ob["role"] = role
            if o.get("facing") in ("left", "right", "away"):
                ob["facing"] = o["facing"]
            if o.get('size'): ob.update(size=o['size'], size_locked=True)
            out.append(ob); continue
        if not outline and models is not None and getattr(fr, "rgb", None) is not None:
            try:
                sm = models.sam_box_mask(fr.rgb, [x0, y0, x1, y1]) & m
                if sm.sum() >= 0.12 * m.sum():
                    m = sm
            except Exception as e:
                print("  annotation %r: no outline (%s) - the whole box is used" % (what, e))
        role = o.get("role") or ("near" if o.get("colour_role") == "orange" else d.get("role"))   # colour card meaning asked for
        colour = None
        src = {"cls": "drawn", "frame": int(f0), "px": [int(v) for v in b], "what": what, "area": 0.0,
               "label": [d["label"], 1.0], "built": d["mode"]}
        k = 1.0                                                   # the measured depth -> the fit's world (where the people
        if fit:                                                   # stand), as for every piece the set finds itself
            R = region(fr, m, indoor)
            if R is not None and R.get("cls") != "sky":
                k = onto_fit_floor(R, C, fr.gp).get("kscale", 1.0)
        V = fr.X[m] - C
        hd = np.hypot(V[:, 0], V[:, 1]) * k
        az = np.degrees(np.arctan2(V[:, 1], V[:, 0]))
        lvl = {"_level": int(o["level"]) if str(o.get("level") or "").isdigit() else None, "_C": [float(v) for v in C],
               "_reach": [float(np.percentile(hd, 30)), float(np.percentile(hd, 90))],
               "_az": [float(np.percentile(az, 2)), float(np.percentile(az, 98))]}   # for by_level
        if d["mode"] == "shape":
            lab = d["label"] or ""
            col = colour or ("tree" if any(w in lab for w in ("plant", "flower", "tree", "bush", "palm", "leaves", "hedge")) else "set")
            ro = relief_object(fr, m, C, "drawn_%d" % (first + i), src, col, lab.split()[-1] if lab else None, k)
            if ro is not None:
                ro.update(frame=int(f0), bbox=[float(v) for v in b], what=what, **lvl)
                if role:
                    ro["role"] = role
                print("  annotation %r: measured shape" % what)
                out.append(ro); continue
            d = dict(d, mode="model", kind=d.get("kind") or props.norm_type(o.get("type")))   # too little depth: a box
        dist = lvl["_reach"][0]                                   # its front, not what shows through behind it
        kind = d.get("kind") or props.norm_type(o.get("type"))
        print("  annotation %r: %s model, %.1f m from the camera" % (what, kind, dist))
        ob = {"name": "drawn_%d" % (first + i), "type": kind, "bbox": [float(v) for v in b], "frame": int(f0),
              "dist": round(max(0.3, dist), 2), "colour": colour or "set", "_src": src, **lvl}
        ob["what"] = what
        if role:
            ob["role"] = role
        if o.get("facing") in ("left", "right", "away"):
            ob["facing"] = o["facing"]
        if o.get('size'): ob.update(size=o['size'], size_locked=True)
        out.append(ob)
    return out


def people_points(data, frames, dx=0.0):
    """the fitted people's body points (shoulders / hips / knees) at these frames, in the set's world"""
    import build_env as BE
    cam = BE.Cam(data)
    return [np.array([p["P"][j] for j in BODY]) - [dx, 0.0, 0.0] for f in frames for p in BE.people_at(data, cam, f, 0.0)]


def push_back(o, s):
    """moved s times as far from the camera along its view: the picture shows it at the same place and size"""
    C = np.array(o["_C"])
    if o.get("type") == "mesh":
        o["v"] = [rnd(C + s * (np.array(q, float) - C), 3) for q in o["v"]]
    else:
        o["dist"] = round(float(o["dist"]) * s, 2)
    o["_reach"] = [r * s for r in o["_reach"]]


def by_level(objs, pts, gap=0.3, pad=0.4):
    """the distance levels set on the clip page (1 near .. 9 far): an object is moved back along the camera's view
    until it is behind every object of a smaller level in the shot.  The shot's biggest level (2 or more) is the back
    of the set: it also goes behind every person who walks inside its view on any frame of the shot."""
    lv = [o for o in objs if o.get("_level")]
    if not lv:
        return
    top = max(o["_level"] for o in lv)
    floor = 0.0
    for L in sorted({o["_level"] for o in lv}):
        back = floor
        for o in (q for q in lv if q["_level"] == L):
            C = np.array(o["_C"]); near = max(0.3, o["_reach"][0])
            need, why = (floor + gap, "behind level %d" % (L - 1)) if floor else (0.0, "")
            if L == top and L >= 2 and pts:
                P = np.vstack(pts); V = P - C
                a0, a1 = o["_az"]; az = np.degrees(np.arctan2(V[:, 1], V[:, 0]))
                inv = ((az - a0 + 3) % 360) <= ((a1 - a0) % 360 + 6)      # inside its view (3 degrees each side)
                if inv.any():
                    far = float(np.hypot(V[inv, 0], V[inv, 1]).max()) + pad
                    if far > need:
                        need, why = far, "behind the people"
            if need > near:
                push_back(o, need / near)
                print("  annotation %r (distance %d): moved back %.1f -> %.1f m, %s" % (o.get("what", ""), L, near, need, why))
            back = max(back, o["_reach"][1])
        floor = back


def obj_box(o):
    """rough world footprint (x0, x1, y0, y1, z1) of an object for duplicate tests"""
    if "pts3" in o or "v" in o:
        P = np.array(o.get("pts3") or o["v"]); return P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(), P[:, 2].max()
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
    def __init__(self, W, H, f, zoom=None):
        self.W, self.H, self.f = W, H, f; self.R = {}; self.C = {}
        self.zoom = zoom or None                                 # focal scale per frame (the lens zooms)
    def f_at(self, fr):
        return self.f * float(self.zoom[min(int(fr), len(self.zoom) - 1)]) if self.zoom else self.f
    def project(self, P, fr):
        R, C = self.R[fr], self.C[fr]; f = self.f_at(fr)
        Xc = (np.atleast_2d(P) - C) @ R
        z = -Xc[:, 2]; zz = np.maximum(z, 1e-6)
        return np.c_[self.W / 2 + f * Xc[:, 0] / zz, self.H / 2 - f * Xc[:, 1] / zz], z
    def ray(self, u, v, fr):
        R, C = self.R[fr], self.C[fr]; f = self.f_at(fr)
        d = R @ np.array([(u - self.W / 2) / f, -(v - self.H / 2) / f, -1.0])
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
    ap.add_argument('--annotations-only', action='store_true', help='Update drawn objects using saved cameras and people; preserve other background objects')
    ap.add_argument("--free-fov", action="store_true", help="also measure MoGe's own field of view on each shot's first "
                    "frame (a second MoGe run; the lens step already measures it)")
    ap.add_argument("--points", type=int, default=None, help="SAM points per side (default: config.json sam_points, 32)")
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
    pts = a.points or int(c.get("sam_points", 32))
    models = Models(pts, config=c, annotations_only=a.annotations_only)
    print("SAM: %dx%d points" % (pts, pts))
    import namer as NM
    models.namer = NM.Namer(models.dev) if (not a.annotations_only and models.seg is None and NM.available()) else None
    print("object names: %s" % ("user annotations (no semantic model)" if a.annotations_only else "Mask2Former ADE20K (models/mask2former_ade)" if models.seg is not None else
                                ("CLIP (models/clip)" if models.namer else "from shape and colour only (CLIP not installed)")))
    print("models loaded on %s in %.1f s" % (models.dev, time.time() - t_load))
    bad = []
    for n in names:
        try:
            for tries in range(4):                               # the GPU is shared: on out-of-memory wait and retry
                try:
                    if a.annotations_only: run_annotations(n, models, c)
                    else: run_clip(n, models, a, c)
                    break
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


def lens_fov(work):
    """MoGe's own field of view of the clip, as the lens step measured it (lens_auto.json): not measured again here"""
    try:
        mm = jload(os.path.join(work, "lens_auto.json"))["lens_mm"]
        return round(2 * math.degrees(math.atan(18.0 / float(mm))), 1)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def sample_set_frames(data, start, end, c, first_only=False, pose=None, picture=None):
    """Keep the usual views and add views when camera position, rotation or zoom changes."""
    if first_only or end - start < 24:
        return [start]
    selected = {start, (start + end) // 2}
    if not c.get("set_adaptive_frames", True):
        return sorted(selected)
    limit = max(2, int(c.get("set_max_frames", 5)))
    candidates = np.unique(np.linspace(start, end, min(33, end - start + 1), dtype=int))
    poses = {int(f): cam_mats(data["cam"][f]) for f in set(candidates) | selected}
    zoom = data.get("zoom") or [1.0] * (end + 1)
    masks = {}
    if pose is not None and picture is not None:
        width, height, y0 = picture
        for f in set(candidates) | selected:
            mask = np.zeros((64, 64), bool)
            for person in pose.get("people", []):
                box = person.get("boxes", {}).get(str(f))
                if box:
                    x0, ya, x1, yb = box[:4]
                    x0, x1 = np.clip(np.array([x0, x1]) * 64 / width, 0, 64).astype(int)
                    ya, yb = np.clip((np.array([ya, yb]) - y0) * 64 / height, 0, 64).astype(int)
                    mask[ya:yb, x0:x1] = True
            masks[int(f)] = mask
    def separation(f, g):
        rf, cf = poses[f]; rg, cg = poses[g]
        angle = math.degrees(math.acos(float(np.clip((np.trace(rf.T @ rg) - 1) / 2, -1, 1))))
        return max(np.linalg.norm(cf - cg) / 0.5, angle / 12.0,
                   abs(math.log(max(zoom[f], 1e-6) / max(zoom[g], 1e-6))) / math.log(1.15))
    while len(selected) < limit:
        hidden = np.logical_and.reduce([masks[g] for g in selected]) if masks else None
        scores = []
        for f in candidates:
            if f in selected:
                continue
            score = min(separation(int(f), g) for g in selected)
            # Occlusion masks share image coordinates only while the view stays nearly fixed.
            if hidden is not None and max(separation(int(f), g) for g in selected) < 0.25:
                score = max(score, float(np.mean(hidden & ~masks[int(f)])) / 0.03)
            scores.append((score, int(f)))
        if not scores:
            break
        score, frame = max(scores)
        if score < 1.0:
            break
        selected.add(frame)
    return sorted(selected)


def run_annotations(name, models, c):
    """Measure only drawn objects, preserving the existing unannotated set and human fit."""
    import copy
    from run_pipeline import marks_dir
    from scene_verify import load_evidence
    from sam3_scene import prefetch
    work=os.path.join(c['work_root'],name)
    data=jload(os.path.join(work,'scene_data.json'));ci=jload(os.path.join(work,'clip_info.json'))
    marks=marks_dir(ci['video'],c);pose=jload(os.path.join(marks,'pose.json'))
    user_path=os.path.join(marks,'env_ai.json')
    annotations=jload(user_path) if os.path.isfile(user_path) else {}
    env_path=os.path.join(output_dir(),name,'env_auto.json')
    prior=jload(env_path) if os.path.isfile(env_path) else {}
    env=copy.deepcopy(prior)
    shots=data.get('shots') or [{'shot':1,'frames':[0,data['frames']-1]}]
    drawn=drawn_frames(annotations,shots)
    frames=read_frames_at(ci['video'],[f for tags in drawn.values() for f in tags.values()])
    models.scene_cache=os.path.join(work,'sam3_scene_cache')
    requests=[]
    for sid,tags in drawn.items():
        for tag,f in tags.items():
            picture=frames[f][ci['y0']:ci['y0']+ci['Hpic']]
            k=min(1.,1024./picture.shape[1])
            if k<1:picture=cv2.resize(picture,(round(picture.shape[1]*k),round(picture.shape[0]*k)),interpolation=cv2.INTER_AREA)
            rgb=cv2.cvtColor(picture,cv2.COLOR_BGR2RGB)
            for obj in annotations['shots'][tag].get('objects',[]):
                if obj.get('bbox') and not obj.get('outline'):
                    requests.append({'rgb':rgb,'automatic':False,'boxes':[[int(round(v*k)) for v in obj['bbox']]]})
    count=prefetch(requests,c,models.scene_cache,models._release_scene_gpu)
    print('  annotated object segmentation: %d uncached box requests'%count,flush=True)
    bundles,_=load_evidence(data,work);shared={}
    for bundle in bundles:
        for i,f in enumerate(bundle['frames']):
            shared[int(f)]={key:bundle[key][i] for key in ('depth','K','valid','labels')}
            shared[int(f)]['names']=bundle['names']
    masks=os.path.join(marks,'sam3_masks.npz')
    if pose.get('tracker_backend')=='sam3' and os.path.isfile(masks):
        from sam3_tracking import read_masks
        pose['_sam3_bundle']=read_masks(masks)
    rv_path=os.path.join(marks,'review.json');rv=jload(rv_path) if os.path.isfile(rv_path) else {}
    pose['_prop_timeline']=rv.get('prop_timeline',[]) if rv.get('prop_mode')!='off' else []
    for shot in shots:
        sid=shot['shot'];dx=200.*(sid-1)
        previous=(prior.get('shots') or {}).get(str(sid),{}) if prior.get('shots') else (prior if sid==1 else {})
        scene=copy.deepcopy(previous) or {'ground':{'size':30,'colour':'ground'},'objects':[]}
        keep=[o for o in scene.get('objects',[]) if (o.get('_src') or {}).get('cls')!='drawn']
        new=[]
        for tag,f in sorted(drawn.get(sid,{}).items()):
            lens=data['lens']*float((data.get('zoom') or [1.]*data['frames'])[f])
            fov=2*math.degrees(math.atan(float(data.get('sensor',36.))/(2*lens)))
            fr=analyse(models,frames[f][ci['y0']:ci['y0']+ci['Hpic']],f,pose,ci,fov,shared=shared.get(f))
            R,C=cam_mats(data['cam'][f]);C=C-np.array([dx,0.,0.])
            scale=1. if fr.shared_metric else (shot.get('ground',{}).get('sam3d_body') or {}).get('meters_per_unit',1.)
            to_world(fr,R@D,C,float(scale))
            new+=user_objects(annotations,tag,fr,C,f,first=len(new),indoor=True,fit=False,models=models)
        deleted=[o for sh in annotations.get('shots',{}).values() for o in sh.get('dropped',[])]
        keep=[o for o in keep if not any((o.get('_src') or {}).get('frame')==d.get('frame') and
              px_iou((o.get('_src') or {}).get('px',[0,0,0,0]),d['bbox'])>.5 for d in deleted)]
        keep=[o for o in keep if not any((o.get('_src') or {}).get('frame')==u.get('frame') and
              px_iou((o.get('_src') or {}).get('px',[0,0,0,0]),u['bbox'])>.3 for u in new)]
        if any(o.get('_level') for o in new):
            by_level(new,people_points(data,range(shot['frames'][0],shot['frames'][1]+1,3),dx))
        for o in new:
            for key in ('_level','_C','_reach','_az'):o.pop(key,None)
        scene['objects']=keep+new
        if len(shots)==1:env=scene
        else:env.setdefault('shots',{})[str(sid)]=scene
    env['annotation_build']={'frames':sorted(frames),'objects':sum(len(sh.get('objects',[])) for sh in annotations.get('shots',{}).values())}
    os.makedirs(os.path.dirname(env_path),exist_ok=True)
    with open(env_path,'w',encoding='utf-8') as target:target.write(dumps_env(env))
    print('  annotated background ready: %d source frames; no person or camera inference'%len(frames),flush=True)


def run_clip(name, models, a, c):
    T0 = time.time()
    work = os.path.join(c["work_root"], name)
    models.scene_cache = os.path.join(work, "sam3_scene_cache")
    print("== %s" % name)
    ci = jload(os.path.join(work, "clip_info.json")); data = jload(os.path.join(work, "scene_data.json"))
    camj = jload(os.path.join(work, "camera.json"))
    shared_frames = {}
    if c.get("metric_scene", True) and a.frame == "fit" and not a.free_fov:
        from scene_verify import load_evidence
        for bundle in load_evidence(data, work)[0]:
            for i, f in enumerate(bundle["frames"]):
                shared_frames[int(f)] = {k: bundle[k][i] for k in ("depth", "K", "valid", "labels")}
                shared_frames[int(f)]["names"] = bundle["names"]
    from run_pipeline import marks_dir
    pp = os.path.join(marks_dir(ci["video"], c), "pose.json")
    pose = jload(pp) if os.path.exists(pp) else {"people": []}
    review = jload(os.path.join(os.path.dirname(pp),'review.json')) if os.path.exists(os.path.join(os.path.dirname(pp),'review.json')) else {}
    pose['_prop_timeline'] = review.get('prop_timeline', []) if review.get('prop_mode') != 'off' else []
    if pose.get('tracker_backend') == 'sam3' and os.path.isfile(os.path.join(os.path.dirname(pp),'sam3_masks.npz')):
        from sam3_tracking import read_masks
        pose['_sam3_bundle'] = read_masks(os.path.join(os.path.dirname(pp),'sam3_masks.npz'))
    hj = os.path.join(output_dir(), name, "held.json")          # objects people carry (held_objects.py): masked out
    import held_objects
    automatic_props = review.get('prop_mode', 'manual' if review.get('prop_timeline') else 'auto') == 'auto' and held_objects.enabled(c,{})
    pose["_held"] = jload(hj) if automatic_props and os.path.exists(hj) else None
    try:                                                        # (and the ones annotated as held on the clip page)
        import held_objects
        ann = held_objects.annotated(ci["video"], c, ci, pose, work) if automatic_props else []
        if ann:
            pose["_held"] = held_objects.with_annotated(pose["_held"], ann)
    except Exception as e:
        print("annotated held objects not read (%s)" % e)
    W, Hp = ci["W"], ci["Hpic"]
    user_env = jload(os.path.join(marks_dir(ci["video"], c), "env_ai.json")) if os.path.exists(os.path.join(marks_dir(ci["video"], c), "env_ai.json")) else {}
    f_full = data["lens"] / data["sensor"] * W
    fov_x = 2 * math.degrees(math.atan(W / 2 / f_full))
    shots = camj.get("shots") or [{"shot": 1, "frames": [0, data["frames"] - 1], "ground": {}}]
    want = {}
    for s in shots:
        a0, a1 = s["frames"]
        want[s["shot"]] = sample_set_frames(data, a0, a1, c, first_only=a.no_mid,
                                            pose=pose, picture=(W, Hp, ci["y0"]))
    drawn = drawn_frames(user_env, shots)                     # frames the Objects step drew boxes on: measured too
    for sid_, fs_ in drawn.items():
        want[sid_] = sorted(set(want.get(sid_, [])) | set(fs_.values()))
    frames = read_frames_at(ci["video"], [f for v in want.values() for f in v])
    # Load SAM 3 once for all automatic views and drawn boxes; release it before geometry work.
    from sam3_scene import prefetch
    requests = []
    for f, image in frames.items():
        picture = image[ci['y0']:ci['y0']+Hp]
        k = min(1.,1024./picture.shape[1]); h,w = picture.shape[:2]
        if k<1: picture=cv2.resize(picture,(round(w*k),round(h*k)),interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(picture,cv2.COLOR_BGR2RGB)
        requests.append({'rgb':rgb})
        for sid, tags in drawn.items():
            for tag, keyframe in tags.items():
                if keyframe != f: continue
                for obj in user_env['shots'][tag].get('objects',[]):
                    if obj.get('bbox') and not obj.get('outline'):
                        box = [int(round(v*k)) for v in obj['bbox']]
                        requests.append({'rgb':rgb,'automatic':False,'boxes':[box]})
    count = prefetch(requests, c, models.scene_cache, models._release_scene_gpu)
    print('  SAM 3 scene batch: %d uncached requests (%s)' % (count, 'one model load' if count else 'all cached'))
    acam = AutoCam(W, Hp, f_full, data.get("zoom"))        # (with the lens zoom: each frame's own focal length)
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
            fov_f = 2 * math.degrees(math.atan(W / 2 / acam.f_at(f)))     # this frame's field of view (zoom)
            fr = analyse(models, frames[f][ci["y0"]:ci["y0"] + Hp], f, pose, ci, fov_f,
                         free_fov=bool(a.free_fov) and f == a0, shared=shared_frames.get(f))
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
                                  "fov_free_deg": round(frs[a0].fov_free, 1) if frs[a0].fov_free else lens_fov(work)}
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
                rs_ = body_scale(ppl, bcam, frs[f], fits[f][2], f) if ppl else None   # shoulders / hips, fitted vs depth
                if rs_ and (f not in feet_scale or abs(feet_scale[f] / rs_ - 1) > 0.3):   # feet hidden, or the depth at the
                    print("  f%d scale from the people's bodies: x%.2f%s" % (f, rs_, "" if f not in feet_scale else   # ankles is
                          " (feet said x%.2f: something in front of them)" % feet_scale[f]))                            # a desk
                    feet_scale[f] = rs_
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
                if fr.shared_metric:
                    sc = 1.                                  # already in the shared scene's metres
                to_world(fr, Rw, Cl, sc, gplane)
                acam.R[f] = Rw @ D; acam.C[f] = Cl + np.array([dx, 0, 0])
                if indoor is None:                               # indoor / outdoor: sky and far depth
                    skyf = 1.0 - fr.valid[~fr.bars].mean()      # the bars are not sky
                    farq = np.percentile(fr.rng[fr.valid], 90) if fr.valid.any() else 0
                    indoor = bool(skyf < 0.02 and farq < 18)
                    rep["indoor"] = indoor; rep["sky_frac"] = round(float(skyf), 3)
                regs = []
                for i in range(fr.n_masks):
                    for mi in split_by_class(fr, fr.lab == i):
                        R = region(fr, mi, indoor)
                        if R is None or R["cls"] == "sky":
                            continue
                        if R["area"] > 0.04 and R["cls"] not in ("vegetation", "pole", "floor", "terrain") and \
                                (not R["planar"] or R["cls"] in ("object", "far", "building", "top")):
                            regs += split_planes(fr, R, indoor)
                        else:
                            regs.append(R)
                    continue
                    R = region(fr, fr.lab == i, indoor)
                    if R is None or R["cls"] == "sky":
                        continue
                    if R["area"] > 0.04 and R["cls"] not in ("vegetation", "pole", "floor", "terrain") and \
                            (not R["planar"] or R["cls"] in ("object", "far", "building", "top")):
                        regs += split_planes(fr, R, indoor)
                    else:
                        regs.append(R)
                regs = merge_regions(regs, fr.H, fr.W)
                if fr.sem is not None:                           # the segmentation model names every piece
                    regs = merge_regions(regs, fr.H, fr.W)
                    for m, nm in fr.soft:                        # plants, curtains...: from the segmentation map
                        R = region(fr, m, indoor)
                        if R is not None and R["cls"] != "sky":
                            R["cls"] = "object"; R["relief"] = True; R["label"] = (nm, 1.0); R["soft"] = True
                            regs.append(R)
                    regs = sem_regions(regs, fr, indoor)
                    n0 = len(regs)
                    regs_by_frame[f] = merge_regions(regs, fr.H, fr.W, join=same_label_join)   # curtain folds: one curtain
                    if len(regs_by_frame[f]) < n0:
                        print("    %d wall pieces of the same kind joined" % (n0 - len(regs_by_frame[f])))
                else:
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
            fpix = acam.f_at(f) * fr.k
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
                        if os.environ.get("ENV_AUTO_DEBUG"):
                            print("    f%d wall area %.3f -> joins %s (area %.3f)" % (f, R["area"], q["name"], Rq["area"]))
                        Rm = dict(Rq); Rm["P"] = np.vstack([Rq["P"], R["P"]]); Rm["Pabs"] = np.vstack([Rq["Pabs"], R["Pabs"]])
                        Rm["N"] = np.vstack([Rq["N"], R["N"]]); Rm["area"] = Rq["area"] + R["area"]
                        Rm["plane"] = plane_fit(Rm["P"]); Rm["ext"] = np.percentile(Rm["P"], 95, 0) - np.percentile(Rm["P"], 5, 0)
                        Rm["z"] = tuple(float(v) for v in np.percentile(Rm["P"][:, 2], [4, 50, 96]))
                        frq = frs[q["_src"]["frame"]]; gk = frq.gp
                        if a.frame == "fit":
                            frq.gp = (0.0, 0.0, 0.0)
                        o2 = make_object(Rm, Cw, indoor, frq, acam.f_at(q["_src"]["frame"]) * frq.k, q["name"])
                        frq.gp = gk
                        if o2 is not None:
                            o2["_R"] = Rm; o2["_src"]["area"] = round(Rm["area"], 3); objs[qi] = o2; merged = True
                        break
                    if merged:
                        continue
                dup = [q for q in objs if f != a0 and overlaps(o, q) > 0.35 and
                       GROUP.get(o["_src"]["cls"], o["_src"]["cls"]) == GROUP.get(q["_src"]["cls"], q["_src"]["cls"])]
                if dup:
                    wide = R["cls"] in ("wall", "building") and R["area"] >= 0.15 and R["area"] > 2 * sum(q["_src"]["area"] for q in dup)
                    if os.environ.get("ENV_AUTO_DEBUG"):
                        print("    f%d %s area %.3f: same place as one already found - %s" % (f, R["cls"], R["area"], "kept too (much more of it seen here)" if wide else "left out"))
                    if not wide:                                 # a wall seen much more fully here (the camera moved): kept as well
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
        gone = [d for sh_ in (user_env.get("shots") or {}).values() for d in (sh_ or {}).get("dropped", []) if len(d.get("bbox") or []) == 4]
        if gone:                                                 # found objects deleted or replaced on the clip page
            n0 = len(keep)
            keep = [o for o in keep if not any(int(d.get("frame", -1)) == o["_src"].get("frame") and px_iou(o["_src"]["px"], d["bbox"]) > 0.5
                                               for d in gone)]
            rep["dropped_on_page"] = n0 - len(keep)
            if n0 > len(keep):
                print("  %d found object(s) deleted or replaced on the clip page" % (n0 - len(keep)))
        keep = keep[:a.max_objects]
        mine = []
        for tag, f in sorted(drawn.get(sid, {}).items()):        # each key frame's boxes, at that frame's view
            mine += user_objects(user_env, tag, frs[f], fits[f][4], f, first=len(mine), indoor=bool(indoor), fit=a.frame == "fit", models=models)
        if any(u.get("_level") for u in mine):                   # distance levels: order them, the back one behind the people
            try:
                by_level(mine, people_points(data, range(s["frames"][0], s["frames"][1] + 1, 3), dx))
            except Exception as e:
                print("  distance levels not applied (%s)" % e)
        for u in mine:
            for k_ in ("_level", "_C", "_reach", "_az"):
                u.pop(k_, None)
        if mine:                                                 # the boxes drawn on the clip page win over what is found there
            keep = [o for o in keep if not any(o["_src"].get("frame", u["frame"]) == u["frame"] and
                                               (px_iou(o["_src"]["px"], u["bbox"]) > 0.3 or px_inside(o["_src"]["px"], u["bbox"]) > 0.6)
                                               for u in mine)] + mine
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
    env["_auto"] = {"tool": "env_auto.py (MoGe-2 vitl-normal + SAM 3 image masks)", "fov_x_deg": round(fov_x, 2), "shots": report,
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

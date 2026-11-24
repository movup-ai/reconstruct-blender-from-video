# -*- coding: utf-8 -*-
"""grey set from a depth model: clean background plate (people removed) -> Depth Anything V2 -> metric depth
(scaled with the ground under the people) -> 2.5D relief mesh seen from the camera."""
import os, json, math, numpy as np, cv2
from scipy.optimize import least_squares
from common import *

MODEL_DIR = os.path.join(AGENT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "depth_anything_v2_small.onnx")

class DepthModel:
    def __init__(self, device="cuda"):
        import onnxruntime as ort
        if device != "cuda":
            raise RuntimeError("the depth model runs on the NVIDIA card only (device %r)" % device)
        gpu_dlls()                                       # CUDA / cuDNN installed by pip (onnxruntime-gpu[cuda,cudnn])
        self.s = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in self.s.get_providers():
            raise RuntimeError("depth model did not load on the NVIDIA card - run Setup.bat (installs onnxruntime-gpu)")
        self.s.disable_fallback()                        # a GPU error must stop the job, not quietly use the CPU
        log("    depth model on %s" % self.s.get_providers()[0])
        self.inp = self.s.get_inputs()[0].name
    def __call__(self, bgr):
        H, W = bgr.shape[:2]
        h = 518; w = int(round(W / H * h / 14.0)) * 14
        w = min(max(w, 14 * 20), 14 * 120)
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        d = self.s.run(None, {self.inp: np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)})[0]
        d = np.squeeze(d).astype(np.float32)
        return cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)      # relative inverse depth (bigger = nearer)

_MODELS = {}
def depth_model(device):
    """the depth model of this process (loaded once: the camera and the set both use it)"""
    if device not in _MODELS:
        _MODELS[device] = DepthModel(device)
    return _MODELS[device]

class FakeDepth:
    """test stand-in (no model): ground plane, a hill band and sky"""
    def __init__(self, world, k):
        self.w, self.k = world, k
    def __call__(self, bgr):
        H, W = bgr.shape[:2]
        ys, xs = np.mgrid[0:H, 0:W]
        d = np.zeros((H, W), np.float32)
        for y in range(0, H, 4):
            for x in range(0, W, 8):
                g = self.w.on_ground(self.k, x, y)
                Z = 900.0 if g is None else min(900.0, np.linalg.norm(g - self.w.C))
                if y < H * 0.30: Z = 1e9
                d[y:y + 4, x:x + 8] = 40.0 / Z + 0.02
        return d

def view_dir(world, i):
    return world.R_cw(i).T @ np.array([0, 0, 1.0])

def plate_keys(world, N, step_deg=12.0):
    keys = [0]
    for i in range(1, N):
        if min(np.degrees(np.arccos(np.clip(view_dir(world, i) @ view_dir(world, k), -1, 1))) for k in keys) > step_deg:
            keys.append(i)
    return keys

def H_between(world, j, k):
    K = world.cam.K()
    return K @ world.R_cw(k) @ world.R_cw(j).T @ np.linalg.inv(K)

def side_bars(img, dark=16, rows=0.97):
    """the black pillarbox bars at the left and right of a 4:3 film inside a wider frame -> (left, right) columns kept"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    col_dark = (g < dark).mean(axis=0) >= rows                  # a bar column is dark almost top to bottom
    W = g.shape[1]; l = 0; r = W
    while l < W // 3 and col_dark[l]:
        l += 1
    while r > W - W // 3 and col_dark[r - 1]:
        r -= 1
    return (l, r) if r - l > W // 3 else (0, W)

def people_mask(shape, bxs, y0, pad=(0.15, 0.1, 0.05)):
    """1 where nobody is: each person's box widened a little (arms, hair, shadows)"""
    H, W = shape
    m = np.ones((H, W), np.uint8)
    for b in bxs:
        x0, x1, ya, yb = b[0], b[2], b[1] - y0, b[3] - y0
        w_, h_ = x1 - x0, yb - ya
        cv2.rectangle(m, (int(x0 - pad[0] * w_), int(ya - pad[1] * h_)), (int(x1 + pad[0] * w_), int(yb + pad[2] * h_)), 0, -1)
    return m

def frame_depth(frames, y0, world, k, boxes, model, nmax=6):
    """Depth for key frame k from the REAL video frames around it (not from the people-removed plate: its smears and
    blur leave the depth model nothing to read, so the set came out as flat walls with no objects).  Each frame's depth
    has the people and the pillarbox bars cut out, is brought onto frame k's view and onto frame k's depth scale, and
    the median over the frames fills each pixel from frames where nobody stands there.
    -> (disparity, hidden) with hidden = pixels covered by a person in every frame (filled from around them)"""
    N = len(frames); Hp, W = world.cam.H, world.cam.W
    near = [j for j in range(N) if np.degrees(np.arccos(np.clip(view_dir(world, j) @ view_dir(world, k), -1, 1))) < 8]
    picks = sorted(set([k] + [near[int(round(i))] for i in np.linspace(0, len(near) - 1, min(nmax, len(near)))]))
    layers = []
    for j in picks:
        img = frames[j][y0:y0 + Hp]
        l, r = side_bars(img)
        d = np.full((Hp, W), np.nan, np.float32)
        d[:, l:r] = model(np.ascontiguousarray(img[:, l:r]))     # the bars are not shown to the model
        d[people_mask((Hp, W), boxes.get(j, []), y0) == 0] = np.nan
        if j != k:
            d = cv2.warpPerspective(d, H_between(world, j, k), (W, Hp), flags=cv2.INTER_NEAREST, borderValue=np.nan)
        layers.append((j, d))
    ref = next(d for j, d in layers if j == k)
    out = []
    for j, d in layers:                                          # relative depth: each frame has its own scale and offset
        if j != k:
            both = np.isfinite(ref) & np.isfinite(d)
            if both.sum() < 2000:
                continue
            x, y = d[both][::7], ref[both][::7]
            a, b = np.polyfit(x, y, 1)
            for _ in range(2):                                   # drop what moved (a door, a reflection) and fit again
                res = np.abs(a * x + b - y); keep = res <= np.percentile(res, 80)
                a, b = np.polyfit(x[keep], y[keep], 1)
            if a <= 0:
                continue
            d = a * d + b
        out.append(d)
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        disp = np.nanmedian(np.stack(out), axis=0)
    hidden = ~np.isfinite(disp)
    if hidden.all():
        return None, hidden
    fill = np.nanmedian(disp)
    disp = cv2.inpaint(np.where(hidden, fill, disp).astype(np.float32), hidden.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
    l, r = side_bars(frames[k][y0:y0 + Hp])
    hidden[:, :l] = False; hidden[:, r:] = False                 # the bars are not "a person standing still"
    return disp, hidden

def make_plate(frames, y0, world, k, boxes, nmax=15):
    N = len(frames); Hp, W = world.cam.H, world.cam.W
    near = [j for j in range(N) if np.degrees(np.arccos(np.clip(view_dir(world, j) @ view_dir(world, k), -1, 1))) < 8]
    near = near[:: max(1, len(near) // nmax)]
    sc = 0.5; Ws, Hs = int(W * sc), int(Hp * sc); S = np.diag([sc, sc, 1])
    stack = np.full((len(near), Hs, Ws, 3), np.nan, np.float32)
    cover = np.zeros((Hs, Ws), np.float32)
    for n, j in enumerate(near):
        Hm = S @ H_between(world, j, k) @ np.linalg.inv(S)
        img = cv2.resize(frames[j][y0:y0 + Hp], (Ws, Hs), interpolation=cv2.INTER_AREA).astype(np.float32)
        m = np.ones((Hs, Ws), np.uint8)
        for b in boxes.get(j, []):
            x0, x1 = b[0] * sc, b[2] * sc; ya, yb = (b[1] - y0) * sc, (b[3] - y0) * sc
            w_, h_ = x1 - x0, yb - ya
            cv2.rectangle(m, (int(x0 - 0.15 * w_), int(ya - 0.1 * h_)), (int(x1 + 0.15 * w_), int(yb + 0.05 * h_)), 0, -1)
        wi = cv2.warpPerspective(img, Hm, (Ws, Hs), flags=cv2.INTER_LINEAR, borderValue=np.nan)
        wm = cv2.warpPerspective(m, Hm, (Ws, Hs), flags=cv2.INTER_NEAREST, borderValue=0)
        wi[wm == 0] = np.nan
        cover += (wm == 0)
        stack[n] = wi
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # pixels covered in every frame: all NaN (filled below)
        plate = np.nanmedian(stack, axis=0)
    holes = np.isnan(plate[..., 0]).astype(np.uint8)
    base = cv2.resize(frames[k][y0:y0 + Hp], (Ws, Hs), interpolation=cv2.INTER_AREA).astype(np.float32)
    plate = np.where(np.isnan(plate), base, plate)
    plate = np.clip(plate, 0, 255).astype(np.uint8)
    if holes.any():
        plate = cv2.inpaint(plate, holes * 255, 5, cv2.INPAINT_TELEA)
    still = ((cover / max(1, len(near))) > 0.3).astype(np.uint8)        # people that never move away stay in the plate
    plate = cv2.resize(plate, (W, Hp), interpolation=cv2.INTER_CUBIC)
    still = cv2.resize(still, (W, Hp), interpolation=cv2.INTER_NEAREST)
    make_plate.unknown = cv2.resize(cv2.dilate(holes, np.ones((9, 9), np.uint8)), (W, Hp), interpolation=cv2.INTER_NEAREST).astype(bool)
    return plate, still

def metric_depth(disp, world, k, anchors, still):
    """fit 1/Z = a*disp + b.  anchors: [(x, y, Z)] in plate-k pixels (ground points under the people)"""
    Hp, W = disp.shape
    pts = list(anchors)
    wts = [1.0] * len(pts)
    for y in range(int(Hp * 0.86), Hp, 6):          # bottom strip: assume ground, weak weight
        for x in range(0, W, 24):
            g = world.on_ground(k, x, y)
            if g is not None:
                Xc = world.R_cw(k) @ (g - world.C)
                if 0 < Xc[2] < 40 and not still[y, x]:
                    pts.append((x, y, Xc[2])); wts.append(0.15)
    if len(pts) < 5:
        return None, {"warning": "no ground anchors for the depth scale"}
    P = np.array(pts); wts = np.array(wts)
    d = np.array([disp[int(np.clip(y, 0, Hp - 1)), int(np.clip(x, 0, W - 1))] for x, y, _ in P])
    iz = 1.0 / P[:, 2]
    s = least_squares(lambda q: wts * (q[0] * d + q[1] - iz) / (iz + 0.01), [1.0 / max(d.max(), 1e-6), 0.0],
                      loss="soft_l1", f_scale=0.1)
    a, b = s.x
    info = {"a": float(a), "b": float(b), "anchors": int(len(anchors)), "strip": int(len(P) - len(anchors))}
    if a <= 0:
        info["warning"] = "depth scale failed (wrong sign) - flat ground used"
        return None, info
    return (a, b), info

def relief(plate_k, Z, world, k, step, far, sky):
    """grid mesh from the camera through every step-th pixel; faces seen edge-on (occlusion walls) are dropped"""
    Hp, W = Z.shape
    xs = np.r_[np.arange(0, W, step), W - 1]; ys = np.r_[np.arange(0, Hp, step), Hp - 1]
    R = world.R_cw(k).T
    V = np.zeros((len(ys), len(xs), 3)); ok = np.zeros((len(ys), len(xs)), bool)
    for a, y in enumerate(ys):
        for b, x in enumerate(xs):
            z = Z[y, x]
            if sky[y, x] or not np.isfinite(z):
                continue
            V[a, b] = world.C + R @ (np.r_[world.cam.norm(x, y), 1.0] * min(z, far))
            ok[a, b] = True
    idx = -np.ones(ok.shape, int); idx[ok] = np.arange(ok.sum())
    verts = V[ok]
    groups = {"ground": [], "set": [], "far": [], "backdrop": []}
    for a in range(len(ys) - 1):
        for b in range(len(xs) - 1):
            q = [(a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1)]
            if not all(ok[i] for i in q):
                continue
            for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                p = [V[i] for i in tri]
                n = np.cross(p[1] - p[0], p[2] - p[0]); ln = np.linalg.norm(n)
                if ln < 1e-9:
                    continue
                n /= ln
                cen = (p[0] + p[1] + p[2]) / 3; ray = cen - world.C; dist = np.linalg.norm(ray); ray /= dist
                if abs(n[2]) < 0.6 and abs(n @ ray) < 0.12:      # stretched across a depth jump
                    continue
                g = "ground" if abs(n[2]) > 0.8 and dist < 0.4 * far else ("backdrop" if dist > 0.9 * far else
                     ("far" if dist > 150 else "set"))
                f = [int(idx[i]) for i in tri]
                if n @ ray > 0:
                    f = f[::-1]
                groups[g].append(f)
    return verts, groups

# ----------------------------------------------------------------------------- L2 block set
def colour_key(bgr, palette, grey=None, prefix="c"):
    """video colour -> palette key (quantised so similar blocks share a material)"""
    if grey:
        return grey
    b, g, r = [int(min(255, max(0, round(float(v) / 12.0) * 12))) for v in bgr]
    k = prefix + "%02x%02x%02x" % (r, g, b)
    palette[k] = [r, g, b]
    return k

def box_mesh(corners):
    """8 corners: front quad (4, around) then back quad (same order) -> verts, outward faces"""
    V = np.asarray(corners, float)
    F = [[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
    cen = V.mean(0); out = []
    for f in F:
        p = V[f]; n = np.cross(p[1] - p[0], p[2] - p[0])
        out.append(f if n @ (p.mean(0) - cen) >= 0 else f[::-1])
    return [[round(float(c), 3) for c in v] for v in V], out

# ----------------------------------------------------------------------------- ground-contact ("pop-up") set
def ground_calibrate(disp, world, k, far, s=6):
    """fit 1/Z = a*disp + b on the pixels that ARE ground: RANSAC against the exact ground depth that the solved camera
    (height + angles) gives for every pixel below the horizon.  Returns (a, b, inlier share) or None."""
    Hp, W = disp.shape; cam = world.cam; R = world.R_cw(k).T; C = world.C
    ys, xs = np.mgrid[0:Hp:s, 0:W:s]
    dc = np.dstack([(xs - W / 2) / cam.f, (ys - Hp / 2) / cam.f, np.ones(xs.shape)])
    dz = (dc @ R.T)[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        Zg = np.where(dz < -1e-4, -C[2] / np.minimum(dz, -1e-4), np.inf)
    m = np.isfinite(Zg) & (Zg > 0.4) & (Zg < min(80.0, far * 0.2))
    d = disp[::s, ::s][m].astype(np.float64); iz = 1.0 / Zg[m]
    if len(d) < 60:
        return None
    rng = np.random.default_rng(3)
    best, bn = None, 0
    for _ in range(500):
        i, j = rng.integers(0, len(d), 2)
        if abs(d[i] - d[j]) < 1e-6:
            continue
        a = (iz[i] - iz[j]) / (d[i] - d[j])
        if a <= 0:
            continue
        b = iz[i] - a * d[i]
        inl = np.abs(a * d + b - iz) < 0.12 * iz
        n = int(inl.sum())
        if n > bn:
            best, bn = inl, n
    if best is None or bn < 40:
        return None
    a, b = np.polyfit(d[best], iz[best], 1)
    if a <= 0:
        return None
    return float(a), float(b), bn / float(len(d))


FLAT = (np.array([0.0, 0.0, 1.0]), 0.0)                              # ground plane n.X = d (level ground z = 0)


def plane_z(plane, x, y):
    n, d = plane
    return (d - n[0] * x - n[1] * y) / n[2]


def fit_ground_plane(Z, world, k, sky, far, s=8):
    """the ground as a plane of any tilt (hillsides, ramps, sloping streets): RANSAC on the 3D points of the lower
    part of the picture, keeping planes within 40 deg of level.  The tilt does not depend on the depth scale.
    Returns (n, d, tilt_deg, inlier_share) or None"""
    Hp, W = Z.shape; cam = world.cam; R = world.R_cw(k).T; C = world.C
    ys, xs = np.mgrid[Hp // 3:Hp:s, 0:W:s]
    z = Z[ys, xs].astype(np.float64)
    ok = np.isfinite(z) & ~sky[ys, xs].astype(bool) & (z > 0.3) & (z < far * 0.15)
    if ok.sum() < 80:
        return None
    dw = np.dstack([(xs - W / 2) / cam.f, (ys - Hp / 2) / cam.f, np.ones(xs.shape)])[ok] @ R.T
    P = C + dw * z[ok][:, None]; zz = z[ok]
    rng = np.random.default_rng(7)
    best, bn = None, 0
    cos40 = np.cos(np.radians(40))
    for _ in range(400):
        i = rng.choice(len(P), 3, replace=False)
        n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]]); ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        if n[2] < 0:
            n = -n
        if n[2] < cos40:
            continue
        inl = np.abs(P @ n - n @ P[i[0]]) < 0.03 * zz + 0.05
        if inl.sum() > bn:
            best, bn = inl, int(inl.sum())
    if best is None or bn < 40:
        return None
    Q = P[best]; cen = Q.mean(0)
    n = np.linalg.svd(Q - cen, full_matrices=False)[2][-1]
    if n[2] < 0:
        n = -n
    return n, float(n @ cen), float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))), bn / float(len(P))


def slope_ground_meshes(plate, ground, sky, dw, plane, s, palette, grey, tag, cell=4):
    """the visible sloping ground as a mesh lying on the fitted plane, one piece per video colour"""
    n, d = plane
    C0 = None
    h, w = ground.shape
    groups = {}
    for b in range(0, w - cell + 1, cell):
        colmask = ground[:, b:b + cell].mean(axis=1) > 0.3
        rows = np.where(colmask)[0]
        if not len(rows):
            continue
        top = rows.min()
        for a in range(top - top % cell, h - cell + 1, cell):          # everything below the highest ground cell
            m = ground[a:a + cell, b:b + cell]
            if sky[a:a + cell, b:b + cell].mean() > 0.5:
                continue
            corners = [dw[min(a, h - 1), b], dw[min(a, h - 1), min(b + cell, w - 1)],
                       dw[min(a + cell, h - 1), min(b + cell, w - 1)], dw[min(a + cell, h - 1), b]]
            pts = []
            for r in corners:
                den = r @ n
                if den > -1e-4:
                    break
                pts.append(r * ((d - n @ C_CAM[0]) / den) + C_CAM[0])
            if len(pts) < 4:
                continue
            px = plate[a * s:(a + cell) * s, b * s:(b + cell) * s].reshape(-1, 3)
            cc = np.median(px, axis=0) if m.any() else px.mean(axis=0)
            key = colour_key(cc, palette, grey and "ground", prefix="g")
            v, f = groups.setdefault(key, ([], []))
            base = len(v)
            v += [[round(float(q), 3) for q in p] for p in pts]
            f.append([base + 3, base + 2, base + 1, base])               # facing up
    return [{"name": "ground%s_%02d" % (tag, i), "v": v, "f": f, "colour": key, "layer": 1}
            for i, (key, (v, f)) in enumerate(groups.items())]


C_CAM = [None]


def fit_surface(P, it=6):
    """smooth ground: z = a + bx + cy + dx^2 + exy + fy^2 + gx^3 + hy^3, robust to grass, rocks and depth noise"""
    P = np.asarray(P, float)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) < 30:
        return None
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    A = np.c_[np.ones_like(x), x, y, x * x, x * y, y * y, x ** 3, y ** 3]
    w = np.ones(len(z))
    coef = None
    for _ in range(it):
        Aw = A * w[:, None]
        coef = np.linalg.lstsq(Aw, z * w, rcond=None)[0]
        r = np.abs(A @ coef - z)
        sc = 1.4826 * np.median(r) + 1e-3
        w = 1.0 / np.maximum(1.0, r / (1.5 * sc))                  # Huber weights
    return coef


def surface_z(coef, x, y):
    return coef[0] + coef[1] * x + coef[2] * y + coef[3] * x * x + coef[4] * x * y + coef[5] * y * y + coef[6] * x ** 3 + coef[7] * y ** 3


def ray_surface(dw, C, coef, t0, it=8):
    """where every camera ray meets the smooth ground (Newton steps from the depth model's guess)"""
    t = np.asarray(t0, float).copy()
    for _ in range(it):
        X = C + dw * t[..., None]
        x, y = X[..., 0], X[..., 1]
        g = X[..., 2] - surface_z(coef, x, y)
        dzx = coef[1] + 2 * coef[3] * x + coef[4] * y + 3 * coef[6] * x * x
        dzy = coef[2] + coef[4] * x + 2 * coef[5] * y + 3 * coef[7] * y * y
        dg = dw[..., 2] - dzx * dw[..., 0] - dzy * dw[..., 1]
        step = np.where(np.abs(dg) > 1e-6, g / dg, 0.0)
        t = np.clip(t - np.clip(step, -0.5 * t, 0.5 * t), 0.3, 400.0)
    return C + dw * t[..., None]


def skyline_meshes(plate, sky, Zm, dw, C, tag, grey, palette, s, step=2):
    """far hills and mountains as low-poly silhouettes (like the client reference): the line where the sky meets the
    land is traced across the picture and stood up as a curtain far away - a nearer ridge layer and a far mountain
    layer, lighter with distance"""
    h, w = sky.shape
    pts = []
    for j in range(0, w, step):
        col = sky[:, j]
        if not col[0]:
            pts.append(None); continue                             # land reaches the top edge here: nothing far to show
        r = int(np.argmax(~col)) if (~col).any() else h
        if r >= h - 1:
            pts.append(None); continue
        zs = Zm[r:min(h, r + 8), j]; zs = zs[np.isfinite(zs)]
        d = float(np.median(zs)) if len(zs) else 2500.0
        if d < 120:
            pts.append(None); continue                             # near land: the terrain / walls already show it
        pts.append((j, r, d))
    meshes = []
    layers = {"ridge": [], "mount": []}
    for p in pts:
        layers["mount" if (p and p[2] > 900) else "ridge"].append(p)
    for lname, lp in layers.items():
        dist = 2600.0 if lname == "mount" else 700.0
        v, f, prev = [], [], None
        for p in pts:
            if p is None or ("mount" if p[2] > 900 else "ridge") != lname:
                prev = None; continue
            j, r, _ = p
            ray = dw[r, j]
            top = C + ray * (dist / max(np.hypot(ray[0], ray[1]), 1e-6))
            bot = top.copy(); bot[2] = -dist * 0.4
            v += [[round(float(x), 2) for x in top], [round(float(x), 2) for x in bot]]
            n = len(v)
            if prev is not None:
                f.append([n - 4, n - 3, n - 1, n - 2])
            prev = n
        if f:
            if grey:
                key = lname
            else:
                key = colour_key(np.median(plate[::s, ::s][~sky], axis=0), palette, None)
            meshes.append({"name": "skyline_%s%s" % (lname, tag), "v": v, "f": f, "colour": key, "layer": 0})
    return meshes


def terrain_meshes(plate, ground, flat_ground, Pm, dw, Zg, C, s, palette, grey, tag, cell=4, clean=True):
    """height field over the visible ground: level ground stays exactly level (camera geometry), slopes follow the
    depth model; one mesh per video colour"""
    h, w = ground.shape
    smooth = cv2.GaussianBlur(cv2.resize(plate, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 6).astype(np.float32)  # surface, not tiles
    if clean:                                                      # gentle heights: a smooth slope, not steps
        Pm = Pm.copy()
        zz = np.where(np.isfinite(Pm[..., 2]), Pm[..., 2], 0).astype(np.float32)
        Pm[..., 2] = cv2.GaussianBlur(zz, (0, 0), 2.5)
    if clean:
        ground = cv2.morphologyEx(ground.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
        px = smooth[ground]
        if len(px) > 30:                                           # 3 main tones of the ground, not a mosaic
            k_ = min(3, len(px))
            _, lab, cen = cv2.kmeans(px.astype(np.float32), k_, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
            q = smooth.copy(); q[ground] = cen[lab.ravel()]
            smooth = q
    V = Pm.copy()
    fl = flat_ground & np.isfinite(Zg)
    V[fl] = C + dw[fl] * Zg[fl][:, None]
    groups = {}
    for a in range(0, h - cell, cell):
        for b in range(0, w - cell, cell):
            if ground[a:a + cell, b:b + cell].mean() < 0.5:
                continue
            q = [V[a, b], V[a, b + cell], V[a + cell, b + cell], V[a + cell, b]]
            if any(not np.all(np.isfinite(p)) for p in q):
                continue
            span = max(np.linalg.norm(q[0] - q[2]), np.linalg.norm(q[1] - q[3]))
            if span > (0.8 if clean else 0.35) * max(1.0, np.linalg.norm(q[0] - C)):   # stretched over a depth jump
                continue
            key = colour_key(smooth[min(h - 1, a + cell // 2), min(w - 1, b + cell // 2)], palette, grey and "ground", prefix="g")
            v, f = groups.setdefault(key, ([], []))
            base = len(v)
            v += [[round(float(x), 3) for x in p] for p in q]
            f.append([base + 3, base + 2, base + 1, base])
    return [{"name": "ground%s_%02d" % (tag, i), "v": v, "f": f, "colour": key, "layer": 1, "smooth": bool(grey)}
            for i, (key, (v, f)) in enumerate(groups.items())]


def popup(plate, disp, ab, world, k, far, c, palette, tag="", plane=None):
    """L2 set from where things meet the ground.  Every image column is walked up from the bottom over the ground;
    the first non-ground pixel is where a wall / object stands, and its distance comes from the camera geometry
    (exact for a known camera height), not from the depth model.  Contact points are joined into straight vertical
    walls; whatever does not touch the visible ground is filled with depth blocks.  Returns meshes, info."""
    grey = c.get("_set_grey")
    clean = c.get("set_detail", "clean") == "clean"                    # readable L2 set (default) or detailed
    Hp, W = disp.shape; cam = world.cam; R = world.R_cw(k).T; C = world.C
    s = 4
    ys, xs = np.mgrid[0:Hp:s, 0:W:s]; h, w = ys.shape
    dw = np.dstack([(xs - W / 2) / cam.f, (ys - Hp / 2) / cam.f, np.ones((h, w))]) @ R.T     # world ray per pixel
    dz = dw[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        Zg = np.where(dz < -1e-4, -C[2] / np.minimum(dz, -1e-4), np.inf)
    if plane is not None:                                             # sloping ground: depth of the fitted plane
        n_, d_ = plane
        den = dw @ n_
        with np.errstate(divide="ignore", invalid="ignore"):
            tpl = (d_ - n_ @ C) / den
        Zg = np.where((den < -1e-4) & (tpl > 0), tpl, np.inf)
    C_CAM[0] = C
    a, b = ab[:2]
    iz = a * disp[::s, ::s].astype(np.float64) + b
    sky = iz <= 1.0 / (far * 1.2)
    # pale / hazy skies are often not "far" for the depth model: also take smooth, bright areas above the horizon
    g = cv2.GaussianBlur(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    grad = cv2.resize(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (w, h), interpolation=cv2.INTER_AREA)
    bright = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    above = ~(dz < -0.02)                                              # at or above the horizon (a little margin)
    smooth = (grad < 6.0) & (bright > max(90.0, np.percentile(bright, 60))) & above
    sky |= cv2.morphologyEx(smooth.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    dsub = disp[::s, ::s].astype(np.float64)
    sky |= above & (dsub < 0.06 * np.percentile(dsub, 95))           # cloudy sky: farthest of all for the depth model
    nl, lab = cv2.connectedComponents(sky.astype(np.uint8))           # sky = regions touching the top edge
    top = set(np.unique(lab[0][sky[0]])) - {0}
    sky = np.isin(lab, list(top)) if top else np.zeros_like(sky)
    if sky.sum() > 50:                                                 # hazy / pale sky: same colour, joined to the sky
        pc = cv2.resize(plate, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        skc = np.median(pc[sky], axis=0)
        cand = sky | (above & (np.abs(pc - skc).max(axis=2) < 32) & (grad < 14))
        nl2, lab2 = cv2.connectedComponents(cand.astype(np.uint8))
        keep2 = set(np.unique(lab2[sky])) - {0}
        sky = np.isin(lab2, list(keep2))
    with np.errstate(divide="ignore"):
        Zm = np.where(iz > 0, 1.0 / np.maximum(iz, 1e-9), np.inf)
    col = plate[::s, ::s].astype(np.float32)
    with np.errstate(invalid="ignore"):
        ground = np.isfinite(Zg) & (Zg < far * 0.3) & (np.abs(Zm / Zg - 1.0) < 0.22)
    ground = cv2.morphologyEx(ground.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ground = cv2.morphologyEx(ground, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool) & np.isfinite(Zg)
    # terrain: walkable surfaces of any gentle slope (hills, ramps, steps), from the depth model's own 3D shape
    Zs = cv2.GaussianBlur(np.where(np.isfinite(Zm), np.minimum(Zm, far), far).astype(np.float32), (7, 7), 0).astype(np.float64)
    Pm = C + dw * Zs[..., None]
    nrm = np.cross(np.gradient(Pm, axis=1), np.gradient(Pm, axis=0))
    nrm /= np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-9)
    terrain = (~sky) & np.isfinite(Zm) & (Zm < far * 0.3) & (np.abs(nrm[..., 2]) > np.cos(np.radians(45 if clean else 35)))
    terrain = cv2.morphologyEx(terrain.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    # only for ground that is really off level (hills, slopes, steps) and joined to the ground under the camera;
    # flat streets keep the clean level plane
    off = terrain & ~ground & (np.abs(Pm[..., 2]) > 0.35 + 0.04 * Zs)
    joined = np.zeros_like(off)
    if off.any():
        nl_, lab_ = cv2.connectedComponents((terrain | ground).astype(np.uint8))
        keep = set(np.unique(lab_[-1][(terrain | ground)[-1]])) - {0}          # touching the bottom edge
        joined = off & np.isin(lab_, list(keep)) if keep else joined
    use_terrain = bool(c.get("terrain", True)) and plane is None and joined.mean() > 0.04
    flat_ground = ground.copy()
    if use_terrain:
        ground = ground | terrain
    info = {"walls": 0, "blocks": 0, "ground_share": round(float(ground.mean()), 3), "terrain": use_terrain}
    meshes = []
    gcol = np.median(col[ground], axis=0) if ground.sum() > 30 else np.array([150, 150, 150])
    G = far * 0.4; cx, cy = C[0], C[1]
    if use_terrain and clean:                                           # one smooth slope, like the client reference
        sel = ground & np.isfinite(Zm) & (Zs < 200)
        coef = fit_surface(Pm[sel][::3]) if sel.sum() > 60 else None
        if coef is not None:
            Pm = ray_surface(dw, C, coef, np.where(sel, Zs, np.nanmedian(Zs[sel])))
            Zs = np.where(ground, np.linalg.norm(Pm - C, axis=2) / np.maximum(np.linalg.norm(dw, axis=2), 1e-9), Zs)
            info["terrain_surface"] = [round(float(x), 5) for x in coef]
    if use_terrain:
        meshes += terrain_meshes(plate, ground, flat_ground, Pm, dw, Zg, C, s, palette, grey, tag, clean=clean)
        tz = Pm[ground & np.isfinite(Zm)][::7]
        info["terrain_xyz"] = [[round(float(v), 3) for v in p] for p in tz[:4000]]
    elif plane is None:
        meshes.append({"name": "ground%s" % tag, "v": [[cx - G, cy - G, 0], [cx + G, cy - G, 0], [cx + G, cy + G, 0], [cx - G, cy + G, 0]],
                       "f": [[0, 1, 2, 3]], "colour": colour_key(gcol, palette, grey and "ground", prefix="g"), "layer": 1})
    else:
        meshes += slope_ground_meshes(plate, ground, sky, dw, plane, s, palette, grey, tag)
        info["slope_deg"] = round(float(np.degrees(np.arccos(np.clip(plane[0][2], -1, 1)))), 1)
    pl = plane or FLAT
    if clean and sky.mean() > 0.02:                                     # distant hills / mountains as silhouettes
        sil = skyline_meshes(plate, sky, Zm, dw, C, tag, grey, palette, s)
        meshes += sil
        info["silhouettes"] = len(sil)
        c["_skyline"] = bool(sil)
    # --- contact + top per column
    cols = [None] * w
    for j in range(w):
        i = h - 1
        if not ground[i, j]:
            continue                                                  # bottom hidden by something: no contact
        while i > 0 and ground[i, j]:
            i -= 1
        if i <= 0 or sky[i, j] or not np.isfinite(Zg[i + 1, j]):
            continue                                                  # open view to the horizon
        Zc = float(Zg[i + 1, j]) if (flat_ground[i + 1, j] or not use_terrain) else float(Zs[i + 1, j])
        if not np.isfinite(Zc):
            continue
        seg = Zm[max(0, i - 6):i + 1, j]; seg = seg[np.isfinite(seg)]
        if not len(seg) or not 0.55 * Zc < float(np.median(seg)) < 1.7 * Zc:
            continue                                                  # what starts here is not standing on this ground
        t, miss, zprev = i, 0, float(np.median(seg))
        while t > 0 and not sky[t - 1, j]:
            win = Zm[max(0, t - 3):t, j]; win = win[np.isfinite(win)]
            zz = float(np.median(win)) if len(win) else np.inf          # 3 rows: window recesses do not count as a step
            # same surface: close to the contact distance AND no big step from the rows below
            # (a step means something else starts there, e.g. a building behind a hedge)
            if np.isfinite(zz) and abs(zz / Zc - 1.0) < 0.5 and abs(zz / zprev - 1.0) < 0.3:
                t -= 1; miss = 0; zprev = 0.7 * zprev + 0.3 * float(zz)
            elif miss < 4:
                t -= 1; miss += 1
            else:
                break
        t += miss
        tsky = i
        while tsky > 0 and not sky[tsky - 1, j]:
            tsky -= 1
        P = C + dw[i + 1, j] * Zc                                     # contact point on the ground
        horiz = np.hypot(P[0] - C[0], P[1] - C[1]); dh = np.hypot(dw[t, j][0], dw[t, j][1])
        dhs = np.hypot(dw[tsky, j][0], dw[tsky, j][1])
        zsky = C[2] + dw[tsky, j][2] * (horiz / max(dhs, 1e-6))       # up to the sky line on the wall's vertical
        ztop = C[2] + dw[t, j][2] * (horiz / max(dh, 1e-6))           # top of the column on the wall's vertical
        cols[j] = (i, t, Zc, P[:2].copy(), float(max(P[2] + 0.25, ztop)), float(P[2]), float(zsky), tsky)
    # --- runs of continuous contact -> straight walls (Douglas-Peucker in the ground plane)
    covered = np.zeros((h, w), bool)
    regions = []
    dbg = np.zeros(col.shape, np.uint8)
    dbg[ground] = (gcol * 0.6 + np.array([0, 60, 0])).clip(0, 255)
    dbg[sky] = (235, 200, 150)
    runs, cur = [], []
    for j in range(w):
        q = cols[j]
        if q is not None and cur and abs(q[2] - cols[cur[-1]][2]) < 0.12 * q[2] + 0.15 and j - cur[-1] <= 2:
            cur.append(j)
        else:
            if len(cur) >= 3:
                runs.append(cur)
            cur = [j] if q is not None else []
    if len(cur) >= 3:
        runs.append(cur)

    def dp(idx, eps):
        if len(idx) <= 2:
            return [idx[0], idx[-1]]
        A, B = cols[idx[0]][3], cols[idx[-1]][3]
        ab_ = B - A; L = np.hypot(*ab_) or 1e-6
        dist = [abs(ab_[0] * (cols[j][3][1] - A[1]) - ab_[1] * (cols[j][3][0] - A[0])) / L for j in idx]
        m = int(np.argmax(dist))
        if dist[m] > eps:
            return dp(idx[:m + 1], eps)[:-1] + dp(idx[m:], eps)
        return [idx[0], idx[-1]]

    # 1) straight pieces of every run
    segs = []
    for ri, run in enumerate(runs):
        zmed = float(np.median([cols[j][2] for j in run]))
        keys = dp(run, max(0.12, 0.025 * zmed))
        for a_, b_ in zip(keys[:-1], keys[1:]):
            part = [j for j in run if a_ <= j <= b_]
            if len(part) < 2:
                continue
            P0, P1 = cols[a_][3].copy(), cols[b_][3].copy()
            if float(np.hypot(*(P1 - P0))) < 0.15:
                continue
            segs.append({"run": ri, "P0": P0, "P1": P1, "part": part})
    # 2) straighten: most streets and rooms are built at right angles - snap each piece to the scene's main
    #    direction (or 90 deg to it) when it is within 15 deg, turning it about its middle
    if segs:
        ang = np.array([np.arctan2(*(s_["P1"] - s_["P0"])[::-1]) for s_ in segs])
        wts = np.array([np.hypot(*(s_["P1"] - s_["P0"])) for s_ in segs])
        main = np.angle(np.sum(wts * np.exp(4j * ang))) / 4.0             # length-weighted, modulo 90 deg
        for s_, a0 in zip(segs, ang):
            quarter = np.round((a0 - main) / (np.pi / 2))
            tgt = main + quarter * np.pi / 2
            if abs(a0 - tgt) < np.radians(15):
                mid = (s_["P0"] + s_["P1"]) / 2; half = np.hypot(*(s_["P1"] - s_["P0"])) / 2
                dvec = np.array([np.cos(tgt), np.sin(tgt)])
                s_["P0"], s_["P1"] = mid - dvec * half, mid + dvec * half
                s_["snapped"] = float(tgt)
        info["main_direction_deg"] = round(float(np.degrees(main)), 1)
    # 3) merge neighbouring pieces of one run that now point the same way (one clean wall instead of several)
    merged = []
    for s_ in segs:
        p = merged[-1] if merged else None
        if (p is not None and p["run"] == s_["run"] and "snapped" in p and "snapped" in s_
                and abs(np.angle(np.exp(1j * (p["snapped"] - s_["snapped"])))) < np.radians(3)):
            dvec = np.array([np.cos(p["snapped"]), np.sin(p["snapped"])])
            pts = [p["P0"], p["P1"], s_["P0"], s_["P1"]]
            org = (p["P0"] + s_["P1"]) / 2
            ts = [float((q - org) @ dvec) for q in pts]
            off = np.mean([q - org - dvec * float((q - org) @ dvec) for q in pts], axis=0)
            p["P0"], p["P1"] = org + off + dvec * min(ts), org + off + dvec * max(ts)
            p["part"] = sorted(set(p["part"]) | set(s_["part"]))
        else:
            merged.append(dict(s_))
    info["wall_pieces"] = len(segs)
    for s_ in merged:
        part, P0, P1 = s_["part"], s_["P0"], s_["P1"]
        if True:
            L = float(np.hypot(*(P1 - P0)))
            if plane is not None:                                       # ground height under each end
                zb0, zb1 = float(plane_z(pl, *P0)), float(plane_z(pl, *P1))
            else:
                zb0 = zb1 = float(np.median([cols[j][5] for j in part]))
            H = float(np.percentile([cols[j][4] for j in part], 85))
            hh = H - max(zb0, zb1)
            if min(cols[j][1] for j in part) <= 1 and hh > 4.0:             # a building running off the top of the frame
                H = max(zb0, zb1) + max(hh * 1.3, hh + 2.0)
            H = max(H, max(zb0, zb1) + 0.3)
            dist = float(np.median([cols[j][2] for j in part]))
            lo = np.concatenate([col[max(cols[j][1], cols[j][0] - (cols[j][0] - cols[j][1]) // 3):cols[j][0] + 1, j] for j in part])
            up = np.concatenate([col[cols[j][1]:cols[j][1] + max(1, (cols[j][0] - cols[j][1]) // 3), j] for j in part])
            one_surface = np.abs(np.median(lo, axis=0) - np.median(up, axis=0)).max() < 45
            building = clean and H - max(zb0, zb1) >= 2.6 and L >= 2.0 and one_surface
            if clean and not building and L >= 0.7 and dist < 12.0 and H - max(zb0, zb1) > 3.0:
                H = max(zb0, zb1) + 3.0                                 # short thing near the camera: street furniture
            if building:                                                # a whole facade, up to where the sky starts
                Hs = float(np.percentile([cols[j][6] for j in part], 70))
                H = min(max(H, Hs), max(zb0, zb1) + 40.0)
            pole = clean and L < 0.7 and H - max(zb0, zb1) > 2.6        # lamp post, pole, awning edge
            if pole:
                mid2 = (P0 + P1) / 2; tv = (P1 - P0) / max(L, 1e-6)
                P0, P1, L = mid2 - tv * 0.075, mid2 + tv * 0.075, 0.15
            pix = np.concatenate([col[cols[j][1]:cols[j][0] + 1, j] for j in part])
            cc = np.median(pix, axis=0)
            mid = (P0 + P1) / 2; tdir = (P1 - P0) / L; n = np.array([-tdir[1], tdir[0]])
            if n @ (mid - C[:2]) < 0:
                n = -n                                                # thickness goes away from the camera
            th = 0.15 if pole else (float(np.clip(0.6 * L, 0.4, 2.0)) if (H < 2.6 and L < 6.0) else 0.35)
            f0 = [np.r_[P0, zb0], np.r_[P1, zb1], np.r_[P1, H], np.r_[P0, H]]
            back = [p - np.r_[-n, 0.0] * th for p in f0]
            v, f = box_mesh(f0 + back)
            wname = "wall%s_%02d" % (tag, info["walls"])
            meshes.append({"name": wname, "v": v, "f": f, "layer": 0 if H > 2.6 else 2,
                           "colour": colour_key(cc, palette, grey and "set")})
            regions.append({"id": "W%d" % info["walls"], "type": "wall", "meshes": [wname], "p0": [round(float(P0[0]), 3), round(float(P0[1]), 3)],
                            "normal": [round(float(-n[0]), 4), round(float(-n[1]), 4)],          # facing the camera side
                            "bbox": [part[0] * s, min(cols[j][1] for j in part) * s, (part[-1] + 1) * s, (max(cols[j][0] for j in part) + 1) * s],
                            "height_m": round(H, 2), "length_m": round(L, 2), "distance_m": round(float(np.median([cols[j][2] for j in part])), 1)})
            info["walls"] += 1
            for j in part:
                covered[(cols[j][7] if building else cols[j][1]):cols[j][0] + 1, j] = True
                dbg[cols[j][1]:cols[j][0] + 1, j] = cc
            for j0, j1 in zip(part[:-1], part[1:]):
                cv2.line(dbg, (j0, cols[j0][0]), (j1, cols[j1][0]), (0, 0, 255), 1)
    # --- the rest (not ground, not sky, not on a wall): depth blocks
    Zfull = cv2.resize(np.where(sky, np.inf, Zm).astype(np.float32), (W, Hp), interpolation=cv2.INTER_NEAREST)
    skyfull = cv2.resize(sky.astype(np.uint8), (W, Hp), interpolation=cv2.INTER_NEAREST).astype(bool)
    excl = cv2.resize((covered | ground).astype(np.uint8), (W, Hp), interpolation=cv2.INTER_NEAREST).astype(bool)
    unk = getattr(make_plate, "unknown", None)
    if unk is not None and unk.shape == excl.shape:
        excl |= unk                                                   # painted over where a person stood: nothing to build
    skycol = np.median(col[sky], axis=0) if sky.sum() > 50 else None
    bm, binf = blocks(plate, Zfull, world, k, skyfull, far, dict(c, _work=None, _clean=clean, _skycol=skycol, _skyline=c.get("_skyline")), palette, tag=tag, use_planes=False,
                      exclude=excl, add_ground=False)
    meshes += bm
    info["blocks"] = binf["blocks"]
    bmask = np.zeros((Hp, W), np.uint8)                               # block groups = connected blocks
    for (x0, y0_, x1, y1, cc) in binf.get("rects", []):
        bmask[y0_:y1, x0:x1] = 1
    nb, blab, bst, _ = cv2.connectedComponentsWithStats(cv2.dilate(bmask, np.ones((9, 9), np.uint8)))
    names = [m["name"] for m in bm]
    for gi in range(1, nb):
        x, y, ww, hh, area = bst[gi]
        if area < 0.002 * W * Hp:
            continue
        members = [names[n] for n, (x0, y0_, x1, y1, _) in enumerate(binf.get("rects", []))
                   if blab[min(Hp - 1, (y0_ + y1) // 2), min(W - 1, (x0 + x1) // 2)] == gi]
        regions.append({"id": "B%d" % len([r for r in regions if r["type"] == "blocks"]), "type": "blocks",
                        "bbox": [int(x), int(y), int(x + ww), int(y + hh)], "meshes": members})
    meshes, added = apply_env_ai(meshes, regions, (c.get("_env_ai") or {}).get(tag), plate, world, k, palette, grey, info, plane,
                                 Z=Zfull, main_deg=info.get("main_direction_deg"))
    if c.get("_work"):
        with open(os.path.join(c["_work"], "regions%s.json" % tag), "w", encoding="utf-8") as fh:
            json.dump({"tag": tag, "size": [W, Hp], "regions": regions, "added": added}, fh, ensure_ascii=False)
    for r in regions:                                                 # numbered outlines (also what the AI is shown)
        x0, y0_, x1, y1 = [v // s for v in r["bbox"]]
        cv2.putText(dbg, r["id"], (x0 + 2, y0_ + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for (x0, y0_, x1, y1, cc) in binf.get("rects", []):
        cv2.rectangle(dbg, (x0 // s, y0_ // s), (x1 // s, y1 // s), (255, 255, 0), 1)
        dbg[y0_ // s:y1 // s, x0 // s:x1 // s][~(covered | ground)[y0_ // s:y1 // s, x0 // s:x1 // s]] = cc
    if sky.mean() > 0.01:
        info["sky_rgb"] = [int(v) for v in np.median(plate[::s, ::s][sky], axis=0)[::-1]]
    if c.get("_work"):
        legend = np.hstack([plate[::s, ::s], dbg])
        cv2.putText(legend, "ground (green tint)  sky (blue)  walls on their ground line (red)  depth blocks (cyan)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        imwrite_u(os.path.join(c["_work"], "blocks%s.jpg" % tag), legend, 85)
    return meshes, info


KIND_DEPTH = {"car": 1.8, "bus": 2.5, "truck": 2.5, "table": 0.9, "bench": 0.5, "door": 0.1, "tree": 0.8, "bush": 0.8,
              "fence": 0.1, "pole": 0.25, "sign": 0.1, "chair": 0.5, "bed": 2.0, "sofa": 0.9, "cabinet": 0.5, "rock": 1.0}


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def apply_env_ai(meshes, regions, env, plate, world, k, palette, grey, info, plane=None, Z=None, main_deg=None):
    """the AI's view of one shot: remove regions it judged wrong, and build every object it names (door, shelf,
    counter, car, tree...) as a recognisable model from props.py, standing where it touches the floor or hanging on
    its wall, sized from the picture.  Depth pieces an object replaces are removed."""
    if not env:
        return meshes, []
    import props
    drop = set()
    for d in env.get("drop", []):
        for r in regions:
            if iou(r["bbox"], d["bbox"]) > 0.45:
                drop.update(r["meshes"]); r["dropped"] = d.get("what", "wrong")
    cam = world.cam; R = world.R_cw(k).T; C = world.C
    pn, pd = plane or FLAT
    walls = [r for r in regions if r.get("type") == "wall" and r.get("normal")]
    added = []

    def ray(x, y):
        return R @ np.r_[cam.norm(x, y), 1.0]

    def contain(inner, outer):
        x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1]); x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
        ar = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
        return max(0, x1 - x0) * max(0, y1 - y0) / ar

    def emit(n, o, kind, parts, base_col):
        role = o.get("colour_role")
        name = "obj_%s_%02d" % (kind, n)
        for pname, V, F, shade in parts:
            if role in ("orange", "pink", "purple", "brown", "white", "dark", "light", "set", "far"):
                colour = role if shade == "main" or role not in ("dark", "light", "set", "far") else {"dark": "dark", "light": "light"}[shade]
            else:
                ref = {"tree": "tree", "pine": "tree", "rock": "rock", "building": "cabin", "fence": "fence", "railing": "fence"}.get(kind)
                gname = (ref if shade == "main" and ref else {"main": "set", "dark": "dark", "light": "light"}[shade])
                if kind == "building" and pname == "roof":
                    gname = "roof"
                colour = colour_key(props.shade_colour(base_col, shade), palette, grey and gname)
            meshes.append({"name": "%s_%s" % (name, pname), "v": V, "f": F, "colour": colour, "layer": 2})
        return name

    for n, o in enumerate(env.get("objects", [])):
        if isinstance(o.get("pos"), (list, tuple)) and len(o["pos"]) >= 2:
            # placed by hand in the shot's world: pos = [x, y] or [x, y, z of its underside] in metres (ground z = 0,
            # the camera is at world.C), yaw in degrees about z (0 = front faces -y), size = [width, height, depth]
            kind = props.norm_type(o.get("type") or o.get("kind"))
            (dw_, dh_, dd_), _ = props.TYPES[kind]
            sz = o.get("size") if isinstance(o.get("size"), (list, tuple)) and len(o.get("size")) == 3 else [dw_, dh_, dd_]
            w_, h_, d_ = [max(0.005, float(v)) for v in sz]
            px, py = float(o["pos"][0]), float(o["pos"][1])
            z0 = float(o["pos"][2]) if len(o["pos"]) > 2 else float((pd - pn[0] * px - pn[1] * py) / pn[2])
            yaw = float(o.get("yaw", 0.0))
            parts = props.place(props.build(kind, w_, h_, d_), (px, py), yaw, z0)
            name = emit(n, o, kind, parts, np.array([160, 160, 160]))
            added.append({"name": name, "what": o.get("what", kind), "kind": kind, "attach": "placed",
                          "size_m": [round(w_, 2), round(d_, 2), round(h_, 2)],
                          "distance_m": round(float(np.hypot(px - C[0], py - C[1])), 1), "yaw_deg": round(yaw, 1)})
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in o["bbox"]]
        except (KeyError, TypeError, ValueError):
            continue
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        kind = props.norm_type(o.get("type") or o.get("kind"))
        (dw_, dh_, dd_), where = props.TYPES[kind]
        attach = o.get("attach") if o.get("attach") in ("floor", "wall") else where
        cx = (x0 + x1) / 2
        wall = None
        if attach == "wall":                                          # the wall it is drawn against
            cands = [r for r in walls if contain([x0, y0, x1, y1], r["bbox"]) > 0.4]
            wall = max(cands, key=lambda r: contain([x0, y0, x1, y1], r["bbox"])) if cands else None
        if wall is not None:
            nn = np.array(wall["normal"], float); p0 = np.array(wall["p0"], float)
            rb = ray(cx, y1); den = rb[0] * nn[0] + rb[1] * nn[1]
            if abs(den) < 1e-6:
                continue
            t = ((p0 - C[:2]) @ nn) / den
            P = C + rb * t; P[:2] += nn * 0.02
            Zc = float(np.linalg.norm(P - C) / max(np.linalg.norm(rb), 1e-6))
            z0 = float(P[2])
        else:
            rb = ray(cx, y1)
            if rb @ pn < -1e-4 and attach == "floor":                  # bottom edge on the ground
                t = (pd - pn @ C) / (rb @ pn)
            elif Z is not None:                                        # hanging: the depth model's distance
                yy, xx = int(min(Z.shape[0] - 1, (y0 + y1) / 2)), int(min(Z.shape[1] - 1, cx))
                t = float(np.nanmedian(Z[max(0, yy - 3):yy + 4, max(0, xx - 3):xx + 4]))
            else:
                continue
            if not np.isfinite(t) or not 0.3 < t < 400:
                continue
            P = C + rb * t; Zc = float(t)
            z0 = float(P[2]) if attach == "wall" else float((pd - pn[0] * P[0] - pn[1] * P[1]) / pn[2])
        # size: how tall it looks at that distance; proportions from the type (or the AI's size)
        hm = (y1 - y0) * Zc / cam.f
        size = o.get("size") if isinstance(o.get("size"), (list, tuple)) and len(o.get("size")) == 3 else None
        base = [float(v) for v in size] if size and all(float(v) > 0 for v in size) else [dw_, dh_, dd_]
        sc = float(np.clip(hm / max(base[1], 1e-3), 0.4, 2.5))
        w_, h_, d_ = base[0] * sc, base[1] * sc, base[2] * sc
        wm = (x1 - x0) * Zc / cam.f                                     # visible width: use it when the object faces us
        # turn: vehicles point their bonnet (+x), everything else its front (-y)
        v = C[:2] - P[:2]
        veh = kind in ("car", "van", "truck", "bus", "bicycle")
        if wall is not None:
            nn = np.array(wall["normal"], float)
            yaw = math.degrees(math.atan2(nn[0], -nn[1]))
        else:
            yaw = math.degrees(math.atan2(v[1], v[0])) if veh else math.degrees(math.atan2(v[0], -v[1]))
            yaw += {"away": 180, "left": 90, "right": -90}.get(str(o.get("facing", "")).lower(), 0)
            if main_deg is not None:                                    # line up with the street / room
                rel = (yaw - main_deg) % 90
                if min(rel, 90 - rel) < 25:
                    yaw = main_deg + round((yaw - main_deg) / 90.0) * 90
        if not veh and str(o.get("facing", "camera")).lower() in ("camera", "towards", "toward camera", "") and wm > 0.2:
            w_ = float(np.clip(wm, w_ * 0.5, w_ * 2.0))
        parts = props.place(props.build(kind, w_, h_, d_), (P[0], P[1]), yaw, z0)
        crop = plate[int(max(0, y0)):int(y1), int(max(0, x0)):int(x1)].reshape(-1, 3)
        base_col = np.median(crop, axis=0) if crop.size else np.array([160, 160, 160])
        name = emit(n, o, kind, parts, base_col)
        for r in regions:                                               # depth pieces this object replaces
            if contain(r["bbox"], [x0, y0, x1, y1]) > 0.6 and r.get("type") != "wall" or \
               (r.get("type") == "wall" and contain(r["bbox"], [x0, y0, x1, y1]) > 0.8):
                drop.update(r["meshes"])
        added.append({"name": name, "what": o.get("what", kind), "kind": kind, "attach": "wall" if wall is not None else attach,
                      "size_m": [round(w_, 2), round(d_, 2), round(h_, 2)], "distance_m": round(Zc, 1), "yaw_deg": round(yaw, 1)})
    meshes = [m for m in meshes if m["name"] not in drop]
    if env.get("only"):                                               # hand-built set: ground + the listed objects only
        keep = ("ground", "obj_") + (("skyline",) if env.get("only") == "keep_skyline" else ())
        drop |= {m["name"] for m in meshes if not m["name"].startswith(keep)}
        meshes = [m for m in meshes if m["name"].startswith(keep)]
        info["set_source"] = "hand-built (only the listed objects)"
    info["ai_dropped"] = len(drop); info["ai_added"] = len(added)
    return meshes, added


def blocks(plate, Z, world, k, sky, far, c, palette, tag="", use_planes=True, exclude=None, add_ground=True):
    """L2 set: vertical planes (walls, facades, fences) fitted to the depth points + the rest as camera-facing
    blocks merged over a grid, each coloured with the video's own colour. Returns meshes, info."""
    grey = c.get("_set_grey")
    Hp, W = Z.shape; cam = world.cam; R = world.R_cw(k).T; C = world.C
    s = 2
    ys, xs = np.mgrid[0:Hp:s, 0:W:s]
    Zs = Z[::s, ::s].astype(np.float64); col = plate[::s, ::s].astype(np.float32)
    ok = np.isfinite(Zs) & ~sky[::s, ::s].astype(bool)
    Zc = np.where(ok, np.minimum(Zs, far * 0.8), 1.0)
    dc = np.dstack([(xs - cam.W / 2) / cam.f, (ys - cam.H / 2) / cam.f, np.ones_like(Zc)])
    P = C + (dc * Zc[..., None]) @ R.T                                   # world points of every pixel
    ground = ok & (P[..., 2] < 0.18 + 0.012 * Zc) & (Zc < far * 0.3)
    struct = ok & ~ground
    if exclude is not None:
        struct &= ~exclude[::s, ::s]
    meshes = []; info = {"planes": 0, "blocks": 0, "rects": []}
    # ground plane, coloured with the road / floor under the people
    gcol = np.median(col[ground], axis=0) if ground.sum() > 50 else np.array([150, 150, 150])
    G = far * 0.4; cx, cy = C[0], C[1]
    if add_ground:
        meshes.append({"name": "ground%s" % tag, "v": [[cx - G, cy - G, 0], [cx + G, cy - G, 0], [cx + G, cy + G, 0], [cx - G, cy + G, 0]],
                       "f": [[0, 1, 2, 3]], "colour": colour_key(gcol, palette, grey and "ground", prefix="g"), "layer": 1})
    # --- vertical planes by RANSAC (near / middle distance only)
    rng = np.random.default_rng(1)
    near = struct & (Zc < 120)
    left = near.copy(); total = max(1, near.sum()); planed = np.zeros_like(near)
    dbg = np.zeros(col.shape, np.uint8) if c.get("_work") else None   # what became what, for checking
    if dbg is not None:
        dbg[ground] = gcol
    for it in range(12 if use_planes else 0):
        idx = np.argwhere(left)
        if len(idx) < max(400, 0.02 * total):
            break
        pts = P[left]; zz = Zc[left]
        best, bn, bpt = None, None, None
        for _ in range(250):
            i, j = rng.integers(0, len(pts), 2)
            d = pts[j] - pts[i]; d[2] = 0; L = np.hypot(d[0], d[1])
            if L < 0.4:
                continue
            n = np.array([-d[1], d[0], 0.0]) / L
            inl = np.abs((pts - pts[i]) @ n) < 0.035 * zz + 0.06
            if best is None or inl.sum() > best.sum():
                best, bn, bpt = inl, n, pts[i]
        if best is None or best.sum() < 0.015 * total:
            break
        q = pts[best]                                                    # refit the plane on its inliers
        uv = q[:, :2] - q[:, :2].mean(0); _, _, vt = np.linalg.svd(uv, full_matrices=False)
        t2 = vt[0]; n = np.array([-t2[1], t2[0], 0.0]); d0 = float(np.median(q @ n))
        inl_full = np.zeros_like(left); inl_full[tuple(idx[best].T)] = True
        m = cv2.morphologyEx(inl_full.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        used = False
        for ci in range(1, ncomp):
            if stats[ci, cv2.CC_STAT_AREA] < 0.006 * total:
                continue
            msk = (lab == ci) & inl_full
            if msk.sum() < 60:
                continue
            pp = P[msk]; t = np.array([t2[0], t2[1], 0.0])
            u = pp @ t; zv = pp[:, 2]
            u0, u1 = np.percentile(u, [2, 98]); z0, z1 = np.percentile(zv, [2, 98])
            if z0 < 0.8: z0 = 0.0                                        # walls stand on the ground
            if u1 - u0 < 0.3 or z1 - z0 < 0.3:
                continue
            nn = n if (C @ n - d0) > 0 else -n                           # nn points to the camera side
            o = n * d0
            th = float(np.clip(0.02 * np.median(Zc[msk]), 0.15, 1.5))
            front = [o + t * u0 + [0, 0, z0], o + t * u1 + [0, 0, z0], o + t * u1 + [0, 0, z1], o + t * u0 + [0, 0, z1]]
            back = [p - nn * th for p in front]
            v, f = box_mesh(front + back)
            meshes.append({"name": "wall%s_%02d" % (tag, info["planes"]), "v": v, "f": f, "layer": 0,
                           "colour": colour_key(np.median(col[msk], axis=0), palette, grey and "set")})
            if dbg is not None:
                dbg[msk] = np.median(col[msk], axis=0)
                yx = np.argwhere(msk).mean(0).astype(int)[::-1]
                cv2.putText(dbg, "W%d" % info["planes"], tuple(int(v) for v in yx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            info["planes"] += 1; used = True
            left &= ~msk; planed |= msk
        if not used:
            left[tuple(idx[best].T)] = False                             # nothing big enough: drop these points
    # --- everything else: camera-facing blocks on a grid, merged along rows then down columns
    rest = struct & ~planed
    clean = c.get("_clean")
    cs = max(6, int(round(W / (36 if clean else 70) / s)))                # cell size in sub-sampled pixels
    if clean:                                                             # one calm colour per mass, not a mosaic
        col = cv2.GaussianBlur(col, (0, 0), 5)
    gh, gw = rest.shape[0] // cs, rest.shape[1] // cs
    cells = {}
    for a in range(gh):
        for b in range(gw):
            m = rest[a * cs:(a + 1) * cs, b * cs:(b + 1) * cs]
            if m.mean() < 0.4:
                continue
            zv = Zc[a * cs:(a + 1) * cs, b * cs:(b + 1) * cs][m]
            cells[(a, b)] = (float(np.median(zv)), np.median(col[a * cs:(a + 1) * cs, b * cs:(b + 1) * cs][m], axis=0))
    def similar(p, q):
        if clean:
            return abs(p[0] - q[0]) / min(p[0], q[0]) < 0.3 and np.abs(p[1] - q[1]).max() < 70
        return abs(p[0] - q[0]) / min(p[0], q[0]) < 0.18 and np.abs(p[1] - q[1]).max() < 42
    runs = []
    for a in range(gh):
        b = 0
        while b < gw:
            if (a, b) not in cells:
                b += 1; continue
            b1 = b; zs_, cs_ = [cells[(a, b)][0]], [cells[(a, b)][1]]
            while (a, b1 + 1) in cells and similar(cells[(a, b1)], cells[(a, b1 + 1)]):
                b1 += 1; zs_.append(cells[(a, b1)][0]); cs_.append(cells[(a, b1)][1])
            runs.append([a, a, b, b1, float(np.median(zs_)), np.median(cs_, axis=0)])
            b = b1 + 1
    merged = []
    for r in runs:                                                        # stack identical spans down the image
        for m_ in merged:
            if m_[1] == r[0] - 1 and m_[2] == r[2] and m_[3] == r[3] and similar((m_[4], m_[5]), (r[4], r[5])):
                m_[1] = r[0]; m_[4] = 0.5 * (m_[4] + r[4]); m_[5] = 0.5 * (m_[5] + r[5]); break
        else:
            merged.append(list(r))
    for a0, a1, b0, b1, z, cc in merged:
        if a0 == a1 and b0 == b1 and not any(abs(m_[0] - a0) <= 1 and m_[2] <= b0 + 1 and m_[3] >= b0 - 1 and m_ is not None
                                             and (m_[0], m_[2]) != (a0, b0) for m_ in merged):
            continue                                                  # a lone single cell: depth noise, not a thing
        x0, x1 = b0 * cs * s, (b1 + 1) * cs * s; y0_, y1_ = a0 * cs * s, (a1 + 1) * cs * s
        if clean and (x1 - x0) * (y1_ - y0_) < 0.004 * W * Hp:
            continue                                                  # small fragment: leave it out of a readable set
        sk = c.get("_skycol")
        if clean and sk is not None and np.abs(np.asarray(cc, float) - sk).max() < 38 and y1_ < 0.6 * Hp:
            continue                                                  # pale sky the sky test missed, not a thing
        th = float(np.clip(0.06 * z, 0.2, 4.0))
        def wp(x, y, d):
            return C + R @ (np.r_[cam.norm(x, y), 1.0] * d)
        front = [wp(x0, y1_, z), wp(x1, y1_, z), wp(x1, y0_, z), wp(x0, y0_, z)]
        back = [wp(x0, y1_, z + th), wp(x1, y1_, z + th), wp(x1, y0_, z + th), wp(x0, y0_, z + th)]
        if clean and c.get("_skyline") and z > 150:
            continue                                                  # far away: the skyline silhouettes show it
        if clean and z < 60:                                          # a solid mass standing on the ground, not floating
            for q in (front[0], front[1], back[0], back[1]):
                q[2] = min(q[2], 0.0)
        v, f = box_mesh(front + back)
        meshes.append({"name": "block%s_%03d" % (tag, info["blocks"]), "v": v, "f": f, "layer": 0 if z > 150 else 2,
                       "colour": colour_key(cc, palette, grey and ("far" if z > 150 else "set"))})
        info["blocks"] += 1
        info["rects"].append((int(x0), int(y0_), int(x1), int(y1_), tuple(int(v) for v in cc)))
        if dbg is not None:
            dbg[a0 * cs:(a1 + 1) * cs, b0 * cs:(b1 + 1) * cs] = cc
            cv2.rectangle(dbg, (b0 * cs, a0 * cs), ((b1 + 1) * cs - 1, (a1 + 1) * cs - 1), (255, 255, 0), 1)
    if dbg is not None:
        imwrite_u(os.path.join(c["_work"], "blocks%s.jpg" % tag), np.hstack([plate[::s, ::s], dbg.astype(np.uint8)]), 85)
    skym = sky[::s, ::s].astype(bool)
    if skym.mean() > 0.01:
        info["sky_rgb"] = [int(v) for v in np.median(plate[::s, ::s][skym], axis=0)[::-1]]
    return meshes, info

def compact(verts, faces):
    used = sorted(set(i for f in faces for i in f))
    m = {o: n for n, o in enumerate(used)}
    return [[round(float(c), 3) for c in verts[i]] for i in used], [[m[i] for i in f] for f in faces]

def flat_set(world, N, far):
    """fallback: ground disc and a far backdrop ring"""
    meshes = []
    R = far * 0.6; seg = 72
    v = [[0, 0, 0]] + [[R * np.cos(2 * np.pi * i / seg), R * np.sin(2 * np.pi * i / seg), 0] for i in range(seg)]
    meshes.append({"name": "ground", "v": v, "f": [[0, 1 + i, 1 + (i + 1) % seg] for i in range(seg)], "colour": "ground", "layer": 1})
    hb = far * 0.05; v = []
    for i in range(seg):
        a = 2 * np.pi * i / seg
        v += [[far * np.cos(a), far * np.sin(a), -5], [far * np.cos(a), far * np.sin(a), hb]]
    f = [[2 * i, 2 * ((i + 1) % seg), 2 * ((i + 1) % seg) + 1, 2 * i + 1] for i in range(seg)]
    meshes.append({"name": "backdrop", "v": v, "f": f, "colour": "backdrop", "layer": 0})
    return meshes

def build_set(frames, y0, world, people_keys, boxes, c, log_=log, fake=False):
    """returns (meshes for DATA, info)"""
    N = len(frames); far = float(c.get("far_distance", 1500.0)); step = int(c.get("mesh_step", 10))
    info = {"plates": []}
    if not fake and not os.path.exists(MODEL_PATH):
        info["warning"] = "depth model missing (run setup.bat) - flat ground + backdrop used"
        return flat_set(world, N, far), info
    model = None if fake else depth_model(pick_device(c.get("device", "cuda")))
    meshes = []
    for k in plate_keys(world, N, c.get("plate_step_deg", 12.0)):
        plate, still = make_plate(frames, y0, world, k, boxes)       # colours for the set pieces
        disp = None
        if not fake and c.get("set_depth_from", "frames") == "frames":   # depth from the real frames, people cut out
            disp, hidden = frame_depth(frames, y0, world, k, boxes, model)
            if disp is not None:
                still = hidden.astype(np.uint8)
        if disp is None:                                             # the older way: depth of the people-removed plate
            disp = (FakeDepth(world, k) if fake else model)(plate)
        disp = cv2.medianBlur(disp.astype(np.float32), 5)
        if still.any():                                  # people standing still: fill their depth from around them
            disp = cv2.inpaint(disp.astype(np.float32), (still > 0).astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
        anchors = []
        for kind, ch in people_keys.items():
            for f in range(0, N, 2):
                if not ch["visible"][f]:
                    continue
                g = np.array(ch["root"][f][:3], float)
                uv, z = world.project(g, k)
                x, y = uv[0]
                if 0 <= x < world.cam.W and 0 <= y < world.cam.H and z[0] > 0.5:
                    anchors.append((x, y, z[0]))
        sc, inf = metric_depth(disp, world, k, anchors, still)
        inf["frame"] = k
        info["plates"].append(inf)
        tag = c.get("_shot_tag", "")
        style = c.get("set_style", "popup")
        gc = ground_calibrate(disp, world, k, far) if style == "popup" else None
        if gc and (gc[2] >= 0.45 or sc is None or len(anchors) < 10):
            sc = (gc[0], gc[1])                                    # depth scale from the ground the camera sees
            inf["ground_fit"] = round(gc[2], 3)
        if sc is None:
            if k == 0:
                info["warning"] = inf.get("warning", "depth scale failed")
                return flat_set(world, N, far), info
            continue
        a, b = sc
        iz = a * disp + b
        sky = iz <= 1.0 / (far * 1.5)
        Z = np.where(sky, np.inf, 1.0 / np.maximum(iz, 1e-9))
        plane = None
        gp = fit_ground_plane(Z, world, k, sky, far) if style == "popup" and c.get("slope_ground", False) else None
        if gp and gp[2] >= 3.0 and gp[3] >= 0.25:
            # sloping ground: the level-ground depth scale is wrong here - use the people's feet when there are
            # enough of them, otherwise put the plane at the camera height below the camera
            n_, d_ = gp[0], gp[1]
            sc2, _ = metric_depth(disp, world, k, anchors, still) if len(anchors) >= 10 else (None, None)
            if sc2 is not None:
                a, b = sc2
            else:
                dist = abs(n_ @ world.C - d_)
                q = world.C[2] / max(dist, 1e-3)
                a, b = a / q, b / q
            iz = a * disp + b
            sky = iz <= 1.0 / (far * 1.5)
            Z = np.where(sky, np.inf, 1.0 / np.maximum(iz, 1e-9))
            gp = fit_ground_plane(Z, world, k, sky, far) or gp
            plane = (gp[0], gp[1])
            info["plane"] = [round(float(x), 5) for x in gp[0]] + [round(gp[1], 4)]
            info["slope_deg"] = round(gp[2], 1)
        if style == "popup":
            first = k == plate_keys(world, N, c.get("plate_step_deg", 12.0))[0]
            bm, binf = popup(plate, disp, (a, b), world, k, far, c, info.setdefault("palette", {}), tag="%s_%03d" % (tag, k), plane=plane)
            if not first:
                bm = [m for m in bm if not m["name"].startswith("ground")]
            meshes += bm
            if "sky_rgb" in binf and "sky_rgb" not in info:
                info["sky_rgb"] = binf["sky_rgb"]
            info.setdefault("walls", 0); info["walls"] += binf["walls"]; info.setdefault("blocks", 0); info["blocks"] += binf["blocks"]
            if binf.get("terrain_xyz") and "terrain_xyz" not in info:
                info["terrain_xyz"] = binf["terrain_xyz"]
            if binf.get("terrain"):
                info["terrain"] = True
            log_("    plate frame %d: %d anchors, ground fit %s, %s, %d walls, %d blocks, ground %.0f%%" % (
                k, len(anchors), "%.0f%%" % (100 * gc[2]) if gc else "none",
                ("sloping ground %.1f deg" % info["slope_deg"]) if plane is not None else ("terrain (hills/slopes)" if binf.get("terrain") else "level ground"),
                binf["walls"], binf["blocks"], 100 * binf["ground_share"]))
        elif style == "blocks":
            bm, binf = blocks(plate, Z, world, k, sky, far, c, info.setdefault("palette", {}), tag="%s_%03d" % (tag, k))
            if k != plate_keys(world, N, c.get("plate_step_deg", 12.0))[0]:
                bm = [m for m in bm if not m["name"].startswith("ground")]          # one ground plane per shot
            meshes += bm
            if "sky_rgb" in binf and "sky_rgb" not in info:
                info["sky_rgb"] = binf["sky_rgb"]
            log_("    plate frame %d: %d anchors, %d wall planes, %d blocks" % (k, len(anchors), binf["planes"], binf["blocks"]))
        else:
            verts, groups = relief(plate, Z, world, k, step, far, sky)
            for g, faces in groups.items():
                if faces:
                    v, f = compact(verts, faces)
                    meshes.append({"name": "set%s_%s_%03d" % (tag, g, k), "v": v, "f": f, "colour": g, "layer": 1 if g == "ground" else 0})
            log_("    plate frame %d: %d anchors, %d faces" % (k, len(anchors), sum(len(x) for x in groups.values())))
        imwrite_u(os.path.join(c["_work"], "plate%s_%03d.jpg" % (tag, k)), plate, 85)
        dv = np.clip(np.where(sky, 0, 255 * (1 - np.log(np.minimum(Z, far)) / np.log(far))), 0, 255).astype(np.uint8)
        imwrite_u(os.path.join(c["_work"], "depth%s_%03d.jpg" % (tag, k)), cv2.applyColorMap(dv, cv2.COLORMAP_TURBO), 85)
    if not meshes:
        info["warning"] = "no set mesh - flat ground used"
        return flat_set(world, N, far), info
    return meshes, info

# -*- coding: utf-8 -*-
"""automatic camera for one shot: rotation (+ zoom) tracking, steadying, and camera height / pitch / roll.

rotation: background features are tracked (people masked with their boxes from nearby frames too), and every frame
is fitted with a camera model that can only TURN and ZOOM (x' ~ S(s) R x).  Fitting that model directly (instead of
a free homography squeezed into a rotation) keeps a zoom from turning into a pan, and keeps the picture's
perspective changes from moving people / water / travel out of the rotation.  Far features (depth model) count more:
camera travel moves near things, turning moves everything alike.  Frame-to-frame steps that disagree with their
neighbours are replaced, and total turns that the features do not support (few inliers, large residual) are damped.

height / pitch / roll: see estimate_ground."""
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d, median_filter
from common import *


# ----------------------------------------------------------------------------- pitch / roll cues from one picture
def up_vector(pitch_deg, roll_deg=0.0):
    """world up in camera coords (x right, y down, z forward); pitch + = looking down (same as hints pitch_down_deg)"""
    return Rotation.from_euler("xz", [np.radians(pitch_deg), np.radians(roll_deg)]).apply([0, -1.0, 0])


def vertical_lines_cue(gray, mask, cam, max_pitch=60.0):
    """pitch / roll from the vertical vanishing point of near-vertical line segments (walls, doors, posts, trunks).
    returns dict(pitch, roll, n, spread, score) or None.  Natural scenes with few straight verticals give None."""
    H, W = gray.shape
    lsd = cv2.createLineSegmentDetector()
    segs = lsd.detect(cv2.GaussianBlur(gray, (3, 3), 0))[0]
    if segs is None:
        return None
    s = segs.reshape(-1, 4)
    L = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    ang = np.degrees(np.arctan2(np.abs(s[:, 2] - s[:, 0]), np.abs(s[:, 3] - s[:, 1])))   # 0 = vertical
    keep = (L > 0.05 * H) & (ang < 25)
    if mask is not None:
        mx = ((s[:, 0] + s[:, 2]) / 2).astype(int).clip(0, W - 1); my = ((s[:, 1] + s[:, 3]) / 2).astype(int).clip(0, H - 1)
        keep &= mask[my, mx] > 0
    s, L = s[keep], L[keep]
    if len(s) < 6:
        return None
    p1 = np.c_[(s[:, 0] - W / 2) / cam.f, (s[:, 1] - H / 2) / cam.f, np.ones(len(s))]
    p2 = np.c_[(s[:, 2] - W / 2) / cam.f, (s[:, 3] - H / 2) / cam.f, np.ones(len(s))]
    mid = (p1 + p2) / 2
    d = p2[:, :2] - p1[:, :2]; d /= np.linalg.norm(d, axis=1, keepdims=True)
    def score(p, r, tol):
        u = up_vector(p, r)
        l = np.cross(mid, u)                                   # line through midpoint and the vanishing point
        e = np.c_[l[:, 1], -l[:, 0]]; e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
        sa = np.abs(d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0])   # sin of the angle between segment and expected line
        return sa, np.sum(L * np.clip(1 - (sa / np.sin(np.radians(tol))) ** 2, 0, None))
    best = (-1, 0, 0)
    for p in np.arange(-30, max_pitch + 0.1, 1.0):
        for r in np.arange(-8, 8.1, 1.0):
            sc = score(p, r, 1.5)[1]
            if sc > best[0]:
                best = (sc, p, r)
    _, p0, r0 = best
    for p in np.arange(p0 - 1, p0 + 1.01, 0.1):
        for r in np.arange(r0 - 1, r0 + 1.01, 0.25):
            sc = score(p, r, 1.0)[1]
            if sc > best[0]:
                best = (sc, p, r)
    sc, p, r = best
    sa, _ = score(p, r, 1.0)
    inl = sa < np.sin(np.radians(1.0))
    xs = mid[inl, 0]
    spread = float(np.sum(L[inl] * np.abs(xs)) / max(np.sum(L[inl]), 1e-6))   # pitch needs lines away from the centre
    # how sharply the score peaks in pitch: compare with pitch +-4 deg
    side = max(score(p - 4, r, 1.0)[1], score(p + 4, r, 1.0)[1])
    return {"pitch": float(p), "roll": float(r), "n": int(inl.sum()), "len": float(L[inl].sum() / H),
            "spread": spread, "peak": float(sc / max(side, 1e-6)), "share": float(L[inl].sum() / L.sum())}


def depth_horizon_cue(disp, mask, cam):
    """pitch / roll from the depth model's ground: on a flat ground the inverse depth is a linear function of the
    pixel position, zero at the horizon (the model gives ~0 for the far sky).  mask: candidate ground pixels.
    returns dict(pitch, roll, inl) or None"""
    H, W = disp.shape
    ys, xs = np.nonzero(mask[::6, ::6]); ys = ys * 6; xs = xs * 6
    if len(xs) < 60:
        return None
    dv = disp[ys, xs].astype(np.float64)
    A = np.c_[xs - W / 2, ys - H / 2, np.ones(len(xs))]
    rng = np.random.default_rng(1); best = None; bn = 0
    tol = 0.04 * (np.percentile(disp, 95) - np.percentile(disp, 5) + 1e-6)
    for _ in range(300):
        i = rng.choice(len(xs), 3, replace=False)
        try:
            q = np.linalg.solve(A[i], dv[i])
        except np.linalg.LinAlgError:
            continue
        if q[1] <= 0:                                           # nearer further down: required for a ground
            continue
        inl = np.abs(A @ q - dv) < tol
        if inl.sum() > bn:
            best, bn = inl, int(inl.sum())
    if best is None:
        return None
    q = np.linalg.lstsq(A[best], dv[best], rcond=None)[0]
    if q[1] <= 1e-9:
        return None
    yh = -q[2] / q[1]
    return {"pitch": float(np.degrees(np.arctan2(-yh, cam.f))), "roll": float(np.degrees(np.arctan2(-q[0], q[1]))),
            "inl": bn / float(len(xs))}


# ----------------------------------------------------------------------------- rotation + zoom tracking
def _boxes_by_frame(pose):
    boxes = {}
    for p in pose["people"]:
        for f, b in p.get("boxes", {}).items():
            boxes.setdefault(int(f), []).append(b)
    return boxes


def _people_mask(boxes, i, y0, W, Hp, N, span=3, px=0.35, py=0.2):
    m = np.full((Hp, W), 255, np.uint8)
    for j in range(max(0, i - span), min(N, i + span + 1)):       # the detector misses people in some frames
        for b in boxes.get(j, []):
            w, h = b[2] - b[0], b[3] - b[1]
            m[max(0, int(b[1] - y0 - py * h)):max(0, int(b[3] - y0 + 0.6 * py * h)), max(0, int(b[0] - px * w)):max(0, int(b[2] + px * w))] = 0
    return m


def _fit_rot_zoom(a, b, w, R0, s0, zoom=True):
    """a, b: normalised coords (ref, cur); fit b ~ pi(diag(s,s,1) R [a,1]).  returns R, s, residuals (normalised)"""
    A = np.c_[a, np.ones(len(a))]
    sw = np.sqrt(w)
    def res(q):
        R = Rotation.from_rotvec(q[:3]).as_matrix()
        X = A @ R.T
        s = np.exp(q[3]) if zoom else 1.0
        p = s * X[:, :2] / X[:, 2:3]
        return ((p - b) * sw[:, None]).ravel()
    q0 = np.r_[Rotation.from_matrix(R0).as_rotvec(), np.log(s0)]
    sol = least_squares(res, q0, loss="huber", f_scale=1.0 / 1500.0, x_scale=np.r_[0.01, 0.01, 0.01, 0.01])
    R = Rotation.from_rotvec(sol.x[:3]).as_matrix(); s = float(np.exp(sol.x[3])) if zoom else 1.0
    X = A @ R.T; p = s * X[:, :2] / X[:, 2:3]
    return R, s, np.linalg.norm(p - b, axis=1)


def track(frames, pose, c, depth=None, log_=None):
    """per-frame rotation R[i] (frame-0 cam -> frame-i cam) and zoom s[i] (focal f_i = s[i] * f_0).
    depth: optional callable(bgr picture) -> relative inverse depth, used to favour far features."""
    x0, y0, W, Hp = pose["picture_area"]
    cam = Camera(W, Hp, c["lens_mm"])
    G = [cv2.cvtColor(f[y0:y0 + Hp], cv2.COLOR_BGR2GRAY) for f in frames]
    N = len(G)
    boxes = _boxes_by_frame(pose)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    border = np.zeros((Hp, W), np.uint8); bw = max(8, W // 100)
    border[bw:Hp - bw, bw:W - bw] = 255

    def new_ref(i):
        m = cv2.bitwise_and(_people_mask(boxes, i, y0, W, Hp, N), border)
        p = cv2.goodFeaturesToTrack(G[i], 1500, 0.01, 12, mask=m)
        p = np.zeros((0, 2), np.float32) if p is None else p.reshape(-1, 2)
        wt = np.ones(len(p))
        if depth is not None and len(p):
            d = depth(frames[i][y0:y0 + Hp])
            dv = d[np.clip(p[:, 1].astype(int), 0, Hp - 1), np.clip(p[:, 0].astype(int), 0, W - 1)]
            lo, hi = np.percentile(d, 2), np.percentile(d, 98)
            near = np.clip((dv - lo) / max(hi - lo, 1e-6), 0, 1)        # 0 far .. 1 near
            wt = 1.0 / (1.0 + 8.0 * near ** 2)
        return p, wt

    R = [np.eye(3)]; S = [1.0]; Rref = np.eye(3); sref = 1.0
    pts_ref, wref = new_ref(0); cur = pts_ref.copy(); alive = np.ones(len(cur), bool)
    resid, inl_share, lost, nlist = [], [], [], []
    Rrel, srel = np.eye(3), 1.0
    for i in range(1, N):
        if len(cur) and alive.any():
            p1, st, _ = cv2.calcOpticalFlowPyrLK(G[i - 1], G[i], cur.reshape(-1, 1, 2).astype(np.float32), None, **lk)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(G[i], G[i - 1], p1, None, **lk)
            p1 = p1.reshape(-1, 2); pb = pb.reshape(-1, 2)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pb - cur, axis=1) < 0.8)
            ok &= (p1[:, 0] > 2) & (p1[:, 0] < W - 2) & (p1[:, 1] > 2) & (p1[:, 1] < Hp - 2)
            mk = _people_mask(boxes, i, y0, W, Hp, N); idx = np.where(ok)[0]
            ok[idx] = mk[p1[idx, 1].astype(int), p1[idx, 0].astype(int)] > 0
            alive &= ok; cur = p1
        n = int(alive.sum()); nlist.append(n)
        got = False
        if n >= 25:
            a = (pts_ref[alive] - [W / 2, Hp / 2]) / cam.f; b = (cur[alive] - [W / 2, Hp / 2]) / cam.f
            Hm, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.5 / cam.f)
            if Hm is not None and inl.sum() >= 20:
                inl = inl.ravel().astype(bool)
                Hn = Hm / np.cbrt(np.linalg.det(Hm)); U, _, Vt = np.linalg.svd(Hn); R0 = U @ Vt
                if np.linalg.det(R0) < 0: R0 = -R0
                s0 = float(np.sqrt(abs(np.linalg.det(Hm[:2, :2] / Hm[2, 2])))) if abs(Hm[2, 2]) > 1e-9 else 1.0
                s0 = float(np.clip(s0, 0.7, 1.4))
                Rr, sr, e = _fit_rot_zoom(a[inl], b[inl], wref[alive][inl], R0, s0, zoom=c.get("camera_zoom", True))
                ein = e * cam.f
                if np.median(ein) < 3.0:
                    Rrel, srel = Rr, sr
                    resid.append(float(np.median(ein))); inl_share.append(float(inl.mean()))
                    got = True
        if not got:
            lost.append(i)                                       # keep the last relative motion's result (hold)
            R.append(R[-1]); S.append(S[-1])
        else:
            R.append(Rrel @ Rref); S.append(srel * sref)
        if n < 300 or not got:
            Rref = R[-1]; sref = S[-1]
            pts_ref, wref = new_ref(i); cur = pts_ref.copy(); alive = np.ones(len(cur), bool)
    return {"R": R, "zoom": S, "resid": resid, "inlier_share": inl_share, "lost": lost, "cam": cam, "y0": y0, "W": W, "H": Hp}


def _flow_basis(x, y, d):
    """image motion (normalised coords) per unit of [wx, wy, wz, zoom, tx, ty, tz] (instantaneous camera motion);
    d = inverse depth (relative).  rotation and zoom move every point alike, travel moves near points more."""
    o = np.zeros_like(x); one = np.ones_like(x)
    Bu = np.c_[x * y, -(1 + x * x), y, x, -d * one, o, d * x]
    Bv = np.c_[1 + y * y, -x * y, -x, y, o, -d * one, d * y]
    return Bu, Bv


def _fit_flow(a, b, d, use_t, thr, wt=None):
    """robust fit of the instantaneous motion model to point moves a -> b.  wt: point weights (far points count
    more: they show the turn without the travel).  returns params, inlier mask"""
    wt = np.ones(len(a)) if wt is None else np.asarray(wt, float)
    x, y = a[:, 0], a[:, 1]
    Bu, Bv = _flow_basis(x, y, d)
    if not use_t:
        Bu, Bv = Bu[:, :4], Bv[:, :4]
    du, dv = b[:, 0] - x, b[:, 1] - y
    n = len(x); k = Bu.shape[1]
    lam = np.r_[np.zeros(4), np.full(k - 4, 1e-3)]                    # travel only when the depth pattern asks for it
    def lsq(idx, w=None):
        A = np.r_[Bu[idx], Bv[idx]]; r = np.r_[du[idx], dv[idx]]
        if w is not None:
            ww = np.r_[w, w]; A = A * ww[:, None]; r = r * ww
        AtA = A.T @ A + np.diag(lam) * max(len(idx), 1) * 1e-4
        return np.linalg.solve(AtA, A.T @ r)
    def err(q):
        return np.hypot(Bu @ q - du, Bv @ q - dv)
    rng = np.random.default_rng(0)
    best, bn = None, -1.0
    m = max(4, (k + 1) // 2 + 1)
    pr = wt / wt.sum()
    for _ in range(150):
        idx = rng.choice(n, m, replace=False, p=pr)
        try:
            q = lsq(idx)
        except np.linalg.LinAlgError:
            continue
        inl = err(q) < thr
        sc = float(wt[inl].sum())
        if sc > bn:
            best, bn = inl, sc
    if best is None or best.sum() < 12:
        return None, None
    idx = np.where(best)[0]; q = lsq(idx, np.sqrt(wt[idx]))
    for _ in range(4):                                                  # Huber IRLS on the inliers
        e = err(q)[idx]; w = np.sqrt(np.minimum(1.0, (0.5 * thr) / np.maximum(e, 1e-9)) * wt[idx])
        q = lsq(idx, w)
    inl = err(q) < thr
    return q, inl


def _grid_features(g, mask, nx=8, ny=5, per=45):
    """corners spread over the whole picture (a textured foreground must not take all of them)"""
    H, W = g.shape; out = []
    for j in range(ny):
        for i in range(nx):
            xa, xb = W * i // nx, W * (i + 1) // nx; ya, yb = H * j // ny, H * (j + 1) // ny
            p = cv2.goodFeaturesToTrack(g[ya:yb, xa:xb], per, 0.003, 6, mask=mask[ya:yb, xa:xb])
            if p is not None:
                out.append(p.reshape(-1, 2) + [xa, ya])
    return np.vstack(out).reshape(-1, 1, 2).astype(np.float32) if out else None


def _dense_points(g0, g1, mask, dis, step=8):
    """point moves g0 -> g1 from dense optical flow, sampled on a grid where the picture has some texture, kept
    only where the backward flow brings the point back (hazy hills and beaches have few corners but do track)"""
    fw = dis.calc(g0, g1, None); bw = dis.calc(g1, g0, None)
    H, W = g0.shape
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    ys = ys.ravel(); xs = xs.ravel()
    ev = cv2.cornerMinEigenVal(g0, 5)
    tex = ev[ys, xs]
    keep = (mask[ys, xs] > 0) & (tex > max(5e-6, 0.01 * float(np.percentile(ev, 99))))
    ys, xs = ys[keep], xs[keep]
    p0 = np.c_[xs, ys].astype(np.float32)
    f = fw[ys, xs]; p1 = p0 + f
    xi = np.clip(np.round(p1[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(p1[:, 1]).astype(int), 0, H - 1)
    back = p1 + bw[yi, xi]
    g = (np.linalg.norm(back - p0, axis=1) < 0.6) & (p1[:, 0] > 1) & (p1[:, 0] < W - 1) & (p1[:, 1] > 1) & (p1[:, 1] < H - 1)
    return p0, p1, g


def track_flow(frames, pose, c, depth=None, scale=0.5):
    """frame-to-frame camera turn + zoom with the camera's TRAVEL separated out by depth (depth model): travel moves
    near points more than far ones, turning and zooming move all points alike.  Without depth this is a plain
    turn + zoom fit.  returns R[i] (frame-0 cam -> frame-i cam), zoom[i], per-step info"""
    x0, y0, W, Hp = pose["picture_area"]
    cam = Camera(W, Hp, c["lens_mm"])
    Ws, Hs = int(W * scale), int(Hp * scale); fs = cam.f * scale
    G = [cv2.cvtColor(cv2.resize(f[y0:y0 + Hp], (Ws, Hs), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames]
    N = len(G)
    boxes = _boxes_by_frame(pose)
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    border = np.zeros((Hs, Ws), np.uint8); bw = max(6, Ws // 80); border[bw:Hs - bw, bw:Ws - bw] = 255
    D = {}
    def disp(i):                                       # relative inverse depth 0 (far) .. 1 (near), every 2nd frame
        if depth is None:
            return None
        k = i - i % 2
        if k not in D:
            dd = cv2.resize(depth(frames[k][y0:y0 + Hp]), (Ws, Hs), interpolation=cv2.INTER_AREA)
            lo, hi = np.percentile(dd, 1), np.percentile(dd, 99)
            D[k] = np.clip((dd - lo) / max(hi - lo, 1e-6), 0, 1.2)
        return D[k]
    R = [np.eye(3)]; S = [1.0]; lost = []; resid = []; trav = []; dbg = []; shares = []
    use_t = depth is not None and c.get("camera_travel_model", True)
    thr = 1.2 / fs
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    masks = {}

    def measure(k, i):
        """camera turn + zoom from keyframe k to frame i: (R_rel, s_rel, inliers, weighted share, median move px)"""
        if k not in masks:
            masks[k] = cv2.bitwise_and(cv2.resize(_people_mask(boxes, k, y0, W, Hp, N, span=1, px=0.1, py=0.08), (Ws, Hs),
                                                  interpolation=cv2.INTER_NEAREST), border)
        p0, p1, g = _dense_points(G[k], G[i], masks[k], dis)
        if g.sum() < 25:
            return None
        a = (p0[g] - [Ws / 2, Hs / 2]) / fs; b = (p1[g] - [Ws / 2, Hs / 2]) / fs
        dm = disp(k)
        d = dm[p0[g, 1].astype(int).clip(0, Hs - 1), p0[g, 0].astype(int).clip(0, Ws - 1)] if dm is not None else np.zeros(len(a))
        wt = 0.1 + np.clip(1.0 - d, 0, 1) ** 2 if dm is not None else np.ones(len(a))
        q, inl = _fit_flow(a, b, d, use_t, thr * (1.0 + 0.1 * min(i - k, 10)), wt)
        if q is None or inl.sum() < 12 or wt[inl].sum() < 0.2 * wt.sum():
            return None
        Bu, Bv = _flow_basis(a[:, 0], a[:, 1], d)             # remove the travel part, fit the exact turn + zoom to the rest
        tr = np.c_[Bu[:, 4:] @ q[4:], Bv[:, 4:] @ q[4:]] if len(q) > 4 else 0.0
        Rr, sr, e = _fit_rot_zoom(a[inl], (b - tr)[inl], wt[inl], np.eye(3), 1.0, zoom=c.get("camera_zoom", True))
        mv = float(np.median(np.linalg.norm(b - a, axis=1)) * fs)
        return Rr, sr, int(inl.sum()), float(wt[inl].sum() / wt.sum()), mv, float(np.median(e) * fs / scale), \
            float(np.linalg.norm(q[4:6])) if len(q) > 4 else 0.0

    k = 0; n_ref = None
    for i in range(1, N):
        if float(np.mean(cv2.absdiff(G[i], G[i - 1]))) < 0.3:          # repeated frame
            R.append(R[-1]); S.append(S[-1]); continue
        m = measure(k, i)
        weak = m is None or (n_ref and m[2] < 0.5 * n_ref) or m[4] > 0.10 * Ws or i - k > 40
        if weak and k != i - 1:                                         # new keyframe: the previous frame
            k = i - 1; n_ref = None
            m = measure(k, i)
        if m is None:
            lost.append(i); R.append(R[-1]); S.append(S[-1]); k = i; n_ref = None
            continue
        if n_ref is None:
            n_ref = m[2]
        R.append(m[0] @ R[k]); S.append(m[1] * S[k])
        resid.append(m[5]); trav.append(m[6]); shares.append(m[3]); dbg.append([i, k, m[2], round(m[3], 2), round(m[4], 1)])
    return {"R": R, "zoom": S, "resid": resid, "inlier_share": shares, "lost": lost,
            "travel": trav, "dbg": dbg, "cam": cam, "y0": y0, "W": W, "H": Hp}


def _clean_steps(R, S, fixed=None, lost=None):
    """frame-to-frame steps; a step far off its neighbours' median (a tracking jump) is replaced by that median.
    fixed: steps that are known (repeated frames = no move) and are left alone"""
    rv = np.array([Rotation.from_matrix(np.asarray(R[i]) @ np.asarray(R[i - 1]).T).as_rotvec() for i in range(1, len(R))])
    ls = np.diff(np.log(np.asarray(S, float)))
    fx = np.zeros(len(rv), bool) if fixed is None else np.asarray(fixed, bool)
    if lost is not None and len(lost):                    # no measurement: carry the motion of the frames around it
        ls_ = np.zeros(len(rv), bool); ls_[[i - 1 for i in lost if 0 < i <= len(rv)]] = True
        good = np.where(~ls_ & ~fx)[0]
        if len(good) >= 2:
            for k in range(3):
                rv[ls_, k] = np.interp(np.where(ls_)[0], good, rv[good, k])
            ls[ls_] = np.interp(np.where(ls_)[0], good, ls[good])
        fx = fx | ls_
    live = np.where(~fx)[0]
    if len(live) < 5:
        return rv, ls
    rl, ll = rv[live], ls[live]
    med = median_filter(rl, size=(7, 1), mode="nearest"); mls = median_filter(ll, size=7, mode="nearest")
    dev = np.linalg.norm(rl - med, axis=1)
    mad = np.median(dev) + 1e-5
    bad = dev > np.maximum(8 * mad, np.radians(0.25))
    rl[bad] = med[bad]
    badz = np.abs(ll - mls) > max(8 * np.median(np.abs(ll - mls)) + 1e-5, 0.004)
    ll[badz] = mls[badz]
    rv[live] = rl; ls[live] = ll
    return rv, ls


def solve(frames, pose, c, depth=None):
    """camera.json-style result (same keys as camera_solve.solve) + "zoom" (focal scale per frame)"""
    if c.get("camera_tracker", "flow") == "flow":
        t = track_flow(frames, pose, c, depth)
    else:
        t = track(frames, pose, c, depth)
    cam = t["cam"]; N = len(t["R"])
    fixed = [np.allclose(t["R"][i], t["R"][i - 1]) and t["zoom"][i] == t["zoom"][i - 1] and i not in t["lost"] for i in range(1, N)]
    rv, ls = _clean_steps(t["R"], t["zoom"], fixed, t["lost"])
    R = [np.eye(3)]
    for v in rv:
        R.append(Rotation.from_rotvec(v).as_matrix() @ R[-1])
    zoom = np.exp(np.r_[0.0, np.cumsum(ls)]) if len(ls) else np.ones(N)
    ang = [float(np.degrees(np.linalg.norm(Rotation.from_matrix(r).as_rotvec()))) for r in R]
    par = float(np.percentile(t["resid"], 90)) if t["resid"] else 0.0
    warn = []
    if t["lost"]:
        warn.append("camera lost at frames %s (cut or fast move) - check this clip" % t["lost"][:10])
    if par > 2.0:
        warn.append("camera seems to MOVE (parallax %.1f px): far set may slide a little" % par)
    if max(ang) > 60 or (len(rv) and np.degrees(np.max(np.linalg.norm(rv, axis=1))) > 2.5):
        warn.append("camera turns %.0f deg (fast or large turn) - check the turn; a camera travelling with the people "
                    "can look like a turn" % max(ang))
    zr = float(np.max(zoom) / np.min(zoom))
    if zr > 1.03:
        warn.append("camera ZOOMS (focal x%.2f over the shot): kept out of the rotation" % (zoom[-1] / zoom[0]))
    return {"R": [r.tolist() for r in R], "zoom": [float(z) for z in zoom], "W": t["W"], "H": t["H"], "y0": t["y0"],
            "f": cam.f, "lens_mm": c["lens_mm"], "max_rotation_deg": max(ang), "parallax_px": par, "cuts": t["lost"],
            "inlier_share": float(np.median(t["inlier_share"])) if t["inlier_share"] else 0.0, "warnings": warn}


# ----------------------------------------------------------------------------- height / pitch / roll
def _person_obs(people, cam, y0):
    """standing-person samples: (unit id, foot px, shoulder px).  unit = person x foot-height band, so a person who
    stands still for 100 frames counts once, and one who walks towards the camera counts several times"""
    obs = []
    for pid, (fs, K) in people.items():
        for f, k in zip(fs, K):
            k = np.array(k, float).copy(); k[:, 1] -= y0
            if min(k[15, 2], k[16, 2]) < 0.4 or k[5, 2] < 0.4 or k[6, 2] < 0.4 or min(k[11, 2], k[12, 2]) < 0.3:
                continue
            foot = (k[15, :2] + k[16, :2]) / 2
            sh = (k[5, :2] + k[6, :2]) / 2
            hp = (k[11, :2] + k[12, :2]) / 2
            leg = np.linalg.norm(foot - hp); torso = np.linalg.norm(sh - hp)
            if torso < 8 or leg < 1.2 * torso or foot[1] > cam.H - 3:     # crouching / bent / feet cut off
                continue
            if abs(sh[0] - foot[0]) > 0.6 * np.linalg.norm(sh - foot):    # lying / falling
                continue
            obs.append((str(pid) + ":%d" % int(foot[1] / (cam.H / 20.0)), int(f), foot, sh))
    return obs


def _upper_obs(people, cam, y0):
    """people seen from the hips up (feet cut off by the frame or hidden), upright, not sitting:
    (unit id, frame, shoulder px, torso length px)"""
    obs = []
    for pid, (fs, K) in people.items():
        for f, k in zip(fs, K):
            k = np.array(k, float).copy(); k[:, 1] -= y0
            if k[5, 2] < 0.5 or k[6, 2] < 0.5 or min(k[11, 2], k[12, 2]) < 0.5:
                continue
            if min(k[15, 2], k[16, 2]) >= 0.4:                          # full body: the feet are better
                continue
            sh = (k[5, :2] + k[6, :2]) / 2; hp = (k[11, :2] + k[12, :2]) / 2
            t = np.linalg.norm(sh - hp)
            if t < 25 or abs(sh[0] - hp[0]) > 0.3 * t or sh[1] > hp[1]:
                continue
            kn = [j for j in (13, 14) if k[j, 2] >= 0.4]
            if kn and np.mean([k[j, 1] for j in kn]) - hp[1] < 0.55 * t:   # thighs near level: sitting
                continue
            obs.append(("u" + str(pid) + ":%d" % int(hp[1] / (cam.H / 10.0)), int(f), sh, t))
    return obs


def upper_heights(obs, cam, pitch_deg, Hsh, T, R=None):
    """camera height each upper-body sample asks for: distance from the torso length (T metres), then the camera
    sits (shoulder height above the ground - shoulder height above the camera)"""
    S = np.array([np.r_[cam.norm(*o[2]), 1.0] for o in obs])
    Z = T * cam.f / np.array([o[3] for o in obs])
    U = _ups(obs, pitch_deg, R)
    Xs = S * Z[:, None]
    return Hsh - np.sum(Xs * U, 1)


def _ups(obs, pitch_deg, R):
    """world up in each sample's camera (the camera may tilt / pan during the shot)"""
    u = up_vector(pitch_deg, 0.0)
    if R is None:
        return np.tile(u, (len(obs), 1))
    n = len(R)
    return np.array([np.asarray(R[min(o[1], n - 1)]) @ u for o in obs])


def people_heights(obs, cam, pitch_deg, roll_deg, Hs, R=None):
    """camera height each sample asks for, if the camera had this pitch / roll (feet on the ground, shoulders Hs above)"""
    U = _ups(obs, pitch_deg, R) if roll_deg == 0 else np.tile(up_vector(pitch_deg, roll_deg), (len(obs), 1))
    F = np.array([np.r_[cam.norm(*o[2]), 1.0] for o in obs]); S = np.array([np.r_[cam.norm(*o[3]), 1.0] for o in obs])
    den = np.sum(F * U, 1)
    ok = den < -1e-4
    Xf = F * (-1.0 / np.where(ok, den, -1.0))[:, None]                # foot with the camera 1 m above the ankle plane
    # shoulder = t * S closest to Xf + k * u   ->  2x2 normal equations
    a11 = np.sum(S * S, 1); a12 = -np.sum(S * U, 1); a22 = np.ones(len(S))
    b1 = np.sum(S * Xf, 1); b2 = -np.sum(Xf * U, 1)
    det = a11 * a22 - a12 * a12
    k = (a11 * b2 - a12 * b1) / np.where(np.abs(det) > 1e-12, det, 1e-12)
    ok &= k > 0.02
    hc = np.where(ok, 0.065 + Hs / np.where(ok, k, 1.0), np.nan)
    return hc


# tuned on the hand-set hints of 71 shots (sandbox\camera\tools\tune2.py)
GROUND_PAR = {"sig_people": 0.15, "invalid": 8.0, "h_prior": (1.5, 0.35), "cue_cap": 9.0,
              "dh_scale": 1.4, "dh_off": 0.0, "dh_sig": 1.0, "h_default": 1.5,
              "upper": True, "upper_if_full_below": 1, "upper_w": 0.5, "sig_upper": 0.3}


def estimate_ground(people, cam, y0, c, cues=(), prior=(5.0, 10.0), R=None):
    """camera height, pitch and roll at the shot's first frame.
    roll is 0 (film cameras are level; the hints never needed a roll).  pitch is the most likely one given:
      - the people: every standing person sample asks for a camera height at each pitch; at the right pitch they
        agree (strong only when people stand at different distances),
      - picture cues (depth-model horizon, vertical lines), each with its own uncertainty,
      - typical film cameras: pitch ~5 +- 10 deg, height ~1.5 m (0.5 - 3 m).
    the height then follows from the people at that pitch (or 1.6 m without people).
    cues: [(pitch_deg, sigma_deg, name)].  returns up (cam0 coords), height, info"""
    Hs = c["person_height"] * 1.45 / 1.75 - 0.065 * c["person_height"] / 1.75
    Hsh = c["person_height"] * 1.45 / 1.75; T = c["person_height"] * 0.29
    obs = _person_obs(people, cam, y0)
    G = GROUND_PAR
    obs_u = _upper_obs(people, cam, y0) if G.get("upper", True) else []
    if len(set(o[0] for o in obs)) >= G.get("upper_if_full_below", 1):
        obs_u = []                                                       # full-body people are enough
    units = sorted(set(o[0] for o in obs)); units_u = sorted(set(o[0] for o in obs_u))
    uidx = np.array([units.index(o[0]) for o in obs], int) if obs else np.zeros(0, int)
    uidx_u = np.array([units_u.index(o[0]) for o in obs_u], int) if obs_u else np.zeros(0, int)
    grid = np.arange(-25.0, 50.01, 0.5)
    cost = np.zeros(len(grid)); hgrid = np.full(len(grid), np.nan)
    pm, ps = prior
    cost += ((grid - pm) / ps) ** 2
    for p, s, nm in cues:
        if nm == "depth horizon":            # the depth model's ground bends towards the horizon: its pitch reads low
            p = G["dh_scale"] * p + G["dh_off"]; s = s * G["dh_sig"]
        cost += np.minimum(((grid - p) / s) ** 2, G["cue_cap"])        # a wrong cue can cost at most 3 sigma
    info = {"samples": len(obs), "units": len(units), "upper_units": len(units_u),
            "cues": [{"name": n, "pitch": round(p, 1), "sigma": round(s, 1)} for p, s, n in cues]}
    if (len(units) >= 1 and len(obs) >= 3) or (len(units_u) >= 1 and len(obs_u) >= 3):
        pc = np.zeros(len(grid))
        wu_full = np.array([min(1.0, np.sum(uidx == j) / 5.0) for j in range(len(units))])
        wu_up = np.array([G.get("upper_w", 0.5) * min(1.0, np.sum(uidx_u == j) / 5.0) for j in range(len(units_u))])
        for gi, p in enumerate(grid):
            hu = np.zeros(0); wu = np.zeros(0); su = np.zeros(0)
            if len(obs):
                h = people_heights(obs, cam, p, 0.0, Hs, R)
                hu = np.array([np.nanmedian(h[uidx == j]) if np.isfinite(h[uidx == j]).any() else np.nan for j in range(len(units))])
                wu = wu_full; su = np.full(len(units), G["sig_people"])
            if len(obs_u):
                h2 = upper_heights(obs_u, cam, p, Hsh, T, R)
                hu = np.r_[hu, [np.nanmedian(h2[uidx_u == j]) for j in range(len(units_u))]]
                wu = np.r_[wu, wu_up]; su = np.r_[su, np.full(len(units_u), G.get("sig_upper", 0.2))]
            good = np.isfinite(hu) & (hu > 0.05)
            bad_w = float(np.sum(wu[~good]))                             # feet above the horizon: impossible
            if good.sum() == 0:
                pc[gi] = G["invalid"] * bad_w + 50.0; continue
            lh = np.log(hu[good]); w = wu[good]
            sg = su[good]
            med = float(np.exp(np.sum(w * lh) / np.sum(w)))
            for _ in range(3):                                           # weighted robust centre
                r = (lh - np.log(med)) / sg
                ww = w / np.sqrt(1 + r ** 2)
                med = float(np.exp(np.sum(ww * lh) / np.sum(ww)))
            r = (lh - np.log(med)) / sg                                  # people's height and pose vary ~8 %
            pc[gi] = float(np.sum(w * 2 * (np.sqrt(1 + r ** 2) - 1))) + G["invalid"] * bad_w
            hgrid[gi] = med
            hm, hs = G["h_prior"]
            cost[gi] += ((np.log(med) - np.log(hm)) / hs) ** 2          # height prior (0.5 - 4 m within 2 sigma)
        cost += pc
        info["people_cost_span"] = float(pc.max() - pc.min())
    gi = int(np.argmin(cost))
    pitch = float(grid[gi])
    if np.isfinite(hgrid[gi]):
        hc = float(np.clip(hgrid[gi], 0.3, 15.0))
        info["source"] = "people + cues" if units else "upper bodies + cues (no full-body people)"
    else:
        hc = float(c.get("default_camera_height", G["h_default"])); info["source"] = "cues only (no full-body people): height %.1f m assumed" % hc
        info["warning"] = "no full-body people to measure the camera height - using %.1f m" % hc
    # 1-sigma range of the pitch (where the cost rises by 1)
    ok = grid[cost <= cost[gi] + 1.0]
    info.update(pitch_deg=pitch, roll_deg=0.0, camera_height=hc, pitch_range=[float(ok.min()), float(ok.max())])
    return up_vector(pitch, 0.0), hc, info


_DEPTH = {}
def get_depth(c):
    if "m" not in _DEPTH:
        try:
            from scene_depth import DepthModel
            _DEPTH["m"] = DepthModel(pick_device(c.get("depth_device", c.get("device", "cuda"))))
        except Exception as e:                                           # no model / no GPU: geometry without it
            log("    camera: depth model not available (%s)" % e)
            _DEPTH["m"] = None
    return _DEPTH["m"]


def picture_cues(frames, pose, y0, cam, R, depth, nf=5):
    """pitch cues (in frame-0 camera terms) from a few early frames"""
    N = len(frames); Hp, W = cam.H, cam.W
    boxes = _boxes_by_frame(pose)
    idx = sorted(set(int(i) for i in np.linspace(0, min(N - 1, 24), nf)))
    dh, vl = [], []
    for i in idx:
        img = frames[i][y0:y0 + Hp]
        pm = _people_mask(boxes, i, y0, W, Hp, N)
        Ri = np.asarray(R[i])
        def to0(p):                                   # pitch seen in frame i -> pitch at frame 0
            u0 = Ri.T @ up_vector(p, 0.0)
            return float(np.degrees(np.arcsin(np.clip(-u0[2], -1, 1))))
        if depth is not None:
            gm = pm > 0; gm[:int(Hp * 0.62)] = False
            d = depth(img)
            r = depth_horizon_cue(d, gm, cam)
            if r and r["inl"] > 0.5:
                dh.append((to0(r["pitch"]), r["inl"]))
        r = vertical_lines_cue(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pm, cam)
        if r and r["n"] >= 12 and r["spread"] > 0.12 and r["peak"] > 1.4:
            vl.append((to0(r["pitch"]), r["n"]))
    cues = []
    if dh:
        p = np.array([a for a, _ in dh]); q = np.array([b for _, b in dh])
        med = float(np.median(p)); mad = float(np.median(np.abs(p - med)))
        sig = 3.0 + 2.0 * mad + 8.0 * max(0.0, 0.9 - float(np.median(q)))
        if len(dh) >= max(2, len(idx) // 2):
            cues.append((med, sig, "depth horizon"))
    if vl:
        p = np.array([a for a, _ in vl]); med = float(np.median(p)); mad = float(np.median(np.abs(p - med)))
        if len(vl) >= max(2, len(idx) // 2) and mad < 3:
            cues.append((med, 2.0 + 1.5 * mad, "vertical lines"))
    return cues


def estimate_shot(frames, pose, people, cam, y0, Hp, c, depth="auto"):
    """everything for one shot: (camj like camera_solve.solve after steadying, up (cam0), height, ground info)"""
    if isinstance(depth, str):
        depth = get_depth(c) if c.get("camera_depth", True) else None
    camj = solve(frames, pose, c, depth if c.get("camera_far_weight", True) else None)
    camj = steady(camj, c)
    cues = picture_cues(frames, pose, y0, cam, camj["R"], depth) if c.get("camera_cues", True) else []
    u, hc, info = estimate_ground(people, cam, y0, c, cues, R=camj["R"])
    camj = level_roll(camj, u, c)
    return camj, u, hc, info


def level_roll(camj, u0, c):
    """film cameras keep the horizon level: the roll (about each frame's optical axis, measured against the world
    up u0) is removed when it stays small (tracking error); a big, deliberate one (> camera_roll_keep_deg) stays"""
    R = [np.asarray(r, float) for r in camj["R"]]
    roll = []
    for r in R:
        u = r @ u0
        roll.append(np.arctan2(u[0], -u[1]))
    roll = np.array(roll)
    if np.degrees(np.max(np.abs(roll))) >= c.get("camera_roll_keep_deg", 4.0):
        camj["roll_kept_deg"] = float(np.degrees(np.max(np.abs(roll))))
        return camj
    camj["R"] = [(Rotation.from_rotvec([0, 0, -a]).as_matrix() @ r).tolist() for a, r in zip(roll, R)]
    camj["roll_removed_deg"] = float(np.degrees(np.max(np.abs(roll))))
    return camj


def steady(camj, c):
    """like run_pipeline.steady_camera, plus: roll is kept only when it is large and steady (film cameras roll very
    rarely; small rolls are tracking error), and a shot whose turn is small and wandering is locked"""
    R = np.array(camj["R"], float)
    if len(R) < 3:
        return camj
    rv = Rotation.from_matrix(R).as_rotvec()
    total = float(np.degrees(np.max(np.linalg.norm(rv, axis=1))))
    sm4 = gaussian_filter1d(rv, 4.0, axis=0, mode="nearest")
    path = float(np.degrees(np.sum(np.linalg.norm(np.diff(sm4, axis=0), axis=1))))
    net = float(np.degrees(np.linalg.norm(sm4[-1] - sm4[0])))
    direct = net / max(path, 1e-6)
    if total < c.get("camera_still_deg", 1.0) or (total < 5.0 and direct < 0.5):
        camj["R"] = [np.eye(3).tolist() for _ in R]; camj["max_rotation_deg"] = 0.0
        camj["steady"] = "locked (only tracking noise: %.2f deg, wandering back and forth)" % total
        return camj
    sig = float(c.get("camera_smooth_frames", 4.0 if direct < 0.8 else 2.0))
    sm = gaussian_filter1d(rv, sig, axis=0, mode="nearest")
    sm -= sm[0]
    camj["R"] = Rotation.from_rotvec(sm).as_matrix().tolist()
    camj["max_rotation_deg"] = float(np.degrees(np.max(np.linalg.norm(sm, axis=1))))
    camj["steady"] = "smoothed"
    if "zoom" in camj:
        z = np.log(np.asarray(camj["zoom"], float))
        z = gaussian_filter1d(z, 3.0, mode="nearest"); z -= z[0]
        if np.max(np.abs(z)) < np.log(1.02):
            z[:] = 0.0
        camj["zoom"] = np.exp(z).tolist()
    return camj

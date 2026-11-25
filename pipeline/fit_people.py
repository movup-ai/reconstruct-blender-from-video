# -*- coding: utf-8 -*-
"""3D pose of every person: the L3 rig is fitted to the 2D marks of each frame (COCO-17),
standing on the ground, with joint limits, pose priors and smooth motion."""
import os
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from rig import Rig, JOINTS
from common import log

# free parameters: (joint, axis, low, high, prior sigma)   (deg)
ANG = [("spine", 0, -95, 30, 20), ("spine", 1, -35, 35, 12), ("spine", 2, -50, 50, 18),
       ("chest", 0, -45, 20, 15), ("chest", 2, -30, 30, 12),
       ("neck", 0, -50, 40, 20), ("head", 0, -45, 45, 20), ("head", 2, -75, 75, 30),
       ("shoulder_L", 0, -70, 180, 45), ("shoulder_L", 1, -20, 165, 35), ("shoulder_L", 2, -90, 90, 40),
       ("shoulder_R", 0, -70, 180, 45), ("shoulder_R", 1, -165, 20, 35), ("shoulder_R", 2, -90, 90, 40),
       ("elbow_L", 0, 0, 150, 50), ("elbow_R", 0, 0, 150, 50),
       ("hip_L", 0, -45, 125, 30), ("hip_L", 1, -20, 45, 12), ("hip_L", 2, -40, 40, 15),
       ("hip_R", 0, -45, 125, 30), ("hip_R", 1, -45, 20, 12), ("hip_R", 2, -40, 40, 15),
       ("knee_L", 0, -150, 0, 40), ("knee_R", 0, -150, 0, 40),
       # whole-body pitch / roll at the pelvis: held at 0 (+-1 deg) unless the person lies down (fit_track widens them)
       ("pelvis", 0, -1, 1, 5), ("pelvis", 1, -1, 1, 5)]
PELVIS_IDX = [i for i, a in enumerate(ANG) if a[0] == "pelvis"]
NB = 4                                   # x, y, yaw, pelvis height
NP = NB + len(ANG)

def unpack(q, rig):
    rot = {j: [0.0, 0.0, 0.0] for j in JOINTS}
    for (j, a, *_), v in zip(ANG, q[NB:]):
        rot[j][a] = v
    for sd in "LR":                                      # keep the feet level with the ground
        rot["ankle_" + sd][0] = float(np.clip(-(rot["hip_" + sd][0] + rot["knee_" + sd][0]), -40, 50))
    return (q[0], q[1], 0.0), q[2], rot, q[3]

def bounds(rig):
    top = float(rig.J[0][2][2] / rig.s)
    # master: deep squats bring the pelvis down to ~0.3 m
    lo = [-1e5, -1e5, -1e5, (0.18 if rig.style in ('master','l3') else 0.50) * rig.s]; hi = [1e5, 1e5, 1e5, (top + 0.04) * rig.s]
    for j, _, l, h, _ in ANG:
        if rig.style in ('master','l3') and j.startswith("knee_"):
            l = -165.0                                   # master: full squats (feet flat, heels under the hips)
        lo.append(l); hi.append(h)
    return np.array(lo, float), np.array(hi, float)

def seg_dist(p1, q1, p2, q2):
    """shortest distance between two segments p1-q1 and p2-q2"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2; den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + d1 * s - p2 - d2 * t))

# body blocks that must not pass through each other: (block a, block b, clearance / s)
# blocks are the bones of the mannequin with the half-thickness of their boxes
NO_MERGE = [("fore_L", "fore_R", 0.115), ("fore_L", "upper_R", 0.125), ("fore_R", "upper_L", 0.125),
            ("fore_L", "head", 0.185), ("fore_R", "head", 0.185), ("thigh_L", "thigh_R", 0.124),
            ("fore_L", "thigh_L", 0.120), ("fore_L", "thigh_R", 0.120), ("fore_R", "thigh_L", 0.120), ("fore_R", "thigh_R", 0.120),
            ("fore_L", "shin_L", 0.120), ("fore_L", "shin_R", 0.120), ("fore_R", "shin_L", 0.120), ("fore_R", "shin_R", 0.120),
            ("shin_L", "shin_R", 0.124), ("thigh_L", "shin_R", 0.124), ("thigh_R", "shin_L", 0.124)]

# the torso block (chest piece of the character): half width, half depth, height / s; limb half thickness / s
TORSO = (0.155, 0.078, 0.511)
LIMB_R = {"elbow": 0.066, "fore": 0.058, "wrist": 0.058, "knee": 0.064, "ankle": 0.064}

def in_torso(W, s, p, r):
    """how deep point p (a limb block of half thickness r) is inside the torso block; 0 when outside"""
    R, t = W["chest"]
    x, y, z = R.T @ (p - t)
    d = min(TORSO[0] * s + r - abs(x), TORSO[1] * s + r - abs(y), z + r, TORSO[2] * s + r - z)
    return max(0.0, float(d))

def body_blocks(W, s):
    """{block: (start, end)} of the posed rig W (rig.fk)"""
    t = lambda j: W[j][1]
    up = W["head"][0] @ np.array([0, 0, 0.14 * s])
    out = {"torso": (t("pelvis") + W["pelvis"][0] @ np.array([0, 0, 0.10 * s]), t("neck") - W["chest"][0] @ np.array([0, 0, 0.10 * s])),
           "head": (t("head") + up, t("head") + up)}
    for sd in "LR":
        out["upper_" + sd] = (t("shoulder_" + sd), t("elbow_" + sd))
        out["fore_" + sd] = (t("elbow_" + sd) + (t("wrist_" + sd) - t("elbow_" + sd)) * 0.25, t("wrist_" + sd))
        out["thigh_" + sd] = (t("hip_" + sd), t("knee_" + sd))
        out["shin_" + sd] = (t("knee_" + sd) + (t("ankle_" + sd) - t("knee_" + sd)) * 0.2, t("ankle_" + sd))
    return out

def merge_residual(W, s):
    """> 0 where two body blocks overlap (how deep, in units of 1.5 cm)"""
    B = body_blocks(W, s)
    out = [max(0.0, c * s - seg_dist(B[a][0], B[a][1], B[b][0], B[b][1])) / 0.015 for a, b, c in NO_MERGE]
    for sd in "LR":                                           # arms and legs stay outside the torso block
        e, w = W["elbow_" + sd][1], W["wrist_" + sd][1]
        out += [in_torso(W, s, e, LIMB_R["elbow"] * s) / 0.015, in_torso(W, s, (e + w) / 2, LIMB_R["fore"] * s) / 0.015,
                in_torso(W, s, w, LIMB_R["wrist"] * s) / 0.015,
                in_torso(W, s, W["knee_" + sd][1], LIMB_R["knee"] * s) / 0.015, in_torso(W, s, W["ankle_" + sd][1], LIMB_R["ankle"] * s) / 0.015]
    return out

# ---- the same model for a batch of parameter vectors at once (B, NP): one numpy pass instead of B Python passes.
# The optimizer's finite-difference Jacobian (NP + 1 evaluations per step) is evaluated as one batch (see _BatchMap),
# which is where nearly all of the fitting time went.  Same numbers as unpack / rig.fk / merge_residual.

def _rot_b(a, b, c):
    """Blender XYZ euler (deg), arrays (B,) -> (B, 3, 3) = Rz @ Ry @ Rx  (rig.Rxyz)"""
    a, b, c = np.radians(a), np.radians(b), np.radians(c)
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    R = np.empty(a.shape + (3, 3))
    R[..., 0, 0] = cc * cb; R[..., 0, 1] = cc * sb * sa - sc * ca; R[..., 0, 2] = cc * sb * ca + sc * sa
    R[..., 1, 0] = sc * cb; R[..., 1, 1] = sc * sb * sa + cc * ca; R[..., 1, 2] = sc * sb * ca - cc * sa
    R[..., 2, 0] = -sb;     R[..., 2, 1] = cb * sa;                R[..., 2, 2] = cb * ca
    return R

_ROT_COLS = {}                                         # joint -> [(axis, column in q)]
for _c, (_j, _a, *_r) in enumerate(ANG):
    _ROT_COLS.setdefault(_j, []).append((_a, NB + _c))
_HIPX = {sd: _ROT_COLS["hip_" + sd][0][1] for sd in "LR"}
_KNEEX = {sd: _ROT_COLS["knee_" + sd][0][1] for sd in "LR"}
_RJ = list(_ROT_COLS) + ["ankle_L", "ankle_R", "root"]
_E_J = np.array([_RJ.index(j) for j, a, *_ in ANG]); _E_A = np.array([a for j, a, *_ in ANG]); _E_C = np.arange(NB, NP)

def fk_b(rig, Q):
    """rig.fk(*unpack(q)) for every row of Q (B, NP) -> {joint: (R (B,3,3), t (B,3))}"""
    B = len(Q); z = np.zeros(B)
    E = np.zeros((B, len(_RJ), 3))                      # euler angles of every turning joint, one rotation pass
    E[:, _E_J, _E_A] = Q[:, _E_C]
    for sd in "LR":                                     # feet level with the ground
        E[:, _RJ.index("ankle_" + sd), 0] = np.clip(-(Q[:, _HIPX[sd]] + Q[:, _KNEEX[sd]]), -40, 50)
    E[:, -1, 2] = Q[:, 2]                               # root: yaw
    Rl = _rot_b(E[..., 0], E[..., 1], E[..., 2])
    W = {"root": (Rl[:, -1], np.c_[Q[:, 0], Q[:, 1], z])}
    for name, parent, off in rig.J:
        Rp, tp = W[parent]
        t = tp + (Rp[:, :, 2] * Q[:, 3:4] if name == "pelvis" else Rp @ off)
        W[name] = (Rp @ Rl[:, _RJ.index(name)] if name in _RJ else Rp, t)
    return W

def coco_b(rig, W):
    """rig.coco_points for a batch -> (B, 17, 3)"""
    return np.stack([W[j][1] + W[j][0] @ o for j, o in rig.coco], axis=1)

def project_b(world, X, f):
    """world.project for points (..., 3) -> uv (..., 2), depth (...)"""
    Xc = (X - world.C_at(f)) @ world.R_cw(f).T
    z = np.maximum(Xc[..., 2], 1e-6)
    cam = world.cam; fl = world.f_at(f)
    return np.stack([Xc[..., 0] / z * fl + cam.W / 2, Xc[..., 1] / z * fl + cam.H / 2], axis=-1), Xc[..., 2]

def _c01(x):
    return np.minimum(np.maximum(x, 0.0), 1.0)

def seg_dist_b(p1, q1, p2, q2):
    """seg_dist for arrays (..., 3) -> (...)"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, b = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    A0, E0 = a < 1e-12, e < 1e-12
    a_ = np.where(A0, 1.0, a); e_ = np.where(E0, 1.0, e)
    den = a * e - b * b; dok = den > 1e-12
    sg = np.where(dok, _c01((b * f - c * e) / np.where(dok, den, 1.0)), 0.0)
    tg = (b * sg + f) / e_
    sg = np.where(tg < 0, _c01(-c / a_), np.where(tg > 1, _c01((b - c) / a_), sg))
    tg = _c01(tg)
    s = np.where(A0, 0.0, np.where(E0, _c01(-c / a_), sg))
    t = np.where(A0, np.where(E0, 0.0, _c01(f / e_)), np.where(E0, 0.0, tg))
    return np.sqrt(((p1 + d1 * s[..., None] - p2 - d2 * t[..., None]) ** 2).sum(-1))

_BLK = ["head"] + [b + "_" + sd for sd in "LR" for b in ("upper", "fore", "thigh", "shin")]
_NM_A = np.array([_BLK.index(a) for a, _, _ in NO_MERGE]); _NM_B = np.array([_BLK.index(b) for _, b, _ in NO_MERGE])
_NM_C = np.array([c for _, _, c in NO_MERGE])
_TOR_R = np.array([LIMB_R[n] for sd in "LR" for n in ("elbow", "fore", "wrist", "knee", "ankle")])

def merge_b(W, s):
    """merge_residual for a batch -> (B, 27): all block pairs and torso points in one pass"""
    t = lambda j: W[j][1]
    hd = t("head") + W["head"][0][:, :, 2] * (0.14 * s)
    st, en, pts = [hd], [hd], []
    for sd in "LR":
        e, w, k, a = t("elbow_" + sd), t("wrist_" + sd), t("knee_" + sd), t("ankle_" + sd)
        st += [t("shoulder_" + sd), e + (w - e) * 0.25, t("hip_" + sd), k + (a - k) * 0.2]
        en += [e, w, k, a]
        pts += [e, (e + w) / 2, w, k, a]
    st, en, pts = np.stack(st, 1), np.stack(en, 1), np.stack(pts, 1)          # (B, 9, 3), (B, 10, 3)
    seg = np.maximum(0.0, _NM_C * s - seg_dist_b(st[:, _NM_A], en[:, _NM_A], st[:, _NM_B], en[:, _NM_B])) / 0.015
    R, tc = W["chest"]                                                       # arms and legs stay outside the torso block
    x = np.einsum("bji,bpj->bpi", R, pts - tc[:, None])
    r = _TOR_R * s
    d = np.minimum(np.minimum(TORSO[0] * s + r - np.abs(x[..., 0]), TORSO[1] * s + r - np.abs(x[..., 1])),
                   np.minimum(x[..., 2] + r, TORSO[2] * s + r - x[..., 2]))
    return np.concatenate([seg, np.maximum(0.0, d) / 0.015], axis=1)

class _BatchMap:
    """least_squares(workers=...) hands the finite-difference points here: they are evaluated as one batch.
    scipy keeps choosing the steps itself, so the Jacobian is the same as with its own 2-point scheme."""
    def __init__(self, fb):
        self.fb = fb
    def __call__(self, fun, xs):
        return list(self.fb(np.array(list(xs), float)))

def lsq_b(fb, q, lo, hi, **kw):
    """least_squares on a batched residual fb((B, n)) -> (B, m)"""
    return least_squares(lambda x: fb(x[None])[0], q, bounds=(lo, hi), workers=_BatchMap(fb), **kw)

def _key(v, f):
    """a hint that is a number, or [[frame, value], ...] keys -> its value at frame f"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    v = sorted(v)
    return float(np.interp(f, [a for a, _ in v], [b for _, b in v]))

def _is_ranges(v):
    """[[first, last, value], ...] (a hint that changes during the shot)"""
    return isinstance(v, list) and len(v) > 0 and all(isinstance(r, (list, tuple)) and len(r) == 3 for r in v)

def _key_angle(v, f):
    """like _key for headings: keys are unwrapped first, so 12 -> 357 turns 15 degrees, not 345 the long way"""
    if v is None or isinstance(v, (int, float)):
        return _key(v, f)
    v = sorted(v)
    a = np.degrees(np.unwrap(np.radians([b for _, b in v])))
    return float(np.interp(f, [k for k, _ in v], a))


def _ranged(v, f, outside=None):
    """a hint that is one value for the whole shot, or [[first, last, value], ...] frame ranges -> its value at frame f
    (outside every range: `outside`)"""
    if not _is_ranges(v):
        return v
    for a, b, val in v:
        if int(a) <= f <= int(b):
            return val
    return outside

def leg_bend_flags(K):
    """per frame: the hip-knee-ankle marks show a deep knee bend (kneeling, crouching, squatting).
    A straight leg projects to a straight line from any side, so a knee angle well below 180 deg in the picture,
    or legs that are short compared to the torso, mean the pelvis is low."""
    K = np.asarray(K, float)
    out = np.zeros(len(K), bool)
    for n, k in enumerate(K):
        ok = k[:, 2] > 0.3
        bent = False
        for h, kn, a in ((11, 13, 15), (12, 14, 16)):
            if ok[h] and ok[kn] and ok[a]:
                u, v = k[h, :2] - k[kn, :2], k[a, :2] - k[kn, :2]
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                if nu > 4 and nv > 4 and np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1, 1))) < 115:
                    bent = True
        if ok[[5, 6, 11, 12]].all() and (ok[15] or ok[16]):
            tor = abs((k[11, 1] + k[12, 1]) / 2 - (k[5, 1] + k[6, 1]) / 2)
            leg = max(k[a, 1] for a in (15, 16) if ok[a]) - (k[11, 1] + k[12, 1]) / 2
            if tor > 8 and leg < 1.25 * tor:                  # standing: legs ~1.5-1.8 x the torso in height
                bent = True                                   # (kneeling seen from the front: ~1.2)
        out[n] = bent
    return out

def lying_flags(K):
    """per frame: the body is horizontal in the picture (shoulders -> hips and hips -> ankles run sideways)"""
    K = np.asarray(K, float)
    out = np.zeros(len(K), bool)
    for n, k in enumerate(K):
        ok = k[:, 2] > 0.3
        if not ok[[5, 6, 11, 12]].all():
            continue
        t = (k[11, :2] + k[12, :2]) / 2 - (k[5, :2] + k[6, :2]) / 2
        if np.linalg.norm(t) < 10 or abs(t[0]) < 1.7 * abs(t[1]):
            continue
        an = [a for a in (15, 16) if ok[a]] or [a for a in (13, 14) if ok[a]]
        if an:
            l = k[an, :2].mean(0) - (k[11, :2] + k[12, :2]) / 2
            if abs(l[0]) < 1.2 * abs(l[1]):
                continue
        out[n] = True
    if len(out) >= 5:                                         # a few flickering frames are not lying down
        from scipy.ndimage import median_filter
        out = median_filter(out.astype(int), size=9, mode="nearest").astype(bool)
    return out

LEG_IDX = [i for i, a in enumerate(ANG) if a[0].startswith(("hip_", "knee_"))]

# ---- SAM 3D Body prior: bone directions of the metric pose, compared in camera axes.
# (name, H36M joint a, H36M joint b, joint that turns the bone (hint keys / held poses on it win), leg bone, COCO marks)
P3_BONES = [("spine", 0, 8, "spine", False, (5, 6, 11, 12)), ("shoulders", 11, 14, "chest", False, (5, 6)),
            ("hips", 4, 1, "root", False, (11, 12)), ("head", 8, 10, "neck", False, (0, 1, 2, 5, 6)),
            ("upper_L", 11, 12, "shoulder_L", False, (5, 7)), ("fore_L", 12, 13, "elbow_L", False, (7, 9)),
            ("upper_R", 14, 15, "shoulder_R", False, (6, 8)), ("fore_R", 15, 16, "elbow_R", False, (8, 10)),
            ("thigh_L", 4, 5, "hip_L", True, (11, 13)), ("shin_L", 5, 6, "knee_L", True, (13, 15)),
            ("thigh_R", 1, 2, "hip_R", True, (12, 14)), ("shin_R", 2, 3, "knee_R", True, (14, 16))]
_P3A = np.array([b[1] for b in P3_BONES]); _P3B = np.array([b[2] for b in P3_BONES])

def p3_points_b(W, s):
    """H36M-17 points of a batch of posed rigs (fk_b) -> (B, 17, 3) (nose / spine unused: zeros)"""
    t = lambda j: W[j][1]
    B = len(t("pelvis")); P = np.zeros((B, 17, 3))
    P[:, 0] = (t("hip_L") + t("hip_R")) / 2; P[:, 8] = (t("shoulder_L") + t("shoulder_R")) / 2
    P[:, 10] = t("head") + W["head"][0][:, :, 2] * (0.27 * s)
    for i, j in ((1, "hip_R"), (2, "knee_R"), (3, "ankle_R"), (4, "hip_L"), (5, "knee_L"), (6, "ankle_L"),
                 (11, "shoulder_L"), (12, "elbow_L"), (13, "wrist_L"), (14, "shoulder_R"), (15, "elbow_R"), (16, "wrist_R")):
        P[:, i] = t(j)
    return P

def p3_dirs_b(W, s, Rcw):
    """unit bone directions (B, nbones, 3) of posed rigs in the axes of camera Rcw (world -> cam)"""
    P = p3_points_b(W, s)
    d = P[:, _P3B] - P[:, _P3A]
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    return d @ Rcw.T

def p3_target(lift, f, k, world):
    """SAM 3D Body bone directions and confidence, already in camera coordinates."""
    fr, X, cf = lift[:3]
    i = int(f - fr[0])
    if i < 0 or i >= len(fr):
        return None
    d = X[i][_P3B] - X[i][_P3A]
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    c = np.minimum(cf[i][_P3A], cf[i][_P3B])
    return d, np.clip(c, 0, 1)

def fit_track(world, y0, fs, K, height, step=2, style="sample", hint=None, lift=None, p3=None):
    """hint (hints.json, written by whoever looked at the clip) fixes what the picture cannot show:
      legs: "seated" | "hidden"   legs are not in the picture: sit / stand them, ignore knee and ankle marks
            or [[first, last, "standing" | "seated" | "hidden"], ...]  per frame range (outside: the picture decides)
      leg_pose: {"hip", "knee"} | "riding", also as [[first, last, value], ...]
      pose: "lying" or [[first, last, "lying"], ...]  lying on the floor (otherwise found from the marks: body
            horizontal in the picture; "lying": false switches that off).  A dict = held joint angles (see below)
      kneeling / crouching is found from the marks (knee angle in the picture, legs short against the torso)
      seat_height                 pelvis height above the floor when seated (m, default 0.55)
      distance                    metres from the camera along the floor (number or [[frame, m], ...])
      yaw                         heading in degrees, 0 = facing +x ... (number or keys); weak guide
      floor_z                     height of what they stand / sit on (m), or [[frame, m], ...] keys (stairs)
      pose3d: false / pose3d_weight   switch off / scale SAM 3D Body direction constraints for this person
    lift: (frames, camera-space joints, confidence) from SAM 3D Body (None = no 3D prior); p3: {"weight", "sigma"} from the config.
    Hints take priority over body constraints: bones turned by keyed / held joints, legs of seated / hidden people, the hip and
    shoulder lines when a yaw is set, and bones of untrusted marks (use_kps / ignore_kps) get no 3D prior."""
    hint = hint or {}
    unanchored = bool(hint.get('_body_unanchored'))
    legs = hint.get("legs", "auto")
    # legs / leg_pose may change during the shot: [[first, last, "standing" | "seated" | "hidden"], ...]
    # (frames outside every range: the picture decides, like "auto" without the cut-off check)
    legs_rng = legs if _is_ranges(legs) else None
    lp_rng = hint.get("leg_pose") if _is_ranges(hint.get("leg_pose")) else None
    if legs_rng:
        legs = "ranges"
    # riding (hint "mount": "horse" or "leg_pose": "riding"): seated on a saddle, hips at seat_height (default 1.55 m),
    # thighs forward and apart round the horse, shins down; knees / ankles are not trusted (the horse hides them)
    riding = hint.get("leg_pose") == "riding" or hint.get("mount") in ("horse", True)
    if riding:
        legs = "seated"
    elif legs == "auto":                                  # cut off by the frame: knees and ankles missing in most frames
        Kc = np.asarray(K, float)
        up = (Kc[:, [5, 6], 2] > 0.3).any(1)
        low = (Kc[:, 13:17, 2] > 0.3).sum(1) >= 2
        if up.sum() >= 4 and (up & ~low).sum() > 0.6 * up.sum():
            legs = "hidden"                               # standing legs by default (hint legs "seated" for sitting)
            log("      legs not in the picture in %d%% of the frames: standing legs assumed" % (100 * (up & ~low).sum() / up.sum()))
    rig = Rig(height, style)
    lo, hi = bounds(rig)
    fs = list(fs); K = np.asarray(K, float).copy(); K[:, :, 1] -= y0
    if unanchored:
        for j in range(11,17):
            if (K[:,j,2]>.5).mean()<.25:
                K[:,j,2]=0  # sparse guesses below a cropped torso must not reposition the person
    fidx = {f_: i_ for i_, f_ in enumerate(fs)}
    def legs_at(f):
        """"seated" | "hidden" | "free" (stand on the floor, the marks decide) at frame f"""
        v = _ranged(legs_rng, f, "free") if legs_rng else legs
        return v if v in ("seated", "hidden", "kneeling") else "free"
    def lp_at(f):
        """leg_pose at frame f ({"hip": .., "knee": ..} or "riding" or None)"""
        return _ranged(lp_rng, f, None) if lp_rng else hint.get("leg_pose")
    # lying down (hint "pose": "lying" or [[first, last, "lying"], ...]; otherwise the marks: body horizontal in the picture)
    ph = hint.get("pose")
    if isinstance(ph, str) or _is_ranges(ph):
        lie_f = np.array([_ranged(ph, f_, None) == "lying" for f_ in fs], bool)
    elif hint.get("lying") is False:
        lie_f = np.zeros(len(fs), bool)
    else:
        lie_f = lying_flags(K)
        if lie_f.any():
            log("      body horizontal (lying) in %d frames" % lie_f.sum())
    bend_f = leg_bend_flags(K) & ~lie_f                     # kneeling / crouching: the pelvis may go low
    if bend_f.any():
        log("      deep knee bend (kneeling / crouching) in %d frames" % bend_f.sum())
    lying_at = lambda f: bool(lie_f[fidx[f]]) if f in fidx else False
    bent_at = lambda f: bool(bend_f[fidx[f]]) if f in fidx else False
    any_seated = legs in ("seated", "kneeling") or bool(legs_rng and any(r[2] in ("seated", "kneeling") for r in legs_rng))
    if any_seated:
        lo[3] = 0.2 * rig.s
    if bend_f.any():
        lo[3] = min(lo[3], 0.22 * rig.s)                    # kneeling upright ~0.5 x height, sitting on the heels ~0.3
    if lie_f.any():
        lo[3] = min(lo[3], 0.04 * rig.s)
        for i_, (lim_lo, lim_hi) in zip(PELVIS_IDX, ((-110.0, 110.0), (-60.0, 60.0))):
            lo[NB + i_], hi[NB + i_] = lim_lo, lim_hi
    ps = hint.get("prior_scale") or {}                     # {"spine": 3}: hold that joint closer to its rest pose
    sig_prior = np.array([s / float(ps.get(j, 1.0)) for j, _, _, _, s in ANG])
    dots = bool(hint.get("follow_dots"))                   # the detected joints win: weak rest-pose prior, strong marks
    if dots:
        sig_prior = sig_prior * 3.0
    kp_gain = float(hint.get("kp_weight", 3.0 if dots else 1.0))
    for i_ in range(len(fs)):
        if legs_at(fs[i_]) in ("seated", "hidden"):
            K[i_, 13:17, 2] = 0.0
        elif legs_at(fs[i_]) == "kneeling":
            K[i_, 15:17, 2] = 0.0                         # ankles hidden behind / under the body: guesses                         # knees and ankles are guesses outside the picture
    if hint.get("use_kps"):                               # only these marks are trusted (e.g. a person seen from behind)
        for i in range(17):
            if i not in hint["use_kps"]:
                K[:, i, 2] = 0.0
    if hint.get("ignore_kps"):                            # the detected joints are not trusted at all for this person
        K[:, :, 2] = 0.0
    head_px = sorted(hint.get("head_px") or [])              # [[frame, u, v], ...] centre of the head in the picture, set by hand
    def fz_at(f):                                          # floor_z: a number or [[frame, m], ...] keys (stairs)
        return float(_key(hint.get("floor_z"), f) or 0.0)
    def pz_target(f):
        if legs_at(f) == "kneeling":                        # kneeling upright: hips about a thigh above the floor
            return float(_ranged(hint.get("seat_height"), f, None) or 0.52 * rig.s) + fz_at(f)
        if legs_at(f) == "seated":
            ride = riding or lp_at(f) == "riding"
            return float(_ranged(hint.get("seat_height"), f, None) or (1.55 if ride else 0.55)) + fz_at(f)
        return rig.J[0][2][2] + fz_at(f)
    pz_t = pz_target(fs[0]) if fs else rig.J[0][2][2] + fz_at(0)
    hi[3] = max([hi[3]] + [pz_target(f_) + 0.1 for f_ in fs])
    if unanchored:
        targets = np.asarray(hint['_body_start'],float)[:,3]
        lo[3] = min(lo[3],float(targets.min())-height)
        hi[3] = max(hi[3],float(targets.max())+height)
    RIDE = {("hip_L", 0): 50.0, ("hip_R", 0): 50.0, ("hip_L", 1): 24.0, ("hip_R", 1): -24.0, ("knee_L", 0): -78.0, ("knee_R", 0): -78.0}
    def leg_targets(f):
        """default hip / knee angles of seated (or riding) legs at frame f"""
        t = np.zeros(len(ANG))
        lp = lp_at(f)
        if riding or lp == "riding":
            for i in LEG_IDX:
                t[i] = RIDE.get(ANG[i][:2], 0.0)
        elif legs_at(f) == "kneeling":
            lp = lp if isinstance(lp, dict) else {}
            for i in LEG_IDX:
                t[i] = float(lp.get("hip", 5.0)) if ANG[i][0].startswith("hip_") and ANG[i][1] == 0 else (float(lp.get("knee", -95.0)) if ANG[i][0].startswith("knee_") else 0.0)
        elif legs_at(f) == "seated":
            lp = lp if isinstance(lp, dict) else {}         # {"hip": 55, "knee": -60}: perched on an edge instead of a chair
            for i in LEG_IDX:
                t[i] = float(lp.get("hip", 85.0)) if ANG[i][0].startswith("hip_") and ANG[i][1] == 0 else (float(lp.get("knee", -85.0)) if ANG[i][0].startswith("knee_") else 0.0)
        return t
    leg_t = leg_targets(fs[0]) if fs else np.zeros(len(ANG))
    pose_t = []                                             # hint "pose": {joint: [x, y, z]} held angles (None = free)
    for jn, vals in (hint.get("pose") if isinstance(hint.get("pose"), dict) else {}).items():
        for ax, v in enumerate(vals):
            for i, a in enumerate(ANG):
                if v is not None and a[0] == jn and a[1] == ax:
                    pose_t.append((i, float(v)))
    key_tracks = [sorted(kt, key=lambda kk: kk[0]) for kt in ([hint.get("keys") or []] + list(hint.get("keys_more") or [])) if kt]
    def key_targets(f):
        out = []
        for kp_keys in key_tracks:                           # several keyed stretches, each with its own frames
            out += key_targets1(f, kp_keys)
        return out
    def key_targets1(f, kp_keys):
        """angles the hand-set key poses ask for at frame f: [(index in ANG, degrees)], blended between keys"""
        if not kp_keys:
            return []
        fr = [kk[0] for kk in kp_keys]
        if f < fr[0] or f > fr[-1]:                          # outside the keyed stretch the detected joints decide
            return []
        if f == fr[-1]:
            a = b = kp_keys[-1]; w = 0.0
        else:
            j = int(np.searchsorted(fr, f, side="right")) - 1
            a, b = kp_keys[j], kp_keys[j + 1]; w = (f - a[0]) / float(b[0] - a[0])
            w = w * w * (3 - 2 * w)                               # ease in and out
        out = []
        for i, an in enumerate(ANG):
            va = (a[1].get(an[0]) or [None] * 3)[an[1]]; vb = (b[1].get(an[0]) or [None] * 3)[an[1]]
            if va is None and vb is None:
                continue
            va = vb if va is None else va; vb = va if vb is None else vb
            out.append((i, float(va) * (1 - w) + float(vb) * w))
        return out
    # 3D pose prior: bone directions of the SAM 3D Body pose; hints keep priority (see the docstring)
    p3 = p3 or {}
    use_p3 = lift is not None and hint.get("pose3d", True) is not False and not hint.get("ignore_kps") and not dots
    p3_w = float(hint.get("pose3d_weight", p3.get("weight", 1.0))) / float(p3.get("sigma", 0.2))
    p3_m0 = np.ones(len(P3_BONES))
    held = {ANG[i_][0] for i_, _ in pose_t}
    for b_, (nm_, _, _, jt_, leg_, mk_) in enumerate(P3_BONES):
        if jt_ in held or (hint.get("use_kps") and not all(m_ in hint["use_kps"] for m_ in mk_)):
            p3_m0[b_] = 0.0
    p3_leg = np.array([0.0 if bn[4] else 1.0 for bn in P3_BONES])
    def p3_frame(f, k, kt, yh):
        """(target directions (nb, 3), weights (nb,)) of frame f, or None"""
        if not use_p3 or p3_w <= 0 or lying_at(f):           # explicit lying poses take priority over body constraints
            return None
        tg = p3_target(lift, f, k, world)
        if tg is None:
            return None
        keyed = {ANG[i_][0] for i_, _ in kt}
        m = p3_m0 * np.array([0.0 if (bn[3] in keyed or (yh is not None and bn[0] in ("hips", "shoulders"))) else 1.0
                              for bn in P3_BONES])
        if legs_at(f) in ("seated", "hidden", "kneeling"):
            m = m * p3_leg
        return (tg[0], tg[1] * m * p3_w) if m.any() else None
    def yaw_hint(f):
        yh = _key_angle(hint.get("yaw"), f)
        yr = hint.get("yaw_range")
        return None if (yh is not None and yr and not (yr[0] <= f <= yr[1])) else yh
    key_sig = float(hint.get("key_strength", 6.0))           # degrees the fit may leave a key pose (smaller = stricter)
    idx = sorted(set(list(range(0, len(fs), step)) + [len(fs) - 1]))
    good = [i for i in idx if (K[i][:, 2] > 0.3).sum() >= 6]   # frames with too few sure marks are filled in from their neighbours
    idx = idx if (hint.get("use_kps") or hint.get("ignore_kps") or len(good) < 2) else good
    hd = hint.get("hand_dist") or {}                         # {"L" / "R": [[frame, metres from the camera], ...]} inside its key range
    def hand_d(sd, f):
        v = hd.get(sd)
        if not v or not (min(a for a, _ in v) <= f <= max(a for a, _ in v)):
            return None
        return _key(v, f)
    def scale_px(k):
        v = k[k[:, 2] > 0.3, :2]
        return max(20.0, np.ptp(v[:, 1]) if len(v) > 2 else 60.0)
    def residual_b(f, k, qprev, dt, q0):
        """the residual of frame f as a function of a batch of parameter vectors (B, NP) -> (B, m).
        Everything that does not depend on the parameters is worked out once here."""
        sp = 0.03 * scale_px(k)
        kw = (np.where(k[:, 2] > 0.3, np.sqrt(np.clip(k[:, 2], 0, 1)), 0.0) / sp)[:, None] * kp_gain
        k2 = k[:, :2]
        kt = key_targets(f)
        lg = legs_at(f); lying = lying_at(f); bent = bent_at(f)
        if lg in ("seated", "hidden", "kneeling"):
            pk = _key(hint.get("pelvis_z"), f)                # pelvis height set by hand (number or keys)
            firm = (pk is not None or lg in ("seated", "kneeling")) and not hint.get("seat_soft")   # otherwise the picture decides the height
            pz_aim, pz_sig = (pz_target(f) if pk is None else pk), (0.02 if firm else 0.12)
            kt_ = dict(kt)                                    # legs keyed by hand (hip / knee angles) are not pulled to the default
            lt = leg_targets(f)
            leg_c = np.array([NB + i_ for i_ in LEG_IDX if i_ not in kt_], int)
            leg_v = np.array([lt[i_] for i_ in LEG_IDX if i_ not in kt_], float)
            leg_s = 2.0 if lg == "hidden" else 4.0
        pv_c = np.array([NB + i_ for i_ in PELVIS_IDX], int)
        sig_f = sig_prior.copy()
        if lying:
            sig_f[PELVIS_IDX] = 1e3                           # pitched / rolled as the marks say
        if bent:
            sig_f[LEG_IDX] *= 2.0                             # deep knee bends are far from the rest pose
        lie_z = 0.10 * rig.s + fz_at(f)                              # pelvis of someone lying on the floor
        pose_c = np.array([NB + i_ for i_, _ in pose_t], int); pose_v = np.array([v_ for _, v_ in pose_t], float)
        key_c = np.array([NB + i_ for i_, _ in kt], int); key_v = np.array([v_ for _, v_ in kt], float)
        D = _key(hint.get("distance"), f)
        Cf = world.C_at(f)
        body_hip = hint.get('_body_hip')
        hip_target = None
        if body_hip:
            target = np.asarray(body_hip,float)
            hip_target = np.array([np.interp(f,target[:,0],target[:,axis]) for axis in (1,2,3)])
        if head_px:
            hf = [h_[0] for h_ in head_px]
            tu = np.interp(f, hf, [h_[1] for h_ in head_px]); tv = np.interp(f, hf, [h_[2] for h_ in head_px])
        Dh = {sd: hand_d(sd, f) for sd in "LR"}
        yh = _key_angle(hint.get("yaw"), f)
        yr = hint.get("yaw_range")
        if yh is not None and yr and not (yr[0] <= f <= yr[1]):
            yh = None
        ysig = float(hint.get("yaw_sigma", 12.0))
        if qprev is not None:
            st = np.r_[0.12 * dt, 0.12 * dt, 12.0 * dt, 0.05 * dt, np.full(len(ANG), 18.0 * dt)]
            if unanchored:
                st[:4]=[.04*dt,.04*dt,(3.+9.*min(k[5,2],k[6,2]))*dt,.025*dt]
                for i,(joint,*_) in enumerate(ANG):
                    if joint.startswith('shoulder_'):
                        a,b=(7,9) if joint.endswith('_L') else (8,10)
                        support=max(k[a,2],k[b,2])
                    elif joint.startswith('elbow_'):
                        support=k[9 if joint.endswith('_L') else 10,2]
                    else: support=.3
                    st[NB+i]=(3.+15.*float(np.clip(support,0,1)))*dt
        p3t = p3_frame(f, k, kt, yh); Rcw = world.R_cw(f)
        contacts = []
        if not unanchored and qprev is not None and dt <= 4 and hint.get("foot_contact", True) and not lying and not bent and lg == "free":
            previous = fk_b(rig, qprev[None])
            for sd, joint in (("L", 15), ("R", 16)):
                if any(ANG[index][0] in ("hip_" + sd, "knee_" + sd, "ankle_" + sd) for index, _ in kt):
                    continue
                ankle = previous["ankle_" + sd][1][0]
                uv, depth = world.project(ankle, f)
                tolerance = max(3.0, 0.012 * scale_px(k))
                previous_index = fidx.get(f - dt)
                previous_seen = previous_index is not None and K[previous_index, joint, 2] >= 0.6
                observed = previous_seen and k[joint, 2] >= 0.6 and np.linalg.norm(uv[0] - k[joint, :2]) <= tolerance
                if observed and depth[0] > 0 and abs(ankle[2] - rig.sole - fz_at(f)) < 0.06:
                    target = ankle.copy()
                    target[2] = rig.sole + fz_at(f)
                    contacts.append((sd, target))
        col = lambda v: v[:, None]

        def fb(Q):
            W = fk_b(rig, Q)
            contact_sigma = float(hint.get("foot_contact_sigma", 0.04))
            uv, _ = project_b(world, coco_b(rig, W), f)
            r = [((uv - k2) * kw).reshape(len(Q), -1), 0.35 * Q[:, NB:] / sig_f,
                 merge_b(W, rig.s)]                            # body blocks do not pass through each other
            for sd, target in contacts:
                r.append((W["ankle_" + sd][1] - target) / max(0.01, contact_sigma))
            if lg in ("seated", "hidden", "kneeling") and not lying:
                r += [col((Q[:, 3] - pz_aim) / pz_sig), (Q[:, leg_c] - leg_v) / leg_s]
            elif lying:                                       # on the floor: pelvis low, nothing below the floor
                za, zb = W["ankle_L"][1][:, 2], W["ankle_R"][1][:, 2]
                low = np.minimum(np.minimum(za, zb), np.minimum(W["head"][1][:, 2], W["chest"][1][:, 2]))
                r += [col((Q[:, 3] - lie_z) / 0.04), col(np.minimum(0.0, low - 0.05 * rig.s - fz_at(f)) / 0.02)]
            elif not unanchored:
                za, zb = W["ankle_L"][1][:, 2], W["ankle_R"][1][:, 2]
                r += [col((np.minimum(za, zb) - rig.sole - fz_at(f)) / 0.03), col(np.minimum(0.0, np.maximum(za, zb) - rig.sole - fz_at(f)) / 0.02),
                      col((Q[:, 3] - rig.J[0][2][2] - fz_at(f)) / (0.6 if bent else 0.25))]
                if not bent:                                  # the old lower bound of a standing pelvis, now soft
                    r.append(col(np.maximum(0.0, 0.50 * rig.s + fz_at(f) - Q[:, 3]) / 0.02))
            if not lying:
                r.append(Q[:, pv_c] / 2.0)                    # the body stays upright
            r += [(Q[:, pose_c] - pose_v) / 3.0, (Q[:, key_c] - key_v) / key_sig]
            if D is not None:
                r.append(col((np.hypot(Q[:, 0] - Cf[0], Q[:, 1] - Cf[1]) - D) / 0.03))
            if hip_target is not None:
                r.append(((W['hip_L'][1]+W['hip_R'][1])/2-hip_target)/float(hint.get('_body_hip_sigma',.15)))
            if head_px:
                hc = W["head"][1] + W["head"][0][:, :, 2] * (0.145 * rig.s)
                huv, _ = project_b(world, hc, f)
                r += [col((huv[:, 0] - tu) / 4.0), col((huv[:, 1] - tv) / 4.0)]
            for sd in "LR":                                   # a hand that reaches something at a known distance
                if Dh[sd] is not None:
                    wp = W["wrist_" + sd][1]
                    r.append(col((np.hypot(wp[:, 0] - Cf[0], wp[:, 1] - Cf[1]) - Dh[sd]) / 0.04))
            if yh is not None:
                r.append(col((((Q[:, 2] - yh) + 180) % 360 - 180) / ysig))
            if qprev is not None:
                d = Q - qprev; d[:, 2] = (d[:, 2] + 180) % 360 - 180
                r.append(d / st)
            if q0 is not None:
                r.append((Q[:, :2] - q0[:2]) / 1.5)
            if p3t is not None:                               # bones follow the measured metric body directions
                r.append(((p3_dirs_b(W, rig.s, Rcw) - p3t[0]) * p3t[1][:, None]).reshape(len(Q), -1))
            return np.concatenate(r, axis=1)
        return fb
    def solve(q, f, k, qprev, dt, q0, it=60):
        q = np.clip(q, lo + 1e-6, hi - 1e-6)
        return lsq_b(residual_b(f, k, qprev, dt, q0), q, lo, hi, loss="soft_l1", f_scale=2.0,
                     max_nfev=it, x_scale="jac", diff_step=1e-3)
    def lift_start(q, f, k, it=30):
        """Initialize heading and joint angles from SAM 3D Body directions
        (place and pelvis height stay; keyed / held angles are set and kept, so are the default legs when seated)"""
        kt = key_targets(f); yh = yaw_hint(f)
        t = p3_frame(f, k, kt, yh)
        if t is None:
            return q
        Dt, w = t; Rcw = world.R_cw(f)
        base = q.copy()
        fixed = {i_: v_ for i_, v_ in pose_t}; fixed.update(dict(kt))
        for i_, v_ in fixed.items():
            base[NB + i_] = v_
        base = np.clip(base, lo + 1e-6, hi - 1e-6)
        free = ([] if yh is not None else [2]) + [NB + i_ for i_ in range(len(ANG)) if i_ not in fixed and
                                                   not (legs_at(f) in ("seated", "hidden", "kneeling") and i_ in LEG_IDX)]
        free = np.array(free, int)
        def fb(X):
            Qf = np.repeat(base[None], len(X), 0); Qf[:, free] = X
            W = fk_b(rig, Qf)
            return np.concatenate([((p3_dirs_b(W, rig.s, Rcw) - Dt) * w[:, None]).reshape(len(X), -1),
                                   0.35 * Qf[:, NB:] / sig_prior, 0.5 * merge_b(W, rig.s)], axis=1)
        s = lsq_b(fb, base[free], lo[free], hi[free], max_nfev=it, x_scale="jac", diff_step=1e-3)
        out = base.copy(); out[free] = s.x
        return out
    def init_q(f, k):
        q = np.zeros(NP); q[3] = pz_target(f) if legs_at(f) in ("seated", "hidden", "kneeling") else rig.J[0][2][2]
        if lying_at(f):
            q[3] = 0.10 * rig.s + fz_at(f)
        lt = leg_targets(f)
        for i, (j, a, l, h, _) in enumerate(ANG):
            q[NB + i] = np.clip(lt[i], l, h)
        if hint.get('_body_start') and not any(k in hint for k in ('distance','mount','floor_z')):
            start=np.asarray(hint['_body_start'],float)
            target=np.array([np.interp(f,start[:,0],start[:,i]) for i in (1,2,3)])
            q[:2]=target[:2]
            if unanchored: q[3]=target[2]-float(np.mean([o[2] for j,p,o in rig.J if j in ('hip_L','hip_R')]))
            return q
        D = _key(hint.get("distance"), f)
        if D is not None:                                   # along the ray through the body, D metres away on the floor
            body = [i for i in (11, 12, 5, 6) if k[i, 2] > 0.3] or list(range(17))
            d = world.ray(f, *k[body, :2].mean(0))
            hd = max(np.hypot(d[0], d[1]), 1e-3)
            Cf = world.C_at(f)
            q[0], q[1] = Cf[0] + d[0] / hd * D, Cf[1] + d[1] / hd * D
            return q
        anc = [i for i in (15, 16) if k[i, 2] > 0.3]
        g = None
        if anc:
            p = k[anc, :2].mean(0); p[1] = k[anc, 1].max()
            g = world.on_ground(f, p[0], p[1])
        if g is None or np.linalg.norm(g - world.C) > 400:
            tor = [i for i in (5, 6, 11, 12) if k[i, 2] > 0.3]
            hp = k[[i for i in (11, 12, 5, 6) if k[i, 2] > 0.3] or [0], :2].mean(0)
            tl = np.linalg.norm(k[[5, 6], :2].mean(0) - k[[11, 12], :2].mean(0)) if len(tor) == 4 else scale_px(k) / 3.3
            Z = world.f_at(f) * 0.50 * rig.s / max(tl, 4.0)
            d = world.ray(f, *hp); g = world.C_at(f) + d * Z / max(d @ (world.R_cw(f).T @ [0, 0, 1.0]), 0.2)
            g[2] = 0.0
        q[0], q[1] = g[0], g[1]
        return q
    def low_starts(q):
        """start poses for a low body from q: kneeling upright (shins on the floor) and a deep squat"""
        out = []
        for pz, hip, knee in ((0.52, 5.0, -95.0), (0.36, 105.0, -135.0)):
            x = q.copy(); x[3] = pz * rig.s + fz_at(f)
            for sd in "LR":
                x[_HIPX[sd]] = hip; x[_KNEEX[sd]] = knee
            out.append(np.clip(x, lo + 1e-6, hi - 1e-6))
        return out
    Q = {}; err = {}
    qprev = None; fprev = None
    for n, i in enumerate(idx):
        f, k = fs[i], K[i]
        if qprev is None:
            q0 = init_q(f, k)
            cam_dir = np.degrees(np.arctan2(*(q0[:2] - world.C[:2])[::-1]))     # heading from the camera to the person
            best = None
            for yaw in np.arange(0, 360, 45):
                for pitch in ((-88.0, 88.0) if lying_at(f) else (0.0,)):
                    q = q0.copy(); q[2] = cam_dir + yaw - 90; q[NB + PELVIS_IDX[0]] = pitch
                    s = solve(lift_start(q, f, k), f, k, None, 1, q0, it=40)
                    if best is None or s.cost < best.cost:
                        best = s
            if bent_at(f) and legs_at(f) == "free":
                for start in low_starts(best.x):
                    s = solve(start, f, k, None, 1, q0, it=40)
                    if s.cost < 0.9 * best.cost:
                        best = s
            s = solve(best.x, f, k, None, 1, q0, it=150)
        elif lying_at(f) != lying_at(fprev):                 # lies down / gets up: search the body's direction again
            dt = max(1, f - fprev)
            best = None
            for yaw in np.arange(0, 360, 45):
                for pitch in ((-88.0, 88.0) if lying_at(f) else (0.0,)):
                    q = qprev.copy(); q[2] += yaw; q[NB + PELVIS_IDX[0]] = pitch
                    q[NB + PELVIS_IDX[1]] = 0.0
                    q[3] = 0.10 * rig.s + fz_at(f) if lying_at(f) else rig.J[0][2][2] + fz_at(f)
                    s = solve(lift_start(q, f, k), f, k, None, 1, None, it=40)
                    if best is None or s.cost < best.cost:
                        best = s
            s = solve(best.x, f, k, None, 1, None, it=120)
        else:
            dt = max(1, f - fprev)
            qs = qprev.copy()
            for i_, v_ in key_targets(f):                    # start from the hand-set pose: the fit refines it, it does not have to find it
                qs[NB + i_] = float(np.clip(v_, lo[NB + i_], hi[NB + i_]))
            # Partial bodies keep the previous solution as their seed. Reinitializing an
            # ambiguous hidden limb from independent model frames caused depth/yaw flips.
            s = solve(qs if unanchored else lift_start(qs, f, k), f, k, qprev, dt, None)
            if bent_at(f) and legs_at(f) == "free":          # kneeling / crouching: a standing body can often explain
                for start in low_starts(qs):                  # the marks too (front view) - try low poses as well
                    s2 = solve(start, f, k, qprev, dt * 6, None)
                    if s2.cost < 0.9 * s.cost:
                        s = s2
            rms = np.sqrt(np.mean(s.fun[:34] ** 2))
            if rms > 2.0 and yaw_hint(f) is None and lift is None:
                                                              # (not when the heading is set by hand: no front/back flips)
                for dy in (90, 180, 270):
                    q = qprev.copy(); q[2] += dy
                    s2 = solve(lift_start(q, f, k), f, k, qprev, dt * 6, None)
                    if s2.cost < s.cost:
                        s = s2
        Q[f] = s.x.copy(); qprev = s.x.copy(); fprev = f
        loc, yaw, rot, pz = unpack(s.x, rig)
        uv, _ = world.project(rig.coco_points(rig.fk(loc, yaw, rot, pz)), f)
        m = k[:, 2] > 0.3
        err[f] = float(np.median(np.linalg.norm(uv[m] - k[m, :2], axis=1))) if m.any() else 0.0
        if n % 20 == 0:
            log("      frame %d  error %.1f px" % (f, err[f]))
    return rig, Q, err

def assemble(rig, Q, N, smooth, span=None, confidence=None, fps=30.):
    """every frame: interpolate the fitted frames, smooth, convert to the Blender keys"""
    fk_ = sorted(Q)
    A = np.array([Q[f] for f in fk_])
    A[:, 2] = np.degrees(np.unwrap(np.radians(A[:, 2])))
    first, last = span or (fk_[0], fk_[-1])
    allf = np.arange(N)
    B = np.array([np.interp(allf, fk_, A[:, c]) for c in range(A.shape[1])]).T
    if smooth > 0 and last - first > 3:
        seg = slice(first, last + 1)
        B[seg] = gaussian_filter1d(B[seg], smooth, axis=0, mode="nearest")
    if confidence is not None and last-first>3:
        from body_constraints import stabilize_arms
        from scipy.ndimage import minimum_filter1d
        original=B[first:last+1].copy()
        proposal=stabilize_arms(original,confidence[first:last+1],ANG,fps)
        limit=np.maximum(.7,merge_b(fk_b(rig,original),rig.s).max(1))
        strength=np.ones(len(original))
        for fallback in (.5,.25,0.):
            trial=original+(proposal-original)*strength[:,None]
            strength[merge_b(fk_b(rig,trial),rig.s).max(1)>limit]=fallback
        strength=minimum_filter1d(strength,size=5,mode='nearest')
        trial=original+(proposal-original)*strength[:,None]
        unsafe=merge_b(fk_b(rig,trial),rig.s).max(1)>limit+.01
        trial[unsafe]=original[unsafe]
        B[first:last+1]=trial
    lo, hi = bounds(rig)
    lo[3] = min(lo[3], 0.04 * rig.s, float(B[:, 3].min()) - 1e-3)   # kneeling / lying people sit lower
    for i_, (lim_lo, lim_hi) in zip(PELVIS_IDX, ((-110.0, 110.0), (-60.0, 60.0))):
        lo[NB + i_], hi[NB + i_] = lim_lo, lim_hi
    hi[3] = max(hi[3], float(B[:, 3].max()) + 1e-3)          # riders sit higher than a standing pelvis
    sg = np.r_[0.03, 0.03, 4.0, 0.03, np.full(len(ANG), 6.0)]
    if last >= first:                                       # smoothing / filling in can push blocks into each other again
        Q0 = np.clip(B[first:last + 1], lo + 1e-6, hi - 1e-6)
        bad = merge_b(fk_b(rig, Q0), rig.s).max(axis=1) > 0.7
        fixed = {}
        for f in np.nonzero(bad)[0] + first:
            q0 = Q0[f - first]
            def res(Qb, q0=q0):
                return np.concatenate([(Qb - q0) / sg, 3.0 * merge_b(fk_b(rig, Qb), rig.s)], axis=1)
            fixed[f] = lsq_b(res, q0, lo, hi, max_nfev=40, x_scale="jac", diff_step=1e-3).x - B[f]
        if fixed:        # blend each frame's correction into its neighbours: no one-frame snap where blocks were pushed apart
            D = np.zeros_like(B); W_ = 10; ff = np.array(sorted(fixed))
            for g in range(first, last + 1):                      # each frame: the nearest corrected frame's correction,
                k = int(np.argmin(np.abs(ff - g))); dist = abs(int(ff[k]) - g)     # fading out over W_ frames
                if dist <= W_:
                    D[g] = fixed[int(ff[k])] * (1.0 - dist / (W_ + 1.0))
            if last - first > 3:                                  # soften the ramp corners too
                D[first:last + 1] = gaussian_filter1d(D[first:last + 1], 1.5, axis=0, mode="nearest")
            B = B + D
    out = {"root": [], "pelvis_z": [], "rot": {j: [] for j in JOINTS}, "visible": []}
    for f in range(N):
        loc, yaw, rot, pz = unpack(B[f], rig)
        out["root"].append([round(float(loc[0]), 4), round(float(loc[1]), 4), 0.0, round(float(yaw), 2)])
        out["pelvis_z"].append(round(float(pz), 4))
        for j in JOINTS:
            out["rot"][j].append([round(float(v), 2) for v in rot[j]])
        out["visible"].append(bool(first <= f <= last))
    return out, B

def cached_fit_track(world, y0, fs, K, height, c, style, hint, lift, p3):
    """Cache each numeric fit independently of character names, colours and other people."""
    import tempfile
    from dependencies import digest
    directory = os.path.join(c['_work'], 'person_fit_cache') if c.get('_work') else None
    path = None
    if directory:
        frames = list(range(max(fs)+1))
        key = digest({'version': 4, 'frames': fs, 'keypoints': K, 'y0': y0, 'height': height,
            'style': style, 'hint': hint, 'lift': lift, 'p3': p3, 'step': c.get('fit_step', 2),
            'camera': [world.cam.K(), [world.C_at(f) for f in frames],
                       [world.R_cw(f) for f in frames], [world.f_at(f) for f in frames]]})
        os.makedirs(directory, exist_ok=True); path = os.path.join(directory, key+'.npz')
        if os.path.exists(path):
            try:
                with np.load(path, allow_pickle=False) as a:
                    fr, values, errors = a['frames'], a['values'], a['errors']
                    if values.shape != (len(fr), NP) or not np.isfinite(values).all(): raise ValueError('Invalid fit cache')
                    log('      numeric person fit: reused')
                    return Rig(height, style), dict(zip(fr.tolist(), values)), dict(zip(fr.tolist(), errors.tolist()))
            except (OSError, ValueError, KeyError):
                pass
    rig, Q, err = fit_track(world, y0, fs, K, height, step=c.get('fit_step', 2), style=style,
                           hint=hint, lift=lift, p3=p3)
    if path:
        fr = sorted(Q)
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.npz', delete=False) as target: temp = target.name
        try:
            np.savez_compressed(temp, frames=fr, values=np.asarray([Q[f] for f in fr]), errors=[err[f] for f in fr])
            os.replace(temp, path)
        finally:
            if os.path.exists(temp): os.remove(temp)
    return rig, Q, err


def fit_all(world, y0, people, N, c):
    """people {id: (frames, kps)} -> DATA['chars'] entries + per-person report"""
    from body_constraints import image_keys
    people={pid:(fs,image_keys(keys,world.cam.W,world.cam.H,y0)) for pid,(fs,keys) in people.items()}
    labels = ["Main character", "Second character"]
    ROLE = {"main": ("Main character", "red"), "second": ("Second character", "blue"), "third": ("Third character", "green"),
            "fourth": ("Fourth character", "yellow"), "extra": ("Extra person", None)}
    rv = (c.get("_review") or {}).get("people", {})          # key points confirmed in the review app
    chars = {}; report = []
    order = sorted(people, key=lambda p: -np.mean([np.ptp(k[k[:, 2] > 0.3, 1]) if (k[:, 2] > 0.3).sum() > 2 else 0
                                                    for k in people[p][1]]) * len(people[p][0]))
    if rv:                                                   # reviewed: drop ignored tracks, main role first
        rank = {"main": 0, "second": 1, "third": 2, "fourth": 3, "extra": 4}
        order = [p for p in order if rv.get(str(p), {}).get("role", "extra") != "ignore"]
        order.sort(key=lambda p: rank.get(rv.get(str(p), {}).get("role", "extra"), 4))
    n_extra = 0
    todo = order[: c["max_people"] if not rv else len(order)]
    c.setdefault('_required_tracks', set()).update(pid for pid in todo if len(people[pid][0]) >= 4)
    lifts, body_info = {}, {}
    body = c.get('_sam3d_body')
    if body is not None:
        import sam3d_body
        for pid in todo:
            prefix = 'p%s'%pid
            if prefix+'_placed' not in body: continue
            fr,xyz = body[prefix+'_f'],body[prefix+'_placed']
            import body_constraints
            dense,interpolated,confidence,info=body_constraints.prepare(world,y0,*people[pid],fr,xyz,c.get('_fps',30.))
            body_info[pid]=info
            lifts[pid]=(dense,sam3d_body.h36m(interpolated),body_constraints.confidence_h36m(confidence),'camera')
    p3 = {"weight": float(c.get("pose3d_weight", 1.0)), "sigma": float(c.get("pose3d_sigma", 0.2))}
    for n, pid in enumerate(todo):
        fs, K = people[pid]
        if len(fs) < 4:
            continue
        r = rv.get(str(pid), {})
        model_height = sam3d_body.height_of(body['p%s_placed'%pid]) if body is not None and 'p%s_placed'%pid in body else None
        info=body_info.get(pid,{})
        trustworthy_height=not (info.get('full_body_fraction',1)<.5 and info.get('placement_confidence',1)<.5)
        actor_heights = c.setdefault('_actor_heights', {})
        height = float(r.get("height") or actor_heights.get(str(pid)) or
                       (model_height if trustworthy_height else None) or c["person_height"])
        actor_heights[str(pid)] = height
        log("    person %s: %d frames%s" % (pid, len(fs), ("  role " + r["role"]) if r.get("role") else ""))
        style = (c.get("_hints") or {}).get("character_style") or c.get("character_style", "sample")
        hint = {"foot_contact": c.get("foot_contact", True),
                **(((c.get("_hints") or {}).get("people") or {}).get(str(pid)) or {})}
        if body is not None and pid in lifts:
            prefix = 'p%s'%pid; xyz = body[prefix+'_placed']
            factor = height/max(.4,model_height or height)
            hip = xyz[:,[11,12]].mean(1)*factor
            if not any(k in hint for k in ('distance','mount','floor_z')):
                targets = []
                for f,p in zip(body[prefix+'_f'],hip):
                    f = int(f); world_point = world.C_at(f)+world.R_cw(f).T @ p
                    targets.append([f,*world_point.tolist()])
                hint['_body_start']=targets
                if info.get('placement_confidence',0)>.35:
                    hint['_body_hip']=targets
                    hint['_body_hip_sigma']=.15/max(.35,info['placement_confidence'])
                if info.get('full_body_fraction',1)<.5 and not any(k in hint for k in ('legs','leg_pose','pelvis_z','seat_height')):
                    hint['_body_unanchored']=True
                    hint['legs']='free'
                    hint.setdefault('kp_weight',2.)
        rig, Q, err = cached_fit_track(world, y0, fs, K, height, c, style, hint, lifts.get(pid), p3)
        sm = (hint or {}).get("smooth", 1.0 if (hint or {}).get("follow_dots") else c.get("smooth_frames", 1.5))
        arm_confidence=None
        if pid in lifts and not any(hint.get(k) for k in ('pose','keys','keys_more','ignore_kps')):
            arm_confidence=np.stack([np.interp(np.arange(N),fs,np.asarray(K)[:,j,2]) for j in range(17)],1)
            arm_confidence[~np.isin(np.arange(N),fs)]=0
        keys, Bq = assemble(rig, Q, N, float(sm), span=(min(fs), max(fs)),
                            confidence=arm_confidence,fps=c.get('_fps',30.))
        if style == 'l3' and c.get('temporal_refinement', True) and not any(
                hint.get(k) for k in ('pose','keys','keys_more','ignore_kps','use_kps','hand_dist','hand_distance')):
            import temporal_refine
            Bq, refinement = temporal_refine.arms(rig, Bq, fs, K, world, y0, c.get('_fps',30.))
            for frame, parameters in enumerate(Bq):
                _, _, rotations, _ = unpack(parameters, rig)
                for joint in ('shoulder_L','elbow_L','shoulder_R','elbow_R'):
                    keys['rot'][joint][frame] = [round(float(v),2) for v in rotations[joint]]
            keys['temporal_refinement'] = refinement
        if c.get('_persistent_tracks'):
            observed = set(fs)
            keys['visible'] = [f in observed for f in range(N)]
        if c.get("motion_prior") or (c.get("_hints") or {}).get("motion_prior"):   # opt-in: real-world movement, soft
            try:
                import motion_prior
                keys = motion_prior.apply(keys, fs, K, rig, c.get("_fps", 30.0), hint, log=log)
            except Exception as e:                           # never let the prior stop a fit
                log("      motion prior skipped (%s)" % str(e)[:200])
        # how far body blocks still overlap after smoothing (cm)
        deep = float(merge_b(fk_b(rig, Bq[min(fs):max(fs) + 1]), rig.s).max()) * 1.5
        log("      body blocks: deepest overlap %.1f cm" % deep)
        if r.get("role") in ROLE:
            base, colour = ROLE[r["role"]]
            colour = r.get("colour") or colour or c["colours"][min(4 + n_extra, len(c["colours"]) - 1)]
            if r["role"] == "extra":
                n_extra += 1; base = "Extra person %d" % n_extra
            label = base + "_" + colour + ("_" + r["name"] if r.get("name") else "")
        else:
            colour = c["colours"][min(n, len(c["colours"]) - 1)]
            label = (labels[n] if n < 2 else "Extra person %d" % (n - 1)) + "_" + colour
        kind = "P%d" % (n + 1)
        keys.update(label=label, colour=colour, height=height, track_id=pid, style=style)
        keys['human_backend'] = 'sam3d_body' if pid in lifts else '2d'
        keys['body_constraints']=dict(info,version=2,unanchored=bool(hint.get('_body_unanchored')),
                                      height_source='review' if r.get('height') else ('body' if trustworthy_height and model_height else 'configured prior'))
        chars[kind] = keys
        e = np.array(list(err.values()))
        report.append({"person": kind, "track": pid, "colour": colour, "frames": len(fs),
                       "median_error_px": round(float(np.median(e)), 1), "worst_error_px": round(float(e.max()), 1)})
        report[-1]['human_backend'] = keys['human_backend']
    return chars, report

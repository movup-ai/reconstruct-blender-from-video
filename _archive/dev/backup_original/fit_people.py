# -*- coding: utf-8 -*-
"""3D pose of every person: the L3 rig is fitted to the 2D marks of each frame (COCO-17),
standing on the ground, with joint limits, pose priors and smooth motion."""
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from rig import Rig, JOINTS
from common import log

# free parameters: (joint, axis, low, high, prior sigma)   (deg)
ANG = [("spine", 0, -30, 95, 30), ("spine", 1, -35, 35, 12), ("spine", 2, -50, 50, 18),
       ("chest", 0, -20, 45, 15), ("chest", 2, -30, 30, 12),
       ("neck", 0, -40, 50, 20), ("head", 0, -45, 45, 20), ("head", 2, -75, 75, 30),
       ("shoulder_L", 0, -70, 180, 45), ("shoulder_L", 1, -20, 165, 35), ("shoulder_L", 2, -90, 90, 40),
       ("shoulder_R", 0, -70, 180, 45), ("shoulder_R", 1, -165, 20, 35), ("shoulder_R", 2, -90, 90, 40),
       ("elbow_L", 0, 0, 150, 50), ("elbow_R", 0, 0, 150, 50),
       ("hip_L", 0, -45, 125, 30), ("hip_L", 1, -20, 45, 12), ("hip_L", 2, -40, 40, 15),
       ("hip_R", 0, -45, 125, 30), ("hip_R", 1, -45, 20, 12), ("hip_R", 2, -40, 40, 15),
       ("knee_L", 0, -150, 0, 40), ("knee_R", 0, -150, 0, 40)]
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
    top = 0.94 if rig.style == "sample" else 0.93
    lo = [-1e5, -1e5, -1e5, 0.50 * rig.s]; hi = [1e5, 1e5, 1e5, (top + 0.04) * rig.s]
    for _, _, l, h, _ in ANG:
        lo.append(l); hi.append(h)
    return np.array(lo, float), np.array(hi, float)

def fit_track(world, y0, fs, K, height, step=2, style="sample"):
    rig = Rig(height, style)
    lo, hi = bounds(rig)
    sig_prior = np.array([s for *_, s in ANG])
    fs = list(fs); K = np.asarray(K, float).copy(); K[:, :, 1] -= y0
    idx = sorted(set(list(range(0, len(fs), step)) + [len(fs) - 1]))
    def scale_px(k):
        v = k[k[:, 2] > 0.3, :2]
        return max(20.0, np.ptp(v[:, 1]) if len(v) > 2 else 60.0)
    def residual(q, f, k, qprev, dt, q0):
        loc, yaw, rot, pz = unpack(q, rig)
        W = rig.fk(loc, yaw, rot, pz)
        P = rig.coco_points(W)
        uv, z = world.project(P, f)
        sp = 0.03 * scale_px(k)
        w = np.where(k[:, 2] > 0.3, np.sqrt(np.clip(k[:, 2], 0, 1)), 0.0)
        r = list(((uv - k[:, :2]) * w[:, None] / sp).ravel())
        r += list(0.35 * q[NB:] / sig_prior)
        za, zb = W["ankle_L"][1][2], W["ankle_R"][1][2]
        r.append((min(za, zb) - rig.sole) / 0.03)
        r.append(min(0.0, max(za, zb) - rig.sole) / 0.02)
        r.append((q[3] - rig.J[0][2][2]) / 0.25)
        if qprev is not None:
            st = np.r_[0.12 * dt, 0.12 * dt, 12.0 * dt, 0.05 * dt, np.full(len(ANG), 18.0 * dt)]
            d = q - qprev; d[2] = (d[2] + 180) % 360 - 180
            r += list(d / st)
        if q0 is not None:
            r += list((q[:2] - q0[:2]) / 1.5)
        return np.array(r)
    def solve(q, f, k, qprev, dt, q0, it=60):
        q = np.clip(q, lo + 1e-6, hi - 1e-6)
        return least_squares(residual, q, bounds=(lo, hi), args=(f, k, qprev, dt, q0), loss="soft_l1", f_scale=2.0,
                             max_nfev=it, x_scale="jac", diff_step=1e-3)
    def init_q(f, k):
        q = np.zeros(NP); q[3] = rig.J[0][2][2]
        for i, (j, a, l, h, _) in enumerate(ANG):
            q[NB + i] = np.clip(0, l, h)
        anc = [i for i in (15, 16) if k[i, 2] > 0.3]
        g = None
        if anc:
            p = k[anc, :2].mean(0); p[1] = k[anc, 1].max()
            g = world.on_ground(f, p[0], p[1])
        if g is None or np.linalg.norm(g - world.C) > 400:
            tor = [i for i in (5, 6, 11, 12) if k[i, 2] > 0.3]
            hp = k[[i for i in (11, 12, 5, 6) if k[i, 2] > 0.3] or [0], :2].mean(0)
            tl = np.linalg.norm(k[[5, 6], :2].mean(0) - k[[11, 12], :2].mean(0)) if len(tor) == 4 else scale_px(k) / 3.3
            Z = world.cam.f * 0.50 * rig.s / max(tl, 4.0)
            d = world.ray(f, *hp); g = world.C + d * Z / max(d @ (world.R_cw(f).T @ [0, 0, 1.0]), 0.2)
            g[2] = 0.0
        q[0], q[1] = g[0], g[1]
        return q
    Q = {}; err = {}
    qprev = None; fprev = None
    for n, i in enumerate(idx):
        f, k = fs[i], K[i]
        if qprev is None:
            q0 = init_q(f, k)
            cam_dir = np.degrees(np.arctan2(*(q0[:2] - world.C[:2])[::-1]))     # heading from the camera to the person
            best = None
            for yaw in np.arange(0, 360, 45):
                q = q0.copy(); q[2] = cam_dir + yaw - 90
                s = solve(q, f, k, None, 1, q0, it=40)
                if best is None or s.cost < best.cost:
                    best = s
            s = solve(best.x, f, k, None, 1, q0, it=150)
        else:
            dt = max(1, f - fprev)
            s = solve(qprev, f, k, qprev, dt, None)
            rms = np.sqrt(np.mean(s.fun[:34] ** 2))
            if rms > 2.0:                                    # lost (turned around fast / left-right swap): retry other headings
                for dy in (90, 180, 270):
                    q = qprev.copy(); q[2] += dy
                    s2 = solve(q, f, k, qprev, dt * 6, None)
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

def assemble(rig, Q, N, smooth):
    """every frame: interpolate the fitted frames, smooth, convert to the Blender keys"""
    fk_ = sorted(Q)
    A = np.array([Q[f] for f in fk_])
    A[:, 2] = np.degrees(np.unwrap(np.radians(A[:, 2])))
    first, last = fk_[0], fk_[-1]
    allf = np.arange(N)
    B = np.array([np.interp(allf, fk_, A[:, c]) for c in range(A.shape[1])]).T
    if smooth > 0 and last - first > 3:
        seg = slice(first, last + 1)
        B[seg] = gaussian_filter1d(B[seg], smooth, axis=0, mode="nearest")
    out = {"root": [], "pelvis_z": [], "rot": {j: [] for j in JOINTS}, "visible": []}
    for f in range(N):
        loc, yaw, rot, pz = unpack(B[f], rig)
        out["root"].append([round(float(loc[0]), 4), round(float(loc[1]), 4), 0.0, round(float(yaw), 2)])
        out["pelvis_z"].append(round(float(pz), 4))
        for j in JOINTS:
            out["rot"][j].append([round(float(v), 2) for v in rot[j]])
        out["visible"].append(bool(first <= f <= last))
    return out, B

def fit_all(world, y0, people, N, c):
    """people {id: (frames, kps)} -> DATA['chars'] entries + per-person report"""
    labels = ["主角色", "第二角色"]
    chars = {}; report = []
    order = sorted(people, key=lambda p: -np.mean([np.ptp(k[k[:, 2] > 0.3, 1]) if (k[:, 2] > 0.3).sum() > 2 else 0
                                                    for k in people[p][1]]) * len(people[p][0]))
    for n, pid in enumerate(order[: c["max_people"]]):
        fs, K = people[pid]
        if len(fs) < 4:
            continue
        log("    person %s: %d frames" % (pid, len(fs)))
        style = c.get("character_style", "sample")
        rig, Q, err = fit_track(world, y0, fs, K, c["person_height"], step=c.get("fit_step", 2), style=style)
        keys, _ = assemble(rig, Q, N, c.get("smooth_frames", 1.5))
        colour = c["colours"][min(n, len(c["colours"]) - 1)]
        label = (labels[n] if n < 2 else "其他人物%d" % (n - 1)) + "_" + colour
        kind = "P%d" % (n + 1)
        keys.update(label=label, colour=colour, height=c["person_height"], track_id=pid, style=style)
        chars[kind] = keys
        e = np.array(list(err.values()))
        report.append({"person": kind, "track": pid, "colour": colour, "frames": len(fs),
                       "median_error_px": round(float(np.median(e)), 1), "worst_error_px": round(float(e.max()), 1)})
    return chars, report

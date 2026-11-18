# -*- coding: utf-8 -*-
"""Feet on the ground and a real stepping gait (client feedback: "äººç‰©çš„æ‚¬æµ®æ„Ÿ/æ¼‚æµ®æ„Ÿå¤ªå¼º", "æ­¥æ€å’ŒåŽŸç‰‡ä¸ç¬¦").
Post-process of the fitted animation, per person; the body (root + upper body) keeps the fitted path, so the 2D
alignment with the film stays.

1. Vertical: every frame the body moves up / down so the lowest sole touches the floor (floor_z or 0); <= 0.35 m.
2. Steps: the lower foot (2 cm hysteresis) is the stance foot.  A stance is split whenever the body has carried the
   planted foot further than a natural stride, so a gliding body gets alternating steps.  Each stance plants its foot
   at one floor point (the fitted foot position at mid-stance).  Between its stances a foot swings from its last plant
   to its next one with a small lift arc.
3. Legs: hip (x, y) and knee angles are solved analytically (two-bone IK in the pelvis frame) to reach the targets;
   the ankle stays level.  Frames where the person sits / kneels / crouches (pelvis low) keep the fitted legs.

Applied by run_pipeline.render_only / render_sample_v1 when the clip's hints.json has "plant_feet": true.
Marks data["feet_planted"]; works on the source-frame timeline (before the pulldown re-time)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from rig import Rig, JOINTS


def _fk(ch, rig, f):
    r = ch["root"][f]
    return rig.fk((r[0], r[1], r[2]), r[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f])


def _leg_ik(v, a, b):
    """hip-local vector v (hip -> ankle) -> (hip_x, hip_y, knee_x) degrees; Rxyz order, leg rest along -Z"""
    d = float(np.linalg.norm(v))
    d = min(max(d, abs(a - b) + 1e-3), a + b - 1e-4)
    v = v / max(np.linalg.norm(v), 1e-9) * d
    cosk = (a * a + b * b - d * d) / (2 * a * b)
    k = np.pi - np.arccos(np.clip(cosk, -1, 1))                    # knee bend (0 = straight)
    p0 = np.array([0.0, -b * np.sin(k), -a - b * np.cos(k)])        # ankle with hip angles 0
    p1z = -np.hypot(v[0], v[2])
    y = np.arctan2(v[0] / p1z, v[2] / p1z) if abs(p1z) > 1e-6 else 0.0
    x = np.arctan2(p1z, v[1]) - np.arctan2(p0[2], p0[1])
    x = (x + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(x), np.degrees(y), -np.degrees(k)


def _hold(ch, ranges, N, blend=4):
    """hints "stand": [[f0, f1], ...] (clip frames): the person stands still there - body position held at its mean,
    legs straight and together; blended in / out over `blend` frames"""
    root = np.array(ch["root"], float); w = np.zeros(N); tgt = root[:, :2].copy()
    for f0, f1 in ranges:
        f0 = max(0, int(f0)); f1 = min(N - 1, int(f1))
        m = root[f0:f1 + 1, :2].mean(0)
        for f in range(max(0, f0 - blend), min(N, f1 + blend + 1)):
            a = 1.0 if f0 <= f <= f1 else 1.0 - (min(abs(f - f0), abs(f - f1)) / float(blend + 1))
            if a > w[f]:
                w[f] = a; tgt[f] = m
    root[:, :2] = root[:, :2] * (1 - w[:, None]) + tgt * w[:, None]
    ch["root"] = root.tolist()
    for j, v in (("hip_L", [3, 2, 0]), ("hip_R", [3, -2, 0]), ("knee_L", [-4, 0, 0]), ("knee_R", [-4, 0, 0]),
                 ("ankle_L", [1, 0, 0]), ("ankle_R", [1, 0, 0])):
        r = np.array(ch["rot"][j], float)
        r = r * (1 - w[:, None]) + np.array(v, float) * w[:, None]
        ch["rot"][j] = r.tolist()
    return w > 0.5


CADENCE_WALK, CADENCE_RUN = 110.0, 170.0                  # steps / min (adults; see docs/MOTION_RESEARCH.md)


def plant(data, max_dz=0.35, stride=0.36, lift=0.09, log=print, hints=None, ankle_vis=None):
    if data.get("feet_planted"):
        return
    fps = float(data.get("fps", 30) or 30)
    ppl = (hints or {}).get("people") or {}
    for kind, ch in data.get("chars", {}).items():
        rig = Rig(ch["height"], ch.get("style", "sample"))
        vis = np.array(ch["visible"], bool); N = len(vis)
        if vis.sum() < 4:
            continue
        ph = ppl.get(str(ch.get("track_id"))) or {}
        if ph.get("plant") is False:                     # hints: this person is left exactly as fitted
            continue
        held = np.zeros(N, bool)
        r0 = np.array(ch["root"], float)
        xy = gaussian_filter1d(r0[:, :2], 3.0, axis=0, mode="nearest")
        speed = gaussian_filter1d(np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)] * fps, 3.0, mode="nearest")
        if ph.get("follow_dots"):                         # the dots keep the pose; legs get real planted steps only
            held = ~(speed > 0.25)                        # while the person really travels (walk / run)
        for f0, f1 in (ph.get("keep_legs") or []):       # hints: hand-set legs here - only the vertical fix, no step IK
            held[max(0, int(f0)):min(N, int(f1) + 1)] = True
        st = ph.get("stand")
        if st:
            held = held | _hold(ch, st, N)
        floor = float(ch.get("floor_z", 0.0) or 0.0)
        Jo = {n: np.asarray(o, float) for n, _, o in rig.J}
        a = float(np.linalg.norm(Jo["knee_L"])); b = float(np.linalg.norm(Jo["ankle_L"]))
        stand = float(rig.J[0][2][2])                                    # pelvis height when standing
        root = np.array(ch["root"], float)
        # 1. vertical
        W = [_fk(ch, rig, f) for f in range(N)]
        S = np.array([[W[f]["ankle_L"][1], W[f]["ankle_R"][1]] for f in range(N)]); S[:, :, 2] -= rig.sole
        dz = np.clip(floor - S[:, :, 2].min(1), -max_dz, max_dz); dz[~vis] = 0.0
        idx = np.nonzero(vis)[0]
        dz[idx] = gaussian_filter1d(dz[idx], 0.8, mode="nearest")
        root[:, 2] += dz
        ch["root"] = root.tolist()
        S[:, :, 2] += dz[:, None]
        walk = vis & (np.array(ch["pelvis_z"]) > 0.70 * stand) & ~held  # standing / walking frames get the step IK
        # 2. stances
        # step length from the real cadence: speed x 60 / cadence (walking 110, running 170 steps / min), capped by
        # hints step_len (kimono / hakama: short steps)
        hs = ch["height"] / 1.75
        run = speed > 2.2
        cad = np.where(run, CADENCE_RUN, CADENCE_WALK)
        cap = float(ph.get("step_len", 1.2)) * hs
        step_f = np.clip(speed * 60.0 / cad, 0.18 * hs, cap)
        lift_f = np.where(run, 0.22, lift) * hs
        lo = np.full(N, -1); cur = int(S[idx[0], :, 2].argmin()); seg_start = idx[0]; segs = []
        for f in range(N):
            if not walk[f]:
                if lo[f - 1] >= 0 if f else False:
                    segs.append([seg_start, f - 1, cur])
                lo[f] = -1; continue
            if f == 0 or lo[f - 1] < 0:
                cur = int(S[f, :, 2].argmin()); seg_start = f
            else:
                switch = False if ph.get("follow_dots") else S[f, 1 - cur, 2] < S[f, cur, 2] - 0.02
                moved = np.linalg.norm(root[f, :2] - root[seg_start, :2])
                carried = moved > (step_f[f] if speed[f] > 0.25 else stride * hs)
                if switch or carried:
                    segs.append([seg_start, f - 1, cur]); cur = 1 - cur; seg_start = f
            lo[f] = cur
        if N and lo[N - 1] >= 0:
            segs.append([seg_start, N - 1, cur])
        T = np.full((N, 2, 3), np.nan)                                   # foot targets
        for s, e, ft in segs:
            m = (s + e) // 2
            P = S[m, ft, :2]
            if speed[m] > 0.25:                                          # the body passes over the planted foot at
                d = xy[min(N - 1, m + 1)] - xy[max(0, m - 1)]            # mid-stance: forward position from the path,
                d = d / max(np.linalg.norm(d), 1e-6); perp = np.array([-d[1], d[0]])   # sideways from the fitted foot
                side = float(np.clip((S[m, ft, :2] - root[m, :2]) @ perp, -0.18 * hs, 0.18 * hs))
                if abs(side) < 0.05 * hs:
                    side = (0.09 if ft == 0 else -0.09) * hs             # perp points to the walker's left
                P = root[m, :2] + perp * side
            T[s:e + 1, ft, :2] = P; T[s:e + 1, ft, 2] = floor
        for ft in (0, 1):                                                # swings between a foot's plants
            have = np.nonzero(~np.isnan(T[:, ft, 0]))[0]
            if not len(have):
                continue
            for f in range(N):
                if not walk[f] or not np.isnan(T[f, ft, 0]):
                    continue
                prv = have[have < f]; nxt = have[have > f]
                if len(prv) and len(nxt):
                    p, n = prv[-1], nxt[0]
                    if n - p > 40:                                       # a long gap (sat down in between): leave it
                        continue
                    t = (f - p) / float(n - p)
                    xy = (1 - t) * T[p, ft, :2] + t * T[n, ft, :2]
                    h = min(lift_f[f], 0.35 * np.linalg.norm(T[n, ft, :2] - T[p, ft, :2])) * np.sin(np.pi * t)
                    T[f, ft] = [xy[0], xy[1], floor + h]
        # 3. leg IK where there is a target
        rot = {j: np.array(ch["rot"][j], float) for j in ch["rot"]}
        n_ik = 0
        for f in range(N):
            if not walk[f]:
                continue
            Wf = _fk({"root": root.tolist(), "rot": ch["rot"], "pelvis_z": ch["pelvis_z"]}, rig, f)
            Rp = Wf["pelvis"][0]
            for ft, sd in ((0, "L"), (1, "R")):
                if np.isnan(T[f, ft, 0]):
                    continue
                tgt = T[f, ft] + [0, 0, rig.sole]
                v = Rp.T @ (tgt - Wf["hip_" + sd][1])
                hx, hy, kx = _leg_ik(v, a, b)
                rot["hip_" + sd][f, :2] = [hx, hy]; rot["hip_" + sd][f, 2] = 0.0
                rot["knee_" + sd][f, 0] = kx
                rot["ankle_" + sd][f, 0] = float(np.clip(-(hx + kx), -40, 50))
                n_ik += 1
        for j in ("hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"):     # light smoothing of the new legs
            if walk.any():
                w = np.nonzero(walk)[0]
                rot[j][w] = gaussian_filter1d(rot[j][w], 1.8, axis=0, mode="nearest")
            ch["rot"][j] = np.round(rot[j], 3).tolist()
        log("  feet: %s %s: %d steps, vertical fix up to %.2f m, legs solved in %d foot-frames" % (
            kind, ch.get("colour"), len(segs), float(np.abs(dz).max()), n_ik))
    data["feet_planted"] = True


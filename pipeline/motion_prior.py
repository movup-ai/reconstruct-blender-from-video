# -*- coding: utf-8 -*-
"""Real-world movement knowledge as a SOFT recommendation for the fitted characters.

The detected joints (dots) stay the main truth. After the fit, every joint of every frame is nudged towards what a
real body does in that motion state, with a weight that is LOW where the controlling dots are confident and higher
only where they are weak or missing (blur, occlusion, dark frames, long robes hiding the legs):

    new = fitted + strength * (1 - dot_confidence) * (recommended - fitted)

Motion state per frame from the fitted body: standing (< 0.25 m/s), walking (0.25 - 2.2 m/s), running (> 2.2 m/s or
very large strides); seated / kneeling / crouched / lying frames (pelvis well below standing) are left alone.

Reference values (rounded; sources from literature, see docs\\MOTION_RESEARCH.md):
- Walking gait cycle: stance ~60 %, swing ~40 %.  Hip ~30 deg flexion at heel contact, ~10-15 deg extension before
  toe-off.  Knee near straight at heel contact, ~10-15 deg flexion in early stance, ~60 deg peak flexion in swing.
- Walking: step length ~0.7-0.85 m for adults (~0.41-0.45 x body height); cadence ~100-120 steps / min.
- Arm swing in walking: shoulder flexion-extension ~20-25 deg peak-to-peak, opposite to the same-side leg; elbow kept
  mostly straight, ~15-30 deg flexion.  Running: elbow bent ~80-90 deg.
- Running: hip ~45-50 deg flexion at foot strike, up to ~55 deg in swing; knee ~25-40 deg at strike/mid-stance,
  ~90+ deg peak swing flexion.
- Trunk: pelvis/trunk near upright when walking (few degrees forward), a little more forward when running.
- Joint ranges (AAOS-type normal values): hip flexion 120 / extension 20 / abduction 45 / adduction 30, knee flexion
  ~135-150, elbow flexion ~140-150, shoulder flexion 180 / extension 60, neck rotation ~70-80.
- Kimono / hakama: shorter steps, smaller knee lift, little arm swing (hands often held at the belt or sword).
Hints per person: "motion_prior": false (off), "motion_strength": 0..1 (default 0.6), "motion_style":
"kimono" (short steps, small arm swing)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

# soft anatomical ranges (deg, rig axes): beyond these the fit is pulled back (not clipped)
ROM = {"hip_L": [(-20, 120), (-30, 45), (-40, 40)], "hip_R": [(-20, 120), (-45, 30), (-40, 40)],
       "knee_L": [(-145, 0)], "knee_R": [(-145, 0)], "elbow_L": [(0, 145)], "elbow_R": [(0, 145)],
       "shoulder_L": [(-60, 180), (-30, 160), (-80, 80)], "shoulder_R": [(-60, 180), (-160, 30), (-80, 80)],
       "head": [(-45, 45), None, (-80, 80)], "spine": [(-90, 25), (-30, 30), (-45, 45)]}

# per motion state: (walk, run) recommended values
REC = {
    "walk": {"knee_swing": -55.0, "knee_stance": -8.0, "hip_amp": 20.0, "arm_gain": 0.55, "arm_max": 14.0,
             "elbow": 22.0, "spine": -4.0},
    "run":  {"knee_swing": -95.0, "knee_stance": -30.0, "hip_amp": 32.0, "arm_gain": 0.8, "arm_max": 35.0,
             "elbow": 85.0, "spine": -12.0},
    "kimono": {"knee_swing": -30.0, "knee_stance": -6.0, "hip_amp": 12.0, "arm_gain": 0.15, "arm_max": 5.0,
               "elbow": None, "spine": -3.0},
}
# dots (COCO) that decide a joint: joint -> keypoint indices
CTRL = {"knee_L": [13, 15], "knee_R": [14, 16], "hip_L": [11, 13], "hip_R": [12, 14], "shoulder_L": [5, 7],
        "shoulder_R": [6, 8], "elbow_L": [7, 9], "elbow_R": [8, 10], "spine": [5, 6, 11, 12], "head": [0, 1, 2, 3, 4]}


def _conf(fs, K, N, idx):
    """per frame mean confidence (0..1) of the given dots; frames without detection = 0"""
    c = np.zeros(N)
    for f, k in zip(fs, K):
        if 0 <= f < N:
            c[f] = float(np.clip(np.asarray(k)[idx, 2], 0, 1).mean())
    return c


def apply(keys, fs, K, rig, fps, hint=None, log=print):
    hint = hint or {}
    if hint.get("motion_prior") is False:
        return keys
    N = len(keys["root"]); fps = float(fps or 30.0)
    vis = np.array(keys["visible"], bool)
    if vis.sum() < 6:
        return keys
    s0 = float(hint.get("motion_strength", 0.6))
    root = np.array(keys["root"], float); pz = np.array(keys["pelvis_z"], float)
    rot = {j: np.array(v, float) for j, v in keys["rot"].items()}
    stand_pz = float(rig.J[0][2][2])
    # motion state
    xy = gaussian_filter1d(root[:, :2], 3.0, axis=0, mode="nearest")
    speed = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)] * fps
    speed = gaussian_filter1d(speed, 3.0, mode="nearest")
    upright = pz > 0.85 * stand_pz
    walk = vis & upright & (speed > 0.25) & (speed <= 2.2)
    run = vis & upright & (speed > 2.2)
    still = vis & upright & (speed <= 0.25)
    style = hint.get("motion_style")
    conf = {j: _conf(fs, K, N, idx) for j, idx in CTRL.items()}
    changed = {}

    def nudge(j, ax, target, mask, strength=1.0):
        if not mask.any():
            return
        w = s0 * strength * (1.0 - conf[j if j in conf else j.split("_")[0]]) * mask
        old = rot[j][:, ax].copy()
        rot[j][:, ax] = old + w * (target - old)
        changed[j] = changed.get(j, 0) + int((np.abs(rot[j][:, ax] - old) > 1.0).sum())

    for state, mask in (("walk", walk), ("run", run)):
        if not mask.any():
            continue
        R = REC["kimono"] if (style == "kimono" and state == "walk") else REC[state]
        mid = 0.5 * (rot["hip_L"][:, 0] + rot["hip_R"][:, 0])
        for sd, other in (("L", "R"), ("R", "L")):
            h = rot["hip_" + sd][:, 0]; ho = rot["hip_" + other][:, 0]
            # legs move in anti-phase around their mean, with a real stride amplitude
            dev = 0.5 * (h - ho)
            amp = np.clip(dev, -R["hip_amp"], R["hip_amp"])
            nudge("hip_" + sd, 0, mid + np.sign(dev) * np.maximum(np.abs(amp), 0.6 * R["hip_amp"] * (np.abs(dev) > 3)), mask, 0.5)
            # knee: bent in swing (the leg moving forward), nearly straight in stance
            hv = np.gradient(gaussian_filter1d(h, 1.5, mode="nearest"))
            swing = hv > 0.4
            ktar = np.where(swing, R["knee_swing"] * np.clip(hv / 4.0, 0.4, 1.0), R["knee_stance"])
            nudge("knee_" + sd, 0, ktar, mask, 0.8)
            # arm swings opposite to the same-side leg (with the other leg), small amplitude
            if R["arm_gain"] > 0:
                atar = np.clip(-R["arm_gain"] * (h - mid), -R["arm_max"], R["arm_max"]) + 6.0
                nudge("shoulder_" + sd, 0, atar, mask, 0.6)
            if R["elbow"] is not None:
                nudge("elbow_" + sd, 0, np.full(N, R["elbow"]), mask, 0.5)
        nudge("spine", 0, np.full(N, R["spine"]), mask, 0.4)
        nudge("spine", 1, np.zeros(N), mask, 0.5)             # no sideways lean while walking / running
    if still.any():                                            # standing: legs under the body, upright
        for sd in "LR":
            nudge("knee_" + sd, 0, np.full(N, -4.0), still, 0.5)
        nudge("spine", 1, np.zeros(N), still, 0.4)
    # soft joint ranges for every visible frame
    for j, axes in ROM.items():
        if j not in rot:
            continue
        for ax, lim in enumerate(axes):
            if lim is None or ax >= rot[j].shape[1]:
                continue
            v = rot[j][:, ax]; lo, hi = lim
            over = np.where(v > hi, v - hi, np.where(v < lo, v - lo, 0.0))
            n = int((np.abs(over) > 0.5).sum())
            if n:
                rot[j][:, ax] = v - 0.8 * over * vis
                changed[j] = changed.get(j, 0) + n
    # real limbs move smoothly: frame-to-frame dot noise is removed, more where the dots are unsure.
    # per joint and frame: sigma 0.8 (sure dots) .. 2.5 frames (weak dots) -> blend of two smoothings
    idx = np.nonzero(vis)[0]
    for j, v in rot.items():
        if len(idx) < 5:
            break
        cj = conf.get(j) if j in conf else conf.get({"neck": "head", "chest": "spine", "wrist_L": "elbow_L", "wrist_R": "elbow_R",
                                                    "ankle_L": "knee_L", "ankle_R": "knee_R", "pelvis": "spine"}.get(j, ""), None)
        cj = np.ones(N) if cj is None else gaussian_filter1d(cj, 2.0, mode="nearest")
        a = gaussian_filter1d(v[idx], 0.8, axis=0, mode="nearest")
        b = gaussian_filter1d(v[idx], 2.5, axis=0, mode="nearest")
        w = np.clip((1.0 - cj[idx]) * 1.4, 0, 1)[:, None] * min(1.0, s0 / 0.6)
        v[idx] = a * (1 - w) + b * w
        # no joint turns faster than a real body (about 20 deg per frame at 30 fps, fast swings included)
        lim = 600.0 / fps
        for t in range(1, len(idx)):
            d = v[idx[t]] - v[idx[t - 1]]
            v[idx[t]] = v[idx[t - 1]] + np.clip(d, -lim, lim)
        keys["rot"][j] = np.round(v, 2).tolist()
    log("      motion prior: walk %d, run %d, still %d frames; nudged %s" % (
        int(walk.sum()), int(run.sum()), int(still.sum()),
        ", ".join("%s %d" % (j, n) for j, n in sorted(changed.items())) or "nothing"))
    return keys

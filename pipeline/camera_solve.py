# -*- coding: utf-8 -*-
"""per-frame camera rotation from tracked background features (pan / tilt / roll shots).
writes camera.json: R[i] (frame-0 camera coords -> frame-i camera coords), picture size, focal, warnings.

solve() now uses camera_estimate (dense flow against keyframes, turn + zoom model with the camera's travel separated
by the depth model, steadied, small rolls dropped).  The old homography tracker is kept as solve_homography()
(config "camera_tracker": "homography")."""
import numpy as np, cv2
from common import *

def solve(frames, pose, c):
    if c.get("camera_tracker") == "homography":
        return solve_homography(frames, pose, c)
    import camera_estimate as CE
    depth = CE.get_depth(c) if c.get("camera_depth", True) else None
    return CE.steady(CE.solve(frames, pose, c, depth), c)

def solve_homography(frames, pose, c):
    x0, y0, W, Hp = pose["picture_area"]
    cam = Camera(W, Hp, c["lens_mm"])
    G = [cv2.cvtColor(f[y0:y0 + Hp], cv2.COLOR_BGR2GRAY) for f in frames]
    N = len(G)
    # mask people (boxes from pose.json, padded)
    boxes = {}
    for p in pose["people"]:
        for f, b in p.get("boxes", {}).items():
            boxes.setdefault(int(f), []).append(b)
    def mask(i):
        m = np.full((Hp, W), 255, np.uint8)
        for b in boxes.get(i, []):
            w, h = b[2] - b[0], b[3] - b[1]
            m[max(0, int(b[1] - y0 - 0.15 * h)):int(b[3] - y0 + 0.1 * h), max(0, int(b[0] - 0.3 * w)):int(b[2] + 0.3 * w)] = 0
        return m
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    R = [np.eye(3)]; ref = 0; Rref = np.eye(3)
    pts_ref = cv2.goodFeaturesToTrack(G[0], 1500, 0.01, 12, mask=mask(0))
    pts_ref = np.zeros((0, 2), np.float32) if pts_ref is None else pts_ref.reshape(-1, 2)
    cur = pts_ref.copy(); alive = np.ones(len(cur), bool)
    resid = []; cuts = []
    for i in range(1, N):
        if len(cur) and alive.any():
            p1, st, _ = cv2.calcOpticalFlowPyrLK(G[i - 1], G[i], cur.reshape(-1, 1, 2).astype(np.float32), None, **lk)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(G[i], G[i - 1], p1, None, **lk)
            p1 = p1.reshape(-1, 2); pb = pb.reshape(-1, 2)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pb - cur, axis=1) < 0.8)
            ok &= (p1[:, 0] > 2) & (p1[:, 0] < W - 2) & (p1[:, 1] > 2) & (p1[:, 1] < Hp - 2)
            mk = mask(i); idx = np.where(ok)[0]
            ok[idx] = mk[p1[idx, 1].astype(int), p1[idx, 0].astype(int)] > 0
            alive &= ok; cur = p1
        n = int(alive.sum())
        Ri = None
        if n >= 25:
            a = np.array([cam.norm(*p) for p in pts_ref[alive]]); b = np.array([cam.norm(*p) for p in cur[alive]])
            Hm, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.5 / cam.f)
            if Hm is not None and inl.sum() >= 20:
                Hm = Hm / np.cbrt(np.linalg.det(Hm)); U, S, Vt = np.linalg.svd(Hm); Rr = U @ Vt
                if np.linalg.det(Rr) < 0: Rr = -Rr
                Ri = Rr @ Rref
                pr = cv2.perspectiveTransform(a.reshape(-1, 1, 2), Hm).reshape(-1, 2)
                resid.append(float(np.median(np.linalg.norm(pr - b, axis=1)) * cam.f))
        if Ri is None:                          # lost the background: a cut or a very fast move
            cuts.append(i); Ri = R[-1]
        R.append(Ri)
        if n < 300 or Ri is R[-2]:              # refresh the reference set
            ref = i; Rref = R[-1]
            p = cv2.goodFeaturesToTrack(G[i], 1500, 0.01, 12, mask=mask(i))
            pts_ref = np.zeros((0, 2), np.float32) if p is None else p.reshape(-1, 2)
            cur = pts_ref.copy(); alive = np.ones(len(cur), bool)
    ang = [float(np.degrees(np.linalg.norm(cv2.Rodrigues(r)[0]))) for r in R]
    par = float(np.percentile(resid, 90)) if resid else 0.0
    warn = []
    if cuts: warn.append("camera lost at frames %s (cut or fast move) - check this clip" % cuts[:10])
    if par > 2.0: warn.append("camera seems to MOVE (parallax %.1f px): far set may slide a little" % par)
    return {"R": [r.tolist() for r in R], "W": W, "H": Hp, "y0": y0, "f": cam.f, "lens_mm": c["lens_mm"],
            "max_rotation_deg": max(ang), "parallax_px": par, "cuts": cuts, "warnings": warn}

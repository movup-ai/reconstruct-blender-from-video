# -*- coding: utf-8 -*-
"""A set piece shaped from what the camera measured, for things no box or library model fits: plants, curtains,
statues, flowers, cushions, heaps.  The depth model's 3D points inside the piece's outline become a surface (a grid
over the outline: its real silhouette and folds), and that surface is given a back, pushed away from the camera, and
sides, so it is a closed solid that casts shadows like the real thing.

  relief_mesh(X, ok, C, thick)  ->  (V, F) in the same world as X, or None
    X: (H, W, 3) world points of the frame, ok: (H, W) bool - the piece's pixels that have depth,
    C: camera position, thick: how deep the solid is behind the seen surface (metres)."""
import numpy as np


def relief_mesh(X, ok, C, thick, cells=36, min_nodes=6, jump=0.12, smooth=2, scale=1.0):
    ys, xs = np.nonzero(ok)
    if len(xs) < 30:
        return None
    C = np.asarray(C, float)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    step = max(2, int(round(max(x1 - x0, y1 - y0) / float(cells))))
    gx = np.arange(x0, x1 + step, step); gy = np.arange(y0, y1 + step, step)
    H, W = ok.shape
    nx, ny = len(gx), len(gy)
    D = np.full((ny, nx), np.nan); U = np.zeros((ny, nx, 3))
    h = step // 2 + 1
    for j, y in enumerate(gy):                                   # a node: the median of the piece's points around it
        for i, x in enumerate(gx):
            a, b = max(0, y - h), min(H, y + h); c, d = max(0, x - h), min(W, x + h)
            m = ok[a:b, c:d]
            if m.size == 0 or m.mean() < 0.35:
                continue
            P = X[a:b, c:d][m]
            v = P - C
            r = np.linalg.norm(v, axis=1)
            k = int(np.argsort(r)[len(r) // 2])
            D[j, i] = r[k]; U[j, i] = v[k] / max(r[k], 1e-9)
    good = np.isfinite(D)
    if good.sum() < min_nodes:
        return None
    lo, hi = np.percentile(D[good], [5, 95])                     # stray far / near points (edges, see-through)
    D = np.where(good, np.clip(D, lo, hi), np.nan)
    for _ in range(smooth):                                      # soften the measuring noise, keep the shape
        P = np.pad(D, 1, constant_values=np.nan)
        nb = np.stack([P[1:-1, 1:-1], P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]])
        D = np.where(good, np.nanmean(nb, axis=0), np.nan)
    D = D * scale
    med = float(np.nanmedian(D))
    cut = max(jump, 0.06 * med)                                  # a bigger step in depth: two things, not one surface
    idx = -np.ones((ny, nx), int); V = []
    for j in range(ny):
        for i in range(nx):
            if good[j, i]:
                idx[j, i] = len(V); V.append(C + U[j, i] * D[j, i])
    F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            q = [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)]
            v = [idx[a, b] for a, b in q]
            have = [k for k in range(4) if v[k] >= 0]
            if len(have) < 3:
                continue
            ds = [D[q[k]] for k in have]
            if max(ds) - min(ds) > cut:
                continue
            if len(have) == 4:
                F += [[v[0], v[3], v[2]], [v[0], v[2], v[1]]]   # front faces the camera (image y runs down)
            else:
                t = [v[k] for k in have]
                F.append([t[0], t[2], t[1]])
    if len(F) < 2:
        return None
    used = sorted({k for f in F for k in f})
    remap = {k: n for n, k in enumerate(used)}
    Vf = np.array([V[k] for k in used]); F = [[remap[k] for k in f] for f in F]
    n = len(Vf)
    dirs = Vf - C; dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    Vb = Vf + dirs * thick                                       # the back: the surface pushed away from the camera
    Fb = [[a + n, c + n, b + n] for a, b, c in F]
    edges = {}
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edges[(a, b)] = edges.get((a, b), 0) + 1
    side = [[a, b, b + n, a + n] for (a, b) in edges if (b, a) not in edges]   # outline edges -> walls round the side
    Vall = np.vstack([Vf, Vb])
    return [[round(float(c), 4) for c in p] for p in Vall], F + Fb + side

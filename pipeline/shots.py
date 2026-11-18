# -*- coding: utf-8 -*-
"""hard-cut detection.  A cut is a frame whose picture has nothing in common with the frame before it: low
correlation of the (small, grey) pictures even after the best whole-picture shift (so whip pans and shakes are not
cuts), and much lower than around it.  Colour-histogram jumps are kept as a second vote, but a cut between two shots
of the same place / light (same colours, different picture) is found by the structure test alone.
Repeated frames (24 -> 30 fps pulldown) are skipped so the comparison is always with the last NEW frame."""
import numpy as np, cv2


def _small(f, y0, Hp):
    g = cv2.cvtColor(cv2.resize(f[y0:y0 + Hp], (192, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)


def _hist(f, y0, Hp):
    hsv = cv2.cvtColor(cv2.resize(f[y0:y0 + Hp], (160, 90)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-6 else 1.0


def _shifted_ncc(a, b):
    """correlation after the best whole-picture shift (phase correlation), on the overlapping part"""
    best = _ncc(a, b)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(a, b)
    except cv2.error:
        return best
    dx, dy = int(round(dx)), int(round(dy))
    H, W = a.shape
    if abs(dx) < W // 2 and abs(dy) < H // 2 and (dx or dy):
        A = a[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        B = b[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
        if A.size > 0.3 * a.size:
            best = max(best, _ncc(A, B))
    return best


def cut_scores(frames, y0, Hp):
    """per frame i (1..N-1): (structure change 1-ncc, colour change, mean abs diff) against the last different frame"""
    G = [_small(f, y0, Hp) for f in frames]
    Hh = [_hist(f, y0, Hp) for f in frames]
    N = len(frames)
    s = np.zeros(N); col = np.zeros(N); mad = np.zeros(N); dup = np.zeros(N, bool)
    for i in range(1, N):
        m = float(np.mean(np.abs(G[i] - G[i - 1])))
        if m < 0.35:                                   # repeated frame (a repeat of i-1 holds the same picture)
            dup[i] = True; continue
        p = i - 1
        s[i] = 1.0 - _shifted_ncc(G[p], G[i])
        col[i] = cv2.compareHist(Hh[p], Hh[i], cv2.HISTCMP_BHATTACHARYYA)
        mad[i] = float(np.mean(np.abs(G[i] - G[p])))
    return s, col, mad, dup


def split_shots(frames, y0, Hp, min_len=6, return_info=False):
    """hard cuts -> [(start, end)]"""
    N = len(frames)
    if N < 2:
        return [(0, N)]
    s, col, mad, dup = cut_scores(frames, y0, Hp)
    live = ~dup; live[0] = False
    base_s = float(np.median(s[live])) if live.any() else 0.0
    base_c = float(np.median(col[live])) if live.any() else 0.0
    base_m = float(np.median(mad[live])) if live.any() else 0.0
    cand = []
    for i in range(1, N):
        if dup[i]:
            continue
        nb = [j for j in range(max(1, i - 6), min(N, i + 7)) if j != i and not dup[j]]
        loc_s = float(np.median(s[nb])) if nb else base_s
        loc_m = float(np.median(mad[nb])) if nb else base_m
        struct = s[i] > 0.45 and s[i] > 3.0 * max(loc_s, 0.02) + 0.15 and mad[i] > max(8.0, 2.5 * loc_m)
        colour = col[i] > max(0.35, 4 * base_c) and mad[i] > max(12.0, 3 * base_m)      # the old test
        if (struct or colour) and s[i] >= max(s[max(1, i - 1)], s[min(N - 1, i + 1)]) - 1e-9:
            cand.append(i)
    bounds = [0]
    for ct in cand:
        if ct - bounds[-1] >= min_len and N - ct >= min_len:
            bounds.append(ct)
    bounds.append(N)
    shots = list(zip(bounds[:-1], bounds[1:]))
    if return_info:
        return shots, {"struct": s.tolist(), "colour": col.tolist(), "mad": mad.tolist(), "candidates": cand}
    return shots

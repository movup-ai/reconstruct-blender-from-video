"""Clean SAM 3 person tracks before they become characters (Stage 1 runs with no human review, so nothing else
catches a ghost or a split actor):

  join_pieces  one actor that SAM 3 lost behind something and found again under a new id: the pieces are joined
               when they never overlap in time, the second starts near where the first ended (allowing for the
               time passed), is about the same size, and looks the same (clothes colour inside the mask).  An
               ambiguous match (two candidates almost as good) is never joined.
  drop_false   tracks that are not a person to animate:
                 fragment   - no reliable head / shoulders / hips (a hand or arm reaching in from the edge, a blob)
                 erratic    - weak joints that jump around from frame to frame (not a body)
                 still      - weak joints that never move while the track stays clear of the side and bottom
                              edges: a poster, painting, statue or mannequin (a real person cut off at the side or
                              bottom of the frame is kept)
                 duplicate  - a second id on a person who is already tracked (same joints in the same frames)

Thresholds were set on the clips in SOURCE (Friends, office, two_men, the task clips): every real person there is
kept, including a man sitting still (strong joints) and a seated person's legs at the frame edge.

tracks: {pid: {frame: (keypoints (17, 2) full-frame px, scores (17,), box [x0, y0, x1, y1] full-frame px)}}
"""
import numpy as np

CORE = (0, 5, 6, 11, 12)                 # nose, shoulders, hips: a body has these, a hand or a blob does not
MIN_CORE = 0.20                          # median core-joint confidence below this: a fragment
WEAK = 0.40                              # "weak" joints for the erratic / still tests
ERRATIC_P50 = 0.25                       # body-relative joint wobble (share of the box height) of a non-body
STILL_P50, STILL_P90 = 0.03, 0.08        # a picture or statue: joints never move relative to its box
STILL_SECONDS = 1.0
EDGE = 0.01                              # box within 1% of the picture edge counts as cut by the frame
JOIN_SECONDS = 20.0                      # longest gap two pieces of one actor may have between them
JOIN_LOOK = 0.60                         # Bhattacharyya distance of the clothes colours: more than this = other clothes
                                         # (one person measured 0.32 median / 0.78 worst start-to-end on the SOURCE
                                         # clips, different people 0.52 median / 0.27 least: colour only vetoes)
WALK = 1.0                               # body heights a second a hidden actor may travel (about 1.7 m/s)
AMBIGUOUS = 0.85                         # best match must be clearly better than the second (score ratio)


def _boxes(tr):
    fs = sorted(tr)
    return fs, np.array([tr[f][2] for f in fs], float)


def core_confidence(tr):
    return float(np.median([np.mean(np.asarray(tr[f][1], float)[list(CORE)]) for f in tr]))


def wobble(tr):
    """(p50, p90) over well-seen joints of how much each joint moves relative to the box (share of box height);
    (nan, nan) when no joint is seen well in most frames"""
    fs, B = _boxes(tr)
    h = np.maximum(B[:, 3] - B[:, 1], 1.0)
    K = np.array([tr[f][0] for f in fs], float); S = np.array([tr[f][1] for f in fs], float)
    rel = np.stack([(K[..., 0] - B[:, [0]]) / h[:, None], (K[..., 1] - B[:, [1]]) / h[:, None]], -1)
    good = S > 0.3
    sd = [float(np.linalg.norm(rel[good[:, j], j].std(0))) for j in range(17)
          if good[:, j].mean() > 0.5 and good[:, j].sum() >= 5]
    if not sd:
        return float("nan"), float("nan")
    return float(np.median(sd)), float(np.percentile(sd, 90))


def inside_share(tr, W, y0, y1):
    """share of frames whose box stays clear of the left, right and bottom picture edges.  A real body cut off by
    the frame continues sideways or downwards out of it; the top edge is ignored because a picture on a wall is
    often cut there (the Friends S01E04 poster touches only the top; a seated person's legs touch the right and
    bottom)"""
    _, B = _boxes(tr)
    mx, my = EDGE * W, EDGE * (y1 - y0)
    clear = (B[:, 0] > mx) & (B[:, 2] < W - mx) & (B[:, 3] < y1 - my)
    return float(clear.mean())


def duplicate_of(a, b):
    """True when track a repeats person b: in most frames they share, a's well-seen joints sit on b's"""
    shared = sorted(set(a) & set(b))
    if len(shared) < max(4, 0.5 * len(a)):
        return False
    same = 0
    for f in shared:
        ka, sa, ba = a[f]; kb, sb, _ = b[f]
        both = (np.asarray(sa) > 0.3) & (np.asarray(sb) > 0.3)
        if both.sum() < 5:
            continue
        h = max(ba[3] - ba[1], 1.0)
        if np.median(np.linalg.norm(np.asarray(ka)[both] - np.asarray(kb)[both], axis=1)) < 0.05 * h:
            same += 1
    return same >= 0.7 * len(shared)


def drop_false(tracks, W, y0, y1, fps):
    """-> (kept tracks, {pid: reason}) for the false ones"""
    gone, stats = {}, {}
    for pid, tr in tracks.items():
        conf = core_confidence(tr); p50, p90 = wobble(tr)
        stats[pid] = (conf, p50, p90)
        if conf < MIN_CORE:
            gone[pid] = "fragment: no reliable head, shoulders or hips (core confidence %.2f)" % conf
        elif conf < WEAK and p50 > ERRATIC_P50:
            gone[pid] = "joints jump around (wobble %.2f, confidence %.2f): not a body" % (p50, conf)
        elif (conf < WEAK and p50 < STILL_P50 and p90 < STILL_P90 and len(tr) >= STILL_SECONDS * fps
              and inside_share(tr, W, y0, y1) >= 0.8):
            gone[pid] = ("never moves and is not cut off at the side or bottom (wobble %.3f, confidence %.2f): "
                         "a picture or statue" % (p50, conf))
    keep = {p: t for p, t in tracks.items() if p not in gone}
    for pid in sorted(keep, key=lambda p: (stats[p][0], len(keep[p]))):          # weakest first
        others = [q for q in keep if q != pid and q not in gone]
        twin = next((q for q in others if duplicate_of(keep[pid], keep[q])), None)
        if twin is not None:
            gone[pid] = "a second id on person %s (same joints in the same frames)" % twin
    return {p: t for p, t in tracks.items() if p not in gone}, gone


# ----------------------------------------------------------------------------------------------- joining pieces
def colour_signature(image, mask):
    """HSV colour histogram of the upper body inside the mask (clothes): None when too little is seen"""
    import cv2
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return None
    top, bottom = ys.min(), ys.max()
    upper = mask.copy(); upper[int(top + 0.7 * (bottom - top)) + 1:] = False          # head and torso, not the floor
    if upper.sum() < 150:
        upper = mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], upper.astype(np.uint8), [18, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, None, 1.0, 0.0, cv2.NORM_L1).astype(np.float32)


def look_distance(a, b):
    import cv2
    if a is None or b is None:
        return None
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def _end(tr, n=5, first=False):
    fs = sorted(tr); use = fs[:n] if first else fs[-n:]
    B = np.array([tr[f][2] for f in use], float)
    h = float(np.median(B[:, 3] - B[:, 1])); cx = (B[:, 0] + B[:, 2]) / 2; cy = (B[:, 1] + B[:, 3]) / 2
    v = np.zeros(2)
    if len(use) >= 2:
        v = np.array([cx[-1] - cx[0], cy[-1] - cy[0]]) / max(1, use[-1] - use[0])
        v *= min(1.0, 0.05 * h / max(np.linalg.norm(v), 1e-9))                # at most 5% of a body height per frame
    return use[0] if first else use[-1], np.array([cx[-1 if not first else 0], cy[-1 if not first else 0]]), h, v


def _not_alone(tracks, a, b, fa, fb):
    """another (real) person in the shot around the hand-over.  Clothes colour cannot tell people apart reliably
    (measured on the SOURCE clips: 18 of 70 simulated hand-overs between different people were joined on colour),
    so pieces are joined only when nobody else could be the one who came back."""
    for q, tq in tracks.items():
        if q in (a, b) or core_confidence(tq) < MIN_CORE:
            continue
        if any(f in tq for f in range(fa - 6, fb + 7)):
            return True
    return False


def join_pieces(tracks, looks, shots, fps, min_gap=0):
    """join the pieces of one actor.  looks {pid: (signature at its start, signature at its end)}.
    SAM 3 usually keeps an actor through a hide itself (a man behind a sliding pillar for 3 s kept his id); this
    joins the pieces it still splits, at any gap of at least min_gap frames - but only when nobody else is in the
    shot around the hand-over.
    -> (joined tracks, {absorbed pid: kept pid}, [log lines])"""
    tracks = {p: dict(t) for p, t in tracks.items()}
    looks = dict(looks)
    shot_of = lambda f: next((i for i, (a, b) in enumerate(shots) if a <= f < b), -1)
    merged, notes = {}, []
    max_gap = int(round(JOIN_SECONDS * fps))
    while True:
        cands = []
        for a, ta in tracks.items():
            if core_confidence(ta) < MIN_CORE:
                continue
            fa, ca, ha, va = _end(ta)
            for b, tb in tracks.items():
                if b == a or core_confidence(tb) < MIN_CORE:
                    continue
                fb, cb, hb, _ = _end(tb, first=True)
                gap = fb - fa - 1
                if fb <= fa or gap < min_gap or gap > max_gap or shot_of(fa) != shot_of(fb) or min(tb) <= max(ta):
                    continue
                grow = 1.67 + 0.5 * (gap + 1) / fps                                 # walking to / from the camera while hidden
                if not 1.0 / grow <= hb / max(ha, 1.0) <= grow:
                    continue
                reach = ha * (0.5 + WALK * (gap + 1) / fps)                         # where a walk could have taken it
                d = float(np.linalg.norm(cb - (ca + va * (gap + 1))))
                look = look_distance(looks.get(a, (None, None))[1], looks.get(b, (None, None))[0])
                if d > reach or look is None or look > JOIN_LOOK:
                    continue
                if _not_alone(tracks, a, b, fa, fb):
                    notes.append("not joined %s + %s: someone else is in the shot then, the identity is not certain" % (a, b))
                    continue
                score = (1 - look / JOIN_LOOK) * (1 - 0.5 * d / reach)              # higher is better
                cands.append((score, a, b, gap, d / ha, look))
        if not cands:
            break
        cands.sort(reverse=True)
        done = False
        for score, a, b, gap, dist, look in cands:
            rivals = [s for s, a2, b2, *_ in cands if (a2 == a) != (b2 == b) and (a2 == a or b2 == b)]
            if rivals and max(rivals) > AMBIGUOUS * score:
                notes.append("not joined %s + %s: another piece matches almost as well" % (a, b))
                continue
            tracks[a].update(tracks.pop(b))
            looks[a] = (looks.get(a, (None, None))[0], looks.get(b, (None, None))[1]); looks.pop(b, None)
            merged[b] = a
            for k, v in list(merged.items()):
                if v == b:
                    merged[k] = a
            notes.append("joined %s + %s: one actor, gap %.1f s, moved %.1f body heights, look distance %.2f" % (
                a, b, gap / fps, dist, look))
            done = True
            break
        if not done:
            break
    return tracks, merged, notes


def looks_of(tracks, bundle, frames, y0, y1, n=4):
    """{pid: (signature over its first frames, signature over its last frames)} from the SAM 3 masks"""
    from sam3_tracking import mask_at
    index = {(int(f), int(p)): i for i, (f, p) in enumerate(zip(bundle["frames"], bundle["ids"]))}
    W, H = frames[0].shape[1], y1 - y0
    out = {}
    for pid, tr in tracks.items():
        fs = sorted(tr)
        sig = []
        for part in (fs[:n], fs[-n:]):
            acc = None; k = 0
            for f in part:
                i = index.get((f, int(pid)))
                if i is None:
                    continue
                s = colour_signature(frames[f][y0:y1], mask_at(bundle, i, W, H))
                if s is not None:
                    acc = s if acc is None else acc + s; k += 1
            sig.append(acc / k if k else None)
        out[pid] = tuple(sig)
    return out


def relabel(bundle, merged):
    """the SAM 3 mask bundle with joined pieces under one id (their frames never overlap, so no clash)"""
    if not merged:
        return bundle
    ids = np.array([merged.get(int(p), int(p)) for p in bundle["ids"]], np.int64)
    out = dict(bundle); out["ids"] = ids
    order = np.lexsort((ids, bundle["frames"]))
    for key in ("frames", "ids", "packed", "probabilities"):
        if key in out:
            out[key] = np.asarray(out[key])[order]
    pairs = list(zip(out["frames"].tolist(), out["ids"].tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("joined tracks overlap in time")
    return out


def clean(tracks, bundle, frames, y0, y1, shots, fps, min_frames=4):
    """join the pieces of one actor, then drop false people and too-short tracks -> (tracks, bundle, report dict)"""
    looks = looks_of(tracks, bundle, frames, y0, y1)
    tracks, merged, notes = join_pieces(tracks, looks, shots, fps)
    bundle = relabel(bundle, merged)
    short = {p: "too short (%d frames)" % len(t) for p, t in tracks.items() if len(t) < min_frames}
    tracks = {p: t for p, t in tracks.items() if p not in short}
    W = frames[0].shape[1]
    tracks, gone = drop_false(tracks, W, y0, y1, fps)
    gone.update(short)
    report = {"joined": {str(k): int(v) for k, v in merged.items()}, "join_notes": notes,
              "dropped": {str(k): v for k, v in sorted(gone.items())}}
    return tracks, bundle, report

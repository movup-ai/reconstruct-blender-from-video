# -*- coding: utf-8 -*-
"""The checks on each object Mask2Former found (ade_find.py) before it is placed (scene_objects.py).  Each
check uses what the run already has.

  size        a placed box outside the real-world size range of its type is left out (a 2 m chair, a 12 cm chair);
              objects.json "size": [[min width, max width], [min height, max height]] in metres sets it for a type.
  split       a box too wide for its type is split by where its 3D points stand on the floor plan (a row of chairs),
              or, for things that stand in rows with no gap (washers), into equal pieces along its width; kept
              only when every piece has a real size.
  people      a non-seat object that people stand inside for more than 15 % of the clip is not where it is drawn;
              people sitting on a seat (sofa, chair, bench, bed) are expected.
config.json "bg_checks": false turns all of them off."""
import json
import os

import cv2
import numpy as np

PEOPLE_MAX = 0.15                     # share of the clip's frames a person may stand inside a non-seat object
SEATS = ("sofa", "chair", "armchair", "office_chair", "stool", "ottoman", "bench", "bed")
# real-world (width of the visible side, height) in metres; a type's own range first, then its role's
TYPE_SIZES = {"ottoman": ((0.35, 1.5), (0.3, 0.6)), "stool": ((0.25, 0.6), (0.4, 1.0)),
              "coffee_table": ((0.4, 2.0), (0.25, 0.6)), "armchair": ((0.6, 1.3), (0.6, 1.3)),
              "office_chair": ((0.4, 0.9), (0.7, 1.4))}
ROLE_SIZES = {"chair": ((0.3, 1.0), (0.6, 1.3)), "sofa": ((0.6, 3.6), (0.5, 1.3)), "table": ((0.35, 3.5), (0.3, 1.15)),
              "desk": ((0.6, 2.5), (0.6, 1.1)), "door": ((0.5, 1.9), (1.7, 2.8)), "bed": ((0.8, 2.5), (0.3, 1.5)),
              "bench": ((0.6, 3.0), (0.3, 1.0)), "counter": ((0.5, 6.0), (0.7, 1.3)), "fridge": ((0.5, 1.2), (1.2, 2.2)),
              "stove": ((0.5, 1.2), (0.7, 1.2)), "cabinet": ((0.3, 3.5), (0.3, 2.6)), "bookshelf": ((0.4, 3.5), (0.8, 2.8)),
              "washer": ((0.5, 0.9), (0.75, 2.0)), "dishwasher": ((0.45, 0.7), (0.75, 0.95))}   # washer: or a stacked pair
SLACK_LOW, SLACK_HIGH = 0.8, 1.25     # measuring tolerance on the ranges
IN_ROWS = ("washer", "dishwasher", "stool")   # stand in rows of identical units, touching: cut into equal units


def enabled(c):
    return bool((c or {}).get("bg_checks", True))


def extras(c=None):
    """objects.json of the asset folder: {type: {...}} (size overrides)"""
    import assets_lib
    p = os.path.join(assets_lib.folder(c), "objects.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8-sig") as fh:
            return {assets_lib.type_of(k): v for k, v in (json.load(fh) or {}).items() if isinstance(v, dict)}
    except (OSError, ValueError):
        return {}


KIND = {"office_chair": "chair", "armchair": "chair", "coffee_table": "table", "kitchen_island": "counter",
        "bookshelf": "bookshelf", "wardrobe": "cabinet", "chest_of_drawers": "cabinet", "display_cabinet": "cabinet"}


def role(t):
    """the size class of a model type: its own name, or the class it is a kind of"""
    return KIND.get(t, t)


def size_range(t, extra=None):
    if extra and extra.get("size"):
        (w0, w1), (h0, h1) = extra["size"]
        return (float(w0), float(w1)), (float(h0), float(h1))
    return TYPE_SIZES.get(t) or ROLE_SIZES.get(role(t) or "")


def size_problem(t, box, extra=None, wide_only=False):
    """why a placed box cannot be a t (outside its real-world size range), or None"""
    r = size_range(t, extra)
    if not r:
        return None
    (w0, w1), (h0, h1) = r
    w, h = float(box["size"][0]), float(box["size"][2])
    name = t.replace("_", " ")
    if wide_only:
        return "too wide" if w > w1 * SLACK_HIGH else None
    if not (w0 * SLACK_LOW <= w <= w1 * SLACK_HIGH):
        return "%.2f m wide, a %s is %.1f-%.1f m" % (w, name, w0, w1)
    if not (h0 * SLACK_LOW <= h <= h1 * SLACK_HIGH):
        return "%.2f m tall, a %s is %.1f-%.1f m" % (h, name, h0, h1)
    return None


def split_points(P, cell=0.15, min_share=0.1):
    """an object's 3D points -> one point set per group standing apart on the floor plan (an empty cell between)"""
    if len(P) < 120:
        return [P]
    xy = P[:, :2]
    ij = np.floor((xy - xy.min(0)) / cell).astype(int)
    shape = tuple(ij.max(0) + 1)
    if shape[0] * shape[1] > 4e6:
        return [P]
    occ = np.zeros(shape, np.uint16)
    np.add.at(occ, (ij[:, 0], ij[:, 1]), 1)
    n, lab = cv2.connectedComponents((occ >= 2).astype(np.uint8), connectivity=8)
    which = lab[ij[:, 0], ij[:, 1]]
    parts = [P[which == k] for k in range(1, n)]
    parts = [q for q in parts if len(q) >= max(60, min_share * len(P))]
    return parts if len(parts) > 1 else [P]


def split_if_too_wide(t, P, fit, extra=None):
    """[P] or its pieces: split only when the whole is too wide for t and every piece then has a real size.
    fit(points) -> box or None"""
    whole = fit(P)
    if whole is None or not size_problem(t, whole, extra, wide_only=True):
        return [P]
    tries = [split_points(P)]
    if t in IN_ROWS:                     # identical units standing touching: cut evenly (never e.g. a "chair" 3 m long)
        tries.append(split_even(P, whole, size_range(t, extra)))
    for pieces in tries:
        if len(pieces) < 2:
            continue
        boxes = [fit(q) for q in pieces]
        if all(b is not None and not size_problem(t, b, extra) for b in boxes):
            return pieces
    return [P]


def split_even(P, whole, rng):
    """a row with no gap (washers side by side): its points cut along its width into the fewest equal pieces that
    are no wider than the type's widest -> list of point sets"""
    if not rng or len(P) < 120:
        return [P]
    n = int(np.ceil(float(whole["size"][0]) / float(rng[0][1])))
    if n < 2 or n > 12:
        return [P]
    yaw = float(whole.get("yaw", 0.0))
    a = np.array([np.cos(yaw), np.sin(yaw)])                       # the box's width axis on the floor
    t = (P[:, :2] - np.asarray((whole.get("centre") or [0, 0])[:2], float)) @ a
    lo, hi = np.percentile(t, 1), np.percentile(t, 99)
    edges = np.linspace(lo, hi, n + 1); edges[0], edges[-1] = -np.inf, np.inf
    parts = [P[(t >= edges[i]) & (t < edges[i + 1])] for i in range(n)]
    return parts if all(len(q) >= 40 for q in parts) else [P]


def is_seat(t):
    return t in SEATS


def people_inside_problem(t, through, frames):
    """a non-seat object people stand inside in more than PEOPLE_MAX of the frames -> why it is left out, or None"""
    if is_seat(t) or through <= PEOPLE_MAX * max(1, frames):
        return None
    return "people stand inside it in %d of %d frames - it is not where they are" % (through, frames)

# -*- coding: utf-8 -*-
"""Stage 2's checks on each SAM 3 object before it is placed (background.py).  SAM 3 matches a name loosely: it can
cover a row of chairs with one mask or call the top of a cabinet a "door".  Each check uses what Stage 2 already has.

  agreement   Mask2Former's labels on the object's pixels (the depth frames' labels) must agree with its type:
              a "door" whose pixels Mask2Former calls cabinet is left out.  Types map to Mask2Former classes through
              semseg.ROLE; objects.json "labels": ["door", "screen door"] sets the classes for a type.  A type with no
              matching class (a bush, a cup) is not checked.
  size        a placed box outside the real-world size range of its type is left out (a 2 m chair, a 12 cm chair);
              objects.json "size": [[min width, max width], [min height, max height]] in metres sets it for a type.
  split       a box too wide for its type is split by where its 3D points stand on the floor plan (a row of chairs),
              kept only when every piece has a real size.
  people      a non-seat object that people stand inside for more than 15 % of the clip is not where it is drawn;
              people sitting on a seat (sofa, chair, bench, bed) are expected.
config.json "bg_checks": false turns all of them off."""
import json
import os

import cv2
import numpy as np

AGREE_MIN = 0.3                       # share of the object's pixels Mask2Former must call this type
PEOPLE_MAX = 0.15                     # share of the clip's frames a person may stand inside a non-seat object
SEATS = ("sofa", "chair", "office_chair", "bench", "bed")
NOT_CHECKED = ("relief", "object")
# library type names that are not Mask2Former class names
SYNONYMS = {"couch": "sofa", "settee": "sofa", "loveseat": "sofa", "floor_lamp": "lamp", "table_lamp": "lamp",
            "pine_tree": "tree", "doorway": "door", "washing_machine": "box", "refrigerator": "fridge",
            "television": "tv", "bush": "relief", "cup": "relief"}
# Mask2Former confuses these with each other: either counts
RELATED = ({"chair", "office_chair"}, {"table", "desk"}, {"cabinet", "bookshelf", "shelf", "counter"})
# real-world (width of the visible side, height) in metres; a type's own range first, then its role's
TYPE_SIZES = {"ottoman": ((0.35, 1.5), (0.3, 0.6)), "stool": ((0.25, 0.6), (0.4, 1.0)),
              "coffee_table": ((0.4, 2.0), (0.25, 0.6)), "armchair": ((0.6, 1.3), (0.6, 1.3)),
              "office_chair": ((0.4, 0.9), (0.7, 1.4))}
ROLE_SIZES = {"chair": ((0.3, 1.0), (0.6, 1.3)), "sofa": ((0.6, 3.6), (0.5, 1.3)), "table": ((0.35, 3.5), (0.3, 1.15)),
              "desk": ((0.6, 2.5), (0.6, 1.1)), "door": ((0.5, 1.9), (1.7, 2.8)), "bed": ((0.8, 2.5), (0.3, 1.5)),
              "bench": ((0.6, 3.0), (0.3, 1.0)), "counter": ((0.5, 6.0), (0.7, 1.3)), "fridge": ((0.5, 1.2), (1.2, 2.2)),
              "stove": ((0.5, 1.2), (0.7, 1.2)), "cabinet": ((0.3, 3.5), (0.3, 2.6)), "bookshelf": ((0.4, 3.5), (0.8, 2.8))}
SLACK_LOW, SLACK_HIGH = 0.8, 1.25     # measuring tolerance on the ranges


def enabled(c):
    return bool((c or {}).get("bg_checks", True))


def extras(c=None):
    """objects.json of the asset folder: {type: {...}} (labels / size overrides)"""
    import assets_lib
    p = os.path.join(assets_lib.folder(c), "objects.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8-sig") as fh:
            return {assets_lib.type_of(k): v for k, v in (json.load(fh) or {}).items() if isinstance(v, dict)}
    except (OSError, ValueError):
        return {}


def role(t):
    """the Mask2Former role of a model type (semseg.ROLE), or None"""
    import semseg
    name = t.replace("_", " ")
    if name in semseg.ROLE:
        return semseg.ROLE[name]
    if t in SYNONYMS:
        return SYNONYMS[t]
    return t if t in set(semseg.ROLE.values()) else None


def accepted_ids(t, names, extra=None):
    """indices of the Mask2Former classes that count as type t, or None: not checked"""
    import semseg
    if extra and extra.get("labels"):
        want = set(extra["labels"])
        return [i for i, nm in enumerate(names) if nm in want] or None
    r = role(t)
    if not r or r in NOT_CHECKED:
        return None
    roles = set().union({r}, *[g for g in RELATED if r in g])
    return [i for i, nm in enumerate(names) if semseg.ROLE.get(nm) in roles] or None


def agreement(mask, labels, acc, names):
    """share of the object's pixels (people left out) that Mask2Former also calls this type, and its most common
    label there.  mask: the object (any size), labels: the class map of that frame"""
    h, w = labels.shape
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    lab = labels[m].astype(np.int64)
    person = [i for i, nm in enumerate(names) if nm == "person"]
    lab = lab[~np.isin(lab, person) & (lab >= 0)]
    if not len(lab):
        return 0.0, "nothing"
    return float(np.isin(lab, acc).mean()), str(names[int(np.bincount(lab).argmax())])


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
    pieces = split_points(P)
    if len(pieces) < 2:
        return [P]
    boxes = [fit(q) for q in pieces]
    if all(b is not None and not size_problem(t, b, extra) for b in boxes):
        return pieces
    return [P]


def is_seat(t):
    return role(t) in SEATS


def people_inside_problem(t, through, frames):
    """a non-seat object people stand inside in more than PEOPLE_MAX of the frames -> why it is left out, or None"""
    if is_seat(t) or through <= PEOPLE_MAX * max(1, frames):
        return None
    return "people stand inside it in %d of %d frames - it is not where they are" % (through, frames)

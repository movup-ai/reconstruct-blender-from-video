# -*- coding: utf-8 -*-
"""Stage 2's prebuilt object library: one .blend per object type in assets\\objects\\ (config "object_assets").

  assets\\objects\\pillar.blend        type "pillar": every object Mask2Former names "column" becomes this model
                                      (ade_objects.py: Mask2Former's 150 names -> the asset placed for each)
  assets\\objects\\objects.json        optional: {"coffee_table": {"role": "fg", "on_floor": true}}
                                      role = a colour-card role that always wins (bg, fg, near, far_i, lamp,
                                      special); on_floor = false for a thing that hangs (default: paintings,
                                      mirrors, windows, clocks, shelves... hang, the rest stands on the floor);
                                      "size" ([[w0, w1], [h0, h1]] m) is read by object_checks.py, not here

How a model should be made: it stands on the ground (its lowest point is the floor), its front faces -Y (Blender's
front view looks at it), any size (it is stretched to the size measured in the video).  Every visible mesh in the
file is used; materials don't matter (the colour card colours it).

Each file is read once with Blender (asset_export.py) and cached in assets\\objects\\_cache\\<type>.json, normalised:
centred on x / y, base at z = 0, size 1 x 1 x 1, and its original size kept ("size") for its proportions."""
import json
import os
import re
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAX_FACES = 200000
NOT_OBJECTS = {"floor", "ground"}                     # Stage 1 already builds the floor
# types that hang on a wall or from the ceiling: placed at the height they are seen.  Every other type stands on the
# floor (its bottom is often hidden behind people or furniture).  objects.json "on_floor": true / false wins.
HANGING = ("painting", "picture", "poster", "frame", "mirror", "clock", "window", "shelf", "sign", "screen", "tv",
           "television", "curtain", "chandelier", "ceiling", "wall_lamp", "sconce", "light", "board")


def on_floor(t, extra=None):
    if extra and isinstance(extra.get("on_floor"), bool):
        return extra["on_floor"]
    words = t.split("_")
    return not any(h in words or h == t for h in HANGING)


def folder(c=None):
    p = (c or {}).get("object_assets") or os.path.join(ROOT, "assets", "objects")
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def type_of(path):
    return re.sub(r"[^a-z0-9_]+", "_", os.path.splitext(os.path.basename(path))[0].strip().lower()).strip("_")


def catalog(c=None):
    """{type: {"blend": path, "role": role or None, "on_floor": bool}} for every .blend in the folder"""
    d = folder(c)
    if not os.path.isdir(d):
        return {}
    extra = {}
    jp = os.path.join(d, "objects.json")
    if os.path.isfile(jp):
        with open(jp, encoding="utf-8-sig") as fh:
            extra = {type_of(k): v for k, v in (json.load(fh) or {}).items() if isinstance(v, dict)}
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(".blend"):
            continue
        t = type_of(fn)
        if not t or t in NOT_OBJECTS:
            continue
        e = extra.get(t, {})
        out[t] = {"blend": os.path.join(d, fn), "role": e.get("role"), "on_floor": on_floor(t, e)}
    return out


def _stamp(path):
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def normalise(raw):
    """raw {"v", "f"} in the file's units -> {"v": unit model (centred x / y, base z = 0, 1 x 1 x 1), "f", "size"}"""
    V = np.asarray(raw["v"], float).reshape(-1, 3)
    F = [list(map(int, f)) for f in raw["f"] if len(f) >= 3]
    if not len(V) or not F:
        raise ValueError("no visible mesh in the file")
    lo, hi = V.min(0), V.max(0)
    size = np.maximum(hi - lo, 1e-6)
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    U = (V - centre) / size
    return {"v": np.round(U, 5).tolist(), "f": F, "size": np.round(size, 5).tolist()}


def load(t, c=None, blender=None):
    """the normalised model of type t (read with Blender once, then from the cache)"""
    cat = catalog(c)
    if t not in cat:
        raise KeyError("no asset for %r in %s" % (t, folder(c)))
    src = cat[t]["blend"]
    cache = os.path.join(folder(c), "_cache", t + ".json")
    stamp = _stamp(src)
    if os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as fh:
                got = json.load(fh)
            if got.get("stamp") == stamp:
                return got
        except (OSError, ValueError):
            pass
    if not blender:
        from common import blender_exe
        blender = blender_exe(c or {})
    if not blender:
        raise RuntimeError("Blender not found - set the Blender path in Settings (needed to read %s)" % src)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + ".raw.json"
    try:
        p = subprocess.run([blender, "-b", src, "--factory-startup", "-P", os.path.join(HERE, "asset_export.py"), "--", tmp],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        if not os.path.isfile(tmp):
            raise RuntimeError("Blender could not read %s: %s" % (src, (p.stderr or p.stdout)[-400:]))
        with open(tmp, encoding="utf-8") as fh:
            raw = json.load(fh)
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)
    if len(raw.get("f", [])) > MAX_FACES:
        raise RuntimeError("%s has %d faces; keep object models under %d" % (os.path.basename(src), len(raw["f"]), MAX_FACES))
    try:
        got = normalise(raw)
    except ValueError as e:
        raise RuntimeError("%s: %s" % (os.path.basename(src), e))
    got.update(stamp=stamp, type=t, source=os.path.basename(src), objects=raw.get("objects", []))
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(got, fh)
    return got


def place(model, centre, yaw, size):
    """the unit model stretched to size (w along its x, d along its y, h), turned by yaw (radians, about z) and
    stood with its base centre at centre (x, y, z) -> vertices (n, 3)"""
    U = np.asarray(model["v"], float) * np.asarray(size, float)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return U @ R.T + np.asarray(centre, float)

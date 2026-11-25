# -*- coding: utf-8 -*-
"""Library of simple, recognisable white-model objects (like the client samples: a door has a frame, a panel and a
handle; a car has a body, a cabin and wheels).  Each builder returns parts in the object's own space:
x = width (left -> right), y = depth (front at -d/2, back at +d/2), z = up from the floor (0).
part = (part name, verts, faces, shade)   shade: "main", "dark" (frames, wheels, handles) or "light" (tops, glass)."""
import math

import numpy as np

# type -> default size (width, height, depth) in metres and where it usually is
TYPES = {
    "palm": ((3.0, 6.0, 3.0), "floor"),
    "door": ((0.9, 2.05, 0.12), "wall"), "window": ((1.2, 1.3, 0.12), "wall"), "cabinet": ((0.8, 0.9, 0.6), "floor"),
    "wall_cabinet": ((0.8, 0.7, 0.35), "wall"), "shelf": ((1.2, 0.9, 0.3), "wall"), "bookshelf": ((0.9, 1.9, 0.35), "floor"),
    "counter": ((1.8, 0.9, 0.6), "floor"), "sink": ((1.2, 0.9, 0.6), "floor"), "table": ((1.2, 0.75, 0.8), "floor"),
    "desk": ((1.4, 0.75, 0.7), "floor"), "chair": ((0.45, 0.9, 0.5), "floor"), "stool": ((0.4, 0.65, 0.4), "floor"),
    "sofa": ((2.0, 0.85, 0.9), "floor"), "bed": ((1.6, 0.55, 2.0), "floor"), "fridge": ((0.7, 1.8, 0.7), "floor"),
    "stove": ((0.6, 0.9, 0.6), "floor"), "tv": ((1.1, 0.65, 0.08), "wall"), "lamp": ((0.4, 1.6, 0.4), "floor"),
    "lamp_post": ((0.3, 4.0, 0.3), "floor"), "car": ((4.4, 1.45, 1.8), "floor"), "van": ((5.2, 2.2, 2.0), "floor"),
    "truck": ((7.0, 3.0, 2.5), "floor"), "bus": ((11.0, 3.1, 2.5), "floor"), "bicycle": ((1.7, 1.0, 0.5), "floor"),
    "tree": ((4.0, 7.0, 4.0), "floor"), "pine": ((2.5, 8.0, 2.5), "floor"), "forest": ((20., 8., 8.), "floor"), "bush": ((1.5, 1.0, 1.2), "floor"), "plant_pot": ((0.5, 1.0, 0.5), "floor"),
    "fence": ((3.0, 1.2, 0.08), "floor"), "wall": ((3.0, 2.5, 0.25), "floor"), "railing": ((3.0, 1.0, 0.08), "floor"),
    "stairs": ((1.2, 1.5, 2.5), "floor"), "bench": ((1.6, 0.85, 0.5), "floor"), "bin": ((0.5, 0.9, 0.5), "floor"),
    "box": ((0.6, 0.5, 0.6), "floor"), "barrel": ((0.6, 0.9, 0.6), "floor"), "sign": ((0.8, 2.4, 0.08), "floor"),
    "umbrella": ((2.5, 2.4, 2.5), "floor"), "awning": ((3.0, 0.6, 1.2), "wall"), "pillar": ((0.5, 3.0, 0.5), "floor"),
    "kettle": ((0.2, 0.25, 0.2), "floor"), "bottle": ((0.08, 0.3, 0.08), "floor"), "bag": ((0.4, 0.35, 0.2), "floor"),
    "building": ((10.0, 9.0, 8.0), "floor"), "rock": ((1.0, 0.6, 0.8), "floor"), "walker": ((0.55, 0.85, 0.5), "floor"),
    "curtain": ((1.4, 2.2, 0.08), "wall"), "mirror": ((0.6, 0.9, 0.04), "wall"), "picture": ((0.6, 0.45, 0.04), "wall"),
    "office_chair": ((0.56, 1.05, 0.58), "floor"),
    "horse": ((2.4, 1.6, 0.6), "floor"), "hedge": ((3.0, 1.2, 0.9), "floor"), "bank": ((6.0, 1.0, 3.0), "floor"),
    "other": ((0.8, 0.8, 0.8), "floor"),
}
ALIASES = {"cupboard": "cabinet", "wardrobe": "bookshelf", "closet": "bookshelf", "kitchen_counter": "counter", "worktop": "counter",
           "streetlight": "lamp_post", "street_lamp": "lamp_post", "lamppost": "lamp_post", "pole": "lamp_post", "post": "lamp_post",
           "conifer": "pine", "fir": "pine", "spruce": "pine", "fir_tree": "pine", "pine_tree": "pine", "automobile": "car", "taxi": "car", "suv": "car", "lorry": "truck", "shrub": "bush", "hedgerow": "hedge", "grass_bank": "bank", "embankment": "bank", "mound": "bank", "berm": "bank", "pony": "horse", "mule": "horse", "donkey": "horse", "plant": "plant_pot", "flowerpot": "plant_pot",
           "column": "pillar", "trash_can": "bin", "dustbin": "bin", "crate": "box", "monitor": "tv", "television": "tv", "screen": "tv",
           "armchair": "sofa", "couch": "sofa", "oven": "stove", "cooker": "stove", "refrigerator": "fridge", "balustrade": "railing",
           "gate": "fence", "staircase": "stairs", "steps": "stairs", "parasol": "umbrella", "house": "building", "shed": "building",
           "cabin": "building", "hut": "building", "boulder": "rock", "basin": "sink", "washbasin": "sink", "board": "sign"}


def norm_type(t):
    t = str(t or "other").strip().lower().replace(" ", "_").replace("-", "_")
    t = ALIASES.get(t, t)
    return t if t in TYPES else "other"


# ----------------------------------------------------------------------------- primitives (outward faces)
def box(x0, x1, y0, y1, z0, z1):
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f


def cyl(cx, cy, r, z0, z1, seg=12, r_top=None):
    rt = r if r_top is None else r_top
    v = [(cx + r * math.cos(2 * math.pi * i / seg), cy + r * math.sin(2 * math.pi * i / seg), z0) for i in range(seg)]
    v += [(cx + rt * math.cos(2 * math.pi * i / seg), cy + rt * math.sin(2 * math.pi * i / seg), z1) for i in range(seg)]
    f = [(i, (i + 1) % seg, seg + (i + 1) % seg, seg + i) for i in range(seg)]
    f.append(tuple(range(seg - 1, -1, -1)))
    if rt > 1e-6:
        f.append(tuple(range(seg, 2 * seg)))
    return v, f



def lump(cx, cy, cz, rx, ry, rz, n=16, jit=0.18, rng=None, flat_bottom=None):
    """faceted low-poly blob (the convex hull of n jittered points on an ellipsoid): reads as a block-style bush,
    canopy or rock instead of a smooth grey ellipsoid.  flat_bottom: clamp points below that z (rocks sit flat)"""
    from scipy.spatial import ConvexHull
    rng = rng if rng is not None else np.random.default_rng(0)
    k = np.arange(n) + 0.5
    th = np.arccos(1 - 2 * k / n); ph = math.pi * (1 + 5 ** 0.5) * k + rng.uniform(0, 2 * math.pi)
    P = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    P *= (1 + rng.uniform(-jit, jit, (n, 1)))
    P = P * [rx, ry, rz] + [cx, cy, cz]
    if flat_bottom is not None:
        P[:, 2] = np.maximum(P[:, 2], flat_bottom)
    h = ConvexHull(P)
    c = P.mean(0); F = []
    for s in h.simplices:                                   # outward winding
        a, b, d = P[s]
        F.append(tuple(int(i) for i in (s if np.cross(b - a, d - a) @ (a - c) > 0 else s[::-1])))
    used = sorted({i for f in F for i in f}); ix = {o: n_ for n_, o in enumerate(used)}
    return [tuple(round(float(x), 4) for x in P[i]) for i in used], [tuple(ix[i] for i in f) for f in F]


def puff(cx, cy, cz, rx, ry, rz, rng=None, seg=11, ring=7):
    """rounded foliage clump (a many-sided ellipsoid, slightly squashed and tilted at random): clusters of these read
    as leaves / bushes / tree crowns, unlike the large flat facets of `lump`, which read as rock"""
    rng = rng if rng is not None else np.random.default_rng(0)
    V, F = icosphere(2)                     # finer sphere, drawn smooth-shaded (build_env FOLIAGE): reads as leaves, not stone
    V = V * (1 + rng.uniform(-0.015, 0.015, (len(V), 1)))
    a = rng.uniform(0, 2 * math.pi); ca, sa = math.cos(a), math.sin(a)
    x, y, z = V[:, 0] * rx, V[:, 1] * ry, V[:, 2] * rz
    # never below the floor (z = 0): the size jitter pushed a clump resting on it a few mm under
    return ([(round(cx + p * ca - q * sa, 4), round(cy + p * sa + q * ca, 4), round(max(cz + s, 0.0), 4)) for p, q, s in zip(x, y, z)], F)


_ICO = {}


def icosphere(sub=1):
    """unit icosphere (subdivided `sub` times): even triangles, no latitude bands -> verts (n, 3) array, faces"""
    if sub in _ICO:
        return _ICO[sub]
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    V = [tuple(np.array(p) / np.linalg.norm(p)) for p in V]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(sub):
        mid = {}

        def m(i, j):
            k = (min(i, j), max(i, j))
            if k not in mid:
                p = (np.array(V[i]) + np.array(V[j])) / 2
                V.append(tuple(p / np.linalg.norm(p))); mid[k] = len(V) - 1
            return mid[k]
        F2 = []
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    _ICO[sub] = (np.array(V), F)
    return _ICO[sub]


def slab(pts, z0, z1):
    """prism over a floor outline [(x, y), ...] (counter-clockwise) from z0 to z1"""
    k = len(pts)
    v = [(x, y, z0) for x, y in pts] + [(x, y, z1) for x, y in pts]
    f = [tuple(range(k - 1, -1, -1)), tuple(range(k, 2 * k))] + [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)]
    return v, f


def bar(p, q, wa, wb=None, taper=1.0):
    """square-section bar from point p to q: wa across (horizontal, perpendicular to p->q), wb the other way;
    taper scales the q end"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    wb = wa if wb is None else wb
    dv = q - p; dv /= max(np.linalg.norm(dv), 1e-9)
    a = np.cross(dv, [0, 0, 1.0]) if abs(dv[2]) < 0.95 else np.cross(dv, [1.0, 0, 0])
    a /= np.linalg.norm(a); b = np.cross(a, dv)
    v = []
    for base, s in ((p, 1.0), (q, taper)):
        for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            v.append(tuple(base + a * i * wa * s / 2 + b * j * wb * s / 2))
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    if np.cross(np.array(v[1]) - v[0], np.array(v[3]) - v[0]) @ (p - q) < 0:   # keep the faces pointing out
        f = [tuple(reversed(x)) for x in f]
    return v, f


HORSE_HIP = {"fl": (0.50, -0.15), "fr": (0.50, 0.15), "hl": (-0.60, -0.15), "hr": (-0.60, 0.15)}   # leg tops (x fwd, y) / scale


def horse_leg(sx, sz, front=True):
    """one horse leg hanging from its hip pivot at (0, 0, 0): upper leg, cannon, hoof (shade, parts)"""
    top = 0.97 * sz
    out = [("upper", bar((0, 0, 0), (0.03 * sx if front else -0.05 * sx, 0, -(top - 0.50 * sz)), 0.15 * sx, 0.20 * sx, 0.8), "main"),
           ("lower", bar((0.03 * sx if front else -0.05 * sx, 0, -(top - 0.52 * sz)), (0, 0, -(top - 0.07 * sz)), 0.10 * sx, 0.11 * sx), "main"),
           ("hoof", box(-0.07 * sx, 0.08 * sx, -0.065 * sx, 0.065 * sx, -top, -(top - 0.08 * sz)), "dark")]
    return out


def horse_parts(L=2.4, H=1.6, legs=True):
    """block horse, head towards +x, the saddle at x = 0, hooves on z = 0 (length L, height at the withers H)"""
    sx, sz = L / 2.4, H / 1.6
    X = lambda x: x * sx
    Z = lambda z: z * sz
    out = [("barrel", box(X(-0.62), X(0.45), X(-0.25), X(0.25), Z(0.82), Z(1.42)), "main"),
           ("rump", box(X(-0.88), X(-0.45), X(-0.26), X(0.26), Z(0.92), Z(1.47)), "main"),
           ("chest", box(X(0.35), X(0.78), X(-0.24), X(0.24), Z(0.92), Z(1.50)), "main"),
           ("neck", bar((X(0.62), 0, Z(1.28)), (X(1.04), 0, Z(1.66)), X(0.20), X(0.34), 0.75), "main"),
           ("mane", bar((X(0.50), 0, Z(1.50)), (X(0.94), 0, Z(1.78)), X(0.08), X(0.08)), "dark"),
           ("head", bar((X(0.98), 0, Z(1.74)), (X(1.36), 0, Z(1.30)), X(0.20), X(0.26), 0.7), "main"),
           ("ear_l", box(X(0.95), X(1.01), X(-0.08), X(-0.03), Z(1.74), Z(1.86)), "dark"),
           ("ear_r", box(X(0.95), X(1.01), X(0.03), X(0.08), Z(1.74), Z(1.86)), "dark"),
           ("tail", bar((X(-0.88), 0, Z(1.40)), (X(-1.00), 0, Z(0.80)), X(0.09), X(0.13), 0.8), "dark"),
           ("saddle", box(X(-0.26), X(0.22), X(-0.27), X(0.27), Z(1.40), Z(1.48)), "dark")]
    if legs:
        for k, (hx, hy) in HORSE_HIP.items():
            for nm, (v, f), sh in horse_leg(sx, sz, k[0] == "f"):
                out.append(("leg_%s_%s" % (k, nm), ([(x + X(hx), y + X(hy), z + Z(0.97)) for x, y, z in v], f), sh))
    return out


def prism(pts, y0, y1):
    """a flat side profile [(x, z), ...] (any simple polygon) pushed along y from y0 to y1"""
    n = len(pts)
    v = [(x, y0, z) for x, z in pts] + [(x, y1, z) for x, z in pts]
    f = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return v, f


def wheel_x(cx, cy, r, w, seg=12):
    """a wheel standing up, axle along y"""
    v = []
    for y in (cy - w / 2, cy + w / 2):
        v += [(cx + r * math.cos(2 * math.pi * i / seg), y, r + r * math.sin(2 * math.pi * i / seg)) for i in range(seg)]
    f = [(i, (i + 1) % seg, seg + (i + 1) % seg, seg + i) for i in range(seg)]
    f += [tuple(range(seg)), tuple(range(2 * seg - 1, seg - 1, -1))]
    return v, f


# ----------------------------------------------------------------------------- builders
def build(kind, w, h, d, seed=0):
    """-> list of (part, verts, faces, shade).  seed varies the faceted shapes (bush, tree, rock, hedge) per object"""
    kind = norm_type(kind)
    W, D = w / 2.0, d / 2.0
    P = []
    rng = np.random.default_rng(abs(int(seed)) % (2 ** 31))

    def add(name, vf, shade="main"):
        P.append((name, vf[0], vf[1], shade))

    if kind == "door":
        t = min(0.08, w * 0.08)
        add("frame_l", box(-W, -W + t, -D, D, 0, h), "dark"); add("frame_r", box(W - t, W, -D, D, 0, h), "dark")
        add("frame_top", box(-W, W, -D, D, h - t, h), "dark")
        add("panel", box(-W + t, W - t, -D * 0.4, D * 0.4, 0, h - t))
        add("handle", box(W - t - 0.14, W - t - 0.06, -D - 0.04, -D * 0.4, h * 0.47, h * 0.5), "dark")
    elif kind in ("window", "mirror", "picture"):
        t = min(0.07, w * 0.08)
        add("frame", box(-W, W, -D, D, 0, t), "dark"); add("frame2", box(-W, W, -D, D, h - t, h), "dark")
        add("frame3", box(-W, -W + t, -D, D, 0, h), "dark"); add("frame4", box(W - t, W, -D, D, 0, h), "dark")
        add("pane", box(-W + t, W - t, -D * 0.2, D * 0.2, t, h - t), "light")
        if kind == "window":
            add("bar", box(-0.02, 0.02, -D, D, t, h - t), "dark")
    elif kind in ("cabinet", "wall_cabinet", "bookshelf", "fridge", "stove"):
        add("body", box(-W, W, -D, D, 0, h))
        doors = 1 if w < 0.65 else 2
        for i in range(doors):                                         # door seams + handles on the front
            x0 = -W + i * w / doors; x1 = x0 + w / doors
            add("seam%d" % i, box(x0 + 0.01, x0 + 0.02, -D - 0.01, -D, 0.03, h - 0.03), "dark")
            hx = x1 - 0.08 if i == 0 and doors == 2 else x0 + 0.06
            add("handle%d" % i, box(hx - 0.015, hx + 0.015, -D - 0.035, -D, h * 0.45, h * 0.6), "dark")
        if kind == "fridge":
            add("split", box(-W, W, -D - 0.01, -D, h * 0.62, h * 0.635), "dark")
        if kind == "stove":
            add("top", box(-W, W, -D, D, h - 0.03, h), "dark")
        if kind == "bookshelf":
            for i in range(1, 5):
                add("board%d" % i, box(-W + 0.02, W - 0.02, -D - 0.005, -D + 0.02, h * i / 5, h * i / 5 + 0.025), "light")
    elif kind == "shelf":
        n = max(2, min(4, int(round(h / 0.35))))
        for i in range(n):
            z = h * i / max(1, n - 1) if n > 1 else 0
            add("board%d" % i, box(-W, W, -D, D, z, z + 0.03), "light")
        add("bracket_l", box(-W + 0.05, -W + 0.08, D - 0.03, D, 0, h), "dark"); add("bracket_r", box(W - 0.08, W - 0.05, D - 0.03, D, 0, h), "dark")
    elif kind in ("counter", "sink"):
        add("body", box(-W, W, -D + 0.03, D, 0, h - 0.04))
        add("top", box(-W, W, -D, D, h - 0.04, h), "light")
        if kind == "sink":
            add("basin", box(-W * 0.45, W * 0.45, -D * 0.6, D * 0.5, h - 0.02, h + 0.005), "dark")
            add("tap", box(-0.02, 0.02, D * 0.6, D * 0.75, h, h + 0.25), "dark")
    elif kind in ("table", "desk", "stool"):
        tt = 0.04
        add("top", box(-W, W, -D, D, h - tt, h), "light")
        lw = 0.05
        for i, (x, y) in enumerate([(-W + lw, -D + lw), (W - lw, -D + lw), (W - lw, D - lw), (-W + lw, D - lw)]):
            add("leg%d" % i, box(x - lw / 2, x + lw / 2, y - lw / 2, y + lw / 2, 0, h - tt), "dark")
    elif kind == "chair":
        seat = 0.46
        add("seat", box(-W, W, -D, D, seat - 0.05, seat))
        add("back", box(-W, W, D - 0.05, D, seat, h))
        for i, (x, y) in enumerate([(-W + 0.03, -D + 0.03), (W - 0.03, -D + 0.03), (W - 0.03, D - 0.03), (-W + 0.03, D - 0.03)]):
            add("leg%d" % i, box(x - 0.02, x + 0.02, y - 0.02, y + 0.02, 0, seat - 0.05), "dark")
    elif kind == "office_chair":                                     # swivel chair: star base, post, seat, tall thin back, arm rests
        seat = min(0.50, h * 0.46)
        add("base", cyl(0, 0, min(W, D) * 0.95, 0.0, 0.05, 10), "dark")
        add("post", cyl(0, 0, 0.035, 0.05, seat - 0.07, 8), "dark")
        add("seat", box(-W * 0.9, W * 0.9, -D * 0.85, D * 0.6, seat - 0.06, seat))
        zb, lean, th = seat + 0.05, D * 0.35, 0.04                      # the back: one thin panel leaning back, narrower at the top
        wt = W * 0.78
        vb = [(-W * 0.92, D * 0.6, zb), (W * 0.92, D * 0.6, zb), (W * 0.92, D * 0.6 + th, zb), (-W * 0.92, D * 0.6 + th, zb),
              (-wt, D * 0.6 + lean, h), (wt, D * 0.6 + lean, h), (wt, D * 0.6 + lean + th, h), (-wt, D * 0.6 + lean + th, h)]
        add("back", (vb, [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]))
        for sx in (-1, 1):
            add("arm%d" % sx, box(sx * W - 0.02, sx * W + 0.02, -D * 0.5, D * 0.6, seat + 0.17, seat + 0.2), "dark")
            add("armpost%d" % sx, box(sx * W - 0.02, sx * W + 0.02, D * 0.3, D * 0.36, seat - 0.03, seat + 0.17), "dark")
    elif kind in ("sofa", "bench"):
        sh = 0.45
        add("base", box(-W, W, -D, D, 0.08 if kind == "bench" else 0, sh))
        add("back", box(-W, W, D - min(0.2, d * 0.3), D, sh, h))
        if kind == "sofa":
            add("arm_l", box(-W, -W + 0.18, -D, D, sh, sh + 0.2)); add("arm_r", box(W - 0.18, W, -D, D, sh, sh + 0.2))
        else:
            add("leg_l", box(-W + 0.1, -W + 0.15, -D, D, 0, 0.08), "dark"); add("leg_r", box(W - 0.15, W - 0.1, -D, D, 0, 0.08), "dark")
    elif kind == "bed":
        add("base", box(-W, W, -D, D, 0, h * 0.6)); add("mattress", box(-W + 0.03, W - 0.03, -D + 0.03, D - 0.05, h * 0.6, h), "light")
        add("head", box(-W, W, D - 0.06, D, 0, h + 0.5), "dark"); add("pillow", box(-W * 0.7, W * 0.7, D - 0.45, D - 0.1, h, h + 0.12), "light")
    elif kind == "tv":
        add("screen", box(-W, W, -D, D, 0, h), "dark")
    elif kind in ("lamp", "lamp_post"):
        r = max(0.04, min(w, d) * 0.18)
        add("pole", cyl(0, 0, r, 0, h * 0.88), "dark")
        add("head", cyl(0, 0, max(w, d) * (0.45 if kind == "lamp" else 0.5), h * 0.82, h, 12,
                        max(w, d) * (0.25 if kind == "lamp" else 0.2)), "light")
    elif kind in ("car", "van", "truck", "bus"):
        # the long side runs along x (width argument = length of the vehicle)
        L2, B2 = w / 2.0, d / 2.0
        rr = min(0.36, h * 0.24)
        body_h = h * (0.55 if kind == "car" else 0.8)
        if kind == "car":                # three-box sedan seen from the side: sloped hood and trunk, raked windscreen
            zb, zl, zs = h * 0.22, h * 0.58, h * 0.63                 # sill, lamp line, belt line
            add("body", prism([(-L2, zb), (L2, zb), (L2, zl), (L2 * 0.93, zl + h * 0.02), (L2 * 0.45, zs),
                               (-L2 * 0.62, zs), (-L2 * 0.93, zl + h * 0.03), (-L2, zl)], -B2, B2))
            add("cabin", prism([(L2 * 0.45, zs), (L2 * 0.10, h), (-L2 * 0.36, h), (-L2 * 0.62, zs)], -B2 * 0.86, B2 * 0.86), "light")
            add("bumper_f", box(L2 - 0.02, L2 + 0.06, -B2, B2, zb + h * 0.02, zb + h * 0.12), "dark")
            add("bumper_r", box(-L2 - 0.06, -L2 + 0.02, -B2, B2, zb + h * 0.02, zb + h * 0.12), "dark")
            rr = min(0.32, h * 0.21)
        elif kind in ("van", "bus", "truck"):
            add("body", box(-L2, L2, -B2, B2, rr * 0.8, body_h))
            add("roof", box(-L2, L2 * (0.7 if kind == "truck" else 1.0), -B2, B2, body_h, h), "main")
        for i, x in enumerate((-L2 * 0.65, L2 * 0.65)):
            for j, y in enumerate((-B2 + 0.12, B2 - 0.12)):
                add("wheel%d%d" % (i, j), wheel_x(x, y, rr, 0.22), "dark")
    elif kind == "bicycle":
        add("wheel_f", wheel_x(W - 0.35, 0, 0.34, 0.05), "dark"); add("wheel_b", wheel_x(-W + 0.35, 0, 0.34, 0.05), "dark")
        add("frame", box(-W + 0.35, W - 0.35, -0.02, 0.02, 0.45, 0.5), "dark"); add("seat", box(-0.25, -0.05, -0.08, 0.08, h - 0.05, h))
    elif kind in ("tree", "pine", "forest"):
        rows = min(3, max(1, math.ceil(d / max(h * .65, .01)))) if kind == "forest" else 1
        columns = min(14, max(2, math.ceil(w / max(h * .45, .01)))) if kind == "forest" else 1
        for row in range(rows):
            for column in range(columns):
                x = (column + .5 - columns / 2) * w / columns
                y = (row + .5 - rows / 2) * d / rows
                height = h if row == column == 0 else h * rng.uniform(.65, .95)
                prefix = f"tree{row}_{column}_" if kind == "forest" else ""
                parts = [("trunk", .1, 0., .23)]
                parts += [(f"tier{i}", radius, bottom, top) for i, (radius, bottom, top) in
                          enumerate(((1., .15, .53), (.82, .34, .72), (.62, .55, 1.)))]
                for part, radius, bottom, top in parts:
                    v, faces = cyl(0, 0, radius, height * bottom, height * top, 8)
                    if part != "trunk":
                        v = v[:8] + [(0., 0., height * top)]
                        faces = [(i, (i + 1) % 8, 8) for i in range(8)] + [faces[-2]]
                    add(prefix + part, ([(x + vx * W / columns, y + vy * D / rows, vz)
                                         for vx, vy, vz in v], faces))
    elif kind == "palm":
        add("trunk", cyl(0, 0, max(.03, w*.04), 0, h*.83, 8), "dark")
        for i in range(8):
            a = i*math.pi/4; u = np.array([math.cos(a)*W, math.sin(a)*D, 0.])
            v = np.array([-math.sin(a)*w*.08, math.cos(a)*d*.08, 0.])
            origin = np.array([0.,0.,h*.86])
            vertices = [origin, origin+u*.55+v+[0,0,h*.14], origin+u, origin+u*.55-v+[0,0,h*.14]]
            add("leaves%d" % i, ([p.tolist() for p in vertices], [[0,1,2,3]]))
    elif kind == "bush":                                             # a mound of rounded faceted clumps
        # one rounded body (the mass of leaves) with smaller rounded clumps along its top outline and front:
        # reads as a shrub / foliage, not as a pile of stones
        D = max(D, min(W, h * 0.5))                                  # a bush is never a thin sheet
        add("bushbody", puff(0, 0, h * 0.4, W * 0.78, D * 0.85, h * 0.4, rng=rng))
        c = min(1.1, max(0.22, min(w, h) * 0.3))                     # clump radius
        n_arc = max(3, int(round(math.pi * W * 0.9 / (c * 1.3))))
        for i in range(n_arc):                                       # clumps along the top outline, left -> right
            a = math.pi * (0.06 + 0.88 * (i + rng.uniform(0.3, 0.7)) / n_arc)
            r = c * rng.uniform(0.85, 1.15)
            cx = -math.cos(a) * (W - r * 0.85); cz = h * 0.4 + math.sin(a) * (h * 0.6 - r * 0.8)
            add("clump%d" % i, puff(cx, rng.uniform(-0.25, 0.25) * D, max(r * 0.7, cz), r, min(r, D * 0.9), r * 0.85, rng=rng))
        n_f = max(1, int(round(w / (3.0 * c))))
        for i in range(n_f):                                         # a few clumps bulging at the front
            cx = ((i + 0.5) / n_f - 0.5) * w * 0.7 + rng.uniform(-0.2, 0.2) * c
            r = c * rng.uniform(0.8, 1.05)
            add("front%d" % i, puff(cx, -D * 0.45, h * rng.uniform(0.3, 0.5), r, min(r, D * 0.7), r * 0.8, rng=rng))
    elif kind == "rock":                                             # irregular faceted block
        add("mass", lump(0, 0, h * 0.42, W * 1.05, D * 1.05, h * 0.62, n=11, jit=0.22, rng=rng, flat_bottom=0.0))
    elif kind in ("hedge", "bank"):                                  # long low faceted mass (hedge row / grass bank), along x
        nseg = max(2, int(round(w / (1.1 if kind == "hedge" else 3.0))))
        for i in range(nseg):
            x0 = -W + w * i / nseg; x1 = -W + w * (i + 1) / nseg; cx = (x0 + x1) / 2
            hh = h * (1 + rng.uniform(-0.12, 0.08))
            if kind == "hedge":
                add("hedgeseg%d" % i, puff(cx, rng.uniform(-0.05, 0.05) * d, hh * 0.45, (x1 - x0) * 0.8, D * 1.05, hh * 0.58, rng=rng))
            else:                                                    # bank: wide base, rounded faceted top
                add("seg%d" % i, lump(cx, 0, 0, (x1 - x0) * 0.8, D, hh, n=16, jit=0.1, rng=rng, flat_bottom=0.0))
    elif kind == "horse":                                            # block horse, length along x (like vehicles), head at +x
        for nm, vf, sh in horse_parts(w, h):
            add(nm, vf, sh)
    elif kind == "plant_pot":
        add("pot", cyl(0, 0, W * 0.6, 0, h * 0.35, 8, W * 0.75), "dark")
        add("leaves", puff(0, 0, h * 0.68, W, D, h * 0.32, rng=rng))
    elif kind in ("fence", "railing"):
        n = max(2, int(w / 1.2) + 1)
        for i in range(n):
            x = -W + i * w / (n - 1)
            add("post%d" % i, box(x - 0.04, x + 0.04, -0.04, 0.04, 0, h), "dark")
        add("rail_top", box(-W, W, -0.03, 0.03, h * 0.85, h * 0.92)); add("rail_low", box(-W, W, -0.03, 0.03, h * 0.35, h * 0.42))
    elif kind == "stairs":
        n = max(3, int(round(h / 0.17)))
        for i in range(n):
            add("step%d" % i, box(-W, W, -D + d * i / n, D, h * i / n, h * (i + 1) / n), "main" if i % 2 else "light")
    elif kind in ("bin", "barrel", "pillar", "kettle", "bottle"):
        add("body", cyl(0, 0, W, 0, h * (0.9 if kind == "bin" else 1.0)))
        if kind == "bin":
            add("lid", cyl(0, 0, W * 1.05, h * 0.9, h), "dark")
        if kind == "kettle":
            add("handle", box(-0.02, 0.02, D * 0.6, D * 1.4, h * 0.3, h * 0.9), "dark")
    elif kind == "sign":
        add("pole", cyl(0, 0, 0.04, 0, h * 0.75), "dark"); add("panel", box(-W, W, -D, D, h * 0.6, h), "light")
    elif kind == "umbrella":
        add("pole", cyl(0, 0, 0.03, 0, h), "dark"); add("canopy", cyl(0, 0, W, h * 0.8, h, 12, 0.05))
    elif kind == "awning":
        add("canopy", box(-W, W, -D, D, h * 0.6, h)); add("valance", box(-W, W, -D - 0.02, -D, 0, h * 0.6), "dark")
    elif kind == "curtain":
        add("cloth", box(-W, W, -D, D, 0, h)); add("rod", box(-W - 0.05, W + 0.05, -D, D, h, h + 0.04), "dark")
    elif kind == "walker":
        for i, (x, y) in enumerate([(-W, -D), (W, -D), (W, D), (-W, D)]):
            add("leg%d" % i, box(x - 0.02, x + 0.02, y - 0.02, y + 0.02, 0, h), "light")
        add("rail_f", box(-W, W, -D - 0.02, -D + 0.02, h * 0.55, h * 0.6), "light"); add("grip", box(-W, W, -D - 0.02, D + 0.02, h - 0.03, h), "light")
        add("rail_l", box(-W - 0.02, -W + 0.02, -D, D, h * 0.3, h * 0.35), "light"); add("rail_r", box(W - 0.02, W + 0.02, -D, D, h * 0.3, h * 0.35), "light")
    elif kind == "building":
        add("block", box(-W, W, -D, D, 0, h))
        add("roof", box(-W - 0.2, W + 0.2, -D - 0.2, D + 0.2, h, h + 0.3), "dark")
        cols_, rows_ = max(1, int(w / 2.5)), max(1, int((h - 1) / 3.0))
        for i in range(cols_):                                         # window grid on the front
            for j in range(rows_):
                x = -W + (i + 0.5) * w / cols_; z = 1.2 + j * (h - 1.5) / rows_
                add("win%d_%d" % (i, j), box(x - 0.45, x + 0.45, -D - 0.03, -D, z, z + 1.2), "dark")
    elif kind == "bag":
        add("body", box(-W, W, -D, D, 0, h * 0.8)); add("strap", box(-W * 0.6, W * 0.6, -0.01, 0.01, h * 0.8, h), "dark")
    else:                                                              # wall, box, other
        add("body", box(-W, W, -D, D, 0, h))
    return P


def place(parts, pos, yaw_deg, z0=0.0):
    """object space -> world (turn about z by yaw, then move to pos with the floor at z0)"""
    a = math.radians(yaw_deg); ca, sa = math.cos(a), math.sin(a)
    out = []
    for name, v, f, shade in parts:
        V = [[round(pos[0] + ca * x - sa * y, 4), round(pos[1] + sa * x + ca * y, 4), round(z0 + z, 4)] for x, y, z in v]
        out.append((name, V, [list(ff) for ff in f], shade))
    return out


def shade_colour(bgr, shade):
    b = np.asarray(bgr, float)
    if shade == "dark":
        return b * 0.55
    if shade == "light":
        return np.minimum(255, b * 0.6 + 255 * 0.4)
    return b

# -*- coding: utf-8 -*-
"""Hand-built set for one clip (the client-sample style: clean grey room / landscape made of simple recognisable
objects).  Reads  output\\<clip>\\env_claude.json, builds the meshes with props.py, puts them into the clip's
scene_data.json (people and camera stay as fitted) and writes an overlay picture to check it against the film.

  python pipeline\\build_env.py <clip name> [--frames 0,60,120] [--marks DIR]

env_claude.json:
  {"ground": {"size": 60, "colour": "ground", "z": 0},           optional floor square around the camera
   "sky": [r, g, b],                                               optional background colour
   "objects": [
     {"type": "box", "pos": [x, y, z], "size": [w, h, d], "yaw": 0, "colour": "set"},     metres; z = underside
     {"type": "rock", "px": [u, v], "frame": 0, "size": [w, h, d]},                    stands on the ground under pixel u,v
     {"type": "picture", "px": [u, v], "dist": 4.2, "size": [...]},                     at that pixel, 4.2 m away (floor distance)
   ]}
World: z up, floor z = 0, camera above the origin looking along +y at frame 0; yaw 0 = the object's front faces -y
(towards the camera), size = [width, height, depth].  colour: set | light | dark | far | ground | white | ... (palette)."""
import argparse, json, math, os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload, jsave, imwrite_u, read_frames_at, output_dir
import props

SHADE = {"main": "set", "dark": "dark", "light": "light"}


def cam_mats(c):
    x, y, z, ex, ey, ez = c
    a, b, g = [math.radians(v) for v in (ex, ey, ez)]
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx, np.array([x, y, z], float)


class Cam:
    def __init__(self, data):
        self.d = data; self.W, self.H = data["res"]; self.f = data["lens"] / data["sensor"] * self.W
    def project(self, P, fr):
        R, C = cam_mats(self.d["cam"][fr])
        Xc = (np.atleast_2d(P) - C) @ R
        z = -Xc[:, 2]
        zz = np.maximum(z, 1e-6)
        return np.c_[self.W / 2 + self.f * Xc[:, 0] / zz, self.H / 2 - self.f * Xc[:, 1] / zz], z
    def ray(self, u, v, fr):
        R, C = cam_mats(self.d["cam"][fr])
        d = R @ np.array([(u - self.W / 2) / self.f, -(v - self.H / 2) / self.f, -1.0])
        return C, d / np.linalg.norm(d)


class Ground:
    """height of the shot's ground at (x, y) (world metres, after the shot's dx shift).
    ground: {"z": 0, "slope_deg": 6, "slope_dir_deg": 0, "origin": [x, y],
             "profile": [[d, z], ...], "heights": [[x, y, dz(, falloff)], ...], "falloff": 8}
      slope_deg / slope_dir_deg: a plane rising slope_deg degrees towards slope_dir_deg (0 = away from the camera (+y),
      90 = to the right (+x), 180 = towards the camera, 270 = left); it passes through z at "origin" (default: where the
      shot's people are, else the point under the camera).
      profile: the ground height z at distance d metres from the point under the camera along slope_dir_deg, straight
      lines in between (flat until a hill foot, then rising: [[0, 0], [9, 0], [14, 1.2], [30, 10]]); added to the plane.
      heights: bumps / hollows: the ground at (x, y) is raised by dz there, blending smoothly to the plane over
      "falloff" metres (default 8).  A heightfield like this makes rolling ground, mounds, a ditch, a hill edge."""
    def __init__(self, g, dx=0.0, origin=None):
        g = g or {}
        self.z0 = float(g.get("z", 0.0)); self.dx = dx
        s = math.tan(math.radians(float(g.get("slope_deg", 0.0)))); a = math.radians(float(g.get("slope_dir_deg", 0.0)))
        self.gx, self.gy = s * math.sin(a), s * math.cos(a)            # dz/dx, dz/dy
        o = g.get("origin") or origin or [0.0, 0.0]
        self.ox, self.oy = float(o[0]) + (dx if g.get("origin") else 0.0), float(o[1])
        self.fall = float(g.get("falloff", 8.0))
        self.H = np.array([[float(h[0]) + dx, float(h[1]), float(h[2]), float(h[3]) if len(h) > 3 else self.fall]
                           for h in (g.get("heights") or [])], float).reshape(-1, 4)
        self.ux, self.uy = math.sin(a), math.cos(a)
        self.prof = sorted([float(q[0]), float(q[1])] for q in (g.get("profile") or []))
        self.active = bool(abs(s) > 1e-6 or len(self.H) or self.prof)
        self.steep_deg = float(g.get("steep_deg", 14.0)); self.slope_colour = g.get("slope_colour", "set")
    def __call__(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        z = self.z0 + self.gx * (x - self.ox) + self.gy * (y - self.oy)
        if self.prof:                                         # height profile along slope_dir_deg
            P = np.array(self.prof)
            z = z + np.interp((x - self.dx) * self.ux + y * self.uy, P[:, 0], P[:, 1])
        if len(self.H):                                       # smooth bumps (Gaussian), each reaching its dz at its centre
            d2 = (x[..., None] - self.H[:, 0]) ** 2 + (y[..., None] - self.H[:, 1]) ** 2
            z = z + (self.H[:, 2] * np.exp(-d2 / (2 * (self.H[:, 3] / 2.0) ** 2))).sum(-1)
        return z
    def at(self, x, y):
        return float(self(x, y))
    def hit(self, C, d, z_off=0.0):
        """where the ray C + t d meets the ground (+ z_off); None above the horizon"""
        t = None
        for _ in range(12):                                   # fixed point on the local height
            zz = (self.at(C[0], C[1]) if t is None else self.at(*(C + d * t)[:2])) + z_off
            if d[2] >= -1e-6:
                return None
            tn = (zz - C[2]) / d[2]
            if tn < 0:
                return None
            if t is not None and abs(tn - t) < 1e-3:
                t = tn; break
            t = tn
        return C + d * t
    def mesh(self, size, colour, tag):
        """the ground: one quad when flat, else a grid (finer near the camera) with flat facets"""
        G = size / 2
        if not self.active:
            return [{"name": "ground" + tag, "v": [[self.dx - G, -G, self.z0], [self.dx + G, -G, self.z0], [self.dx + G, G, self.z0], [self.dx - G, G, self.z0]],
                    "f": [[0, 1, 2, 3]], "colour": colour, "layer": 1}]
        t = np.linspace(-1, 1, 61); t = np.sign(t) * np.abs(t) ** 1.8 * G      # denser in the middle
        X, Y = np.meshgrid(t + self.dx, t)
        Z = self(X, Y)
        n = len(t)
        V = [[round(float(X[j, i]), 4), round(float(Y[j, i]), 4), round(float(Z[j, i]), 4)] for j in range(n) for i in range(n)]
        F = [[j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i] for j in range(n - 1) for i in range(n - 1)]
        # steep facets (hillsides, bank faces) get their own grey so the slope reads against the pale floor
        Va = np.asarray(V); flat, steep = [], []
        for f in F:
            nrm = np.cross(Va[f[1]] - Va[f[0]], Va[f[3]] - Va[f[0]]); ln = np.linalg.norm(nrm)
            (steep if ln > 1e-9 and abs(nrm[2]) / ln < math.cos(math.radians(self.steep_deg)) else flat).append(f)
        out = [{"name": "ground" + tag, "v": V, "f": flat, "colour": colour, "layer": 1}]
        if steep:
            out.append({"name": "ground_slope" + tag, "v": V, "f": steep, "colour": self.slope_colour, "layer": 1})
        return out


def where(o, cam, zg, frame0=0, dx=0.0, gz=None):
    """object's base point in the world (pos is in the shot's own metres: dx moves it to the shot's place).
    gz (Ground, sloped / uneven): "pos" z and "z" are heights above the ground under the object"""
    if "pos" in o:
        p = list(o["pos"]) + [zg] * (3 - len(o["pos"]))
        p = np.array(p[:3], float)
        if not o.get("_world"):
            p[0] += dx
            if gz is not None and gz.active and not o.get("abs_z"):
                p[2] = gz.at(p[0], p[1]) + (p[2] - zg)
        return p
    u, v = o["px"]; fr = int(o.get("frame", frame0))
    C, d = cam.ray(u, v, fr)
    if gz is not None and gz.active and "dist" not in o:     # on the sloped / uneven ground under the pixel
        q = gz.hit(C, d, float(o.get("z", zg)) - zg)
        if q is not None:
            return q
    if "dist" in o:                                          # that far away along the floor
        t = float(o["dist"]) / max(math.hypot(d[0], d[1]), 1e-6)
        p = C + d * t
        if "z" in o: p[2] = float(o["z"])
        return p
    z = float(o.get("z", zg))
    if d[2] >= -1e-5:                                       # above the horizon with this camera: put it far away instead
        print("warning: pixel %s of %s is above the horizon - placed 300 m away (give it a 'dist')" % (o["px"], o.get("type") or o.get("name")))
        t = 300.0 / max(math.hypot(d[0], d[1]), 1e-6)
        p = C + d * t; p[2] = float(o.get("z", zg))
        return p
    return C + d * (z - C[2]) / d[2]


def from_bbox(o, cam, zg, frame0=0, dx=0.0):
    """an object drawn as a box in the picture: {"bbox": [x0, y0, x1, y1], "frame": 0} plus one of
      "depth": metres forward (a flat front on the plane y = depth: walls, windows, shelves against a wall)
      "z":     height of the surface it stands on (things on a desk or the floor; turned to the camera)
      "dist":  metres from the camera along the floor (turned to the camera)
    -> the same object with pos / size / yaw in metres"""
    x0, y0, x1, y1 = [float(v) for v in o["bbox"]]; fr = int(o.get("frame", frame0))
    o = dict(o, _world=True); thick = o.get("thick")
    if "depth" in o:
        def hit(u, v):
            C, d = cam.ray(u, v, fr); return C + d * (float(o["depth"]) - C[1]) / d[1]
        a, b, t = hit(x0, y1), hit(x1, y1), hit((x0 + x1) / 2, y0)
        th = float(thick if thick is not None else 0.1)
        o["pos"] = [(a[0] + b[0]) / 2, float(o["depth"]) + th / 2, min(a[2], b[2])]
        o["size"] = [abs(b[0] - a[0]), max(0.02, t[2] - min(a[2], b[2])), th]
        o.setdefault("yaw", 0.0)
        return o
    if "plane" in o:                                         # on an upright wall through two floor points [xa, ya, xb, yb]
        xa, ya, xb, yb = [float(v) for v in o["plane"]]; xa += dx; xb += dx
        A = np.array([xa, ya, 0.0]); u = np.array([xb - xa, yb - ya, 0.0]); u /= np.linalg.norm(u)
        n = np.array([-u[1], u[0], 0.0])
        def hitp(px, py):
            C, d = cam.ray(px, py, fr); return C + d * ((A - C) @ n) / (d @ n)
        a, b, t = hitp(x0, y1), hitp(x1, y1), hitp((x0 + x1) / 2, y0)
        C, _ = cam.ray(0, 0, fr)
        th = float(thick if thick is not None else 0.1)
        mid = (a + b) / 2
        away = n if (mid - C) @ n > 0 else -n                 # the wall's far side
        yaw = math.degrees(math.atan2(u[1], u[0]))
        if (math.sin(math.radians(yaw)) * -away[0] + -math.cos(math.radians(yaw)) * -away[1]) < 0:
            yaw += 180.0                                     # front (-y of the object) towards the camera
        zb = float(o["z0"]) if "z0" in o else min(a[2], b[2])
        o["pos"] = [float(mid[0] + away[0] * th / 2), float(mid[1] + away[1] * th / 2), zb]
        o["size"] = [float(np.linalg.norm((b - a)[:2])), max(0.02, float(t[2] - zb)), th]
        o["yaw"] = yaw
        return o
    if o.get("anchor") == "top":                             # the box's top edge is what the picture shows; it stands on the floor
        C, d = cam.ray((x0 + x1) / 2, y0, fr)
        p = C + d * float(o["dist"]) / max(math.hypot(d[0], d[1]), 1e-6)
        rng = float(np.linalg.norm(p - C))
        w = (x1 - x0) * rng / cam.f
        o["pos"] = [float(p[0]), float(p[1]), zg]
        o["size"] = [w, max(0.05, float(p[2] - zg)), float(thick if thick is not None else w)]
        if "yaw" not in o:
            o["face"] = "camera"
        return o
    if "z" in o:
        C, d = cam.ray((x0 + x1) / 2, y1, fr)
        if d[2] >= -1e-5 or (float(o["z"]) - C[2]) / d[2] > 300:   # bottom above the horizon with this camera
            print("warning: bbox %s of %s cannot stand on z=%s with this camera - placed 20 m away" % (o["bbox"], o.get("name"), o["z"]))
            p = C + d * 20.0 / max(math.hypot(d[0], d[1]), 1e-6)
        else:
            p = C + d * (float(o["z"]) - C[2]) / d[2]
    else:
        C, d = cam.ray((x0 + x1) / 2, y1, fr)
        p = C + d * float(o["dist"]) / max(math.hypot(d[0], d[1]), 1e-6)
    rng = float(np.linalg.norm(p - C))
    w, h = (x1 - x0) * rng / cam.f, (y1 - y0) * rng / cam.f
    o["pos"] = [float(p[0]), float(p[1]), float(p[2])]
    o["size"] = [w, h, float(thick if thick is not None else w)]
    if "yaw" not in o:
        o["face"] = "camera"
    return o


def build(env, cam, frame0=0, dx=0.0, tag="", origin=None):
    g = env.get("ground") or {}
    zg = float(g.get("z", 0.0))
    gz = Ground(g, dx, origin)
    meshes = []
    if g is not None and g.get("size", 60):
        meshes += gz.mesh(float(g.get("size", 60)), g.get("colour", "ground"), tag)
    for n, o in enumerate(env.get("objects", [])):
        if o.get("off"):
            continue
        if o.get("type") == "patch":                        # flat shape on the floor: {"px": [[u, v], ...]} or {"pts": [[x, y], ...]}
            z = float(o.get("z", zg + 0.02))
            if "pts3" in o:                                  # any flat polygon in metres [[x, y, z], ...] (env_auto: walls with
                V = [[float(q[0]) + dx, float(q[1]), float(q[2])] for q in o["pts3"]]   # sloped tops, sloped ground)
                F = [list(range(len(V)))]
                if o.get("thick"):                           # slab: the outline pushed back along its normal, away from the camera
                    P_ = np.array(V); nrm = np.zeros(3)
                    for i in range(len(P_)):                 # Newell normal
                        a_, b_ = P_[i], P_[(i + 1) % len(P_)]
                        nrm += [(a_[1] - b_[1]) * (a_[2] + b_[2]), (a_[2] - b_[2]) * (a_[0] + b_[0]), (a_[0] - b_[0]) * (a_[1] + b_[1])]
                    nrm /= max(np.linalg.norm(nrm), 1e-9)
                    Cs = cam.ray(cam.W / 2, cam.H / 2, frame0)[0]
                    if (P_.mean(0) - Cs) @ nrm < 0:
                        nrm = -nrm
                    k = len(V); V = V + [list(map(float, q + nrm * float(o["thick"]))) for q in P_]
                    F += [list(range(2 * k - 1, k - 1, -1))] + [[i, k + i, k + (i + 1) % k, (i + 1) % k] for i in range(k)]
                meshes.append({"name": "obj%s_%02d_%s" % (tag, n, o.get("name", "patch")), "v": V, "f": F,
                               "colour": o.get("colour", "set"), "layer": 2 if o.get("thick") else 1})
                continue
            if "pts" in o:
                V = [[float(q[0]) + dx, float(q[1]), z + (gz.at(float(q[0]) + dx, float(q[1])) - zg if gz.active else 0.0)] for q in o["pts"]]
            else:
                V = [[float(v) for v in where({"px": q, "frame": o.get("frame", frame0), "z": z}, cam, zg, frame0, dx, gz)] for q in o["px"]]
            F = [list(range(len(V)))]
            if o.get("thick"):                               # a slab (table top): the same outline lower down + sides
                k = len(V); V = V + [[q[0], q[1], q[2] - float(o["thick"])] for q in V]
                F += [list(range(2 * k - 1, k - 1, -1))] + [[i, k + i, k + (i + 1) % k, (i + 1) % k] for i in range(k)]
            meshes.append({"name": "obj%s_%02d_%s" % (tag, n, o.get("name", "patch")), "v": V, "f": F,
                           "colour": o.get("colour", "far"), "layer": 1})
            continue
        if "bbox" in o:
            o = from_bbox(o, cam, zg, frame0, dx)
        kind = props.norm_type(o.get("type"))
        (dw, dh, dd), _ = props.TYPES[kind]
        w, h, d = [float(v) for v in (o.get("size") or [dw, dh, dd])]
        p = where(o, cam, zg, frame0, dx, gz)
        yaw = float(o.get("yaw", 0.0))
        if o.get("face") == "camera":                        # turn the front to the camera
            Cs = cam.ray(cam.W / 2, cam.H / 2, frame0)[0]
            yaw = math.degrees(math.atan2(-(p[0] - Cs[0]), p[1] - Cs[1]))
        if gz.active and "bbox" not in o and not o.get("abs_z"):   # on a slope: sink to the lowest ground under the footprint
            a_ = math.radians(yaw); ca, sa = math.cos(a_), math.sin(a_)
            cx_ = [p[0] + ca * sx * w / 2 - sa * sy * d / 2 for sx in (-1, 1) for sy in (-1, 1)]
            cy_ = [p[1] + sa * sx * w / 2 + ca * sy * d / 2 for sx in (-1, 1) for sy in (-1, 1)]
            p = p.copy(); p[2] += min(0.0, float(gz(cx_, cy_).min()) - gz.at(p[0], p[1]))
        col = o.get("colour")
        seed = n * 7919 + sum(ord(ch_) for ch_ in str(o.get("name", ""))) + int(o.get("seed", 0))
        for pname, V, F, shade in props.place(props.build(kind, w, h, d, seed), (p[0], p[1]), yaw, p[2]):
            if col in ("bg", "fg", "near", "far_i", "lamp", "special"):   # LZHD meaning colours: the whole object
                c_ = o.get("colour_" + shade) or col
            else:
                c_ = (col if shade == "main" else (o.get("colour_" + shade) or SHADE[shade])) if col else SHADE[shade]
            m_ = {"name": "obj%s_%02d_%s_%s" % (tag, n, o.get("name", kind), pname), "v": V, "f": F, "colour": c_, "layer": 2}
            if pname.rstrip("0123456789") in FOLIAGE:            # leafy clumps (props.puff) are smooth-shaded
                m_["smooth"] = True
            meshes.append(m_)
    return meshes


FOLIAGE = {"crown", "clump", "bushbody", "front", "hedgeseg", "leaves"}


def top_faces(m):
    """the upward-facing faces of a set mesh (what someone could stand on): {"name", "faces": [(xy polygon, point,
    normal)], "area" (m2, seen from above), bounding box} or None"""
    V = np.asarray(m["v"], float); faces = []; ar = [0.0, 0.0]
    for f in m["f"]:
        P = V[list(f)]
        if len(P) < 3:
            continue
        n = np.zeros(3)
        for i in range(len(P)):                               # Newell normal (any polygon)
            a, b = P[i], P[(i + 1) % len(P)]
            n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
        ln = np.linalg.norm(n)
        if ln < 1e-9 or abs(n[2]) / ln < 0.7:                 # walls and steep slopes: not a top
            continue
        faces.append((P[:, :2], P.mean(0), n / ln))
        ar[int(n[2] > 0)] += abs(n[2]) / 2
    if not faces:
        return None
    xy = np.vstack([fc[0] for fc in faces])
    # bottom faces count too (normal sign depends on the winding): surface_z takes the highest face under the point;
    # the area is that of one side (top and bottom of a block point opposite ways)
    return {"name": m["name"], "faces": faces, "area": float(max(ar)),
            "x0": xy[:, 0].min(), "x1": xy[:, 0].max(), "y0": xy[:, 1].min(), "y1": xy[:, 1].max()}

def _in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside

def surface_z(sf, x, y):
    """height of the highest top face of surface sf right above / below (x, y), or None when none is there"""
    best = None
    for poly, p0, n in sf["faces"]:
        if _in_poly(x, y, poly):
            z = p0[2] - (n[0] * (x - p0[0]) + n[1] * (y - p0[1])) / n[2]
            best = z if best is None else max(best, z)
    return best

NEAR = 0.15                                                   # m in front of the camera: the preview's near plane

def clip_near(P, z, near=NEAR):
    """the part of polygon P (world points, depths z) in front of the near plane (Sutherland-Hodgman, one plane;
    depth is affine in world space, so the crossing point is found on the world edge)"""
    out = []
    n = len(P)
    for i in range(n):
        a, b, za, zb = P[i], P[(i + 1) % n], z[i], z[(i + 1) % n]
        if za >= near:
            out.append(a)
        if (za >= near) != (zb >= near):
            t = (near + 1e-3 - za) / (zb - za) if za < near else (near + 1e-3 - za) / (zb - za)
            out.append(a + (b - a) * float(np.clip(t, 0, 1)))
    return np.array(out, float)

def overlay(meshes, cam, frame_img, fr, palette, alpha=0.62, extra=None):
    """painter's picture of the set over the film frame (what the render will roughly look like).
    extra: more (depth, pixel polygon, BGR colour) to sort in with the set (the people's bodies)"""
    img = frame_img.copy(); lay = img.copy()
    L = np.array([0.35, -0.5, 0.8]); L /= np.linalg.norm(L)
    polys = list(extra or [])
    for m in meshes:
        V = np.array(m["v"], float)
        uv, z = cam.project(V, fr)
        col = np.array(palette.get(m["colour"], [170, 170, 170]), float)[::-1]
        for f in m["f"]:
            zf = np.array([z[i] for i in f], float)
            if zf.min() < NEAR and zf.max() < 0.5:
                continue
            P = V[list(f)]
            nrm = np.cross(P[1] - P[0], P[2] - P[0]); ln = np.linalg.norm(nrm)
            sh = 0.55 + 0.45 * abs(float(nrm @ L) / ln) if ln > 1e-9 else 1.0
            if zf.min() < NEAR:                              # reaches behind the camera (long walls, floors): clip the
                Pc = clip_near(P, zf, 0.5)                    # face at a near plane instead of dropping it (0.5 m:
                                                              # pixels stay in range for the drawing)
                if len(Pc) < 3:
                    continue
                uvc, zc = cam.project(Pc, fr)
                polys.append((float(np.mean(zc)), uvc.astype(np.int32), col * sh))
                continue
            polys.append((float(np.mean(zf)), uv[list(f)].astype(np.int32), col * sh))
    for _, pts, c_ in sorted(polys, key=lambda t: -t[0]):
        if np.abs(pts).max() > 20000:                         # (a clipped face can still reach far out: squeeze it in)
            if np.abs(pts).max() > 2e6:
                continue
            pts = np.clip(pts, -20000, 20000).astype(np.int32)
        cv2.fillPoly(lay, [pts], tuple(int(v) for v in c_), cv2.LINE_AA)
        cv2.polylines(lay, [pts], True, (60, 60, 60), 1, cv2.LINE_AA)
    return cv2.addWeighted(lay, alpha, img, 1 - alpha, 0)


# ---- the people in env_check.jpg: each visible mannequin projected like compare.rig_uv, so floating / buried feet
# and people inside furniture show up while the set is built ----
P_BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
P_COL = {"red": (60, 62, 235), "blue": (230, 120, 50), "green": (90, 190, 70), "yellow": (60, 210, 235), "purple": (190, 90, 150)}
P_NAMES = {5: "shoulder_L", 6: "shoulder_R", 7: "elbow_L", 8: "elbow_R", 9: "wrist_L", 10: "wrist_R", 11: "hip_L", 12: "hip_R",
           13: "knee_L", 14: "knee_R", 15: "ankle_L", 16: "ankle_R", 17: "head"}


def people_at(data, cam, fr, zg):
    """every person visible at frame fr: world points (COCO-17 on the rig, head centre, soles) and their pixels"""
    from rig import Rig, JOINTS
    out = []
    for kind, ch in (data.get("chars") or {}).items():
        try:
            if fr >= len(ch["root"]) or not ch["visible"][fr]:
                continue
            rig = Rig(ch["height"], ch.get("style", "sample"))
            r = ch["root"][fr]
            Wj = rig.fk((r[0], r[1], r[2]), r[3], {j: ch["rot"][j][fr] for j in JOINTS}, ch["pelvis_z"][fr])
            P = rig.coco_points(Wj)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        head = (P[3] + P[4]) / 2; hr = max(0.05, 0.8 * float(np.linalg.norm(P[3] - P[4])) / 2)
        soles = P[[15, 16]] - [0, 0, rig.sole]
        gfun = zg if callable(zg) else (lambda x, y, z_=zg: np.full(np.shape(x), float(z_)))   # Ground or a flat height
        floor = soles.copy(); floor[:, 2] = gfun(soles[:, 0], soles[:, 1])
        ring = np.array([[floor[k, 0] + 0.16 * math.cos(t), floor[k, 1] + 0.16 * math.sin(t), floor[k, 2]]
                         for k in (0, 1) for t in np.linspace(0, 2 * math.pi, 17)[:-1]])
        uv, z = cam.project(np.vstack([P, head[None], soles, floor, ring]), fr)
        if z[17] < 0.15:
            continue
        out.append({"name": kind, "col": P_COL.get(ch.get("colour"), (200, 200, 200)), "s": rig.s, "P": P, "head": head,
                    "uv": uv[:17], "z": z[:17], "head_uv": uv[17], "head_r": cam.f * hr / z[17], "head_z": z[17],
                    "sole_uv": uv[18:20], "sole_z": z[18:20], "floor_uv": uv[20:22], "ring_uv": uv[22:].reshape(2, 16, 2),
                    "ring_z": z[22:].reshape(2, 16), "gap": float((soles[:, 2] - floor[:, 2]).min())})
    return out


def body_polys(people, cam):
    """the bodies as flat limbs / torso / head for overlay()'s painter sort (so the set hides what is behind it)"""
    polys = []
    for p in people:
        uv, z = p["uv"], p["z"]
        col = np.array(p["col"], float) * 0.75 + 255 * 0.25
        for i, j in P_BONES:
            if min(z[i], z[j]) < 0.15:
                continue
            a, b = uv[i], uv[j]; d = b - a; n = np.linalg.norm(d)
            nn = np.array([-d[1], d[0]]) / n if n > 1e-6 else np.array([1.0, 0.0])
            wa, wb = cam.f * 0.06 * p["s"] / z[i], cam.f * 0.06 * p["s"] / z[j]
            pts = np.array([a + nn * wa, b + nn * wb, b - nn * wb, a - nn * wa])
            polys.append((float((z[i] + z[j]) / 2), pts.astype(np.int32), col))
        if min(z[[5, 6, 11, 12]]) >= 0.15:
            polys.append((float(np.mean(z[[5, 6, 11, 12]])), uv[[5, 6, 12, 11]].astype(np.int32), col))
        c = p["head_uv"]
        if np.abs(c).max() < 20000:
            polys.append((float(p["head_z"]), cv2.ellipse2Poly((int(c[0]), int(c[1])), (max(2, int(p["head_r"])),) * 2, 0, 0, 360, 15), col))
    return polys


def solid_parts(meshes):
    """convex hulls of the set's objects (shrunk 5 cm) to find body points inside furniture"""
    from scipy.spatial import Delaunay
    parts = []
    for m in meshes:
        if m.get("layer") != 2:
            continue
        V = np.asarray(m["v"], float)
        if len(V) < 4:
            continue
        c = V.mean(0); d = V - c; ln = np.linalg.norm(d, axis=1, keepdims=True)
        Vs = c + d * np.clip(1 - 0.05 / np.maximum(ln, 1e-6), 0, 1)
        try:
            tri = Delaunay(Vs)
        except Exception:                                    # flat part: nothing can be inside it
            continue
        nm = m["name"][3:] if m["name"].startswith("obj") else m["name"]
        parts.append((nm.lstrip("_"), Vs.min(0), Vs.max(0), tri))
    return parts


def inside_hits(p, parts):
    """[(point index, part name)] for the person's joints / head centre inside a set object"""
    Q = np.vstack([p["P"][5:17], p["head"][None]]); idx = list(range(5, 18)); hits = []
    for nm, lo, hi, tri in parts:
        box = np.all((Q >= lo) & (Q <= hi), axis=1)
        if not box.any():
            continue
        ins = np.zeros(len(Q), bool); ins[box] = tri.find_simplex(Q[box]) >= 0
        hits += [(idx[k], nm) for k in np.nonzero(ins)[0]]
    return hits


def draw_people(img, people, parts, legend=False):
    """stick figure + head circle on top of the picture, feet as dots, a white ring on the floor under each foot
    (gap = feet above (+) / below (-) the floor in metres), red X = body point inside a set object"""
    H, W = img.shape[:2]
    k_ = max(1.0, W / 640.0)                                 # line / text sizes that survive the sheet's downscale
    def th(v):
        return max(1, int(round(v * k_)))
    def ok(q):
        return np.isfinite(q).all() and np.abs(q).max() < 20000
    def pt(q):
        return int(round(q[0])), int(round(q[1]))
    def text(t, x, y, c_, sc):
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc * k_, (0, 0, 0), th(3), cv2.LINE_AA)
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc * k_, c_, th(1.2), cv2.LINE_AA)
    used = []                                                # label corners already taken (people close together)
    for p in people:
        uv, z, col = p["uv"], p["z"], p["col"]
        for i, j in P_BONES:
            if min(z[i], z[j]) >= 0.15 and ok(uv[i]) and ok(uv[j]):
                cv2.line(img, pt(uv[i]), pt(uv[j]), (20, 20, 20), th(3.5), cv2.LINE_AA)
                cv2.line(img, pt(uv[i]), pt(uv[j]), col, th(2), cv2.LINE_AA)
        if ok(p["head_uv"]):
            if min(z[5], z[6]) >= 0.15:
                cv2.line(img, pt((uv[5] + uv[6]) / 2), pt(p["head_uv"]), col, th(2), cv2.LINE_AA)
            r = max(3, int(p["head_r"]))
            cv2.circle(img, pt(p["head_uv"]), r, (20, 20, 20), th(3.5), cv2.LINE_AA)
            cv2.circle(img, pt(p["head_uv"]), r, col, th(2), cv2.LINE_AA)
        for k in (0, 1):
            if p["ring_z"][k].min() >= 0.15 and ok(p["ring_uv"][k]):
                ring = p["ring_uv"][k].astype(np.int32)
                cv2.polylines(img, [ring], True, (20, 20, 20), th(2.5), cv2.LINE_AA)
                cv2.polylines(img, [ring], True, (255, 255, 255), th(1.2), cv2.LINE_AA)
            if p["sole_z"][k] >= 0.15 and ok(p["sole_uv"][k]):
                if abs(p["gap"]) >= 0.03 and ok(p["floor_uv"][k]):
                    cv2.line(img, pt(p["sole_uv"][k]), pt(p["floor_uv"][k]), (255, 255, 255), th(1.2), cv2.LINE_AA)
                cv2.circle(img, pt(p["sole_uv"][k]), th(4.5), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, pt(p["sole_uv"][k]), th(3.5), col, -1, cv2.LINE_AA)
        hits = inside_hits(p, parts)
        for k, nm in hits:
            q = p["head_uv"] if k == 17 else uv[k]
            if ok(q):
                x, y = pt(q); d = th(6)
                cv2.line(img, (x - d, y - d), (x + d, y + d), (0, 0, 255), th(2.5), cv2.LINE_AA)
                cv2.line(img, (x - d, y + d), (x + d, y - d), (0, 0, 255), th(2.5), cv2.LINE_AA)
        txt = [p["name"]]
        if abs(p["gap"]) >= 0.05:
            txt.append("feet %+.2fm" % p["gap"])
        if hits:
            txt.append("in " + ",".join(sorted({nm for _, nm in hits}))[:40])
        if ok(p["head_uv"]):
            lh = th(22)
            x = int(min(max(p["head_uv"][0] - p["head_r"], 4), W - th(170)))
            y = int(min(max(p["head_uv"][1] - p["head_r"] - th(6), lh * len(txt) + th(30)), H - th(20)))
            x_h, y_h = x, y
            def free(x_, y_):
                return not any(abs(x_ - ux) < th(160) and abs(y_ - uy) < lh * 3 for ux, uy in used)
            if not free(x, y):                               # two heads close together: the label goes above / below
                for y_try in (y - lh * (len(txt) + 1), y + int(2 * p["head_r"]) + lh * (len(txt) + 1)):
                    if lh * len(txt) <= y_try <= H - th(20) and free(x, y_try):
                        y = y_try
                        break
            for _ in range(8):
                if free(x, y):
                    break
                if x + th(165) <= W - th(170):
                    x += th(165)
                else:
                    y = min(y + lh * 3, H - th(20))
            used.append((x, y))
            if (x, y) != (x_h, y_h):                         # a moved label points at its head (no label without a body)
                yl = y if y > p["head_uv"][1] else y + th(4)
                cv2.line(img, (x + th(10), yl - th(6)), pt(p["head_uv"]), col, th(1.2), cv2.LINE_AA)
            for n_, t in enumerate(txt):
                c_ = (0, 0, 255) if t.startswith("in ") else ((0, 255, 255) if t.startswith("feet") else col)
                text(t, x, y - lh * (len(txt) - 1 - n_), c_, 0.6)
    if legend:
        text("ring = floor under foot, feet +m above / -m below floor, X = inside set object", th(6), H - th(8), (255, 255, 255), 0.42)
    return img

# ---- mounts: hints.json person {"mount": "horse"} -> a block horse under the person for the whole shot, following the
# person's root position and heading every frame (legs swing with the ground speed).  The rider's own legs are posed by
# the fit (leg_pose "riding", see fit_people.py).
def _rxyz(rx, ry, rz):
    from rig import Rxyz
    return Rxyz((rx, ry, rz))


def load_hints(clip, video, c):
    from run_pipeline import marks_dir
    for p in (os.path.join(marks_dir(video, c), "hints.json"), os.path.join(output_dir(), clip, "hints.json")):
        if os.path.exists(p):
            return jload(p)
    return {}


def mount_props(kind, ch, hint, ground_at, fps=24.0):
    """the props (DATA['props'], per-frame keys loc / euler deg / scale) of the animal under one person.
    hint: {"mount": "horse", "mount_size": [length, withers height], "mount_yaw": deg offset, "mount_colour": "set",
           "mount_shift": metres forward (+) / back of the saddle under the pelvis, "mount_gait": false}"""
    from scipy.ndimage import gaussian_filter1d
    L, Hh = [float(v) for v in (hint.get("mount_size") or [2.4, 1.6])]
    sx, sz = L / 2.4, Hh / 1.6
    root = np.asarray(ch["root"], float); N = len(root)
    vis = np.asarray(ch.get("visible") or [True] * N, bool)
    if not vis.any():
        return []
    a0, a1 = int(np.argmax(vis)), int(N - 1 - np.argmax(vis[::-1]))
    yaw = np.degrees(np.unwrap(np.radians(root[:, 3])))
    xy = root[:, :2].copy()
    if a1 - a0 > 3:                                            # a horse turns and moves smoothly
        yaw[a0:a1 + 1] = gaussian_filter1d(yaw[a0:a1 + 1], 5.0, mode="nearest")
        xy[a0:a1 + 1] = gaussian_filter1d(xy[a0:a1 + 1], 2.0, axis=0, mode="nearest")
    yaw += float(hint.get("mount_yaw", 0.0))
    shift = float(hint.get("mount_shift", 0.0))
    colour = hint.get("mount_colour", "set")
    parts = props.horse_parts(L, Hh, legs=False)
    body = {"main": [], "dark": []}
    for _, vf, sh in parts:
        body["dark" if sh == "dark" else "main"].append(vf)
    legs = {}
    for k in props.HORSE_HIP:
        pl = props.horse_leg(sx, sz, k[0] == "f")
        legs[k] = {"main": [vf for _, vf, sh in pl if sh != "dark"], "dark": [vf for _, vf, sh in pl if sh == "dark"]}
    def merged(vfs):
        V, F = [], []
        for v, f in vfs:
            n0 = len(V); V += [[round(float(q), 4) for q in p] for p in v]; F += [[n0 + i for i in ff] for ff in f]
        return V, F
    # walking: phase from the distance travelled (one stride ~ 1.7 m at a walk), swing grows with speed
    step = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    step[~vis] = 0.0
    speed = gaussian_filter1d(step * fps, 3.0)
    stride = np.where(speed > 3.0, 2.6, 1.7) * sx
    phase = np.cumsum(step / stride)
    amp = np.clip(speed / 1.2, 0, 1) * 24.0 * (0.0 if hint.get("mount_gait") is False else 1.0)
    PH = {"hl": 0.0, "fl": 0.25, "hr": 0.5, "fr": 0.75}        # lateral four-beat walk
    keys_body, keys_leg = [], {k: [] for k in legs}
    for f in range(N):
        h = math.radians(yaw[f] + 90.0)                       # the horse model faces +x; the rider faces +y at yaw 0
        fx, fy = math.cos(h), math.sin(h)
        cx, cy = xy[f, 0] + fx * shift, xy[f, 1] + fy * shift
        zc = ground_at(f, cx, cy)
        zf, zh = ground_at(f, cx + fx * 0.5 * sx, cy + fy * 0.5 * sx), ground_at(f, cx - fx * 0.6 * sx, cy - fy * 0.6 * sx)
        pitch = math.degrees(math.atan2(zf - zh, 1.1 * sx))       # nose up on an uphill
        s = 1.0 if vis[f] else 0.0
        kb = [round(cx, 4), round(cy, 4), round(zc, 4), 0.0, round(-pitch, 2), round(yaw[f] + 90.0, 2), s, s, s]
        keys_body.append(kb)
        R = _rxyz(0.0, -pitch, yaw[f] + 90.0)
        for k, (hx, hy) in props.HORSE_HIP.items():
            P = np.array([cx, cy, zc]) + R @ np.array([hx * sx, hy * sx, 0.97 * sz])
            sw = amp[f] * math.sin(2 * math.pi * (phase[f] + PH[k]))
            keys_leg[k].append([round(float(P[0]), 4), round(float(P[1]), 4), round(float(P[2]), 4), 0.0, round(-pitch - sw, 2),
                                round(yaw[f] + 90.0, 2), s, s, s])
    out = []
    for sh, vfs in body.items():
        V, F = merged(vfs)
        out.append({"name": "%s_horse_body_%s" % (kind, sh), "v": V, "f": F, "colour": colour if sh == "main" else "dark", "keys": keys_body})
    for k, d_ in legs.items():
        for sh, vfs in d_.items():
            if vfs:
                V, F = merged(vfs)
                out.append({"name": "%s_horse_leg_%s_%s" % (kind, k, sh), "v": V, "f": F, "colour": colour if sh == "main" else "dark", "keys": keys_leg[k]})
    return out


def prop_at(p, f):
    """an animated prop's mesh in the world at frame f (as blender_build places it)"""
    k = p["keys"][f]
    if k[6] == 0:
        return None
    R = _rxyz(k[3], k[4], k[5])
    V = (np.asarray(p["v"], float) * k[6:9]) @ R.T + k[0:3]
    return {"name": p["name"], "v": V.tolist(), "f": p["f"], "colour": p["colour"], "layer": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--frames", default="")
    ap.add_argument("--marks", default=None)
    ap.add_argument("--alpha", type=float, default=0.62)
    a = ap.parse_args()
    c = cfg()
    work = os.path.join(c["work_root"], a.name)
    dp = os.path.join(work, "scene_data.json")
    data = jload(dp)
    ci = jload(os.path.join(work, "clip_info.json"))
    from run_pipeline import marks_dir
    marks = a.marks or os.path.join(output_dir(), a.name)
    if not os.path.exists(os.path.join(marks, "env_claude.json")):        # also accepted in the clip's marks folder
        marks = marks_dir(ci["video"], c)
    env = jload(os.path.join(marks, "env_claude.json"))
    cam = Cam(data)
    shots = (jload(os.path.join(work, "camera.json")) if os.path.exists(os.path.join(work, "camera.json")) else {}).get("shots") or [{"shot": 1, "frames": [0, data["frames"] - 1]}]
    def people_mid(s):                                      # where the shot's people are (a slope passes through z there)
        a0, a1 = s["frames"]; P = []
        for ch in data["chars"].values():
            P += [r[:2] for f, r in enumerate(ch["root"][a0:a1 + 1], a0) if (ch.get("visible") or [True] * (f + 1))[f]]
        return np.median(np.asarray(P, float), 0).tolist() if P else None
    grounds = {}                                            # shot -> Ground
    if env.get("shots"):                                    # one set per shot (clips with cuts): {"shots": {"1": {...}, "2": {...}}}
        meshes = []
        for s in shots:
            se = env["shots"].get(str(s["shot"]))
            dx_ = 200.0 * (s["shot"] - 1)
            if se:
                meshes += build(se, cam, s["frames"][0], dx_, "_s%d" % s["shot"], people_mid(s))
            grounds[s["shot"]] = Ground((se or {}).get("ground"), dx_, people_mid(s))
    else:
        meshes = build(env, cam, origin=people_mid(shots[0]) if len(shots) == 1 else None)
        for s in shots:
            grounds[s["shot"]] = Ground(env.get("ground"), 0.0, people_mid(shots[0]) if len(shots) == 1 else None)
    def shot_of(f):
        for s in shots:
            if s["frames"][0] <= f <= s["frames"][1]:
                return s["shot"]
        return shots[-1]["shot"]
    def ground_at(f, x, y):
        return grounds[shot_of(f)].at(x, y)
    data["meshes"] = meshes
    # feet on the hand-built floor: the fit may have lifted people onto the automatic depth set's terrain, which this
    # set replaces (flat floor at the shot's ground z, 0 unless the set says otherwise)
    # Each person stands on the highest hand-built surface under them that is not far above where the fit put them
    # (stepped hillsides, terraces, slabs), otherwise on the floor.
    surfs = [sf for sf in (top_faces(m) for m in meshes if len(m["v"]) and not m["name"].startswith("ground")) if sf]
    from scipy.ndimage import median_filter
    hints = load_hints(a.name, ci["video"], c)
    for s in shots:
        se = (env.get("shots") or {}).get(str(s["shot"]), {}) if env.get("shots") else env
        zg = float((se.get("ground") or {}).get("z", 0.0))
        follow = (se.get("ground") or {}).get("follow", (env.get("ground") or {}).get("follow", True)) is not False
        tag = "_s%d" % s["shot"] if env.get("shots") else None
        mine = [b for b in surfs if tag is None or tag in b["name"]]
        a0, a1 = s["frames"]
        for kind, ch in data["chars"].items():
            if "root_z_fit" not in ch:                       # the fit's own heights, kept so rebuilding never compounds
                ch["root_z_fit"] = [r[2] for r in ch["root"]]
            fr = list(range(a0, min(a1 + 1, len(ch["root"]))))
            h_ = ((hints.get("people") or {}).get(str(ch.get("track_id"))) or {})
            keep = h_.get("keep_height", not follow)          # hint / set: keep the fit's own height, no ground following
            zs, zfit = [], []
            for f in fr:
                x, y, zl = ch["root"][f][0], ch["root"][f][1], ch["root_z_fit"][f]
                zfit.append(zl)
                # only large surfaces (terraces, steps, slabs, hill blocks: top faces >= 3 m2), never furniture, only
                # when the fit had lifted the person off the floor, and only where a top face really is under them
                tops = []
                if zl > zg + 0.3:
                    for b in mine:
                        if b["area"] >= 3.0 and b["x0"] <= x <= b["x1"] and b["y0"] <= y <= b["y1"]:
                            zt = surface_z(b, x, y)
                            if zt is not None and zt <= zl + 0.3 and zt < zg + 8.0:
                                tops.append(zt)
                zs.append(max([grounds[s["shot"]].at(x, y)] + tops))      # the (sloped / uneven) ground under them
            if not zs:
                continue
            zs = median_filter(np.asarray(zs), size=min(9, len(zs)), mode="nearest")
            vis = [f_ for f_ in range(len(fr)) if (ch.get("visible") or [True] * len(ch["root"]))[fr[f_]]]
            if keep:
                print("%s (track %s) shot %d: keeps the fit's own height (keep_height / ground follow false)" % (kind, ch.get("track_id"), s["shot"]))
                for f in fr:
                    ch["root"][f][2] = round(float(ch["root_z_fit"][f]), 4)
                continue
            zf_ = np.asarray(zfit, float)
            if vis and np.abs(zf_).max() > 1e-3:              # the fit had its own heights: say when the set disagrees
                dz = (zs - zf_)[vis]
                if np.median(np.abs(dz)) > 0.3:
                    print("WARNING: %s (track %s) shot %d: the ground under them is %+.2f m from the fit's height "
                          "(median, frames %d-%d) - moved onto the ground; hints people {\"%s\": {\"keep_height\": true}} "
                          "keeps the fit's height" % (kind, ch.get("track_id"), s["shot"], float(np.median(dz)), fr[0], fr[-1], ch.get("track_id")))
            for f, z in zip(fr, zs):
                ch["root"][f][2] = round(float(z), 4)
    # animals / vehicles carrying a person (hints.json people entry "mount": "horse")
    data["props"] = [p for p in (data.get("props") or []) if "_horse_" not in p["name"]]
    for kind, ch in data["chars"].items():
        h_ = ((hints.get("people") or {}).get(str(ch.get("track_id"))) or {})
        if h_.get("mount") in ("horse", True):
            mp = mount_props(kind, ch, h_, ground_at, float(data.get("fps", 24)))
            data["props"] += mp
            print("mount: horse under %s (track %s)" % (kind, ch.get("track_id")))
    if env.get("sky"):
        data["sky"] = env["sky"]
    for k, v in (env.get("palette") or {}).items():
        data["palette"][k] = v
    jsave(dp, data)
    N = data["frames"]
    frs = [int(x) for x in a.frames.split(",") if x != ""] or sorted({v for s in shots for v in (s["frames"][0], (s["frames"][0] + s["frames"][1]) // 2, s["frames"][1])})[:9]
    frames = read_frames_at(ci["video"], frs)
    y0, Hp = ci["y0"], ci["Hpic"]
    tiles = []
    try:
        parts = solid_parts(meshes)
    except Exception as e:                                   # the check picture must never stop the set build
        print("env_check: no inside-object test (%s)" % e); parts = []
    for fr in frs:
        src = frames[fr][y0:y0 + Hp]
        gzf = grounds[shot_of(fr)]
        try:
            ppl = people_at(data, cam, fr, gzf); bodies = body_polys(ppl, cam)
        except Exception as e:
            print("env_check: people not drawn at f%d (%s)" % (fr, e)); ppl, bodies = [], []
        mf = meshes + [m for m in (prop_at(p, fr) for p in data["props"] if p.get("keys")) if m]
        ov = overlay(mf, cam, src, fr, data["palette"], a.alpha)       # film person stays visible: stick figure only
        solid = overlay(mf, cam, np.full_like(src, 225), fr, data["palette"], 1.0, bodies)
        try:
            draw_people(ov, ppl, parts); draw_people(solid, ppl, parts, legend=True)
        except Exception as e:
            print("env_check: people not drawn at f%d (%s)" % (fr, e))
        for im in (src, ov, solid):
            cv2.putText(im, "f%d" % fr, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(np.hstack([src, ov, solid]))
    out = os.path.join(work, "env_check.jpg")
    sheet = np.vstack(tiles)
    s = min(1.0, 2400.0 / sheet.shape[1])
    imwrite_u(out, cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 88)
    nobj = sum(len(se.get("objects", [])) for se in (env.get("shots") or {}).values()) if env.get("shots") else len(env.get("objects", []))
    print("%d meshes from %d objects in %d shot(s) -> %s\n%s" % (len(meshes), nobj, len(shots), dp, out))


if __name__ == "__main__":
    main()

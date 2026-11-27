# -*- coding: utf-8 -*-
"""The objects of a clip's set, part of the one run (run_pipeline.py): your prebuilt models (assets\\objects) placed
where the video shows the things they stand for.

  load_assets(c)   every assets\\objects\\<type>.blend (assets_lib.py) and the Mask2Former panoptic model must be
                   there; read before any GPU work, so a missing one fails the clip at once
  find(...)        Mask2Former panoptic (ADE20K, 150 names) outlines and names every object, a few frames per second
                   of every shot, plus the depth step's view frames (ade_find.py); ade_objects.py turns each name into
                   its asset (people and scenery are not placed; a name whose .blend is missing is reported).  The
                   same pass names every pixel: floor, wall and ceiling for the depth step's gravity and floor
  place(...)       per object its best views (big, whole, not hidden by people); MoGe depth on those frames
                   (metric_observe.py); each frame's depth matched to the fitted floor (the floor pixels land on z = 0,
                   so objects stand on the floor the people stand on); the object's pixels -> 3D points (the fitted
                   camera of that frame, with its zoom) -> a box on the floor (or on the wall, when it hangs) facing
                   the camera -> the asset stretched into that box.  Checked first (object_checks.py): the box must
                   have a real size for its type (too wide: split by its 3D points), and people must not stand in it

Files: Work\\<clip>\\objects\\ (objects_found.npz and depth.npz caches, preview.jpg)."""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from common import log, imwrite_u
from camera_model import Cam, cam_mats
from profiling import stage
import assets_lib
import object_checks

VERSION = 2
# ADE names of the ground objects stand on (their floor scale)
FLOOR_NAMES = {"floor", "rug", "stage", "pier", "road", "sidewalk", "earth", "field", "sand", "path", "land", "dirt track",
               "runway", "grass", "water", "sea", "lake", "river", "swimming pool"}
SURFACES = ("wall",)            # large surfaces: they never count as a duplicate of the things standing against them


def folder(work):
    return os.path.join(work, "objects")


# ------------------------------------------------------------------ geometry
def pixel_dirs(cam, fr, u, v):
    """world direction per pixel (u, v: picture-area pixels), scaled so that 1 unit along it is 1 m of camera depth"""
    R, C = cam_mats(cam.d["cam"][fr]); f = cam.f_at(fr)
    dc = np.stack([(u - cam.W / 2.0) / f, -(v - cam.H / 2.0) / f, -np.ones_like(u, dtype=float)], -1)
    return C, dc @ R.T


def floor_scale(cam, fr, depth, floor):
    """how much this frame's depth must be multiplied by so its floor pixels lie on z = 0 -> (scale, pixels) or None"""
    h, w = depth.shape
    vv, uu = np.nonzero(floor)
    if len(uu) < 300:
        return None
    if len(uu) > 20000:
        k = np.random.default_rng(0).choice(len(uu), 20000, replace=False); uu, vv = uu[k], vv[k]
    sx, sy = cam.W / float(w), cam.H / float(h)
    C, D = pixel_dirs(cam, fr, (uu + .5) * sx, (vv + .5) * sy)
    down = D[:, 2] < -1e-4
    if C[2] <= 0.05 or down.sum() < 300:
        return None
    zt = -C[2] / D[down, 2]                                # camera depth at which the ray meets z = 0
    zm = depth[vv[down], uu[down]].astype(float)
    ok = (zm > 0.05) & (zt < 80.0)
    if ok.sum() < 300:
        return None
    r = zt[ok] / zm[ok]
    return float(np.median(r)), int(ok.sum())


def object_points(cam, fr, depth, valid, mask, scale, max_points=4000):
    """an object's pixels at one frame -> world points (n, 3); its outline and depth jumps are left out"""
    h, w = depth.shape
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((3, 3), np.uint8)).astype(bool) & valid
    z = depth.astype(float)
    gx = np.abs(cv2.Sobel(z, cv2.CV_64F, 1, 0, ksize=3)); gy = np.abs(cv2.Sobel(z, cv2.CV_64F, 0, 1, ksize=3))
    m &= (np.hypot(gx, gy) / np.maximum(z, 1e-3)) < 0.35          # depth jumps: the edge between object and wall
    vv, uu = np.nonzero(m)
    if len(uu) > max_points:
        k = np.random.default_rng(fr).choice(len(uu), max_points, replace=False); uu, vv = uu[k], vv[k]
    if not len(uu):
        return np.zeros((0, 3))
    sx, sy = cam.W / float(w), cam.H / float(h)
    C, D = pixel_dirs(cam, fr, (uu + .5) * sx, (vv + .5) * sy)
    return C + D * (scale * z[vv, uu])[:, None]


def fit_box(P, cams, model, stands=True, ground_gap=0.3, raised=1.2):
    """3D points of the visible side of one object + the camera positions that saw it -> its box
    {centre (base centre), yaw, size (w, d, h), grounded} or None.  The hidden depth comes from the asset's own
    proportions (a pillar seen from the front is as deep as it is wide).
    stands: a type that stands on the floor reaches down to it (its bottom is often hidden), unless all of it is
    seen more than <raised> m up (on a counter or shelf); a hanging type stays at the height it is seen."""
    if len(P) < 60:
        return None
    # stray points go (depth bleeding off the edge onto the wall far behind); the object's own spread is kept whole,
    # also when most of its pixels are at one end (a counter running away from the camera)
    med = np.median(P, 0); mad = np.median(np.abs(P - med), 0)
    lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
    P = P[np.all((np.abs(P - med) <= 8 * 1.4826 * mad + 0.3) & (P >= lo) & (P <= hi), 1)]
    if len(P) < 40:
        return None
    zl, zh = float(np.percentile(P[:, 2], 1)), float(np.percentile(P[:, 2], 99))
    if zh <= 0.05:
        return None                                        # all of it under the floor: the depth is wrong here
    grounded = zl < ground_gap or (stands and zl < raised)
    base = 0.0 if grounded else zl
    H = max(0.05, zh - base)
    Q = P[:, :2]; mu = Q.mean(0)
    ev, evec = np.linalg.eigh(np.cov((Q - mu).T) + 1e-9 * np.eye(2))
    a1 = evec[:, 1]; n = evec[:, 0]                        # a1: along the visible side, n: across it
    view = np.mean(np.asarray(cams, float)[:, :2], 0) - mu
    if np.dot(n, view) < 0:
        n = -n                                             # n points toward the cameras: the object's front
    t1 = (Q - mu) @ a1; tn = (Q - mu) @ n
    w = max(0.05, float(np.percentile(t1, 99) - np.percentile(t1, 1)))
    dm = float(np.percentile(tn, 97) - np.percentile(tn, 3))
    sx, sy, sz = (float(x) for x in model["size"])
    if sy < 0.2 * sx:                                      # a flat thing (wall, window, picture, door, blind): its
        d = max(0.02, w * sy / max(sx, 1e-6))              # depth is its own; the measured spread is noise and bleed
    else:
        d = max(0.05, dm, w * sy / max(sx, 1e-6))
    front = float(np.percentile(tn, 90))
    c1 = (np.percentile(t1, 99) + np.percentile(t1, 1)) / 2.0
    cxy = mu + a1 * c1 + n * (front - d / 2.0)
    yaw = math.atan2(n[0], -n[1])                          # the model's front (-Y) turned toward the cameras
    return {"centre": [float(cxy[0]), float(cxy[1]), base], "yaw": yaw, "size": [w, d, H], "grounded": bool(grounded),
            "points": int(len(P)), "seen_z": [zl, zh]}


def footprint(box):
    """the box's floor corners (4, 2)"""
    w, d = box["size"][0] / 2.0, box["size"][1] / 2.0
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    R = np.array([[c, -s], [s, c]])
    return np.array([[-w, -d], [w, -d], [w, d], [-w, d]]) @ R.T + np.asarray(box["centre"][:2])


def overlap(a, b):
    """share of the smaller box's footprint (axis-aligned) that the other one covers, and the overlap of the heights
    at which they were seen (not the boxes: a standing object's box reaches down to the floor)"""
    A, B = footprint(a), footprint(b)
    lo, hi = np.maximum(A.min(0), B.min(0)), np.minimum(A.max(0), B.max(0))
    inter = float(np.prod(np.maximum(hi - lo, 0)))
    small = min(float(np.prod(A.max(0) - A.min(0))), float(np.prod(B.max(0) - B.min(0))))
    za = a.get("seen_z") or (a["centre"][2], a["centre"][2] + a["size"][2])
    zb = b.get("seen_z") or (b["centre"][2], b["centre"][2] + b["size"][2])
    zo = max(0.0, min(za[1], zb[1]) - max(za[0], zb[0])) / max(1e-6, min(za[1] - za[0], zb[1] - zb[0]))
    return inter / max(small, 1e-9), zo


# ------------------------------------------------------------------ steps
def tracks_of(bundle, types, people_mask, min_frames=2, min_area=0.002, best_area=0.004, max_hidden=0.4):
    """Mask2Former detections -> {id: {"type", "frames": [clip frames], "rows": [bundle rows], "views": [how well it is
    seen], "probs"}}.  A view mostly covered by a person is not used (the outline is then often part of the person);
    an object never seen clearly at <best_area> of the picture or more is left out."""
    out = {}
    shape = tuple(int(x) for x in bundle["shape"])
    total = float(np.prod(shape))
    for i in range(len(bundle["frames"])):
        oid = int(bundle["ids"][i])
        t = out.setdefault(oid, {"type": types[int(bundle["types"][i])], "frames": [], "rows": [], "views": [], "probs": []})
        m = np.unpackbits(bundle["packed"][i], count=int(total)).reshape(shape).astype(bool)
        area = m.sum() / total
        if area < min_area:
            continue
        f = int(bundle["frames"][i])
        edge = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean()       # cut by the picture's edge
        pm = people_mask(f, shape)
        hidden = (m & pm).sum() / max(1, m.sum())
        t["frames"].append(f); t["rows"].append(i); t["probs"].append(float(bundle["probabilities"][i]))
        t["views"].append(0.0 if hidden > max_hidden else area * (1 - min(1.0, 4 * edge)) * (1 - hidden))
        t.setdefault("best", 0.0)
        if hidden <= max_hidden:
            t["best"] = max(t["best"], area)
    return {k: v for k, v in out.items() if len(v["frames"]) >= min_frames and v.get("best", 0) >= best_area
            and max(v["views"]) > 0}


def best_views(t, k=3):
    """up to k frames where the object is seen best, spread over the time it is seen"""
    order = np.argsort(t["views"])[::-1]
    span = max(1, (t["frames"][-1] - t["frames"][0]) // (2 * k))
    pick = []
    for j in order:
        if t["views"][j] <= 0:
            break
        if all(abs(t["frames"][j] - t["frames"][p]) >= span for p in pick):
            pick.append(int(j))
        if len(pick) == k:
            break
    return pick


def run_depth(images, fovs, out, c):
    """MoGe depth of the chosen frames (metric_observe.py, cached by images + fov + version)"""
    import moge_model
    mv = moge_model.version(c)
    refine = int(c.get("moge_refine_steps", 3))
    digest = hashlib.sha256(images.tobytes() + np.asarray(fovs, float).tobytes() + ("%s/%d/%d" % (mv, refine, VERSION)).encode()).hexdigest()
    path = os.path.join(out, "depth.npz")
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as got:
            if str(got["key"]) == digest:
                return {k: got[k].copy() for k in got.files}
    with tempfile.TemporaryDirectory(dir=out, prefix="depth_") as tmp:
        inp, res = os.path.join(tmp, "input.npz"), os.path.join(tmp, "result.npz")
        np.savez_compressed(inp, images=images, fov=np.asarray(fovs, float))
        p = subprocess.run([sys.executable, os.path.join(HERE, "metric_observe.py"), "--input", inp, "--output", res,
                            "--moge", mv, "--refine", str(refine)], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=1800)
        if p.returncode:
            raise RuntimeError("depth failed: " + (p.stderr or p.stdout)[-600:])
        with np.load(res, allow_pickle=False) as got:
            result = {k: got[k].copy() for k in got.files if k != "images"}
    np.savez_compressed(path, key=digest, **result)
    return result


def preview(pics, cam, meshes, path, n=4):
    fs = sorted(pics)
    if not fs:
        return
    tiles = []
    for f in [fs[int(i)] for i in np.linspace(0, len(fs) - 1, min(n, len(fs)))]:
        img = pics[f].copy()
        for m in meshes:
            V = np.asarray(m["v"], float)
            uv, z = cam.project(V, f)
            col = (40, 220, 255)
            for face in m["f"][::max(1, len(m["f"]) // 400)]:
                if all(z[i] > 0.2 for i in face):
                    pts = np.round(uv[face]).astype(np.int32)
                    cv2.polylines(img, [pts], True, col, 3, cv2.LINE_AA)
            if (z > 0.2).any():
                q = uv[z > 0.2].mean(0)
                cv2.putText(img, m["name"], (int(q[0]), int(q[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(img, m["name"], (int(q[0]), int(q[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
        s = 960.0 / img.shape[1]
        tiles.append(cv2.resize(img, (960, int(img.shape[0] * s))))
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]          # two per row
    imwrite_u(path, np.vstack(rows), 85)


def walk_through(box, data, step=2):
    """frames in which a person's root stands inside a tall object's footprint"""
    if box["size"][2] < 0.5 or not box["grounded"]:
        return 0
    A = footprint(box)
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    hits = 0
    for ch in (data.get("chars") or {}).values():
        vis = ch.get("visible") or [True] * len(ch["root"])
        for f in range(0, len(ch["root"]), step):
            if not vis[f]:
                continue
            p = np.asarray(ch["root"][f][:2]) - np.asarray(box["centre"][:2])
            x, y = p[0] * c + p[1] * s, -p[0] * s + p[1] * c
            if abs(x) < box["size"][0] / 2 - 0.05 and abs(y) < box["size"][1] / 2 - 0.05:
                hits += 1
    return hits * step


# ------------------------------------------------------------------ the steps of the run
def load_assets(c):
    """-> (catalog, {type: model}, [types]).  Every model is read before the GPU work, so a broken one fails at once"""
    import ade_find
    if not ade_find.available():
        raise RuntimeError("Mask2Former panoptic is missing: put config.json, preprocessor_config.json and model.safetensors "
                           "of facebook/mask2former-swin-large-ade-panoptic in %s" % ade_find.MODEL_DIR)
    cat = assets_lib.catalog(c)
    if not cat:
        raise RuntimeError("no object models: put one .blend per object type in %s (e.g. pillar.blend, wall.blend)"
                           % assets_lib.folder(c))
    log("  assets: %d object models in %s" % (len(cat), assets_lib.folder(c)))
    models = {t: assets_lib.load(t, c) for t in cat}
    return cat, models, list(cat)


def people_masks(marks):
    """f, (h, w) -> where people are in clip frame f (SAM 3's masks, a little wider), or nothing"""
    people = None
    mp = os.path.join(marks, "sam3_masks.npz")
    if os.path.isfile(mp):
        from sam3_tracking import read_masks
        people = read_masks(mp)

    def at(f, shape):
        if people is None:
            return np.zeros(shape, bool)
        from sam3_tracking import union_at
        m = union_at(people, f, shape[1], shape[0])
        return cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return at


@stage("scene_finder")
def find(pics, shots, types, video, work, c):
    """Mask2Former panoptic on pics {clip frame: picture} -> the bundle of ade_find.find (objects + every frame's name
    map); cached by the frames, the models and the settings"""
    import ade_find
    out = folder(work); os.makedirs(out, exist_ok=True)
    key = hashlib.sha256(json.dumps([VERSION, ade_find.VERSION, sorted(pics), types, os.path.getsize(video), os.path.getmtime(video),
                                     c.get("bg_detect_fps", 5), c.get("bg_find_width", 1024), c.get("bg_ade_threshold", .5),
                                     c.get("bg_min_area", .002), c.get("bg_link_iou", .3), c.get("sam3_mask_side", 640)]).encode()).hexdigest()
    dp = os.path.join(out, "objects_found.npz")
    if os.path.isfile(dp):
        with np.load(dp, allow_pickle=False) as got:
            if str(got["key"]) == key:
                bundle = {k: got[k].copy() for k in got.files if k not in ("missing", "key")}
                missing = json.loads(str(got["missing"]))
                log("  Mask2Former: %d frames (kept from the last run)" % len(pics))
                return bundle, missing
    log("  Mask2Former: objects and surfaces on %d frames ..." % len(pics))
    bundle, info = ade_find.find(pics, shots, types, c)
    np.savez_compressed(dp, key=key, missing=json.dumps(info["missing"]), **bundle)
    log("    Mask2Former %.0fs: %d object(s)" % (info["seconds"], info["objects"]))
    return bundle, info["missing"]


@stage("objects")
def place(data, pics, bundle, missing, people_mask, assets, work, c):
    """the found objects -> (meshes for scene_data, objects report, warnings).  data: the fitted scene (cameras,
    people, shots); pics {clip frame: picture} as given to find()"""
    import ade_find
    cat, models, types = assets
    out = folder(work); os.makedirs(out, exist_ok=True)
    N = int(data["frames"])
    cam = Cam(data)
    shots = [(int(s["frames"][0]), int(s["frames"][1]) + 1) for s in data.get("shots") or []] or [(0, N)]
    warns = []
    if missing:
        log("    no model for: %s" % ", ".join("%s (add %s.blend)" % (n, a) for n, a in sorted(missing.items())))
    tracks = tracks_of(bundle, types, people_mask)
    log("  objects found: %d%s" % (len(tracks), "" if not tracks else " - " + ", ".join(
        "%s %d" % (t, sum(1 for x in tracks.values() if x["type"] == t)) for t in types if any(x["type"] == t for x in tracks.values()))))

    # depth on each object's best views
    views = {oid: [t["frames"][j] for j in best_views(t)] for oid, t in tracks.items()}
    key_frames = sorted({f for fs in views.values() for f in fs})
    if len(key_frames) > int(c.get("bg_max_depth_frames", 48)):
        keep = set(np.asarray(key_frames)[np.linspace(0, len(key_frames) - 1, int(c.get("bg_max_depth_frames", 48))).astype(int)])
        views = {oid: [f for f in fs if f in keep] or fs[:1] for oid, fs in views.items()}
        key_frames = sorted({f for fs in views.values() for f in fs})
    objects, meshes = [], []
    if not key_frames:
        warns.append("no object of the asset types was found: the scene has people and the floor only")
    else:
        width = int(c.get("bg_depth_width", 640)); width = min(width, cam.W); height = max(16, round(cam.H * width / cam.W))
        images = np.asarray([cv2.cvtColor(cv2.resize(pics[f], (width, height)), cv2.COLOR_BGR2RGB) for f in key_frames])
        fovs = [math.degrees(2 * math.atan(cam.W / (2 * cam.f_at(f)))) for f in key_frames]
        log("  object depth on %d frame(s) ..." % len(key_frames))
        td = time.time()
        depth = run_depth(images, fovs, out, c)
        log("    depth %.0fs" % (time.time() - td))
        names = [str(x) for x in bundle["names"]]
        floor_ids = [i for i, nm in enumerate(names) if nm in FLOOR_NAMES]
        # scale: each frame's floor onto z = 0
        scales = {}
        for k, f in enumerate(key_frames):
            D, ok = depth["depth"][k], depth["valid"][k]
            lab = ade_find.labels_at(bundle, f, ok.shape[1], ok.shape[0])
            fl = np.isin(lab, floor_ids) & ok if lab is not None and floor_ids else np.zeros_like(ok)
            fl &= ~cv2.resize(people_mask(f, fl.shape).astype(np.uint8), (fl.shape[1], fl.shape[0])).astype(bool)
            s = floor_scale(cam, f, D, fl)
            if s:
                scales[f] = s[0]
        shot_of = lambda f: next((i for i, (a, b) in enumerate(shots) if a <= f < b), 0)

        def scale_at(f):
            if f in scales:
                return scales[f], "floor"
            same = [v for g, v in scales.items() if shot_of(g) == shot_of(f)]
            if same:
                return float(np.median(same)), "floor of other frames in the shot"
            if scales:
                return float(np.median(list(scales.values()))), "floor of other shots"
            return 1.0, "depth model's own metres (no floor seen)"
        idx = {f: k for k, f in enumerate(key_frames)}
        # place (object_checks.py: each object must have a real size for its type, and people not stand in it)
        checks = object_checks.enabled(c)
        extra = object_checks.extras(c) if checks else {}
        boxes = []
        for oid, t in tracks.items():
            ty = t["type"]; ex = extra.get(ty, {})
            P, cams, how = [], [], set()
            for f in views[oid]:
                row = t["rows"][t["frames"].index(f)]
                m = ade_find.mask_of(bundle, row, cam.W, cam.H) & ~people_mask(f, (cam.H, cam.W))
                k = idx[f]; s, why = scale_at(f); how.add(why)
                P.append(object_points(cam, f, depth["depth"][k], depth["valid"][k], m, s))
                cams.append(cam_mats(data["cam"][f])[1])
            P = np.vstack(P) if P else np.zeros((0, 3))
            fit = lambda Q: fit_box(Q, cams, models[ty], cat[ty]["on_floor"])
            parts = object_checks.split_if_too_wide(ty, P, fit, ex) if checks else [P]
            for j, Pj in enumerate(parts):
                box = fit(Pj)
                if box is None:
                    warns.append("%s #%d: too little depth to place it (left out)" % (ty, oid)); continue
                dist = float(np.linalg.norm(np.asarray(box["centre"][:2]) - np.mean(np.asarray(cams)[:, :2], 0)))
                if box["size"][2] > 40 or dist > 0.6 * float(c.get("far_distance", 1500.0)):
                    warns.append("%s #%d: measured out of range (left out)" % (ty, oid)); continue
                bad = object_checks.size_problem(ty, box, ex) if checks else None
                if bad:
                    warns.append("%s #%d: %s (left out)" % (ty, oid, bad)); continue
                box.update(type=ty, id=oid * 100 + j if len(parts) > 1 else oid, frames=[t["frames"][0], t["frames"][-1]],
                           seen=len(t["frames"]), views=views[oid], prob=float(np.mean(t["probs"])), scale=sorted(how),
                           shot=shot_of(t["frames"][0]), pts=Pj)
                boxes.append(box)
        # one object found twice (lost behind a person and found again, or two names for one thing): keep the one
        # seen most surely and most often
        boxes.sort(key=lambda b: -(b["prob"] * b["seen"]))
        kept = []
        for b in boxes:
            dup = None
            for a in kept:
                if (a["type"] in SURFACES) != (b["type"] in SURFACES):
                    continue                     # furniture against a wall is not the wall seen twice
                fo, zo = overlap(a, b)
                if a["shot"] == b["shot"] and fo > 0.5 and zo > 0.5:
                    dup = a; break
            if dup is None:
                kept.append(b)
            elif dup["type"] == b["type"]:                       # the same object: measure it from both sets of views
                cams = [cam_mats(data["cam"][f])[1] for f in dup["views"] + b["views"]]
                nb = fit_box(np.vstack([dup["pts"], b["pts"]]), cams, models[dup["type"]], cat[dup["type"]]["on_floor"])
                if nb and checks and object_checks.size_problem(dup["type"], nb, extra.get(dup["type"], {})):
                    nb = None                    # measured together it has no real size: keep the first measurement
                if nb:
                    for k_ in ("centre", "yaw", "size", "grounded", "points", "seen_z"):
                        dup[k_] = nb[k_]
                dup["views"] = sorted(set(dup["views"] + b["views"])); dup["seen"] += b["seen"]
                dup["frames"] = [min(dup["frames"][0], b["frames"][0]), max(dup["frames"][1], b["frames"][1])]
            else:
                warns.append("%s #%d is where %s #%d is: kept the %s" % (b["type"], b["id"], dup["type"], dup["id"], dup["type"]))
        count = {}
        for b in kept:
            through = walk_through(b, data)
            bad = object_checks.people_inside_problem(b["type"], through, N) if checks else None
            if bad:
                warns.append("%s #%d: %s (left out)" % (b["type"], b["id"], bad)); continue
            count[b["type"]] = count.get(b["type"], 0) + 1
            nm = "%s_%d" % (b["type"], count[b["type"]]) + ("_s%d" % (b["shot"] + 1) if len(shots) > 1 else "")
            V = assets_lib.place(models[b["type"]], b["centre"], b["yaw"], b["size"])
            mesh = {"name": nm, "v": np.round(V, 4).tolist(), "f": models[b["type"]]["f"], "colour": "set", "layer": 2,
                    "kind": b["type"], "label": b["type"], "obj": nm, "asset": models[b["type"]]["source"]}
            if cat[b["type"]].get("role"):
                mesh["role"] = cat[b["type"]]["role"]
            meshes.append(mesh)
            if through and not (checks and object_checks.is_seat(b["type"])):     # sitting on a seat is expected
                warns.append("%s: a person stands inside it in %d frame(s) - check its place" % (nm, through))
            objects.append({"name": nm, "type": b["type"], "asset": models[b["type"]]["source"],
                            "position_m": [round(x, 3) for x in b["centre"]], "yaw_deg": round(math.degrees(b["yaw"]), 1),
                            "size_m": [round(x, 3) for x in b["size"]], "on_floor": b["grounded"],
                            "seen_frames": b["frames"], "detections": b["seen"], "depth_frames": b["views"],
                            "probability": round(b["prob"], 3), "depth_scale_from": b["scale"], "points": b["points"],
                            "person_inside_frames": through})
    log("  placed: %d object(s)" % len(objects))
    for o in objects:
        log("    %-18s %5.2f x %5.2f x %5.2f m at (%.2f, %.2f, %.2f)%s" % (o["name"], *o["size_m"], *o["position_m"],
                                                                         "" if o["on_floor"] else "  (hanging)"))
    preview({f: pics[f] for f in sorted(pics)}, cam, meshes, os.path.join(out, "preview.jpg"))
    return meshes, objects, warns

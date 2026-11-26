# -*- coding: utf-8 -*-
"""Stage 2 "Background": Stage 1's scene + your prebuilt objects -> a second v2.1 zip (Stage 1's zip is kept).

  python run_pipeline.py VIDEO --stage background --name CLIP

  1. assets      every assets\\objects\\<type>.blend (assets_lib.py); none -> the run stops and says so
  2. find        SAM 3 looks for each type by name, a few frames per second of every shot (sam3_objects.py):
                 an object that only comes into view later (zoom, pan) is found where it appears
  3. depth       per object its best views (big, whole, not hidden by people); MoGe (the configured version) depth +
                 Mask2Former floor labels on those frames (metric_observe.py)
  4. scale       each frame's depth is matched to Stage 1's floor: the floor pixels must land on z = 0 of the scene,
                 so objects stand on the floor the people stand on
  5. place       the object's pixels -> 3D points (Stage 1's camera of that frame, with its zoom) -> a box on the
                 floor (or on the wall, when it hangs) facing the camera -> the asset stretched into that box
  6. build       Stage 1's scene_data.json + the objects -> Work\\<clip>\\background\\scene_data.json -> Blender
                 (colour card) -> SUBMIT\\_background\\<clip>.zip -> the v2.1 package check

Files: Work\\<clip>\\background\\ (objects.json report, preview.jpg, caches, render) and SUBMIT\\_background\\<clip>\\.
Stage 1's files are only read."""
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from common import jload, jsave, log, read_frames_at, imwrite_u
from camera_model import Cam, cam_mats
import assets_lib

VERSION = 1
FLOOR_ROLES = ("floor", "ground")


def bg_root(c):
    """where the Stage 1 + Stage 2 packages go: SUBMIT\\_background (the "_" keeps it out of Stage 1's list)"""
    return os.path.join(c["submit_root"], "_background")


def bg_work(work):
    return os.path.join(work, "background")


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
    """SAM 3 detections -> {id: {"type", "frames": [clip frames], "rows": [bundle rows], "views": [how well it is
    seen], "probs"}}.  A view mostly covered by a person is not used (SAM 3 then often names part of the person);
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
    """MoGe depth + Mask2Former labels of the chosen frames (metric_observe.py, cached by images + fov + version)"""
    try:                                                   # the MoGe version switch (moge_model.py), when present
        import moge_model
        mv = moge_model.version(c)
    except ImportError:
        mv = None                                          # metric_observe.py's own MoGe
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
        p = subprocess.run([sys.executable, os.path.join(HERE, "metric_observe.py"), "--input", inp, "--output", res] +
                           (["--moge", mv, "--refine", str(refine)] if mv else []), capture_output=True, text=True, encoding="utf-8",
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


# ------------------------------------------------------------------ the run
def run(name, video, work, marks, c, render_and_pack):
    """Stage 2 for one clip -> report dict.  render_and_pack(name, video, work, c) is Stage 1's (run_pipeline)"""
    import gpu_lease
    t0 = time.time()
    sp, cp = os.path.join(work, "scene_data.json"), os.path.join(work, "clip_info.json")
    if not (os.path.isfile(sp) and os.path.isfile(cp)):
        raise RuntimeError("Stage 1 (Movement) has not finished for this clip: run it first")
    s1zip = os.path.join(c["submit_root"], name + ".zip")
    if not os.path.isfile(s1zip):
        raise RuntimeError("Stage 1's zip is missing (%s): run Movement first" % s1zip)
    out = bg_work(work); os.makedirs(out, exist_ok=True)
    data = jload(sp); ci = jload(cp)
    cat = assets_lib.catalog(c)
    if not cat:
        raise RuntimeError("no object models: put one .blend per object type in %s (e.g. pillar.blend, wall.blend)"
                           % assets_lib.folder(c))
    log("  assets: %s" % ", ".join("%s (\"%s\")" % (t, cat[t]["prompt"]) for t in cat))
    models = {t: assets_lib.load(t, c) for t in cat}          # read every model before the GPU work: fail early
    types = list(cat)
    y0, Hp, N, fps = int(ci["y0"]), int(ci["Hpic"]), int(data["frames"]), float(data["fps"])
    cam = Cam(data)
    shots = [(int(s["frames"][0]), int(s["frames"][1]) + 1) for s in data.get("shots") or []] or [(0, N)]
    import sam3_objects
    sample = sam3_objects.sample_frames(shots, fps, c)
    gpu_lease.acquire()
    log("  reading %d frames ..." % len(sample))
    pics = {f: img[y0:y0 + Hp] for f, img in read_frames_at(video, sample).items()}

    # people's masks (Stage 1's SAM 3 tracks): an object hidden by a person is measured around them
    people = None
    mp = os.path.join(marks, "sam3_masks.npz")
    if os.path.isfile(mp):
        from sam3_tracking import read_masks, union_at
        people = read_masks(mp)

    def people_mask(f, shape):
        if people is None:
            return np.zeros(shape, bool)
        m = union_at(people, f, shape[1], shape[0])
        return cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    # 2. find (cached: frames, prompts, settings)
    prompts = [cat[t]["prompt"] for t in types]
    key = hashlib.sha256(json.dumps([VERSION, sample, prompts, os.path.getsize(video), os.path.getmtime(video),
                                     c.get("bg_probability", .5), c.get("bg_max_objects", 12), c.get("sam3_mask_side", 640)]).encode()).hexdigest()
    dp = os.path.join(out, "sam3_objects.npz")
    bundle = None
    if os.path.isfile(dp):
        with np.load(dp, allow_pickle=False) as got:
            if str(got["key"]) == key:
                bundle = {k: got[k].copy() for k in got.files}
    if bundle is None:
        log("  SAM 3: looking for %s ..." % ", ".join(prompts))
        tf = time.time()
        bundle = sam3_objects.detect(pics, shots, prompts, out, c)
        np.savez_compressed(dp, key=key, **bundle)
        log("    SAM 3 %.0fs" % (time.time() - tf))
    tracks = tracks_of(bundle, types, people_mask)
    log("  found: %d object(s)%s" % (len(tracks), "" if not tracks else " - " + ", ".join(
        "%s %d" % (t, sum(1 for x in tracks.values() if x["type"] == t)) for t in types if any(x["type"] == t for x in tracks.values()))))

    # 3. depth on each object's best views
    views = {oid: [t["frames"][j] for j in best_views(t)] for oid, t in tracks.items()}
    key_frames = sorted({f for fs in views.values() for f in fs})
    if len(key_frames) > int(c.get("bg_max_depth_frames", 48)):
        keep = set(np.asarray(key_frames)[np.linspace(0, len(key_frames) - 1, int(c.get("bg_max_depth_frames", 48))).astype(int)])
        views = {oid: [f for f in fs if f in keep] or fs[:1] for oid, fs in views.items()}
        key_frames = sorted({f for fs in views.values() for f in fs})
    objects, meshes, warns = [], [], []
    if key_frames:
        width = int(c.get("bg_depth_width", 640)); width = min(width, cam.W); height = max(16, round(cam.H * width / cam.W))
        images = np.asarray([cv2.cvtColor(cv2.resize(pics[f], (width, height)), cv2.COLOR_BGR2RGB) for f in key_frames])
        fovs = [math.degrees(2 * math.atan(cam.W / (2 * cam.f_at(f)))) for f in key_frames]
        log("  depth on %d frame(s) ..." % len(key_frames))
        td = time.time()
        depth = run_depth(images, fovs, out, c)
        log("    depth %.0fs" % (time.time() - td))
        gpu_lease.release()
        names = [str(x) for x in depth.get("names", [])]
        import semseg
        floor_ids = [i for i, nm in enumerate(names) if semseg.role(nm) in FLOOR_ROLES]
        # 4. scale: each frame's floor onto z = 0
        scales = {}
        for k, f in enumerate(key_frames):
            D, ok = depth["depth"][k], depth["valid"][k]
            fl = np.isin(depth["labels"][k], floor_ids) & ok if floor_ids else np.zeros_like(ok)
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
        # 5. place
        boxes = []
        for oid, t in tracks.items():
            P, cams, how = [], [], set()
            for f in views[oid]:
                row = t["rows"][t["frames"].index(f)]
                m = sam3_objects.mask_of(bundle, row, cam.W, cam.H) & ~people_mask(f, (cam.H, cam.W))
                k = idx[f]; s, why = scale_at(f); how.add(why)
                P.append(object_points(cam, f, depth["depth"][k], depth["valid"][k], m, s))
                cams.append(cam_mats(data["cam"][f])[1])
            P = np.vstack(P) if P else np.zeros((0, 3))
            box = fit_box(P, cams, models[t["type"]], cat[t["type"]]["on_floor"])
            if box is None:
                warns.append("%s #%d: too little depth to place it (left out)" % (t["type"], oid)); continue
            dist = float(np.linalg.norm(np.asarray(box["centre"][:2]) - np.mean(np.asarray(cams)[:, :2], 0)))
            if box["size"][2] > 40 or dist > 0.6 * float(c.get("far_distance", 1500.0)):
                warns.append("%s #%d: measured out of range (left out)" % (t["type"], oid)); continue
            box.update(type=t["type"], id=oid, frames=[t["frames"][0], t["frames"][-1]], seen=len(t["frames"]),
                       views=views[oid], prob=float(np.mean(t["probs"])), scale=sorted(how), shot=shot_of(t["frames"][0]), pts=P)
            boxes.append(box)
        # one object found twice (SAM 3 lost and found it again, or two names for one thing): keep one
        boxes.sort(key=lambda b: -(b["prob"] * b["seen"]))
        kept = []
        for b in boxes:
            dup = None
            for a in kept:
                fo, zo = overlap(a, b)
                if a["shot"] == b["shot"] and fo > 0.5 and zo > 0.5:
                    dup = a; break
            if dup is None:
                kept.append(b)
            elif dup["type"] == b["type"]:                       # the same object: measure it from both sets of views
                cams = [cam_mats(data["cam"][f])[1] for f in dup["views"] + b["views"]]
                nb = fit_box(np.vstack([dup["pts"], b["pts"]]), cams, models[dup["type"]], cat[dup["type"]]["on_floor"])
                if nb:
                    for k_ in ("centre", "yaw", "size", "grounded", "points", "seen_z"):
                        dup[k_] = nb[k_]
                dup["views"] = sorted(set(dup["views"] + b["views"])); dup["seen"] += b["seen"]
                dup["frames"] = [min(dup["frames"][0], b["frames"][0]), max(dup["frames"][1], b["frames"][1])]
            else:
                warns.append("%s #%d is where %s #%d is: kept the %s" % (b["type"], b["id"], dup["type"], dup["id"], dup["type"]))
        count = {}
        for b in kept:
            count[b["type"]] = count.get(b["type"], 0) + 1
            nm = "%s_%d" % (b["type"], count[b["type"]]) + ("_s%d" % (b["shot"] + 1) if len(shots) > 1 else "")
            V = assets_lib.place(models[b["type"]], b["centre"], b["yaw"], b["size"])
            mesh = {"name": nm, "v": np.round(V, 4).tolist(), "f": models[b["type"]]["f"], "colour": "set", "layer": 2,
                    "kind": b["type"], "label": b["type"], "obj": nm, "asset": models[b["type"]]["source"]}
            if cat[b["type"]].get("role"):
                mesh["role"] = cat[b["type"]]["role"]
            meshes.append(mesh)
            through = walk_through(b, data)
            if through:
                warns.append("%s: a person stands inside it in %d frame(s) - check its place" % (nm, through))
            objects.append({"name": nm, "type": b["type"], "asset": models[b["type"]]["source"],
                            "position_m": [round(x, 3) for x in b["centre"]], "yaw_deg": round(math.degrees(b["yaw"]), 1),
                            "size_m": [round(x, 3) for x in b["size"]], "on_floor": b["grounded"],
                            "seen_frames": b["frames"], "detections": b["seen"], "depth_frames": b["views"],
                            "probability": round(b["prob"], 3), "depth_scale_from": b["scale"], "points": b["points"],
                            "person_inside_frames": through})
    else:
        gpu_lease.release()
        warns.append("no object of the asset types was found: the package is Stage 1's scene re-rendered")
    log("  placed: %d object(s)" % len(objects))
    for o in objects:
        log("    %-18s %5.2f x %5.2f x %5.2f m at (%.2f, %.2f, %.2f)%s" % (o["name"], *o["size_m"], *o["position_m"],
                                                                         "" if o["on_floor"] else "  (hanging)"))
    preview(pics, cam, meshes, os.path.join(out, "preview.jpg"))

    # 6. build: Stage 1's scene + the objects -> render, pack, check (into SUBMIT\_background)
    scene = copy.deepcopy(data)
    scene["meshes"] = [m for m in scene.get("meshes", []) if m.get("layer", 2) != 2] + meshes
    scene["stage"] = "background"
    for fn in ("clip_info.json",):
        shutil.copyfile(os.path.join(work, fn), os.path.join(out, fn))
    jsave(os.path.join(out, "scene_data.json"), scene)
    rep = {"clip": name, "video": video, "stage": "background", "assets": {t: models[t]["source"] for t in types},
           "objects": objects, "warnings": warns, "status": "rendering"}
    jsave(os.path.join(out, "report.json"), rep)
    cb = dict(c, submit_root=bg_root(c))
    os.makedirs(cb["submit_root"], exist_ok=True)
    rep.update(render_and_pack(name, video, out, cb))
    rep.update(status="ok", seconds=round(time.time() - t0, 1), stage1_zip=s1zip)
    jsave(os.path.join(out, "report.json"), rep)
    log("  done in %.0fs%s" % (rep["seconds"], ("   WARNINGS: " + " | ".join(warns)) if warns else ""))
    return rep

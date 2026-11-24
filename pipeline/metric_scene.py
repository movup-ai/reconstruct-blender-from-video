"""MAVE-style shared metric evidence using the installed MoGe, poses and semantics.

The fitted mannequin supplies a scale reference, not SAM 3D Body surface geometry.
All arrays use OpenCV cameras and the pipeline's z-up world, in metres.
"""
import hashlib
import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from scipy.optimize import least_squares
from profiling import stage


VERSION = 1
FLOOR = {"floor", "ground", "road", "sidewalk", "earth", "grass", "field", "path", "rug", "stage"}
WALL = {"wall", "building", "house", "skyscraper"}
SUPPORT = FLOOR | {"table", "desk", "countertop", "step", "stairs", "stairway", "bed", "bench"}


def points(depth, K):
    y, x = np.indices(depth.shape)
    return np.stack(((x - K[0, 2]) / K[0, 0] * depth,
                     (y - K[1, 2]) / K[1, 1] * depth, depth), -1)


def resample_depth(depth, K, shape):
    """Keep the same optical rays when reusing observations at the set builder's size."""
    h, w = shape; scaled = np.asarray(K, float).copy()
    scaled[0] *= w/depth.shape[1]; scaled[1] *= h/depth.shape[0]
    resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return points(resized, scaled)


def robust_scale(ratios, minimum=3):
    values = np.asarray(ratios, float)
    values = values[np.isfinite(values) & (values > .02) & (values < 50)]
    if len(values) < minimum:
        return None
    logs = np.log(values); center = np.median(logs)
    mad = np.median(np.abs(logs - center))
    keep = np.abs(logs - center) <= max(.12, 3 * 1.4826 * mad)
    if keep.sum() < minimum or np.median(np.abs(logs[keep] - np.median(logs[keep]))) > .25:
        return None
    return float(np.exp(np.median(logs[keep])))


def person_mask(pose, frame, shape, y0, source_size):
    """Conservative box mask, including a margin; never relies on SAM 3 masks."""
    mask = np.zeros(shape, np.uint8); W, H = source_size
    for person in pose.get("people", []):
        box = person.get("boxes", {}).get(str(frame))
        if box is None:
            keys = person.get("frames", {}).get(str(frame))
            if keys is None:
                continue
            k = np.asarray(keys); k = k[k[:, 2] > .3]
            if len(k) < 3:
                continue
            box = [*k[:, :2].min(0), *k[:, :2].max(0)]
        x0, ya, x1, yb = np.asarray(box[:4], float)
        margin = max(4, .08 * (yb - ya))
        xa, xb = np.clip(np.array([x0-margin, x1+margin]) * shape[1] / W, 0, shape[1]).astype(int)
        ya, yb = np.clip(np.array([ya-y0-margin, yb-y0+margin]) * shape[0] / H, 0, shape[0]).astype(int)
        mask[ya:yb, xa:xb] = 1
    return mask.astype(bool)


@stage("metric_observation")
def observe(frames, pose, cam, y0, camj, c):
    """Run isolated GPU observations, cached by the actual images and intrinsics."""
    count = min(len(frames), max(1, int(c.get("metric_max_views", 3))))
    fs = np.unique(np.linspace(0, len(frames)-1, count, dtype=int))
    width = min(cam.W, int(c.get("metric_resolution", 384)))
    height = max(16, round(cam.H * width / cam.W))
    images = np.asarray([cv2.cvtColor(cv2.resize(frames[f][y0:y0+cam.H], (width, height)), cv2.COLOR_BGR2RGB) for f in fs])
    zoom = camj.get("zoom", [1.] * len(frames))
    fovs = [math.degrees(2 * math.atan(cam.W / (2 * cam.f * zoom[f]))) for f in fs]
    digest = hashlib.sha256(images.tobytes() + np.asarray(fovs).tobytes() + str(VERSION).encode()).hexdigest()
    path = os.path.join(c["_work"], "metric_observation%s.npz" % c.get("_shot_tag", ""))
    source = None
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["key"]) == digest:
                source = {k: cached[k].copy() for k in cached.files}
    if source is None:
        with tempfile.TemporaryDirectory(dir=c["_work"], prefix="metric_") as tmp:
            inp, out = os.path.join(tmp, "input.npz"), os.path.join(tmp, "result.npz")
            np.savez_compressed(inp, images=images, fov=fovs)
            proc = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "metric_observe.py"),
                                   "--input", inp, "--output", out], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=600)
            if proc.returncode:
                raise RuntimeError("metric observation failed: " + (proc.stderr or proc.stdout)[-600:])
            with np.load(out, allow_pickle=False) as result:
                source = {k: result[k].copy() for k in result.files if k != "images"}
            np.savez_compressed(path, key=digest, **source)
    source["frames"] = fs
    Ks, people, groups = [], [], []
    for f, label in zip(fs, source["labels"]):
        K = cam.K().copy(); K[0, 0] *= zoom[f]; K[1, 1] *= zoom[f]
        K[0] *= width / cam.W; K[1] *= height / cam.H
        Ks.append(K)
        mask = person_mask(pose, f, (height, width), y0, (cam.W, cam.H))
        group = np.full(label.shape, 3, np.int8)
        for i, name in enumerate(source["names"]):
            if name in FLOOR: group[label == i] = 0
            if name in WALL: group[label == i] = 1
            if name == "ceiling": group[label == i] = 2
            if name == "person": mask |= label == i
            if name in ("sky", "water", "sea", "mirror", "windowpane"):
                source["valid"][len(groups)][label == i] = False
        people.append(mask); groups.append(group)
    source.update(K=np.asarray(Ks), person=np.asarray(people), groups=np.asarray(groups), source="MoGe-2")
    return source


def solve_up(normals, categories, prior, body_up=()):
    prior = np.asarray(prior, float); prior /= np.linalg.norm(prior)
    normals = np.asarray(normals, float).reshape(-1, 3); categories = np.asarray(categories)
    horizontal = normals[(categories == 0) | (categories == 2)].copy()
    horizontal *= np.where((horizontal @ prior)[:, None] >= 0, 1, -1)
    horizontal = horizontal[horizontal @ prior > math.cos(math.radians(25))]
    wall = normals[categories == 1]; wall = wall[np.abs(wall @ prior) < .4]
    body = np.asarray(body_up, float).reshape(-1, 3)
    if len(horizontal) < 30 and len(wall) < 60:
        return prior.copy()
    def residual(v):
        up = v / max(np.linalg.norm(v), 1e-9)
        return np.r_[(np.cross(horizontal, up) / math.sqrt(max(1, len(horizontal)))).ravel(),
                     wall @ up / math.sqrt(max(1, len(wall))),
                     (.2 * np.cross(body, up) / math.sqrt(max(1, len(body)))).ravel(),
                     .35 * (up-prior), np.linalg.norm(v)-1]
    up = least_squares(residual, prior, loss="soft_l1", f_scale=.05, max_nfev=60).x
    up /= np.linalg.norm(up)
    return up if up @ prior > math.cos(math.radians(12)) else prior.copy()


def align(observation, camj, prior, height, c, people=None, cam=None, y0=0):
    """Combine scene normals in cam0; retain the human-derived absolute height."""
    normals, categories, floors, walls, upright = [], [], [], [], []
    for f, depth, K, valid, person, group in zip(observation["frames"], observation["depth"], observation["K"],
                                               observation["valid"], observation["person"], observation["groups"]):
        xyz = points(depth, K)
        n = np.cross(np.gradient(xyz, axis=1), np.gradient(xyz, axis=0))
        norm = np.linalg.norm(n, axis=-1)
        n /= np.maximum(norm[..., None], 1e-9)
        # Discontinuities at people and depth edges do not supply orientation evidence.
        edge = np.maximum(np.abs(np.gradient(depth, axis=0)), np.abs(np.gradient(depth, axis=1)))
        selected = valid & ~person & np.isfinite(depth) & (depth > .1) & (depth < 80) & (norm > 1e-8) & (edge < .06 * depth)
        selected = cv2.erode(selected.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        R = np.asarray(camj["R"][f])
        n0 = n[selected] @ R
        normals.append(n0[::8]); categories.append(group[selected][::8])
        floors.append((xyz[selected & (group == 0)] @ R)[::8])
        walls.append((n[selected & (group == 1)] @ R)[::8])
        if cam is not None:
            for fs, keys in (people or {}).values():
                if f not in fs: continue
                key = np.asarray(keys[list(fs).index(f)])[[5,6,11,12]]
                if np.min(key[:,2]) < .7: continue
                x = np.clip((key[:,0]*depth.shape[1]/cam.W).astype(int),0,depth.shape[1]-1)
                y = np.clip(((key[:,1]-y0)*depth.shape[0]/cam.H).astype(int),0,depth.shape[0]-1)
                if not valid[y,x].all(): continue
                p = xyz[y,x]; vector = (p[:2].mean(0)-p[2:].mean(0)) @ R
                length = np.linalg.norm(vector)
                if length > .1 and vector @ prior/length > .9:
                    upright.append(vector/length)
    up = solve_up(np.concatenate(normals), np.concatenate(categories), prior, upright)
    # MoGe scale is calibrated from the recovered camera height and semantic floor.
    ground = np.concatenate(floors); ground_scale = None
    if len(ground) >= 30:
        offsets = -(ground @ up)
        offsets = offsets[offsets > .2]
        if len(offsets) >= 30:
            med = np.median(offsets)
            if np.median(np.abs(offsets-med)) / med < .2:
                ground_scale = height / med
    observation["scale"] = float(ground_scale or 1.)
    observation["scale_source"] = "semantic floor / fitted camera height" if ground_scale else "MoGe metric prior"
    # Build the same initial horizontal axes as World without constructing camera state.
    forward = np.array([0., 0., 1.]) - up * up[2]
    if np.linalg.norm(forward) < 1e-6: forward = np.array([0., 1., 0.])
    forward /= np.linalg.norm(forward)
    M = np.c_[np.cross(forward, up), forward, up]
    walls = np.concatenate(walls)
    yaw, strength = 0., 0.
    if len(walls) >= 60:
        local = walls @ M
        local = local[np.abs(local[:, 2]) < .25]
        if len(local) >= 60:
            vote = np.mean(np.exp(4j * np.arctan2(local[:, 1], local[:, 0])))
            strength = float(abs(vote))
            if strength > .65:
                yaw = float(np.angle(vote) / 4)
    camj["world_yaw"] = yaw
    return up, {"source": "MoGe-2 + semantic floor/ceiling/wall normals", "world_yaw_deg": math.degrees(yaw),
                "manhattan_confidence": strength, "ground_samples": len(ground), "scale_source": observation["scale_source"]}


def body_scale(observation, world, chars):
    """Calibrate depth to fitted torso proxy surfaces; retain one scale per shot.

    Restrict samples to separated torsos so overlaps cannot calibrate to another person.
    A mannequin proxy is less accurate than the paper's dense SAM 3D Body mesh.
    """
    from rig import Rig, JOINTS
    ratios = []
    for f, depth, K in zip(observation["frames"], observation["depth"], observation["K"]):
        torsos = []
        for ch in chars.values():
            if not ch.get("visible", [True] * len(ch["root"]))[f]: continue
            rig = Rig(ch["height"], ch.get("style", "sample")); root = ch["root"][f]
            joints = rig.fk(root[:3], root[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f])
            p = rig.coco_points(joints)[[5, 6, 11, 12]]
            camera = (p-world.C_at(f)) @ world.R_cw(f).T
            if np.any(camera[:, 2] < .2): continue
            uv = camera @ K.T; uv = uv[:, :2] / uv[:, 2:]
            torsos.append((ch, camera, uv))
        for ch, camera, uv in torsos:
            center = uv.mean(0); radius = max(2., np.linalg.norm(uv[0]-uv[1]) / 3)
            if any(other is not ch and np.linalg.norm(other_uv.mean(0)-center) < 2*radius
                   for other, _, other_uv in torsos): continue
            x, y = np.rint(center).astype(int); r = max(1, int(radius/2))
            if not (r <= x < depth.shape[1]-r and r <= y < depth.shape[0]-r): continue
            patch = depth[y-r:y+r+1, x-r:x+r+1]
            valid = observation["valid"][list(observation["frames"]).index(f)][y-r:y+r+1, x-r:x+r+1]
            measured = patch[valid & np.isfinite(patch) & (patch > .1)]
            if len(measured) < 5: continue
            med = np.median(measured)
            if np.median(np.abs(measured-med)) / med > .08: continue
            surface = float(camera[:, 2].mean() - .055 * ch["height"])
            ratios.append(surface / med)
    scale = robust_scale(ratios)
    prior = observation.get("scale", 1.)
    if scale is not None and .5 < scale / prior < 2.:
        observation["scale"], observation["scale_source"] = scale, "fitted torso proxy surfaces"
    return {"samples": len(ratios), "meters_per_unit": observation.get("scale", 1.),
            "source": observation.get("scale_source", "MoGe metric prior")}


def capture(observation, world, work, tag="", start=0, dx=0.):
    fs = observation["frames"]
    centers = np.array([world.C_at(f) for f in fs]); centers[:, 0] += dx
    path = os.path.join(work, "metric_evidence%s.npz" % tag)
    np.savez_compressed(path, version=VERSION, frames=fs+start,
        depth=observation["depth"] * observation.get("scale", 1.), K=observation["K"],
        R=np.asarray([world.R_cw(f) for f in fs]), C=centers,
        person=observation["person"], valid=observation["valid"], groups=observation["groups"],
        labels=observation["labels"], names=observation["names"])
    return path


def plane_proposals(bundle, minimum=40):
    """Measured planar patches as editable meshes, separated by semantic component.

    Retain openings by using each component's convex hull only when it is mostly filled.
    Non-planar furniture/vegetation remains the existing SAM 2 environment's job.
    """
    candidates = []
    for f, depth, K, R, C, valid, person, labels in zip(bundle["frames"], bundle["depth"], bundle["K"], bundle["R"],
                                                      bundle["C"], bundle["valid"], bundle["person"], bundle["labels"]):
        xyz = points(depth, K) @ R + C
        valid = valid & ~person & np.isfinite(xyz).all(-1) & (depth > .1) & (depth < 80)
        for label in np.unique(labels[valid]):
            if label < 0 or label >= len(bundle["names"]): continue
            name = str(bundle["names"][label])
            if name not in SUPPORT | WALL | {"ceiling"}: continue
            count, components = cv2.connectedComponents((valid & (labels == label)).astype(np.uint8))
            for component in range(1, count):
                mask = components == component
                if mask.sum() < minimum: continue
                yy, xx = np.nonzero(mask)
                hull = cv2.convexHull(np.c_[xx, yy].astype(np.int32))
                if cv2.contourArea(hull) > 1.3 * mask.sum(): continue
                cloud = xyz[mask][::max(1, int(mask.sum())//2000)]
                center = np.median(cloud, axis=0)
                _, _, vt = np.linalg.svd(cloud-center, full_matrices=False)
                distance = (cloud-center) @ vt[2]
                if np.percentile(np.abs(distance), 85) > max(.08, .025*np.median(depth[mask])): continue
                local = (cloud-center) @ vt[:2].T
                hull = cv2.convexHull(local.astype(np.float32)).reshape(-1, 2)
                vertices = hull @ vt[:2] + center
                # Trim extreme hull points rather than creating huge surfaces at depth outliers.
                if np.max(np.linalg.norm(vertices-center, axis=1)) > 40: continue
                candidates.append({"name": "evidence_%s_f%d_c%d" % (name.replace(" ", "_"), f, component),
                    "v": vertices.tolist(), "f": [list(range(len(vertices)))], "colour": "set", "role": "bg",
                    "_evidence": {"frame": int(f), "semantic": name, "samples": int(mask.sum())}})
    return sorted(candidates, key=lambda m: -m["_evidence"]["samples"])[:60]

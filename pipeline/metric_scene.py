"""Metric scene evidence: MoGe-2 depth (metric_observe.py) + what each pixel is
(floor, wall, ceiling, person, sky...) from Mask2Former panoptic's name maps (ade_find.py, run once per clip).

observe() caches the depth in metric_observation<shot>.npz in the clip's work folder.  view_frames() are the frames
it looks at (the scene finder looks at them too).  align() / camera_height() give gravity and the camera height; body_scale() is the
fallback scale when SAM 3D Body is not calibrated.  All arrays use OpenCV cameras and the pipeline's z-up world, in metres.
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
import moge_model


VERSION = 1
FLOOR = {"floor", "ground", "road", "sidewalk", "earth", "grass", "field", "path", "rug", "stage"}
WALL = {"wall", "building", "house", "skyscraper"}


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


def view_frames(n, c):
    """the shot frames (0 .. n-1) the depth step looks at"""
    count = min(n, max(1, int(c.get("metric_max_views", 3))))
    return np.unique(np.linspace(0, n-1, count, dtype=int))


@stage("metric_observation")
def observe(frames, pose, cam, y0, camj, c, labels_at):
    """MoGe depth of the view frames (an isolated GPU worker, cached by the actual images and intrinsics).
    labels_at(shot frame, width, height) -> (name map, names): Mask2Former panoptic's name of every pixel"""
    fs = view_frames(len(frames), c)
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
                source = {k: cached[k].copy() for k in cached.files if k not in ("labels", "names")}
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
                source = {k: result[k].copy() for k in result.files if k not in ("images", "labels", "names")}
            np.savez_compressed(path, key=digest, **source)
    source["frames"] = fs
    mask_bundle = None
    if pose.get('tracker_backend') == 'sam3' and c.get('_marks_dir'):
        from sam3_tracking import read_masks,union_at
        path = os.path.join(c['_marks_dir'],'sam3_masks.npz')
        if os.path.isfile(path): mask_bundle = read_masks(path)
    Ks, people, groups = [], [], []
    for f in fs:
        label, names = labels_at(int(f), width, height)
        K = cam.K().copy(); K[0, 0] *= zoom[f]; K[1, 1] *= zoom[f]
        K[0] *= width / cam.W; K[1] *= height / cam.H
        Ks.append(K)
        mask = person_mask(pose, f, (height, width), y0, (cam.W, cam.H))
        if mask_bundle is not None:
            ids = {int(p['sam3_id']) for p in pose.get('people',[]) if 'sam3_id' in p}
            mask = union_at(mask_bundle,int(f)+c.get('_shot_start',0),width,height,ids)
        group = np.full(label.shape, 3, np.int8)
        for i, name in enumerate(names):
            if name in FLOOR: group[label == i] = 0
            if name in WALL: group[label == i] = 1
            if name == "ceiling": group[label == i] = 2
            if name == "person": mask |= label == i
            if name in ("sky", "water", "sea", "mirror", "windowpane"):
                source["valid"][len(groups)][label == i] = False
        people.append(mask); groups.append(group)
    source.update(K=np.asarray(Ks), person=np.asarray(people), groups=np.asarray(groups), source=moge_model.NAME)
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
        safe_depth = np.where(valid & np.isfinite(depth),depth,0.)
        xyz = points(safe_depth, K)
        n = np.cross(np.gradient(xyz, axis=1), np.gradient(xyz, axis=0))
        norm = np.linalg.norm(n, axis=-1)
        n /= np.maximum(norm[..., None], 1e-9)
        # Discontinuities at people and depth edges do not supply orientation evidence.
        edge = np.maximum(np.abs(np.gradient(safe_depth, axis=0)), np.abs(np.gradient(safe_depth, axis=1)))
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
    for key,values in c.get('_sam3d_body',{}).items():
        if not key.endswith('_j'): continue
        for f,joints in zip(c['_sam3d_body'][key[:-2]+'_f'],values):
            vector = (joints[[5,6]].mean(0)-joints[[11,12]].mean(0)) @ np.asarray(camj['R'][int(f)])
            length = np.linalg.norm(vector)
            if length>.1 and vector @ prior/length>.85: upright.append(vector/length)
    up = solve_up(np.concatenate(normals), np.concatenate(categories), prior, upright)
    # MoGe scale is calibrated from the recovered camera height and the Mask2Former floor.
    ground = np.concatenate(floors); ground_scale = None
    if len(ground) >= 30:
        offsets = -(ground @ up)
        offsets = offsets[offsets > .2]
        if len(offsets) >= 30:
            med = np.median(offsets)
            if np.median(np.abs(offsets-med)) / med < .2:
                ground_scale = height / med
    observation["scale"] = float(ground_scale or 1.)
    observation["scale_source"] = "Mask2Former floor / fitted camera height" if ground_scale else "MoGe metric prior"
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
    return up, {"source": "%s + Mask2Former floor/ceiling/wall normals" % observation.get("source", "MoGe"), "world_yaw_deg": math.degrees(yaw),
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


def camera_height(observation, camj, up):
    offsets = []
    for f,d,K,valid,person,groups in zip(observation['frames'],observation['depth'],observation['K'],
                                        observation['valid'],observation['person'],observation['groups']):
        mask = valid & ~person & (groups==0) & np.isfinite(d) & (d>.05) & (d<100)
        if mask.sum()<30: continue
        xyz = points(np.where(mask,d,0),K)[mask]*observation.get('scale',1.)
        center = np.asarray(camj.get('travel_cam0',np.zeros((len(camj['R']),3)))[int(f)])
        values = -((xyz @ np.asarray(camj['R'][int(f)])+center) @ up)
        values = values[np.isfinite(values)&(values>.2)&(values<12)]
        if len(values)>=30: offsets.extend(values[::8])
    if len(offsets)<30: return None
    median = float(np.median(offsets))
    return median if np.median(np.abs(np.asarray(offsets)-median))<.15*median else None

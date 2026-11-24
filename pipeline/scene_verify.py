"""Evidence-based scene checks and bounded local repair, without an API or SAM 3."""
import argparse
import copy
import glob
import hashlib
import os

import cv2
import numpy as np

from common import cfg, jload, jsave
from profiling import stage


def triangles(faces):
    return np.asarray([(face[0], face[i], face[i+1]) for face in faces for i in range(1, len(face)-1)], int).reshape(-1, 3)


def _clip_near(poly, near=.01):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        if a[2] >= near: out.append(a)
        if (a[2] >= near) != (b[2] >= near):
            out.append(a + (b-a) * ((near-a[2]) / (b[2]-a[2])))
    return np.asarray(out)


def rasterize(vertices, faces, K, R, C, shape):
    """Near-clipped, perspective-correct software z-buffer, OpenCV camera convention."""
    camera = (np.asarray(vertices, float)-C) @ R.T
    depth = np.full(shape, np.inf)
    for face in triangles(faces):
        poly = camera[face]
        if not np.isfinite(poly).all(): continue
        poly = _clip_near(poly)
        for i in range(1, len(poly)-1):
            xyz = poly[[0, i, i+1]]
            uv = xyz @ K.T; uv = uv[:, :2] / uv[:, 2:]
            low = np.maximum(np.floor(uv.min(0)), 0).astype(int)
            high = np.minimum(np.ceil(uv.max(0)), [shape[1]-1, shape[0]-1]).astype(int)
            if np.any(high < low): continue
            x, y = np.meshgrid(np.arange(low[0], high[0]+1)+.5, np.arange(low[1], high[1]+1)+.5)
            a, b, c = uv
            den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(den) < 1e-9: continue
            u = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
            v = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
            w = 1-u-v
            inside = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
            inverse = u/xyz[0, 2] + v/xyz[1, 2] + w/xyz[2, 2]
            value = np.full_like(inverse, np.inf)
            np.divide(1, inverse, out=value, where=inside & (inverse > 0))
            region = depth[low[1]:high[1]+1, low[0]:high[0]+1]
            np.minimum(region, value, out=region)
    return depth


def inside_mesh(point, vertices, faces):
    """Ray parity for closed meshes; open measured planes have no enclosed volume."""
    vertices = np.asarray(vertices, float)
    if np.any(point <= vertices.min(0)+.01) or np.any(point >= vertices.max(0)-.01): return False
    edges = {}
    for face in faces:
        for a, b in zip(face, face[1:]+face[:1]):
            key = tuple(sorted((a, b))); edges[key] = edges.get(key, 0)+1
    if not edges or any(n != 2 for n in edges.values()): return False
    tri = vertices[triangles(faces)]
    direction = np.array([1., .371, .193])
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    h = np.cross(direction, e2); det = np.einsum("ij,ij->i", e1, h)
    inv = np.divide(1, det, out=np.zeros_like(det), where=np.abs(det) > 1e-9)
    delta = point-tri[:, 0]; u = np.einsum("ij,ij->i", delta, h)*inv
    q = np.cross(delta, e1); v = (q @ direction)*inv
    t = np.einsum("ij,ij->i", e2, q)*inv
    hits = t[(np.abs(det) > 1e-9) & (u >= 0) & (v >= 0) & (u+v <= 1) & (t > 1e-7)]
    return len(np.unique(np.round(hits, 6))) % 2 == 1


def support_height(xy, vertices, faces, maximum):
    tri = np.asarray(vertices, float)[triangles(faces)]
    if not len(tri): return None
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    normal = np.cross(b-a, c-a)
    horizontal = np.abs(normal[:, 2]) > .7*np.linalg.norm(normal, axis=1)
    den = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0])+(c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
    valid = horizontal & (np.abs(den) > 1e-9)
    inv = np.divide(1, den, out=np.zeros_like(den), where=valid)
    u = ((b[:, 1]-c[:, 1])*(xy[0]-c[:, 0])+(c[:, 0]-b[:, 0])*(xy[1]-c[:, 1]))*inv
    v = ((c[:, 1]-a[:, 1])*(xy[0]-c[:, 0])+(a[:, 0]-c[:, 0])*(xy[1]-c[:, 1]))*inv
    z = u*a[:, 2]+v*b[:, 2]+(1-u-v)*c[:, 2]
    selected = z[valid & (u >= -1e-7) & (v >= -1e-7) & (u+v <= 1+1e-7) & (z <= maximum)]
    return float(selected.max()) if len(selected) else None


def compare_depth(observed, predicted, person, valid=None):
    valid = np.isfinite(observed) & (observed > .05) & (observed < 100) & ~person & (True if valid is None else valid)
    tolerance = np.maximum(.15, .05*observed)
    overlap = valid & np.isfinite(predicted)
    conflicts = overlap & (predicted < observed-tolerance)
    covered = overlap & (np.abs(predicted-observed) <= tolerance)
    return {"conflict_fraction": float(conflicts.sum()/max(overlap.sum(), 1)),
            "coverage_fraction": float(covered.sum()/max(valid.sum(), 1)), "pixels": int(overlap.sum()),
            "observed_pixels": int(valid.sum())}, conflicts


def load_evidence(data, work):
    """Reject measurements from a different camera/shot after a manual edit or stale run."""
    from build_env import Cam, cam_mats
    cam = Cam(data); accepted, skipped = [], []
    for path in sorted(glob.glob(os.path.join(work, "metric_evidence*.npz"))):
        with np.load(path, allow_pickle=False) as source:
            bundle = {k: source[k].copy() for k in source.files}
        good = True
        for f, K, R, C, depth in zip(bundle["frames"], bundle["K"], bundle["R"], bundle["C"], bundle["depth"]):
            if f < 0 or f >= data["frames"]: good = False; break
            Rb, center = cam_mats(data["cam"][f]); expected_R = np.diag([1., -1., -1.]) @ Rb.T
            h, w = depth.shape
            if not (np.allclose(C, center, atol=.005) and np.allclose(R, expected_R, atol=.001)
                    and np.isclose(K[0, 0], cam.f_at(f)*w/cam.W, rtol=.005)
                    and np.isclose(K[1, 1], cam.f_at(f)*h/cam.H, rtol=.005)):
                good = False; break
        if good: accepted.append(bundle)
        else: skipped.append(os.path.basename(path))
    return accepted, skipped


def _body(ch, f):
    from rig import Rig, JOINTS
    rig = Rig(ch["height"], ch.get("style", "sample")); root = ch["root"][f]
    joints = rig.fk(root[:3], root[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f])
    body = rig.coco_points(joints)
    # Sample along limbs too: endpoints alone miss a wall crossing a limb.
    bones = [(5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
    samples = np.vstack([body] + [body[a]+(body[b]-body[a])*t for a, b in bones for t in (.25, .5, .75)])
    feet = [joints["ankle_"+side][1]-[0, 0, rig.sole] for side in ("L", "R")]
    return samples, feet


def verify(data, work, justifications=None, bundles=None, resolution=160, raster_cache=None):
    issues, evidence, frames = [], [], set()
    if bundles is None: bundles, skipped = load_evidence(data, work)
    else: skipped = []
    def issue(kind, obj, frame, detail):
        key = "%s|%s|%s" % (kind, obj, frame)
        issues.append({"id": hashlib.sha1(key.encode()).hexdigest()[:12], "kind": kind,
                       "object": obj, "frame": int(frame), "detail": detail})
    meshes, mesh_keys = [], {}
    seen = {}
    for mesh in data.get("meshes", []):
        vertices = np.asarray(mesh.get("v", []), float)
        faces = mesh.get("f", [])
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all() or not faces:
            issue("invalid_geometry", mesh.get("name", "mesh"), 0, "Missing or non-finite mesh geometry"); continue
        tri = triangles(faces)
        if not len(tri):
            issue("invalid_geometry", mesh.get("name", "mesh"), 0, "Mesh has no polygon surfaces"); continue
        if tri.min() < 0 or tri.max() >= len(vertices):
            issue("invalid_geometry", mesh.get("name", "mesh"), 0, "Mesh face index is out of bounds"); continue
        key = hashlib.sha256(vertices.tobytes()+tri.tobytes()).hexdigest()
        if key in seen:
            issue("duplicate", mesh["name"], 0, "Same geometry as "+seen[key])
        seen[key] = mesh["name"]; meshes.append(mesh)
        mesh_keys[id(mesh)] = key
    # Global completeness cues supplement per-camera depth checks. These are review
    # flags, since lights, signs and suspended scenery may legitimately be isolated.
    bounds = [(np.min(m["v"], axis=0), np.max(m["v"], axis=0)) for m in meshes]
    adjacent = [set() for _ in meshes]
    for i, (low, high) in enumerate(bounds):
        for j in range(i):
            other_low, other_high = bounds[j]
            gap = np.maximum(np.maximum(low-other_high, other_low-high), 0)
            if np.linalg.norm(gap) < .12:
                adjacent[i].add(j); adjacent[j].add(i)
    anchored = {i for i,(low,high) in enumerate(bounds) if low[2] <= .1 and high[2] >= -.1}
    reached = set(anchored); pending = list(anchored)
    while pending:
        for j in adjacent[pending.pop()]-reached:
            reached.add(j); pending.append(j)
    if anchored:
        for i,mesh in enumerate(meshes):
            low,high = bounds[i]
            if i not in reached and low[2] > .15 and np.min(high-low) > .05:
                center = (low+high)/2
                f = min(range(len(data["cam"])), key=lambda f: np.linalg.norm(np.asarray(data["cam"][f][:3])-center))
                issue("isolated_geometry", mesh["name"], f,
                      "No contact path to measured ground; inspect floating or intentionally suspended geometry")
    for bundle in bundles:
        for f, obs, K, R, C, person, valid in zip(bundle["frames"], bundle["depth"], bundle["K"], bundle["R"], bundle["C"], bundle["person"], bundle["valid"]):
            f = int(f); frames.add(f)
            h, w = obs.shape; scale = min(1., resolution/max(h, w))
            shape = (max(1, round(h*scale)), max(1, round(w*scale)))
            obs = cv2.resize(obs, shape[::-1], interpolation=cv2.INTER_NEAREST)
            person = cv2.resize(person.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            valid = cv2.resize(valid.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            K = K.copy(); K[0] *= shape[1]/w; K[1] *= shape[0]/h
            combined = np.full(shape, np.inf)
            for mesh in meshes:
                key = (mesh_keys[id(mesh)], f, shape)
                predicted = raster_cache.get(key) if raster_cache is not None else None
                if predicted is None:
                    predicted = rasterize(mesh["v"], mesh["f"], K, R, C, shape)
                    if raster_cache is not None and len(raster_cache) < 128:
                        raster_cache[key] = predicted
                measured, _ = compare_depth(obs, predicted, person, valid)
                if measured["pixels"] >= 20 and measured["conflict_fraction"] > .1:
                    issue("occlusion", mesh["name"], f, "%.0f%% of overlapping pixels lie in front of observed surfaces" % (100*measured["conflict_fraction"]))
                np.minimum(combined, predicted, out=combined)
            measured, _ = compare_depth(obs, combined, person, valid)
            evidence.append(dict(frame=f, **measured))
            if measured["observed_pixels"] >= 20 and measured["coverage_fraction"] < .7:
                issue("coverage", "environment", f, "%.0f%% of static depth evidence is covered" % (100*measured["coverage_fraction"]))
    if not frames:
        frames = set(np.linspace(0, max(0, data["frames"]-1), min(5, data["frames"]), dtype=int).tolist())
    # Include each character's own visible endpoints: sparse keyframes can miss short tracks.
    for name, ch in data.get("chars", {}).items():
        visible = np.flatnonzero(ch.get("visible", [True]*len(ch["root"])))
        selected = frames | set(visible[np.linspace(0, len(visible)-1, min(3, len(visible)), dtype=int)].tolist())
        if not .4 <= ch["height"] <= 2.5: issue("human_height", name, 0, "Height %.2f m requires review" % ch["height"])
        for f in sorted(selected):
            if f >= len(ch["root"]) or (ch.get("visible") and not ch["visible"][f]): continue
            body, feet = _body(ch, f)
            for mesh in meshes:
                if any(inside_mesh(p, mesh["v"], mesh["f"]) for p in body):
                    issue("penetration", mesh["name"], f, "Body or limb samples of %s intersect this object" % name)
            gaps = []
            for foot in feet:
                supports = [support_height(foot[:2], m["v"], m["f"], foot[2]+.15) for m in meshes]
                supports = [z for z in supports if z is not None]
                gaps.append(foot[2]-max(supports) if supports else float("inf"))
            if min(gaps) > .08:
                issue("foot_support", name, f, "Neither foot has measured support within 8 cm; inspect jumping, riding or missing geometry")
    justifications = justifications or {}
    for row in issues:
        explanation = justifications.get(row["id"])
        row["status"] = "justified" if isinstance(explanation, str) and explanation.strip() else "unresolved"
        if row["status"] == "justified": row["justification"] = explanation.strip()[:1000]
    return {"version": 1, "issues": issues, "evidence": evidence, "skipped_stale_evidence": skipped,
            "unresolved": [r["id"] for r in issues if r["status"] == "unresolved"],
            "depth_checked": bool(evidence), "human_geometry": "fitted mannequin proxy"}


def scale_track(ch, cameras, factor):
    """Equation 10: one constant scale about the corresponding camera at every frame."""
    if not np.isfinite(factor) or factor <= 0: raise ValueError("human scale must be positive and finite")
    from build_env import cam_mats
    for f, root in enumerate(ch["root"]):
        _, C = cam_mats(cameras[f])
        root[:3] = (C+factor*(np.asarray(root[:3])-C)).tolist()
        if "root_z_fit" in ch:
            ch["root_z_fit"][f] = float(C[2]+factor*(ch["root_z_fit"][f]-C[2]))
    ch["height"] *= factor
    ch["pelvis_z"] = (np.asarray(ch["pelvis_z"])*factor).tolist()
    ch["metric_scale"] = ch.get("metric_scale", 1.)*factor


def place_people(data, hints):
    """Explicit constant scale/height or named support placement; no guessing at jumps."""
    from build_env import cam_mats
    changes = []
    for name, ch in data.get("chars", {}).items():
        hint = (hints.get("people") or {}).get(str(ch.get("track_id")), {})
        requested = hint.get("metric_scale")
        if "metric_height" in hint:
            requested = float(hint["metric_height"]) / ch["height"] * ch.get("metric_scale", 1.)
        support = hint.get("support_object")
        if support and requested is None:
            meshes = [m for m in data["meshes"] if m["name"] == support]
            ratios = []
            for f in np.flatnonzero(ch.get("visible", [True]*len(ch["root"])))[::max(1, len(ch["root"])//12)]:
                _, feet = _body(ch, f); _, C = cam_mats(data["cam"][f])
                for foot in feet:
                    tops = [support_height(foot[:2], m["v"], m["f"], C[2]-.05) for m in meshes]
                    tops = [z for z in tops if z is not None]
                    if tops and abs(foot[2]-C[2]) > .1:
                        ratios.append((max(tops)-C[2])/(foot[2]-C[2]))
            from metric_scene import robust_scale
            factor = robust_scale(ratios)
            if factor is not None: requested = factor*ch.get("metric_scale", 1.)
        if requested is None: continue
        requested = float(requested)
        delta = requested/ch.get("metric_scale", 1.)
        if not np.isfinite(delta) or not .4 <= ch["height"]*delta <= 2.5:
            raise ValueError("metric scale for %s gives an implausible human height" % name)
        if abs(delta-1) < 1e-6: continue
        scale_track(ch, data["cam"], delta)
        scale_attached_props(data, ch, delta)
        changes.append({"character": name, "scale": requested, "source": "support_object" if support else "explicit height/scale"})
    return changes


def scale_attached_props(data, ch, factor):
    from build_env import cam_mats
    for prop in data.get("props", []):
        if str(prop.get("track_id")) != str(ch.get("track_id")): continue
        for f, key in enumerate(prop.get("keys", [])):
            _, C = cam_mats(data["cam"][f])
            key[:3] = (C+factor*(np.asarray(key[:3])-C)).tolist()
            key[6:9] = (np.asarray(key[6:9])*factor).tolist()


def support_scale(data, ch):
    """Small constant scale suggested by sustained support, excluding likely airborne motion."""
    from build_env import cam_mats
    from metric_scene import robust_scale
    visible = np.flatnonzero(ch.get("visible", [True]*len(ch["root"])))
    if len(visible) < 3: return None
    selected = visible[np.linspace(0, len(visible)-1, min(8, len(visible)), dtype=int)]
    ratios, heights = [], []
    for f in selected:
        _, feet = _body(ch, f); _, C = cam_mats(data["cam"][f])
        foot = min(feet, key=lambda p: p[2]); heights.append(foot[2])
        supports = [support_height(foot[:2], m["v"], m["f"], C[2]-.1) for m in data["meshes"]]
        supports = [z for z in supports if z is not None and abs(z-foot[2]) < .3]
        if supports and abs(foot[2]-C[2]) > .2:
            ratio = (max(supports)-C[2])/(foot[2]-C[2])
            if ratio > 0: ratios.append(ratio)
    if np.ptp(heights) > .12 or len(ratios) < .75*len(selected): return None
    if np.std(np.log(ratios)) > .03: return None
    factor = robust_scale(ratios)
    return factor if factor is not None and .9 <= factor <= 1.1 and abs(factor-1) > .01 else None


def _score(result):
    e = result["evidence"]
    coverage = np.mean([r["coverage_fraction"] for r in e]) if e else 0.
    conflict = np.mean([r["conflict_fraction"] for r in e]) if e else 0.
    body = sum(r["kind"] in ("penetration", "foot_support") for r in result["issues"])
    return float(coverage-2*conflict-.05*body)


def depth_correction(group, bundles):
    """A bounded translation of a whole primitive toward its measured surface."""
    shifts = []
    for bundle in bundles:
        for obs,K,R,C,person,valid in zip(bundle['depth'],bundle['K'],bundle['R'],bundle['C'],bundle['person'],bundle['valid']):
            h,w = obs.shape; shape = (max(1,round(h*min(1,160/max(h,w)))),max(1,round(w*min(1,160/max(h,w)))))
            K = K.copy(); K[0] *= shape[1]/w; K[1] *= shape[0]/h
            obs = cv2.resize(obs,shape[::-1],interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize((valid & ~person).astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
            predicted = np.full(shape,np.inf)
            for mesh in group:
                np.minimum(predicted,rasterize(mesh['v'],mesh['f'],K,R,C,shape),out=predicted)
            overlap = mask & np.isfinite(predicted) & np.isfinite(obs) & (obs>.05) & (obs<100)
            conflict = overlap & (predicted < obs-np.maximum(.15,.05*obs))
            if conflict.sum() >= 20 and conflict.sum() > .15*overlap.sum():
                shifts.append(np.median((obs-predicted)[conflict])*R[2])
    if not shifts: return None
    shift = np.median(shifts,axis=0); length = np.linalg.norm(shift)
    if length < .05 or np.median(np.linalg.norm(np.asarray(shifts)-shift,axis=1)) > .5*length: return None
    return shift*min(1.,2./length)


@stage("scene_verification")
def process(data, work, c, hints=None, repair=False):
    """Verify edits, keep only evidence-improving proposals, persist all unresolved issues."""
    hints = hints or {}; bundles, skipped = load_evidence(data, work)
    justification = hints.get("scene_justifications", {})
    scale_changes = place_people(data, hints)
    cache = {}
    result = verify(data, work, justification, bundles=bundles, raster_cache=cache)
    repairs = []
    if bundles:
        from metric_scene import plane_proposals
        proposals = [p for b in bundles for p in plane_proposals(b)]
        jsave(os.path.join(work, "scene_proposals.json"), {"meshes": proposals})
    else:
        proposals = []
    if repair and bundles:
        groups = {}
        conflicting = {i['object'] for i in result['issues'] if i['kind']=='occlusion'}
        for mesh in data['meshes']:
            groups.setdefault(mesh.get('_primitive',mesh['name']),[]).append(mesh)
        attempted = 0
        for group in groups.values():
            if not any(m['name'] in conflicting for m in group): continue
            if any(m['name'].startswith('ground') or '_evidence' in m for m in group): continue
            if attempted >= max(0,int(c.get('scene_repair_limit',4))): break
            attempted += 1
            shift = depth_correction(group,bundles)
            if shift is None: continue
            replaced = {id(m): dict(m,v=(np.asarray(m['v'])+shift).tolist()) for m in group}
            trial = dict(data,meshes=[replaced.get(id(m),m) for m in data['meshes']])
            checked = verify(trial,work,justification,bundles=bundles,raster_cache=cache)
            old_bad = {(i['kind'],i['object'],i['frame']) for i in result['issues'] if i['kind']!='coverage'}
            new_bad = {(i['kind'],i['object'],i['frame']) for i in checked['issues'] if i['kind']!='coverage'}
            old_coverage = np.mean([e['coverage_fraction'] for e in result['evidence']])
            new_coverage = np.mean([e['coverage_fraction'] for e in checked['evidence']])
            if _score(checked)>_score(result)+.01 and new_coverage>=old_coverage-.02 and not (new_bad-old_bad):
                data['meshes']=trial['meshes']; result=checked
                repairs.append({'object':group[0].get('_primitive',group[0]['name']),
                                'action':'depth-aligned primitive translation','translation':shift.tolist()})
        existing = {m["name"] for m in data["meshes"]}
        for proposal in proposals[:max(0, int(c.get("scene_repair_limit", 4)))]:
            if proposal["name"] in existing: continue
            trial = dict(data, meshes=data["meshes"]+[proposal])
            checked = verify(trial, work, justification, bundles=bundles, raster_cache=cache)
            old_bad = {(r["kind"], r["object"], r["frame"]) for r in result["issues"] if r["kind"] != "coverage"}
            new_bad = {(r["kind"], r["object"], r["frame"]) for r in checked["issues"] if r["kind"] != "coverage"}
            if _score(checked) > _score(result)+.005 and not (new_bad-old_bad):
                data["meshes"].append(proposal); result = checked
                repairs.append({"object": proposal["name"], "action": "added measured planar surface"})
    if repair:
        for name, ch in list(data.get("chars", {}).items()):
            hint = (hints.get("people") or {}).get(str(ch.get("track_id")), {})
            # Explicit choices and mounted people are never superseded by inferred support.
            if any(k in hint for k in ("metric_height", "metric_scale", "support_object", "mount", "keep_height")): continue
            factor = support_scale(data, ch)
            if factor is None: continue
            trial = dict(data, chars=dict(data["chars"]))
            candidate = copy.deepcopy(ch); trial["chars"][name] = candidate
            scale_track(candidate, data["cam"], factor)
            checked = verify(trial, work, justification, bundles=bundles, raster_cache=cache)
            old_bad = {(i["kind"],i["object"],i["frame"]) for i in result["issues"] if i["kind"] != "foot_support"}
            new_bad = {(i["kind"],i["object"],i["frame"]) for i in checked["issues"] if i["kind"] != "foot_support"}
            if _score(checked) > _score(result)+.01 and not (new_bad-old_bad):
                data["chars"][name] = candidate
                scale_attached_props(data, candidate, factor)
                result = checked
                repairs.append({"object": name, "action": "constant support scale", "factor": factor})
    result.update(repairs=repairs, human_scale_changes=scale_changes, skipped_stale_evidence=skipped)
    overview(data, os.path.join(work,"scene_overview.jpg"))
    jsave(os.path.join(work, "scene_verification.json"), result)
    data["scene_verification"] = {"file": "scene_verification.json", "unresolved": len(result["unresolved"]),
                                   "depth_checked": result["depth_checked"], "repairs": len(repairs)}
    return result


def overview(data, path):
    """Three global inspection views per shot, independent of the source camera."""
    from common import imwrite_u
    shots = data.get("shots") or [{"frames":[0,data["frames"]-1]}]
    rows = []
    origins = [np.asarray(data["cam"][s["frames"][0]][:3]) for s in shots]
    valid_meshes = [m for m in data.get("meshes",[]) if np.asarray(m.get("v",[])).ndim == 2
                   and np.asarray(m["v"]).shape[1] == 3 and np.isfinite(m["v"]).all()]
    for si,shot in enumerate(shots[:6]):
        meshes = [m for m in valid_meshes if len(m["v"]) and
                  int(np.argmin([np.linalg.norm(np.mean(m["v"],axis=0)-C) for C in origins])) == si]
        if not meshes: continue
        cloud = np.concatenate([np.asarray(m["v"],float) for m in meshes])
        if not np.isfinite(cloud).all(): continue
        panels = []
        for label, axes in (("Top",np.array([[1,0,0],[0,1,0]])),
                            ("Front",np.array([[1,0,0],[0,0,1]])),
                            ("Oblique",np.array([[.707,-.707,0],[-.35,-.35,.866]]))):
            flat = cloud @ axes.T
            low,high = np.percentile(flat,[1,99],axis=0)
            span = np.maximum(high-low,.5); scale = min(280/span[0],220/span[1])
            center = (low+high)/2
            panel = np.full((280,320,3),245,np.uint8)
            def project(vertices):
                xy = (np.asarray(vertices) @ axes.T-center)*scale
                xy = np.clip(xy, -10000,10000)
                return np.rint(xy*[1,-1]+[160,150]).astype(np.int32)
            for mesh in meshes:
                xy = project(mesh["v"])
                for face in mesh["f"]:
                    if len(face)>=3 and min(face)>=0 and max(face)<len(xy):
                        cv2.polylines(panel,[xy[face]],True,(150,150,150),1,cv2.LINE_AA)
            for ch in data.get("chars",{}).values():
                a,b = shot["frames"]
                fs = [f for f in range(a,min(b+1,len(ch["root"]))) if ch.get("visible",[True]*len(ch["root"]))[f]]
                if fs:
                    body,_ = _body(ch,fs[len(fs)//2])
                    for p in project(body[:17]): cv2.circle(panel,tuple(p),2,(65,65,210),-1)
            cv2.putText(panel,"Shot %d - %s"%(si+1,label),(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1,cv2.LINE_AA)
            panels.append(panel)
        rows.append(np.hstack(panels))
    if rows: imwrite_u(path,np.vstack(rows),88)


def feedback(work, start=0, end=None):
    path = os.path.join(work, "scene_verification.json")
    if not os.path.isfile(path): return "No numerical verification yet."
    result = jload(path)
    issues = [i for i in result["issues"] if i["status"] == "unresolved" and i["frame"] >= start and (end is None or i["frame"] <= end)]
    return "Numerical evidence checks (fitted human proxy, estimated depth):\n" + "\n".join(
        "%s: %s f%d: %s" % (i["kind"], i["object"], i["frame"], i["detail"]) for i in issues[:30])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args(); c = cfg()
    work = os.path.join(c["work_root"], args.clip); path = os.path.join(work, "scene_data.json")
    data = jload(path)
    result = process(data, work, c, repair=args.repair)
    jsave(path, data)
    print("Scene verification: %d unresolved issues, %d repairs, depth checked: %s" %
          (len(result["unresolved"]), len(result["repairs"]), result["depth_checked"]))


if __name__ == "__main__":
    main()

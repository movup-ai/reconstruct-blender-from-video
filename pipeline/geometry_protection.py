"""Bounded scene corrections that preserve people visible in the source video."""
import cv2
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=4)
def character_parts(style):
    from common import AGENT, jload
    import os
    asset = {'sample':'sample_character.json', 'base21':'base_character_v21.json', 'master':'human_master.json'}.get(style)
    return jload(os.path.join(AGENT,'assets',asset))['parts'] if asset else None


def rendered_people_depth(data, frame, K, R, C, shape):
    """Rasterize the same local character parts used by Blender, without starting Blender."""
    from rig import Rig, JOINTS
    from scene_verify import rasterize
    import props
    result = np.full(shape,np.inf)
    for ch in data.get('chars',{}).values():
        if not ch.get('visible',[True]*data['frames'])[frame]: continue
        rig=Rig(ch['height'],ch.get('style','sample')); root=ch['root'][frame]
        joints=rig.fk(root[:3],root[3],{j:ch['rot'][j][frame] for j in JOINTS},ch['pelvis_z'][frame])
        parts=character_parts(rig.style)
        if parts:
            for name, part in parts.items():
                if name not in joints: continue
                rotation,center=joints[name]
                vertices=np.asarray(part['v'])*rig.s @ rotation.T+center
                np.minimum(result,rasterize(vertices,part['f'],K,R,C,shape),out=result)
        else:
            # A conservative limb proxy for alternative rigs without extracted parts.
            for name,parent,offset in rig.J:
                if parent=='root': continue
                if np.linalg.norm(joints[parent][1]-joints[name][1]) < 1e-6: continue
                vertices,faces=props.bar(joints[parent][1],joints[name][1],.1*rig.s)
                np.minimum(result,rasterize(vertices,faces,K,R,C,shape),out=result)
    return result


def person_conflict(predicted, depth, person, valid=None, rendered=None):
    observed = person & np.isfinite(depth) & (depth > .05) & (depth < 100)
    if valid is not None: observed &= valid
    covered = observed & np.isfinite(predicted)
    wrong = covered & (predicted < depth - np.maximum(.15, .06 * depth))
    if rendered is not None:
        wrong |= covered & np.isfinite(rendered) & (predicted < rendered-.03)
    return int(wrong.sum()), int(observed.sum()), wrong


def protect(data, bundles, max_shift=3.):
    """Try a small depth correction; suppress a group if it still hides a source-visible person.

    Genuine source occlusion is retained: the source person mask includes visible human pixels only.
    Dynamic props are rendered separately and are never treated as background geometry here.
    """
    from scene_verify import rasterize
    if not bundles or not data.get('chars'): return []
    observations = []
    for bundle in bundles:
        for f, depth, K, R, C, person, valid in zip(bundle['frames'], bundle['depth'], bundle['K'], bundle['R'],
                                                   bundle['C'], bundle['person'], bundle['valid']):
            if person.sum() < 10: continue
            h, w = depth.shape; scale = min(1., 160/max(h, w)); shape = (max(1, round(h*scale)), max(1, round(w*scale)))
            k = np.asarray(K).copy(); k[0] *= shape[1]/w; k[1] *= shape[0]/h
            rendered = rendered_people_depth(data,int(f),k,np.asarray(R),np.asarray(C),shape)
            observations.append((int(f), cv2.resize(depth, shape[::-1], interpolation=cv2.INTER_NEAREST), k,
                np.asarray(R), np.asarray(C), cv2.resize(person.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST)>0,
                cv2.resize(valid.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST)>0, rendered))
    if not observations: return []
    groups = {}
    for mesh in data['meshes']:
        if mesh['name'].startswith('ground'): continue
        groups.setdefault(mesh.get('_primitive', mesh['name']), []).append(mesh)
    actions, removed, replacements = [], set(), {}
    for name, group in groups.items():
        def measure(parts):
            total, gaps, wrong_views = 0, [], []
            for f, depth, K, R, C, person, valid, rendered in observations:
                predicted = np.full(depth.shape, np.inf)
                for mesh in parts:
                    if mesh.get('visible_frames') and not mesh['visible_frames'][0] <= f <= mesh['visible_frames'][1]: continue
                    np.minimum(predicted, rasterize(mesh['v'], mesh['f'], K, R, C, depth.shape), out=predicted)
                n, count, bad = person_conflict(predicted, depth, person, valid, rendered)
                if n >= max(8, .025 * count):
                    total += n; wrong_views.append(f)
                    target=np.where(np.isfinite(rendered),np.maximum(depth,rendered),depth)
                    gaps.append((float(np.percentile((target-predicted)[bad], 90))+.2, R[2]))
            return total, gaps, wrong_views
        score, gaps, frames = measure(group)
        if not score: continue
        # Preserve the footprint and physical dimensions when correcting depth.
        distance, forward = max(gaps, key=lambda g: g[0])
        shift = np.asarray(forward) * min(max_shift, max(.2, distance))
        shifted = [dict(m, v=(np.asarray(m['v'], float)+shift).tolist()) for m in group]
        after, _, _ = measure(shifted)
        if after == 0 and distance <= max_shift:
            for before, new in zip(group, shifted): replacements[id(before)] = new
            actions.append({'object': name, 'action': 'moved behind source-visible people', 'translation': shift.tolist(), 'frames': frames})
        else:
            removed.update(id(m) for m in group)
            saved = {'object': name, 'reason': 'would hide source-visible people', 'frames': frames, 'meshes': group}
            prior = [x for x in data.get('suppressed_geometry', []) if x['object'] != name]
            data['suppressed_geometry'] = prior + [saved]
            actions.append({k: v for k, v in saved.items() if k != 'meshes'})
    data['meshes'] = [replacements.get(id(m), m) for m in data['meshes'] if id(m) not in removed]
    data['geometry_protection'] = data.get('geometry_protection', []) + actions
    return actions


def match_source_box(group, source, data):
    """Shrink an automatic primitive that projects beyond its measured image footprint."""
    from build_env import Cam
    box, f = source.get('px'), source.get('frame')
    if not group or box is None or f is None or len(box) != 4: return group
    cam = Cam(data)
    points = np.concatenate([np.asarray(m['v'], float) for m in group])
    # Cam.project returns pixel coordinates and camera depth.
    uv, depth = cam.project(points, int(f))
    uv = uv[depth > .05]
    if len(uv) < 3: return group
    extent = np.ptp(uv, axis=0)
    target = np.array([box[2]-box[0], box[3]-box[1]], float)
    factor = min(1., float(np.min(target / np.maximum(extent, 1.))) * 1.08)
    if factor >= .9: return group
    center = (points.min(0)+points.max(0))/2
    if any(m.get('prototype') for m in group): center[2] = points[:,2].min()
    return [dict(m, v=(center+(np.asarray(m['v'])-center)*max(.01, factor)).tolist(),
                 _footprint_scale=float(factor)) for m in group]

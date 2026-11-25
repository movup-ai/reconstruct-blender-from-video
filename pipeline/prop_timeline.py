"""Independent prop identities with explicit ownership events and sparse spatial keyframes."""
import math
import re

import numpy as np

STATES = ('held', 'worn', 'free', 'resting', 'hidden')
JOINTS = ('left', 'right', 'both', 'back')
SHAPES = ('box', 'flat', 'cyl', 'ball', 'bag', 'rod')


def finite_vector(value, length, label):
    try: a = np.asarray(value, dtype=float)
    except (TypeError, ValueError): raise ValueError(label + ' must contain numbers') from None
    if a.shape != (length,) or not np.isfinite(a).all():
        raise ValueError(label + ' must contain %d finite numbers' % length)
    return a.tolist()


def validate(objects, frames=None):
    if not isinstance(objects, list) or len(objects) > 40:
        raise ValueError('Prop timeline must be a list of at most 40 objects')
    result, ids = [], set()
    def frame(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or int(v) != v:
            raise ValueError('Timeline frames must be whole numbers')
        if v < 0 or v >= (frames if frames is not None else 10_000_000):
            raise ValueError('Timeline frame is outside this clip')
        return int(v)
    for item in objects:
        if not isinstance(item, dict): raise ValueError('Each prop must be an object')
        pid = str(item.get('id', ''))
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', pid) or pid in ids:
            raise ValueError('Each prop needs a unique ID using letters, numbers, _ or -')
        ids.add(pid)
        name = str(item.get('name', '')).strip()[:120]
        if not name: raise ValueError('Name each prop')
        shape = item.get('shape', 'box')
        if shape not in SHAPES: raise ValueError('Unknown prop prototype')
        size = item.get('size_m')
        if size is not None:
            size = finite_vector(size, 3, 'Prop dimensions (width, height, depth)')
            if min(size) <= 0 or max(size) > 20: raise ValueError('Prop dimensions must be between 0 and 20 metres')
        events, used = [], set()
        if not isinstance(item.get('events', []), list) or len(item.get('events', [])) > 200:
            raise ValueError('Each prop supports up to 200 events')
        for event in item.get('events', []):
            if not isinstance(event, dict): raise ValueError('Each event must be an object')
            f = frame(event.get('frame'))
            if f in used: raise ValueError('Only one event per prop per frame is allowed')
            used.add(f)
            state = event.get('state')
            if state not in STATES: raise ValueError('Unknown prop state')
            owner = event.get('person')
            joint = event.get('joint', 'right')
            if joint not in JOINTS: raise ValueError('Choose left, right, both hands or back')
            if state in ('held', 'worn'):
                if isinstance(owner, bool) or not str(owner).isdigit() or int(owner) < 1:
                    raise ValueError('Choose a character for held or worn objects')
                owner = int(owner)
            else: owner = None
            events.append({'frame': f, 'state': state, 'person': owner, 'joint': joint})
        keys, used = [], set()
        if not isinstance(item.get('keyframes', []), list) or len(item.get('keyframes', [])) > 500:
            raise ValueError('Each prop supports up to 500 spatial keyframes')
        for key in item.get('keyframes', []):
            if not isinstance(key, dict): raise ValueError('Each spatial keyframe must be an object')
            f = frame(key.get('frame'))
            if f in used: raise ValueError('Only one spatial keyframe per prop per frame is allowed')
            used.add(f); k = {'frame': f}
            if key.get('box') is not None:
                k['box'] = finite_vector(key['box'], 4, 'Prop box')
                if min(k['box'][:2]) < 0 or k['box'][2] <= k['box'][0] or k['box'][3] <= k['box'][1]:
                    raise ValueError('Draw a box with positive width and height')
            if key.get('position') is not None: k['position'] = finite_vector(key['position'], 3, 'Prop position')
            if key.get('depth_m') is not None:
                try: depth = float(key['depth_m'])
                except (TypeError, ValueError): raise ValueError('Prop depth must be a number') from None
                if not math.isfinite(depth) or not .05 < depth < 100: raise ValueError('Prop depth must be between .05 and 100 metres')
                k['depth_m'] = depth
            if not ('box' in k or 'position' in k): raise ValueError('Each spatial keyframe needs a box or world position')
            keys.append(k)
        result.append({'id': pid, 'name': name, 'shape': shape, 'size_m': size,
                       'events': sorted(events, key=lambda e: e['frame']),
                       'keyframes': sorted(keys, key=lambda k: k['frame'])})
    return result


def state_at(item, frame):
    active = {'frame': 0, 'state': 'hidden', 'person': None, 'joint': 'right'}
    for event in item['events']:
        if event['frame'] > frame: break
        active = event
    return active


def interpolate(values, f):
    keys = sorted(values)
    return np.array([np.interp(f, keys, [values[k][i] for k in keys]) for i in range(len(values[keys[0]]))])


def boxes_at(objects, tracked, f):
    out = []
    for item in objects:
        if state_at(item, f)['state'] == 'hidden': continue
        measured = (tracked.get(item['id'], {}) or {}).get(str(f))
        if measured: out.append(measured['box'])
        else:
            keys = {k['frame']: k['box'] for k in item['keyframes'] if k.get('box') and state_at(item, k['frame']) == state_at(item, f)}
            if keys: out.append(interpolate(keys, f).tolist())
    return out


def restore_background(data):
    """A prop edit can be undone without rebuilding or resegmenting the background."""
    names={m['name'] for m in data.get('meshes',[])}
    for mesh in data.pop('prop_background_backup',[]):
        if mesh['name'] not in names:
            data.setdefault('meshes',[]).append(mesh); names.add(mesh['name'])


def exclude_background(data, objects, tracked, y0):
    """Remove a previous static copy of an explicitly annotated prop using its source footprint."""
    remove=set()
    for mesh in data.get('meshes',[]):
        src=mesh.get('_source',{}); box=src.get('px'); f=src.get('frame')
        if box is None or f is None: continue
        label=str(mesh.get('label') or '').lower()
        if mesh.get('cls') in ('wall','building','floor','ground','ceiling'): continue
        static=np.asarray(box,float)+[0,y0,0,y0]
        for item in objects:
            # Background detections have semantic labels; never remove an unrelated wall/furniture group.
            if not label or not set(re.findall(r'[a-z]+',label)) & set(re.findall(r'[a-z]+',item['name'].lower())): continue
            for moving in boxes_at([item],tracked,int(f)):
                moving=np.asarray(moving,float)
                low=np.maximum(static[:2],moving[:2]); high=np.minimum(static[2:],moving[2:])
                intersection=float(np.prod(np.maximum(0,high-low)))
                union=float(np.prod(static[2:]-static[:2])+np.prod(moving[2:]-moving[:2])-intersection)
                if intersection/max(1.,union) > .3:
                    remove.add(mesh.get('_primitive',mesh['name']))
    data['prop_background_backup']=[m for m in data.get('meshes',[]) if m.get('_primitive',m['name']) in remove]
    data['meshes']=[m for m in data.get('meshes',[]) if m.get('_primitive',m['name']) not in remove]


def apply(data, objects, tracked=None, y0=0):
    """Bake ownership changes without teleporting released objects or merging different props."""
    import build_env as BE
    from held_objects import box_mesh, cyl_mesh, euler_xyz
    from rig import Rig, JOINTS as RIG_JOINTS
    objects = validate(objects, data['frames']); tracked = tracked or {}
    restore_background(data)
    data['props'] = [p for p in data.get('props', []) if not p.get('timeline')]
    cam = BE.Cam(data); N = data['frames']; warnings = []; baked = []
    by_track = {}
    for ch in data.get('chars', {}).values(): by_track.setdefault(ch.get('track_id'), []).append(ch)
    fk_cache = {}
    def anchor(event, f):
        for ch in by_track.get(event['person'], []):
            if f >= len(ch['root']) or not ch.get('visible', [True] * N)[f]: continue
            key = (id(ch), f)
            if key not in fk_cache:
                rig = Rig(ch['height'], ch.get('style', 'sample')); root = ch['root'][f]
                fk_cache[key] = rig.fk(root[:3], root[3], {j: ch['rot'][j][f] for j in RIG_JOINTS}, ch['pelvis_z'][f])
            joints = fk_cache[key]
            if event['state'] == 'worn' or event['joint'] == 'back':
                R, pos = joints['chest']; return np.asarray(R), np.asarray(pos) + np.asarray(R) @ [0, -.18, 0]
            if event['joint'] == 'both':
                return np.asarray(joints['wrist_R'][0]), (joints['wrist_L'][1] + joints['wrist_R'][1]) / 2
            R, pos = joints['wrist_L' if event['joint'] == 'left' else 'wrist_R']
            return np.asarray(R), np.asarray(pos)
        return None
    for item in objects:
        measured = {int(f): v['box'] for f, v in tracked.get(item['id'], {}).items() if 0 <= int(f) < N}
        measured.update({k['frame']: k['box'] for k in item['keyframes'] if 'box' in k})
        positions = {k['frame']: k['position'] for k in item['keyframes'] if 'position' in k}
        sizes, offsets = [], {}
        for f, box in sorted(measured.items()):
            event = state_at(item, f)
            a = anchor(event, f) if event['state'] in ('held', 'worn') else None
            if a is None: continue
            R, hand = a; Rc, C = BE.cam_mats(data['cam'][f]); forward = Rc @ [0., 0., -1.]
            z = float((hand - C) @ forward)
            if z <= .05: continue
            _, ray = cam.ray((box[0] + box[2]) / 2, (box[1] + box[3]) / 2 - y0, f)
            point = C + ray * z / max(.05, float(ray @ forward))
            off = R.T @ (point - hand)
            if np.linalg.norm(off) < .8:
                offsets[f] = off
            if box[0] > 1 and box[2] < cam.W-1 and box[1]-y0 > 1 and box[3]-y0 < cam.H-1:
                sizes.append([(box[2]-box[0])*z/cam.f_at(f), (box[3]-box[1])*z/cam.f_at(f)])
        if item['size_m']:
            w, h, d = item['size_m']
        else:
            w, h = np.clip(np.median(sizes, axis=0), .015, 2.) if sizes else (.12, .18)
            d = min(w, h) * (.12 if item['shape'] == 'flat' else .6)
        if item['shape'] == 'cyl': V, F = cyl_mesh(w/2, h)
        elif item['shape'] == 'ball':
            import props
            vv, F = props.icosphere(1)
            V = (np.asarray(vv) * [w/2, d/2, h/2]).tolist()
        else: V, F = box_mesh(w, d, h)
        depths = {k['frame']: k['depth_m'] for k in item['keyframes'] if 'depth_m' in k}
        keys, last, last_depth, missing = [], None, None, False
        def same_shot(a, b):
            return not any(min(a,b) < sh['frames'][0] <= max(a,b) for sh in data.get('shots', [])[1:])
        for f in range(N):
            event = state_at(item, f); state = event['state']; P, rot, visible = None, [0., 0., 0.], state != 'hidden'
            if any(sh['frames'][0] == f for sh in data.get('shots', [])[1:]): last = last_depth = None
            if state in ('held', 'worn'):
                a = anchor(event, f)
                if a is not None:
                    R, hand = a
                    interval_offsets = {k: v for k, v in offsets.items() if state_at(item, k) == event and same_shot(k,f)}
                    P = hand + R @ (interpolate(interval_offsets, f) if interval_offsets else np.zeros(3))
                    rot = euler_xyz(R)
                else: visible = False
            else:
                interval_positions = {k: v for k, v in positions.items() if state_at(item, k) == event and same_shot(k,f)}
                interval_boxes = {k: v for k, v in measured.items() if state_at(item, k) == event and same_shot(k,f)}
                interval_depths = {k: [v] for k,v in depths.items() if state_at(item,k) == event and same_shot(k,f)}
                depth = float(interpolate(interval_depths, f)[0]) if interval_depths else last_depth
                if interval_positions: P = interpolate(interval_positions, f)
                elif state == 'resting' and last is not None: P = last[:3]
                elif interval_boxes and depth is not None:
                    box = interpolate(interval_boxes, f); R, C = BE.cam_mats(data['cam'][f])
                    _, ray = cam.ray((box[0]+box[2])/2, (box[1]+box[3])/2-y0, f)
                    P = C + ray * depth / max(.05, float(ray @ (R @ [0.,0.,-1.])))
                elif last is not None:
                    P = last[:3]
                    if state == 'free' and not missing:
                        warnings.append('%s: free motion needs a box or position keyframe; depth is held at release' % item['name']); missing = True
            if P is None:
                P = last[:3] if last is not None else np.zeros(3)
                if state != 'hidden':
                    visible = False
                    if not missing: warnings.append('%s: add a box and depth or world position for this shot' % item['name']); missing = True
            key = [*np.asarray(P, float).tolist(), *rot, *([1.,1.,1.] if visible else [0.,0.,0.])]
            if visible:
                last = np.asarray(key[:6]); R, C = BE.cam_mats(data['cam'][f])
                last_depth = max(.05, float((last[:3]-C) @ (R @ [0.,0.,-1.])))
            keys.append(key)
        prop = {'name': 'timeline_'+item['id'], 'label': item['name'], 'object_id': item['id'],
                'timeline': True, 'held': True, 'v': V, 'f': F, 'colour': 'near', 'keys': keys,
                'events': item['events'], 'size_m': [float(w), float(h), float(d)]}
        data['props'].append(prop); baked.append(prop['name'])
    data['prop_timeline_warnings'] = warnings
    exclude_background(data,objects,tracked,y0)
    return {'objects': baked, 'warnings': warnings}

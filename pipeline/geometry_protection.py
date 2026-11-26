"""Bounded scene corrections that preserve people visible in the source video."""
import cv2
import numpy as np
from functools import lru_cache
from profiling import stage,counter


@lru_cache(maxsize=4)
def character_parts(style):
    from common import AGENT, jload
    import os
    asset = {'sample':'sample_character.json', 'base21':'base_character_v21.json',
             'master':'human_master.json','l3':'l3/character.json'}.get(style)
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


@stage('occlusion_protection')
def protect(data, bundles, max_shift=3., max_groups=None, refine=True, resolution=160, only_objects=None):
    """Correct depth first, then trim only unsupported conflicting surface fragments.

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
            h, w = depth.shape; scale = min(1., resolution/max(h, w)); shape = (max(1, round(h*scale)), max(1, round(w*scale)))
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
    actions, replacements, attempted = [], {}, 0
    for name, group in groups.items():
        if max_groups is not None and attempted>=max_groups:break
        if only_objects is not None and not any(m['name'] in only_objects for m in group):continue
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
        attempted+=1
        # Preserve the footprint and physical dimensions when correcting depth.
        distance, forward = max(gaps, key=lambda g: g[0])
        shift = np.asarray(forward) * min(max_shift, max(.2, distance))
        shifted = [dict(m, v=(np.asarray(m['v'], float)+shift).tolist()) for m in group]
        after, _, _ = measure(shifted)
        candidates=[]
        if after==0 and distance<=max_shift:
            candidates.append((shifted,{'action':'moved behind source-visible people','translation':shift.tolist()}))
        # Compare joint depth/scale placement with rigid translation using all source masks.
        # A depth correction must not silently destroy a fitted object's screen alignment.
        anchor=next(o for o in observations if o[0] in frames)
        C=anchor[4];z=float(np.median((np.concatenate([np.asarray(m['v']) for m in group])-C)@anchor[3][2]))
        factor=1+(distance+.2)/max(.2,z)
        if factor<=2 and not any(m.get('size_locked') for m in group):
            ray_scaled=[dict(m,v=(C+(np.asarray(m['v'])-C)*factor).tolist(),_occlusion_scale=factor) for m in group]
            if measure(ray_scaled)[0]==0:
                candidates.append((ray_scaled,{'action':'depth and scale aligned along source rays','factor':factor}))
        from object_fit import mesh_metrics,placement_error
        original_alignment=mesh_metrics(group)
        candidates=[candidate for candidate in candidates if not any(
            placement_error(b)>max(.04,placement_error(a)+.01)
            for a,b in zip(original_alignment,mesh_metrics(candidate[0])))]
        if candidates:
            def alignment(candidate):
                views=mesh_metrics(candidate[0])
                return sum(1-v['iou']+v['center_error_px']/100+v['size_error_fraction']+10*placement_error(v) for v in views)
            corrected,action=min(candidates,key=alignment)
            for before,new in zip(group,corrected):replacements[id(before)]=new
            actions.append(dict(action,object=name,frames=frames))
        elif refine:
            corrected=[clip_unsupported_faces(mesh,observations) for mesh in group]
            if measure(corrected)[0]<score and any(m['f'] for m in corrected):
                for before,new in zip(group,corrected):replacements[id(before)]=new
                archive=[a for a in data.get('occlusion_original_geometry',[]) if a['object']!=name]
                data['occlusion_original_geometry']=archive+[{'object':name,'meshes':group}]
                actions.append({'object':name,'action':'trimmed unsupported occluding fragments; object retained','frames':frames})
            else:
                actions.append({'object':name,'action':'retained for review: depth conflict could not be resolved','frames':frames})
        else:
            actions.append({'object':name,'action':'retained for review: normal repair budget; detailed refinement available in audit mode','frames':frames})
    data['meshes'] = [replacements.get(id(m), m) for m in data['meshes']]
    data['geometry_protection'] = data.get('geometry_protection', []) + actions
    return actions


@stage('occlusion_clipping')
def clip_unsupported_faces(mesh, observations, levels=6, max_triangles=100000):
    """Keep visible/supported geometry; remove only fragments in front of observed human pixels.

    Evidence for the object in another view protects a fragment from destructive trimming.
    Original geometry is retained by protect() for recovery and further fitting.
    """
    from object_fit import unbits
    from scene_verify import triangles
    supports=[]
    for view in mesh.get('_observations',[]):
        supports.append((unbits(view['mask'],view['shape']),np.asarray(view['K']),np.asarray(view['R']),np.asarray(view['C'])))
    masks=[]
    for f,depth,K,R,C,person,valid,rendered in observations:
        target=np.where(np.isfinite(rendered),np.maximum(depth,rendered),depth)
        area=cv2.dilate((person&valid).astype(np.uint8),np.ones((5,5),np.uint8))>0
        masks.append((area,target,K,R,C))
    def sample(points):
        bad=np.zeros(len(points),bool);supported=np.zeros(len(points),bool)
        for mask,K,R,C in supports:
            camera=(points-C)@R.T;z=camera[:,2];uv=camera@K.T
            xy=np.rint(uv[:,:2]/np.maximum(z[:,None],.02)).astype(int)
            inside=(z>.02)&(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]<mask.shape[0])
            ids=np.flatnonzero(inside);supported[ids]|=mask[xy[ids,1],xy[ids,0]]
        for mask,target,K,R,C in masks:
            camera=(points-C)@R.T;z=camera[:,2];uv=camera@K.T
            xy=np.rint(uv[:,:2]/np.maximum(z[:,None],.02)).astype(int)
            inside=(z>.02)&(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]<mask.shape[0])
            ids=np.flatnonzero(inside)
            bad[ids]|=mask[xy[ids,1],xy[ids,0]]&(z[ids]<target[xy[ids,1],xy[ids,0]]-.03)
        return bad&~supported
    vertices=np.asarray(mesh['v'],float);kept=[]
    # Project many probes together instead of doing seven tiny matrix operations
    # per triangle, per recursive child, per camera. Preserve the same split rules.
    integrals=[cv2.integral(mask.astype(np.uint8)) for mask,*_ in masks]
    pending=[(vertices[triangles(mesh['f'])],levels)];processed=0;limited=False
    while pending:
        current,level=pending.pop()
        for start in range(0,len(current),512):
            batch=current[start:start+512];a,b,c=batch[:,0],batch[:,1],batch[:,2]
            if processed+len(batch)>max_triangles:
                kept.append(current[start:]);kept.extend(tris for tris,_ in pending)
                pending.clear();limited=True;counter('occlusion_clip_budget_exhausted');break
            processed+=len(batch)
            ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
            probes=np.stack([a,b,c,ab,bc,ca,(a+b+c)/3],axis=1)
            bad=sample(probes.reshape(-1,3)).reshape(-1,7)
            clear=~bad.any(1);intersects=np.zeros(len(batch),bool)
            if level:
                # A triangle whose vertices all sample the same pixel in every
                # view cannot conceal an unsampled mask feature. Depth is linear
                # over the triangle, and the corner probes already test it.
                constant=clear.copy()
                for mask,K,R,C in supports+[(m,k,r,c) for m,_,k,r,c in masks]:
                    ids=np.flatnonzero(constant)
                    if not len(ids):break
                    cam=(batch[ids]-C)@R.T;z=cam[:,:,2];uv=cam@K.T
                    xy=np.rint(uv[:,:,:2]/np.maximum(z[:,:,None],.02)).astype(int)
                    same=(xy==xy[:,:1]).all((1,2))
                    outside=(xy[:,:,0]<0).all(1)|(xy[:,:,1]<0).all(1)|(xy[:,:,0]>=mask.shape[1]).all(1)|(xy[:,:,1]>=mask.shape[0]).all(1)
                    constant[ids]&=(z<=.02).all(1)|((z>.02).all(1)&(same|outside))
                for integral,(mask,target,K,R,C) in zip(integrals,masks):
                    ids=np.flatnonzero(clear&~constant&~intersects)
                    if not len(ids):break
                    cam=(batch[ids]-C)@R.T;uv=cam@K.T;uv=uv[:,:,:2]/np.maximum(cam[:,:,2,None],.02)
                    lo=np.maximum(0,np.floor(uv.min(1))).astype(int)
                    hi=np.minimum([mask.shape[1]-1,mask.shape[0]-1],np.ceil(uv.max(1))).astype(int)
                    valid=(hi>=lo).all(1);rows=np.flatnonzero(valid)
                    low,high=lo[rows],hi[rows]
                    count=integral[high[:,1]+1,high[:,0]+1]-integral[low[:,1],high[:,0]+1]-integral[high[:,1]+1,low[:,0]]+integral[low[:,1],low[:,0]]
                    for row in rows[count>0]:
                        x0,y0=lo[row];x1,y1=hi[row]
                        local=mask[y0:y1+1,x0:x1+1]
                        if cam[row,:,2].min()<np.max(target[y0:y1+1,x0:x1+1][local])-.03:
                            intersects[ids[row]]=True
            accepted=clear&~intersects
            if accepted.any():kept.append(batch[accepted])
            split=~bad.all(1)&~accepted
            if level and split.any():
                children=np.stack([np.stack([a,ab,ca],1),np.stack([ab,b,bc],1),
                                   np.stack([ca,bc,c],1),np.stack([ab,bc,ca],1)],axis=1)
                pending.append((children[split].reshape(-1,3,3),level-1))
    if not kept:return dict(mesh,f=[])
    points=np.concatenate(kept).reshape(-1,3)
    unique,inverse=np.unique(np.round(points,7),axis=0,return_inverse=True)
    result=dict(mesh,v=unique.tolist(),f=inverse.reshape(-1,3).tolist(),_occlusion_clipped=True)
    result.pop('_occlusion_clip_limited',None)
    if limited:result['_occlusion_clip_limited']=True
    return result


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

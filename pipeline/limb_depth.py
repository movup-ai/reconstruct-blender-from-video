"""Local, reversible ordinal limb-depth edits on an already fitted scene.

No inference, camera changes, root translations or bone scaling occur here.
"""
import copy
import math
import numpy as np
from scipy.optimize import least_squares
from rig import Rig, Rxyz

PARTS = {'left_upper_arm': ('shoulder_L', 'elbow_L'),
         'left_forearm': ('elbow_L', 'wrist_L'),
         'right_upper_arm': ('shoulder_R', 'elbow_R'),
         'right_forearm': ('elbow_R', 'wrist_R'),
         'torso': ('pelvis', 'neck')}


def validate(items, people=None, frames=None):
    if not isinstance(items, list) or len(items) > 200:
        raise ValueError('Limb depth timeline must contain at most 200 ranges')
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Invalid limb depth range')
        row = {}
        for key in ('person', 'other'):
            row[key] = str(item.get(key, ''))
            if not row[key] or (people is not None and row[key] not in people):
                raise ValueError('Limb depth character is not in this clip')
        if row['person'] == row['other']:
            raise ValueError('Choose two different characters')
        for key in ('part', 'other_part'):
            row[key] = item.get(key)
            if row[key] not in PARTS or (key == 'part' and row[key] == 'torso'):
                raise ValueError('Choose an arm segment (or target torso)')
        row['order'] = item.get('order', 'front')
        if row['order'] not in ('front', 'behind'):
            raise ValueError('Depth order must be front or behind')
        for key in ('start', 'end'):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value) or value < 0:
                raise ValueError('Depth frames must be nonnegative integers')
            row[key] = int(value)
        if row['end'] < row['start'] or (frames and row['end'] >= frames):
            raise ValueError('Limb depth range is outside this clip')
        margin = float(item.get('margin_m', .08))
        if not math.isfinite(margin) or not .01 <= margin <= .5:
            raise ValueError('Depth clearance must be between 1 and 50 cm')
        row['margin_m'] = margin
        result.append(row)
    # Contradictory constraints on the same pair cannot be satisfied.
    for i, a in enumerate(result):
        a_pair = ((a['person'], a['part']), (a['other'], a['other_part']))
        a_front = a_pair if a['order'] == 'front' else a_pair[::-1]
        for b in result[:i]:
            b_pair = ((b['person'], b['part']), (b['other'], b['other_part']))
            b_front = b_pair if b['order'] == 'front' else b_pair[::-1]
            if a_front == b_front[::-1] and max(a['start'], b['start']) <= min(a['end'], b['end']):
                raise ValueError('Conflicting front/behind ranges overlap')
    return result


def apply(data, items):
    """Jointly solve involved arms against cached image projections and ordinal depth.

    The immutable rotation baseline makes removing/repeating edits reversible.
    Depth is evaluated at the nearest projected points of the selected segments.
    Unsatisfied constraints are reported rather than moving entire actors.
    """
    from fit_people import ANG, merge_residual
    items = validate(items, frames=data['frames'])
    chars = data.get('chars', {})
    for ch in chars.values():
        if '_limb_depth_base' in ch:
            ch['rot'] = copy.deepcopy(ch['_limb_depth_base'])
        elif items:
            ch['_limb_depth_base'] = copy.deepcopy(ch['rot'])
    report = {'ranges': len(items), 'solved_frames': 0, 'unresolved': [], 'skipped_frames': 0}
    rigs = {k: Rig(ch['height'], ch.get('style', 'sample')) for k, ch in chars.items()}
    padding = max(1, round(data.get('fps', 30)*.15))
    previous = {}
    active_frames = sorted({f for r in items for f in range(max(0,r['start']-padding), min(data['frames'],r['end']+padding+1))})
    for f in active_frames:
        rows = [r for r in items if r['start']-padding <= f <= r['end']+padding]
        def find(pid):
            return next((k for k,c in chars.items() if str(c.get('track_id')) == pid and c.get('visible',[True]*data['frames'])[f]
                         and c.get('source_range',[0,data['frames']-1])[0] <= f <= c.get('source_range',[0,data['frames']-1])[1]), None)
        pairs = [(r,find(r['person']),find(r['other'])) for r in rows]
        pairs = [(r,a,b) for r,a,b in pairs if a is not None and b is not None]
        if not pairs:
            report['skipped_frames'] += 1
            continue
        cam = data['cam'][f]
        rotation = Rxyz(cam[3:6])
        center = np.asarray(cam[:3])
        focal = data['res'][0]*data['lens']/data.get('sensor',36)*data.get('zoom',[1]*data['frames'])[f]
        def camera(points):
            p = (points-center) @ rotation
            return np.c_[p[:,0],p[:,1],-p[:,2]]
        def project(points):
            return points[:,:2]/np.maximum(points[:,2:],.05)*focal
        involved = {}
        for r,a,b in pairs:
            for k,part in ((a,r['part']),(b,r['other_part'])):
                if part != 'torso':
                    side = 'L' if part.startswith('left') else 'R'
                    involved.setdefault(k,set()).update(('shoulder_'+side,'elbow_'+side))
        variables = [(k,j,axis,lo,hi) for k,joints in involved.items() for j,axis,lo,hi,_ in ANG if j in joints]
        initial = np.array([chars[k]['rot'][j][f][axis] for k,j,axis,_,_ in variables])
        def points(x):
            rotations = {k:{j:list(v[f]) for j,v in c['rot'].items()} for k,c in chars.items() if k in involved or any(k==b for _,_,b in pairs)}
            for v,(k,j,axis,_,_) in zip(x,variables): rotations[k][j][axis]=v
            result = {}
            for k,rot in rotations.items():
                c=chars[k]; root=c['root'][f]
                w=rigs[k].fk(root[:3],root[3],rot,c['pelvis_z'][f])
                result[k]={j:camera(np.asarray([p]))[0] for j,(_,p) in w.items()}
                result[k]['_collision']=np.asarray(merge_residual(w,rigs[k].s))
            return result
        base=points(initial)
        baseline={k:project(np.array([base[k][j] for j in ('elbow_L','wrist_L','elbow_R','wrist_R')])) for k in involved}
        samples=[]
        t=np.linspace(0,1,17)
        for r,a,b in pairs:
            pa=np.array([base[a][j] for j in PARTS[r['part']]])
            pb=np.array([base[b][j] for j in PARTS[r['other_part']]])
            sa=pa[0]+t[:,None]*(pa[1]-pa[0]); sb=pb[0]+t[:,None]*(pb[1]-pb[0])
            distances=np.linalg.norm(project(sa)[:,None]-project(sb)[None,:],axis=2)
            i,j=np.unravel_index(distances.argmin(),distances.shape)
            fade=min(1.,(f-r['start']+padding)/padding,(r['end']+padding-f)/padding)
            samples.append((r,a,b,float(t[i]),float(t[j]),max(0.,fade)))
        def gaps(p):
            values=[]
            for r,a,b,ta,tb,fade in samples:
                aa,ab=PARTS[r['part']]; ba,bb=PARTS[r['other_part']]
                za=(1-ta)*p[a][aa][2]+ta*p[a][ab][2]
                zb=(1-tb)*p[b][ba][2]+tb*p[b][bb][2]
                values.append((zb-za)*(1 if r['order']=='front' else -1))
            return np.array(values)
        base_gap=gaps(base)
        targets=np.array([max(g,r['margin_m']) if r['start']<=f<=r['end'] else g+fade*(max(g,r['margin_m'])-g)
                          for g,(r,_,_,_,_,fade) in zip(base_gap,samples)])
        prior=np.array([previous.get((k,j,axis),(f-2,0))[1] if previous.get((k,j,axis),(f-2,0))[0]==f-1 else 0 for k,j,axis,_,_ in variables])
        def residual(x):
            p=points(x)
            repro=np.concatenate([(project(np.array([p[k][j] for j in ('elbow_L','wrist_L','elbow_R','wrist_R')]))-baseline[k]).ravel()/6 for k in involved])
            collision=np.concatenate([np.maximum(0,p[k]['_collision']-np.maximum(.7,base[k]['_collision'])) for k in involved])
            fade=max(sample[-1] for sample in samples)
            return np.r_[repro,np.maximum(0,targets-gaps(p))/.008,(x-initial)/max(.05,60*fade),(x-initial-prior)/12,collision*2]
        lo=np.array([min(l,v-1e-5) for v,(_,_,_,l,_) in zip(initial,variables)])
        hi=np.array([max(h,v+1e-5) for v,(_,_,_,_,h) in zip(initial,variables)])
        solved=least_squares(residual,initial,bounds=(lo,hi),max_nfev=60)
        if not np.all(np.isfinite(solved.x)):
            report['unresolved'].append({'frame':f,'reason':'Nonfinite fit'}); continue
        final_gap=gaps(points(solved.x))
        for v,old,(k,j,axis,_,_) in zip(solved.x,initial,variables):
            chars[k]['rot'][j][f][axis]=float(v)
            previous[k,j,axis]=(f,float(v-old))
        report['solved_frames']+=1
        for gap,(r,a,b,_,_,_) in zip(final_gap,samples):
            if r['start']<=f<=r['end'] and gap < r['margin_m']-.02:
                report['unresolved'].append({'frame':f,'person':r['person'],'other':r['other'],'gap_m':round(float(gap),4)})
    data['limb_depth_report']=report
    data['limb_depth_timeline']=copy.deepcopy(items)
    return report


def preview(data, video, path, y0=0):
    """Refresh the existing pose-overlay preview using only decoded source frames."""
    import cv2
    from build_env import Cam
    from common import imwrite_u
    cam=Cam(data)
    capture=cv2.VideoCapture(video)
    tiles=[]
    marked=[(r['start']+r['end'])//2 for r in data.get('limb_depth_timeline',[])][:3]
    selected=list(dict.fromkeys(marked+np.linspace(0,data['frames']-1,6).astype(int).tolist()))[:6]
    selected+= [selected[-1]]*(6-len(selected))
    for f in sorted(selected):
        capture.set(cv2.CAP_PROP_POS_FRAMES,int(f))
        ok,image=capture.read()
        if not ok:
            capture.release()
            raise RuntimeError('Cannot decode source frame for depth preview')
        image=image[y0:y0+data['res'][1]].copy()
        for ch in data['chars'].values():
            if not ch['visible'][f]: continue
            rig=Rig(ch['height'],ch.get('style','sample')); root=ch['root'][f]
            world=rig.fk(root[:3],root[3],{j:v[f] for j,v in ch['rot'].items()},ch['pelvis_z'][f])
            names=[j for j,_,_ in rig.J]; positions=np.array([world[j][1] for j in names])
            uv,z=cam.project(positions,int(f)); lookup={j:i for i,j in enumerate(names)}
            colour={'red':(60,60,230),'blue':(230,110,40),'green':(70,190,70)}.get(ch.get('colour'),(60,200,240))
            for j,parent,_ in rig.J:
                if parent in lookup:
                    a,b=lookup[j],lookup[parent]
                    if z[a]>.05 and z[b]>.05:
                        cv2.line(image,tuple(np.clip(uv[a],-10000,10000).astype(int)),tuple(np.clip(uv[b],-10000,10000).astype(int)),colour,3,cv2.LINE_AA)
        cv2.putText(image,'Depth edit - frame %d'%f,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        tiles.append(cv2.resize(image,(data['res'][0]//2,data['res'][1]//2)))
    capture.release()
    imwrite_u(path,np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,6,2)]),85)

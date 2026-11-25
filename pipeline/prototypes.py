"""Recognizable vegetation prototypes sized from measured regions, not noisy relief meshes."""
import re
import numpy as np
from profiling import stage


def vegetation_kind(label, indoor=False):
    text = str(label or '').lower()
    if re.search(r'\b(potted|houseplant|house plant|plant pot|flowerpot)\b', text): return 'plant_pot'
    if text.strip()=='pot':return 'pot'
    words = set(re.findall(r'[a-z]+', text))
    def has(*terms): return any(t in words or t+'s' in words for t in terms)
    if has('forest', 'woodland', 'grove', 'woods'): return 'forest'
    if has('palm'): return 'palm'
    if has('bush', 'shrub', 'bushes', 'shrubbery'): return 'bush'
    if has('hedge', 'hedgerow'): return 'hedge'
    if has('tree', 'pine', 'fir', 'spruce', 'conifer'): return 'pine'
    if has('plant', 'flower', 'foliage', 'leaves'):
        return 'plant_pot' if indoor else 'bush'
    return None


@stage('vegetation_classification')
def automatic_kind(label, points, mask, indoor=False, ground=(0.,0.,0.)):
    """ADE's generic tree/plant labels also contain shrubs; distinguish shape and support locally."""
    kind=vegetation_kind(label,indoor);text=str(label or '').strip().lower()
    evidence={'rule':'named vegetation','kind':kind}
    if text not in ('tree','plant','flower','foliage','leaves','vegetation'):return kind,evidence
    if indoor and text!='tree':return kind,dict(evidence,rule='indoor plant')
    p=np.asarray(points,float);p=p[np.isfinite(p).all(1)];y,x=np.nonzero(mask)
    if len(p)<16 or not len(x):return kind,evidence
    lo,hi=np.percentile(p,[5,95],axis=0);a,b,c=ground
    floor=a*np.median(p[:,0])+b*np.median(p[:,1])+c
    height=max(.05,float(hi[2]-lo[2]));top=float(hi[2]-floor);base=float(lo[2]-floor)
    width,height_px=float(x.max()-x.min()+1),float(y.max()-y.min()+1);aspect=width/height_px
    rows=np.bincount(y,minlength=mask.shape[0]);start,stop=int(y.min()),int(y.max())+1
    middle=rows[start+int(height_px*.15):start+int(height_px*.65)];bottom=rows[start+int(height_px*.75):stop]
    trunk=float(bottom.mean()/max(1.,middle.mean())) if len(bottom) and len(middle) else 1.
    evidence.update(height_above_ground=top,base_above_ground=base,aspect=aspect,trunk_ratio=trunk)
    compact=aspect>=.8 and not (aspect<1.4 and trunk<.25)
    low=top<=2.4 and base<.8 and y.min()>1
    if low and compact:
        kind='hedge' if aspect>=2.8 else 'bush';rule='low compact foliage without a tall trunk'
    elif top>2.4 and (hi[:2]-lo[:2]).max()>2.5*height and aspect>2.:
        kind='forest';rule='wide elevated tree canopy'
    elif text=='tree' or (top>2.4 and aspect<.8):
        kind='pine';rule='upright or elevated tree canopy'
    else:rule='generic outdoor plant'
    evidence.update(kind=kind,rule=rule)
    return kind,evidence


@stage('vegetation_instance_separation')
def split_shrubs(fr, mask, label, indoor=False, min_area=.0015):
    """Retain SAM shrub instances that a connected semantic tree region would otherwise absorb."""
    if label not in ('tree','plant','foliage') or (indoor and label=='plant'):return [(mask,label)]
    candidates=[]
    for instance in getattr(fr,'vegetation_instances',[]):
        piece=instance&mask
        if piece.sum()<max(30,min_area*mask.size) or piece.sum()>.85*mask.sum():continue
        kind,_=automatic_kind(label,fr.X[piece&fr.valid],piece,indoor,fr.gp)
        if kind in ('bush','hedge'):candidates.append(piece)
    claimed=np.zeros(mask.shape,bool);pieces=[]
    for piece in sorted(candidates,key=lambda p:-p.sum()):
        if (piece&claimed).sum()>.5*piece.sum():continue
        piece=piece&~claimed;claimed|=piece;pieces.append((piece,label))
    if not pieces:return [(mask,label)]
    import cv2
    n,regions,stats,_=cv2.connectedComponentsWithStats((mask&~claimed).astype(np.uint8),connectivity=8)
    pieces += [(regions==i,label) for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=min_area*mask.size]
    return pieces


def measured_object(kind, points, name, source):
    p = np.asarray(points, float)
    p = p[np.isfinite(p).all(1)]
    if len(p) < 8: return None
    lo, hi = np.percentile(p, [5, 95], axis=0)
    width, depth = np.maximum(hi[:2]-lo[:2], .08)
    height = max(.12, float(hi[2]-lo[2]))
    if kind in ('pine', 'palm'):
        depth = max(depth, .35 * width)
    elif kind in ('bush', 'hedge'):
        height = min(height, max(.25, width*1.2)); depth = max(depth, width*.5)
    center = (lo+hi)/2
    return {'name': name, 'type': kind, 'prototype': kind,
            'pos': [float(center[0]), float(center[1]), float(lo[2])],
            'size': [float(width), height, float(depth)], 'yaw': 0., 'colour': 'tree', '_src': source}

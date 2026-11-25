"""Joint, visibility-aware fitting of recognizable prototypes to image/depth observations."""
import base64
import copy
import hashlib
import math
import os

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares,minimize
from scipy.spatial import cKDTree

import props
from common import jsave
from profiling import stage

VERSION = 3
FURNITURE = {'chair','office_chair','table','desk','counter','sofa','bed','cabinet','shelf','bookshelf','bench','stool'}
ROUND = {'plant_pot','pine','palm','bush','vase','bottle','pot'}


def bits(mask):
    return base64.b64encode(np.packbits(np.asarray(mask,bool)).tobytes()).decode('ascii')


def unbits(value, shape):
    return np.unpackbits(np.frombuffer(base64.b64decode(value),np.uint8),count=int(np.prod(shape))).reshape(shape).astype(bool)


def room(fr):
    """Classify the observed room, independently of missing/reflective depth pixels."""
    labels=getattr(fr,'sem',None);seg=getattr(fr,'segm',None)
    if labels is None or seg is None:return False,{'source':'semantic evidence unavailable','confidence':0.}
    valid=~fr.bars & ~fr.person
    if not valid.any():return False,{'source':'no static pixels','confidence':0.}
    values=labels[valid];counts=np.bincount(values.astype(int),minlength=len(seg.names))/len(values)
    def share(names):return float(sum(counts[i] for i,n in enumerate(seg.names) if n in names))
    ceiling=share({'ceiling'});windows=share({'windowpane','blind'});floor=share({'floor','rug'})
    furniture=share({'table','desk','chair','armchair','swivel chair','sofa','bed','cabinet','shelf','bookcase'})
    structure=share({'wall','windowpane','blind','door','ceiling'});sky=share({'sky'})
    indoor=ceiling>.015 or (windows>.03 and furniture>.025) or (floor>.08 and structure>.1 and furniture>.025)
    if not indoor:indoor=furniture>.25 and structure>.15 and sky<.05
    return bool(indoor),{'source':'semantic room surfaces','ceiling':ceiling,'windows':windows,'floor':floor,
                         'furniture':furniture,'structure':structure,'sky':sky,'confidence':.85 if indoor else .65}


def observation(fr, mask, partial=False, parts=None):
    """Compact mask, calibrated camera, reliable depth, and known foreground occluders."""
    mask=np.asarray(mask,bool)&~fr.person&~fr.bars
    if mask.sum()<16:return None
    h,w=fr.H,fr.W;scale=min(1.,256/w);shape=(max(1,round(h*scale)),max(1,round(w*scale)))
    def resize(m):return cv2.resize(np.asarray(m,np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    p=fr.P[mask&fr.valid];z=p[:,2]*fr.s
    z=z[np.isfinite(z)&(z>.05)]
    if len(z)<8:return None
    med=float(np.median(z));mad=float(np.median(np.abs(z-med)))
    reliable=np.abs(z-med)<max(.15,3*mad)
    med=float(np.median(z[reliable])) if reliable.any() else med
    fx=fr.fpix
    K=np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1.]])
    K[0]*=shape[1]/w;K[1]*=shape[0]/h
    occluder=fr.person | (fr.valid & (fr.P[:,:,2]*fr.s<med-max(.15,.06*med)) & ~mask)
    small=resize(mask)
    if small.sum()<4:return None
    return {'frame':int(fr.f),'shape':list(shape),'mask':bits(small),'occluder':bits(resize(occluder)),
            'K':K.tolist(),'R':fr.Rw.T.tolist(),'C':fr.Cl.tolist(),
            'depth':med,'depth_sigma':max(.15,.08*med,min(mad*2,.4*med)),
            'partial':bool(partial),'parts':parts,'source_scale':float(w/shape[1]/fr.k)}


def attach(o, fr, mask):
    """Seed a complete prototype from its observed surface, retaining hidden-part uncertainty."""
    kind=props.norm_type(o.get('type'))
    if o.get('type') in ('mesh','patch') or kind not in props.TYPES:return o
    mask=np.asarray(mask,bool)&~fr.person&~fr.bars
    if mask.sum()<16:return o
    P=fr.X[mask&fr.valid];P=P[np.isfinite(P).all(1)]
    if len(P)<16:return o
    default=np.asarray(props.TYPES[kind][0],float)
    lo,hi=np.percentile(P,[5,95],axis=0);anchor=np.median(P,axis=0)
    observed_h=max(.04,float(hi[2]-lo[2]))
    partial=kind in ('chair','office_chair') and observed_h<.72*default[1]
    parts=['back'] if partial else (['top'] if kind in ('table','desk','counter') else None)
    direction=anchor[:2]-fr.Cl[:2]
    yaw=math.degrees(math.atan2(-direction[0],direction[1]))
    across=np.array([math.cos(math.radians(yaw)),math.sin(math.radians(yaw)),0.])
    width=max(.06,float(np.ptp(np.percentile(P@across,[3,97]))))
    dims=np.asarray(o.get('size') or default,float).copy()
    if not o.get('size_locked'):
        dims=default.copy();dims[0]=width
        if kind in ROUND:
            dims[1]=max(.12,observed_h);dims[2]=width
        elif kind not in FURNITURE:
            dims[1]=max(.1,observed_h);dims[2]=max(.04,min(float(hi[1]-lo[1]),dims[0]))
        else:
            dims[2]=max(.15,min(width*default[2]/default[0],2*default[2]))
    # Place the observed part, not the guessed floor, at the measured surface.
    zfraction=.745 if partial else (.98 if parts==['top'] else .5)
    position=anchor-[0,0,dims[1]*zfraction]
    if o.get('_level') and o.get('dist'):
        reach=np.linalg.norm((anchor-fr.Cl)[:2]);factor=float(o['dist'])/max(.1,reach)
        position=fr.Cl+(position-fr.Cl)*factor
        if not o.get('size_locked'):dims*=factor
    if o.get('facing') in ('left','right','away'):
        yaw+={'left':-90,'right':90,'away':180}[o['facing']]
    o.update(pos=position.tolist(),size=dims.tolist(),yaw=yaw,abs_z=True)
    obs=observation(fr,mask,partial,parts)
    if obs:
        o.setdefault('_observations',[]).append(obs)
        if (o.get('_src') or {}).get('cls')=='drawn' and o.get('bbox'):
            obs['annotation_box']=(np.asarray(o['bbox'],float)/obs['source_scale']).tolist()
            obs['primary']=True
            if kind in ('pine','tree','palm','forest') and not o.get('_level') and not o.get('size_locked'):
                obs['ground_plane']=list(getattr(fr,'gp',(0.,0.,0.)))
                if getattr(fr,'annotation_ground',None):obs['ground_definition']=fr.annotation_ground
            seed_annotation(o,obs)
            occluder=fr.person | (fr.valid & (fr.P[:,:,2]*fr.s<obs['depth']-max(.15,.06*obs['depth'])) & ~mask)
            obs['occluder']=bits(cv2.resize(occluder.astype(np.uint8),tuple(obs['shape'])[::-1],interpolation=cv2.INTER_NEAREST)>0)
        elif kind in ('bush','hedge'):
            y,x=np.nonzero(unbits(obs['mask'],obs['shape']))
            obs.update(detection_box=[int(x.min()),int(y.min()),int(x.max()),int(y.max())],primary=True)
            a,b,c=getattr(fr,'gp',(0.,0.,0.))
            if lo[2]-(a*anchor[0]+b*anchor[1]+c)<.35:obs['ground_plane']=[a,b,c]
            seed_annotation(o,obs)
    return o


def bounds(points, view):
    if not len(points):return None
    uv,z=_uv(points,view)
    if not len(uv) or (z<=.02).any():return None
    lo=np.maximum(0,uv.min(0));hi=np.minimum(np.asarray(view['shape'])[::-1]-1,uv.max(0))
    return np.r_[lo,hi]


def fit_box(view):
    return view.get('annotation_box') or (view.get('detection_box') if view.get('primary') else None)


def box_error(points, view):
    predicted=bounds(points,view);target=np.asarray(fit_box(view),float)
    if predicted is None:return np.full(4,10.)
    return (predicted-target)/np.tile(np.maximum(target[2:]-target[:2],2.),2)


def seed_annotation(o, view):
    """Use drawn image extent for size/placement; depth determines metres, not the silhouette."""
    kind=props.norm_type(o['type']);box=np.asarray(fit_box(view));K=np.asarray(view['K']);R=np.asarray(view['R']);C=np.asarray(view['C'])
    span=box[2:]-box[:2];z=float(view['depth']);dims=np.asarray(o['size'],float).copy()
    if view.get('ground_plane') and box[3]<view['shape'][0]*.97:
        a,b,c=view['ground_plane'];normal=np.array([-a,-b,1.])
        ray=np.linalg.solve(K,[(box[0]+box[2])/2,box[3],1.])@R
        denominator=ray@normal
        foot_depth=(c-C@normal)/denominator if denominator<-.01 else -1.
        if view.get('ground_definition'):
            from build_env import Ground
            hit=Ground(view['ground_definition']).hit(C,ray)
            foot_depth=float((hit-C)@R[2]) if hit is not None else -1.
        if .3<foot_depth<100 and .25<foot_depth/z<4.:
            view.setdefault('measured_depth',z);view['depth']=z=float(foot_depth)
            view['depth_sigma']=max(view['depth_sigma'],z*.2);view['ground_contact']=True
    if not o.get('size_locked'):
        dims[0]=max(.04,span[0]*z/K[0,0])
        dims[1]=max(.08,span[1]*z/K[1,1]/max(.3,abs(R[1,2])))
        if view.get('parts')==['back']:dims[1]/=.51
        elif view.get('parts')==['top']:dims[1]=props.TYPES[kind][0][1]
        if kind in ROUND:dims[2]=dims[0]
        elif kind in ('wall','window','door','curtain','picture','mirror','building'):
            dims[2]=min(props.TYPES[kind][0][2],max(.08,dims[0]*.08))
        else:dims[2]=max(.06,min(dims[2],dims[0]))
    parts=_parts(kind,dims);points=np.concatenate([v for name,v,_ in parts if not view.get('parts') or name in view['parts']])
    yaw=math.radians(o.get('yaw',0.));q=np.r_[[0,0,0],np.log(dims),yaw]
    local=_transform(points/dims[[0,2,1]],q);camera=local@R.T
    center_depth=z-float(np.percentile(camera[:,2],15))
    pixel=np.r_[(box[:2]+box[2:])/2,1.]
    center=C+(np.linalg.solve(K,pixel)*center_depth)@R
    o.update(pos=(center-(local.min(0)+local.max(0))/2).tolist(),size=dims.tolist(),abs_z=True)
    q=np.r_[o['pos'],np.log(dims),yaw];canonical=points/dims[[0,2,1]]
    radius=max(1.,z*.5);lower=q-np.r_[radius,radius,radius,.8,.8,.8,.01];upper=q+np.r_[radius,radius,radius,.8,.8,.8,.01]
    if o.get('size_locked'):lower[3:6]=q[3:6]-1e-6;upper[3:6]=q[3:6]+1e-6
    def residual(trial):
        world=_transform(canonical,trial);_,depths=_uv(world,view)
        depth_weight=.15 if o.get('size_locked') else 1.
        errors=[box_error(world,view)*3,[(np.percentile(depths,15)-z)/max(.15,view['depth_sigma'])*depth_weight]]
        if view.get('ground_contact'):
            errors.append([ground_error(trial,view)/max(.1,np.exp(trial[4])*.1)*2])
        if kind in ROUND:errors.append([(trial[5]-trial[3])*.5])
        return np.concatenate(errors)
    result=least_squares(residual,q,bounds=(lower,upper),max_nfev=70,diff_step=.001)
    o.update(pos=result.x[:3].tolist(),size=dims.tolist() if o.get('size_locked') else np.exp(result.x[3:6]).tolist(),yaw=float(np.degrees(result.x[6])))


def _uv(points, obs):
    camera=(np.asarray(points)-obs['C'])@np.asarray(obs['R']).T
    z=camera[:,2];uv=camera@np.asarray(obs['K']).T
    return uv[:,:2]/np.maximum(z[:,None],.02),z


def _parts(kind,dims):
    return [(name,np.asarray(v,float),f) for name,v,f,_ in props.build(kind,*dims,seed=0)]


def _transform(points,q):
    dims=np.exp(q[3:6])[[0,2,1]];a=q[6];ca,sa=np.cos(a),np.sin(a)
    return (points*dims)@np.array([[ca,sa,0],[-sa,ca,0],[0,0,1.]])+q[:3]


def _outline(points, count=80):
    if len(points)<3:return points
    hull=cv2.convexHull(np.asarray(points,np.float32)).reshape(-1,2)
    edge=np.roll(hull,-1,axis=0)-hull;length=np.linalg.norm(edge,axis=1)
    distance=np.linspace(0,length.sum(),count,endpoint=False)
    start=np.r_[0,np.cumsum(length)];i=np.minimum(np.searchsorted(start,distance,side='right')-1,len(hull)-1)
    t=(distance-start[i])/np.maximum(length[i],1e-6)
    return hull[i]+t[:,None]*edge[i]


def _prepare(obs):
    obs=dict(obs);mask=unbits(obs['mask'],obs['shape']);occluder=unbits(obs['occluder'],obs['shape'])
    y,x=np.nonzero(mask);span=max(6.,float(max(np.ptp(x),np.ptp(y))))
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    target=_outline(np.concatenate(contours).reshape(-1,2),80)
    near=cv2.dilate(occluder.astype(np.uint8),np.ones((3,3),np.uint8))>0
    xy=np.rint(target).astype(int);target=target[~near[np.clip(xy[:,1],0,mask.shape[0]-1),np.clip(xy[:,0],0,mask.shape[1]-1)]]
    if len(target)<8:target=_outline(np.c_[x,y],40)
    obs.update(target=target,span=span,mask_array=mask,occluder_array=occluder,
               outside=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3))
    if obs.get('ground_definition'):
        from build_env import Ground
        obs['_ground']=Ground(obs['ground_definition'])
    return obs


def ground_error(q, view):
    ground=view.get('_ground')
    if ground is None and view.get('ground_definition'):
        from build_env import Ground
        ground=Ground(view['ground_definition'])
    if ground is not None:return q[2]-ground.at(q[0],q[1])
    a,b,c=view['ground_plane'];return q[2]-a*q[0]-b*q[1]-c


@stage('prototype_fitting')
def fit(o, max_evaluations=45, _recovery=True):
    """Optimize x/y/z, width/height/depth and yaw against all linked visible observations."""
    if not o.get('_observations') or o.get('type') in ('mesh','patch'):return o
    kind=props.norm_type(o['type']);obs=[_prepare(s) for s in o['_observations'][:5]]
    dims=np.maximum(.03,o['size']);q0=np.r_[o['pos'],np.log(dims),math.radians(o.get('yaw',0))]
    parts=_parts(kind,dims)
    canonical={name:v/np.asarray(dims)[[0,2,1]] for name,v,_ in parts}
    sets=[]
    for view in obs:
        chosen=[v for name,v in canonical.items() if not view.get('parts') or name in view['parts']]
        if not chosen:chosen=list(canonical.values())
        sets.append(np.concatenate(chosen))
    annotated=any(v.get('primary') for v in obs)
    weights=np.array([1. if not annotated or v.get('primary') else v.get('fit_weight',.12) for v in obs])
    def residual(q):
        residuals=[]
        for view,points,weight in zip(obs,sets,weights):
            world=_transform(points,q);uv,z=_uv(world,view)
            outline=_outline(uv,80)
            # Symmetric contour error prevents shrinking a prototype to a few interior points.
            dist=cKDTree(outline).query(view['target'])[0]
            outside=map_coordinates(view['outside'],[outline[:,1],outline[:,0]],order=1,mode='constant',cval=view['span'])
            occluded=map_coordinates(view['occluder_array'].astype(float),[outline[:,1],outline[:,0]],order=0,mode='constant',cval=0)>.5
            outside[occluded]=0
            residuals.extend((dist/view['span']*math.sqrt(weight),outside/view['span']*math.sqrt(weight)))
            if not view['partial']:
                residuals.append((np.median(uv,axis=0)-np.median(view['target'],axis=0))/view['span']*math.sqrt(weight))
            residuals.append(np.array([(np.percentile(z,15)-view['depth'])/view['depth_sigma']])*(.6 if view.get('primary') else .15)*math.sqrt(weight))
            if fit_box(view):residuals.append(box_error(world,view)*3)
            if view.get('ground_contact'):
                residuals.append(np.array([ground_error(q,view)/max(.1,np.exp(q[4])*.1)*2]))
            residuals.append(np.maximum(.02-z,0)*2)
        residuals.append((q[3:6]-q0[3:6])*(100 if o.get('size_locked') else .18))
        if kind in ROUND:residuals.append(np.array([(q[5]-q[3])*.7]))
        residuals.append(np.array([(q[6]-q0[6])*(10 if o.get('facing') in ('left','right','away') else .015)]))
        return np.concatenate(residuals)
    radius=max(1.,float(np.linalg.norm(np.asarray(o['pos'])-obs[0]['C']))*.6)
    lower=q0-np.r_[radius,radius,radius,1.1,1.1,1.1,np.pi]
    upper=q0+np.r_[radius,radius,radius,1.1,1.1,1.1,np.pi]
    if kind in ('chair','office_chair') and not o.get('size_locked'):
        lower[4]=math.log(.6);upper[4]=math.log(1.6);q0[4]=np.clip(q0[4],lower[4]+1e-5,upper[4]-1e-5)
    if o.get('size_locked'):lower[3:6]=q0[3:6]-1e-5;upper[3:6]=q0[3:6]+1e-5
    if o.get('facing') in ('left','right','away'):lower[6]=q0[6]-1e-5;upper[6]=q0[6]+1e-5
    best=q0.copy();initial=float(np.mean(residual(q0)**2));score=initial;solutions=[q0.copy()]
    starts=[0.] if kind in ROUND or o.get('facing') in ('left','right','away') else [0.,np.pi/2,-np.pi/2]
    for heading in starts:
        start=np.clip(q0+np.r_[np.zeros(6),heading],lower+1e-6,upper-1e-6)
        result=least_squares(residual,start,bounds=(lower,upper),max_nfev=max_evaluations,
                             diff_step=.002,loss='soft_l1',f_scale=.15)
        value=float(np.mean(residual(result.x)**2))
        solutions.append(result.x)
        if value<score:best,score=result.x,value
    def silhouette_objective(q):
        trial=dict(o,pos=q[:3].tolist(),size=np.exp(q[3:6]).tolist(),yaw=float(np.degrees(q[6])))
        errors=[]
        for view,points in zip(obs,sets):
            predicted=raster(trial,view);m=metrics(predicted,view)
            center=m['center_error_px']/max(1.,view['span']*view['source_scale'])
            world=_transform(points,q);_,depths=_uv(world,view)
            depth_error=abs(np.percentile(depths,15)-view['depth'])/view['depth_sigma']
            value=1-m['iou']+.5*min(3.,center)+.12*min(3.,m['size_error_fraction'])+.08*min(5.,depth_error)
            if fit_box(view):
                actual_dims=np.exp(q[3:6]);actual_parts=_parts(kind,actual_dims)
                actual_points=np.concatenate([v for name,v,_ in actual_parts if not view.get('parts') or name in view['parts']])
                actual_world=_transform(actual_points/actual_dims[[0,2,1]],q)
                error=np.abs(box_error(actual_world,view))
                value+=2*error.mean()+40*max(0.,error.max()-.025)
            if view.get('ground_contact'):
                value+=2*abs(ground_error(q,view))/max(.1,np.exp(q[4])*.1)
            errors.append(value)
        regularization=np.mean((q[3:6]-q0[3:6])**2)*(.3 if o.get('size_locked') else .025)
        if kind in ROUND:regularization+=(q[5]-q[3])**2*.08
        return float(np.average(errors,weights=weights)+.25*max(np.asarray(errors)*weights)+regularization)
    initial_silhouette=silhouette_objective(q0)
    best=min(solutions,key=silhouette_objective).copy()
    # Projected rectangles have multiple orientation minima; compare fitted headings.
    seeds=sorted(solutions,key=silhouette_objective)[:(3 if kind in ('table','desk','counter','ceiling_light') else 1)]
    for seed in seeds:
        refined=minimize(silhouette_objective,seed,method='Powell',bounds=list(zip(lower,upper)),
                         options={'maxfev':350,'xtol':.002,'ftol':.002})
        if silhouette_objective(refined.x)<silhouette_objective(best):best=refined.x
    if annotated and not o.get('size_locked'):
        primary_error=max(np.abs(box_error(_transform(points,best),view)).max() for view,points in zip(obs,sets) if fit_box(view))
        if primary_error>.04:best=q0.copy()
    score=float(np.mean(residual(best)**2))
    o.update(pos=best[:3].tolist(),size=np.exp(best[3:6]).tolist(),yaw=float(np.degrees(best[6])),abs_z=True)
    if o.get('size_locked'):o['size']=np.asarray(dims).tolist()
    # Bbox placement is now a joint fit; build_env must not replace it with a second guess.
    if 'bbox' in o:o['_annotation_box']=o.pop('bbox')
    o.pop('face',None)
    o['_fit']={'version':VERSION,'before':initial,'after':score,'observations':len(obs),
               'silhouette_before':initial_silhouette,'silhouette_after':silhouette_objective(best),
               'annotation_anchored':any(v.get('annotation_box') for v in obs),
               'detection_anchored':any(v.get('detection_box') for v in obs),
               'verification_only_frames':[v['frame'] for v in obs if v.get('fit_weight')==0],
               'partial':any(v['partial'] for v in obs),'parameters':['x','y','z','width','height','depth','yaw']}
    if _recovery and o.get('_observations') and o['_observations'][0].get('detection_box'):
        recover_shrub(o,max_evaluations)
    return o


@stage('vegetation_recovery')
def recover_shrub(o,max_evaluations=25):
    """Bounded local-model recovery: compare shrub prototypes against reliable masks without user edits."""
    if o.get('type') not in ('bush','hedge') or (o.get('_src') or {}).get('cls')=='drawn':return
    views=[v for v in o['_observations'] if v.get('primary') or v.get('fit_weight',.12)>0]
    def quality(obj):return float(np.average([metrics(raster(obj,v),v)['iou'] for v in views],weights=[1. if v.get('primary') else .12 for v in views]))
    before=quality(o)
    if before>=.55:return
    alternative='hedge' if o['type']=='bush' else 'bush';original=o['type']
    trial=copy.deepcopy(o);trial.update(type=alternative,prototype=alternative)
    seed_annotation(trial,trial['_observations'][0]);fit(trial,min(25,max_evaluations),_recovery=False)
    after=quality(trial)
    if after>before+.04:o.clear();o.update(trial)
    o['_fit']['automatic_recovery']={'initial_type':original,'selected_type':o['type'],'attempts':1,
                                    'iou_before':before,'iou_after':after if o['type']==alternative else before}


def vet_observation(o, view):
    """Retain doubtful views for verification, without letting them displace the edited anchor."""
    dims=np.asarray(o['size']);parts=view.get('parts')
    points=np.concatenate([v for name,v,_ in _parts(props.norm_type(o['type']),dims) if not parts or name in parts])
    q=np.r_[o['pos'],np.log(dims),np.radians(o.get('yaw',0))]
    projected=bounds(_transform(points/dims[[0,2,1]],q),view)
    y,x=np.nonzero(unbits(view['mask'],view['shape']))
    target=np.array([x.min(),y.min(),x.max(),y.max()]);span=np.maximum(target[2:]-target[:2],2.)
    if projected is None or np.any(projected[2:]<=projected[:2]):bad=True
    else:
        center=np.abs((projected[:2]+projected[2:]-target[:2]-target[2:])/2)/span
        ratio=(projected[2:]-projected[:2])/span
        bad=center.max()>.75 or ratio.max()>2.5 or ratio.min()<.4
    view['fit_weight']=0. if bad else .12
    if bad:view['association_warning']='Mask extent disagrees with the saved camera projection; verification only.'
    return view


def annotation_replaces(previous, edited):
    """An edited instance replaces overlapping automatic prototypes even from another sampled frame."""
    views=edited.get('_observations',[])
    if not views or not views[0].get('annotation_box') or previous.get('type') in ('mesh','patch'):return False
    if not previous.get('pos') or not previous.get('size'):return False
    def family(kind):
        kind=props.norm_type(kind)
        if kind in ('pine','tree','palm','forest'):return 'tree'
        if kind in ('bush','hedge'):return 'shrub'
        if kind in ('car','truck','van','bus'):return 'vehicle'
        if kind in ('chair','office_chair'):return 'chair'
        return kind
    if family(previous.get('type'))!=family(edited.get('type')):return False
    dims=np.asarray(previous['size']);points=np.concatenate([v for _,v,_ in _parts(props.norm_type(previous['type']),dims)])
    q=np.r_[previous['pos'],np.log(dims),np.radians(previous.get('yaw',0))]
    a=bounds(_transform(points/dims[[0,2,1]],q),views[0]);b=np.asarray(views[0]['annotation_box'])
    if a is None:return False
    overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return overlap/max(1.,min(np.maximum(0,a[2:]-a[:2]).prod(),np.maximum(0,b[2:]-b[:2]).prod()))>.55


_flows = {}


def flow_mask(view, first, second):
    """Associate static instances using image evidence when camera parallax is uncertain."""
    shape=tuple(view['shape'])
    def gray(image):
        image=cv2.resize(image,shape[::-1],interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    a,b=gray(first),gray(second)
    key=(shape,hashlib.sha1(a.tobytes()).digest(),hashlib.sha1(b.tobytes()).digest())
    if key not in _flows:
        if len(_flows)>=16:_flows.clear()
        _flows[key]=cv2.calcOpticalFlowFarneback(a,b,None,.5,4,21,5,7,1.5,0)
    mask=unbits(view['mask'],shape);y,x=np.nonzero(mask)
    xy=np.rint(np.c_[x,y]+_flows[key][y,x]).astype(int)
    valid=(xy[:,0]>=0)&(xy[:,0]<shape[1])&(xy[:,1]>=0)&(xy[:,1]<shape[0])
    result=np.zeros(shape,np.uint8);result[xy[valid,1],xy[valid,0]]=1
    return cv2.morphologyEx(result,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0


def match(objects, candidate, images=None):
    if candidate.get('type') in ('mesh','patch'):return None
    if not candidate.get('_observations'):return None
    obs=candidate['_observations'][0];mask=unbits(obs['mask'],obs['shape']);y,x=np.nonzero(mask)
    center=np.array([np.median(x),np.median(y)]);span=max(6.,float(max(np.ptp(x),np.ptp(y))))
    best=None;score=.8
    for previous in objects:
        if not previous.get('_observations'):continue
        if previous.get('type')!=candidate.get('type') and {previous.get('type'),candidate.get('type')}-{'bush','hedge'}:continue
        same_frame=any(v['frame']==obs['frame'] for v in previous['_observations'])
        if same_frame and candidate['type'] not in ('table','desk','counter'):continue
        old=previous['_observations'][0]
        # Reproject the actual observed surface anchor, independently of invisible prototype parts.
        target=np.mean(np.asarray(_prepare(old)['target']),axis=0)
        camera=np.linalg.solve(np.asarray(old['K']),np.r_[target,1.])*old['depth']
        world=np.asarray(old['C'])+camera@np.asarray(old['R'])
        uv,z=_uv(world[None],obs)
        value=np.linalg.norm(uv[0]-center)/span+abs(float(z[0])-obs['depth'])/max(1.,obs['depth'])*.5
        oldmask=unbits(old['mask'],old['shape']);yy,xx=np.nonzero(oldmask)
        take=np.linspace(0,len(xx)-1,min(400,len(xx)),dtype=int)
        rays=np.c_[xx[take],yy[take],np.ones(len(take))]@np.linalg.inv(np.asarray(old['K'])).T
        world_points=np.asarray(old['C'])+(rays*old['depth'])@np.asarray(old['R'])
        pixels,depths=_uv(world_points,obs);projected=np.zeros(tuple(obs['shape']),np.uint8)
        if len(pixels)>=3:cv2.fillConvexPoly(projected,cv2.convexHull(np.rint(pixels).astype(np.int32)),1)
        if images is not None and old['frame'] in images and obs['frame'] in images:
            projected=flow_mask(old,images[old['frame']],images[obs['frame']])
        intersection=(projected.astype(bool)&mask).sum()
        overlap=intersection/max(1,min(projected.sum(),mask.sum()))
        iou=intersection/max(1,(projected.astype(bool)|mask).sum())
        if same_frame and mask.sum()<.35*oldmask.sum():
            # Table edges are often segmented separately from the reflective top.
            adjacent=cv2.dilate(oldmask.astype(np.uint8),np.ones((7,7),np.uint8))>0
            overlap=max(overlap,float((adjacent&mask).sum()/max(1,mask.sum())))
        if overlap>.45 and abs(float(z[0])-obs['depth'])<max(.4,.25*obs['depth']):
            value=.65*(1-iou)+.15*min(value,2)
        else:continue
        if same_frame:
            # Only absorb a small coplanar fragment, never two complete adjacent furniture instances.
            ratio=mask.sum()/max(1,oldmask.sum())
            if ratio>.35 or overlap<.35:continue
        if value<score:best,score=previous,value
    return best


def planar_surface(R,fr,name,source,colour):
    """A clean planar silhouette, instead of extending noisy wall points down to a guessed floor."""
    mask=R['mask']&fr.valid&~fr.bars
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=[c for c in contours if cv2.contourArea(c)>=16]
    if not contours:return None
    normal=np.asarray(R['plane'][0],float).copy();normal[2]=0
    center=np.median(R['Pabs'],axis=0);direction=center-fr.Cl
    if np.linalg.norm(normal)<1e-8:normal=direction.copy();normal[2]=0
    normal/=max(1e-8,np.linalg.norm(normal))
    if abs(normal@direction)/max(.1,np.linalg.norm(direction))<.2:
        normal=direction.copy();normal[2]=0;normal/=max(1e-8,np.linalg.norm(normal))
    vertices=[];faces=[]
    for contour in contours:
        polygon=cv2.approxPolyDP(contour,max(1.,.003*cv2.arcLength(contour,True)),True).reshape(-1,2)
        if len(polygon)<3:continue
        rays=np.c_[(polygon[:,0]-fr.W/2)/fr.fpix,(polygon[:,1]-fr.H/2)/fr.fpix,np.ones(len(polygon))]@fr.Rw.T
        denominator=rays@normal;numerator=(center-fr.Cl)@normal
        distance=np.divide(numerator,denominator,out=np.zeros(len(rays)),where=np.abs(denominator)>1e-6)
        fallback=float(np.median(fr.P[mask,2]))*fr.s
        invalid=(distance<.1)|(distance>max(2.,R['rng']*3))
        distance[invalid]=fallback
        world=fr.Cl+rays*distance[:,None];offset=len(vertices)
        vertices.extend(world.tolist());faces.extend([[offset+i for i in face] for face in triangulate(polygon)])
    if not faces:return None
    result={'name':name,'type':'mesh','v':vertices,'f':faces,'colour':colour,'_src':source,'_mask_patch':True,
            '_fit':{'version':VERSION,'planar':True,'observations':1}}
    view=observation(fr,R['mask'])
    if view:result['_observations']=[view]
    return result


def triangulate(polygon):
    """Ear clipping retains concave silhouettes; triangle fans fill outside concave walls."""
    p=np.asarray(polygon,float);ids=list(range(len(p)));result=[]
    area=sum(p[i,0]*p[(i+1)%len(p),1]-p[(i+1)%len(p),0]*p[i,1] for i in ids)
    sign=1 if area>0 else -1
    for _ in range(len(p)**2):
        if len(ids)<=3:break
        found=False
        for k,b in enumerate(ids):
            a=ids[k-1];c=ids[(k+1)%len(ids)]
            ab=p[b]-p[a];ac=p[c]-p[a];cross=ab[0]*ac[1]-ab[1]*ac[0]
            if cross*sign<=1e-8:continue
            rest=[i for i in ids if i not in (a,b,c)];q=p[rest]-p[a]
            u=(q[:,0]*ac[1]-q[:,1]*ac[0])/cross
            v=(ab[0]*q[:,1]-ab[1]*q[:,0])/cross
            if ((u>1e-7)&(v>1e-7)&(u+v<1-1e-7)).any():continue
            result.append([a,b,c]);ids.pop(k);found=True;break
        if not found:break
    if len(ids)==3:result.append(ids)
    return result


def link(previous,candidate):
    for view in candidate['_observations']:
        old=next((v for v in previous['_observations'] if v['frame']==view['frame']),None)
        if old is None:
            if view.get('detection_box'):
                view=dict(view,primary=False);vet_observation(previous,view)
            previous['_observations'].append(view);continue
        mask=unbits(old['mask'],old['shape'])|unbits(view['mask'],view['shape'])
        excluded=unbits(old['occluder'],old['shape'])&~mask
        old.update(mask=bits(mask),occluder=bits(excluded))


def projected_box(o,data,frame,dx=0.):
    """A static prototype's visible component projected into another source frame."""
    from build_env import Cam
    dims=np.asarray(o['size']);parts=o['_observations'][0].get('parts')
    chosen=[v for name,v,_ in _parts(props.norm_type(o['type']),dims) if not parts or name in parts]
    if not chosen:return None
    q=np.r_[o['pos'],np.log(dims),np.radians(o.get('yaw',0))]
    points=_transform(np.concatenate(chosen)/dims[[0,2,1]],q)+[dx,0,0]
    uv,z=Cam(data).project(points,frame)
    if (z<=.1).any():return None
    lo,hi=uv.min(0),uv.max(0)
    pad=np.maximum(5.,(hi-lo)*(.08 if o['_observations'][0].get('primary') else .25))
    low=np.maximum([0,0],lo-pad);high=np.minimum(data['res'],hi+pad)
    if np.any(high-low<5):return None
    return [*low,*high]


def raster(o, view, selected=True):
    shape=tuple(view['shape']);result=np.zeros(shape,np.uint8)
    dims=np.asarray(o['size']);q=np.r_[o['pos'],np.log(dims),np.radians(o.get('yaw',0))]
    for name,v,faces in _parts(props.norm_type(o['type']),dims):
        if selected and view.get('parts') and name not in view['parts']:continue
        uv,z=_uv(_transform(v/dims[[0,2,1]],q),view)
        if (z<=.02).any():continue
        hull=cv2.convexHull(np.clip(np.rint(uv),-10000,10000).astype(np.int32))
        cv2.fillConvexPoly(result,hull,1)
    result[unbits(view['occluder'],shape)]=0
    return result.astype(bool)


def metrics(predicted, view):
    target=unbits(view['mask'],view['shape']);union=(target|predicted).sum();intersection=(target&predicted).sum()
    def measure(mask):
        y,x=np.nonzero(mask)
        return (np.array([x.mean(),y.mean()]),np.array([np.ptp(x)+1,np.ptp(y)+1])) if len(x) else (np.array([1e4,1e4]),np.zeros(2))
    tc,ts=measure(target);pc,ps=measure(predicted);scale=view['source_scale']
    return {'frame':view['frame'],'iou':float(intersection/max(1,union)),
            'center_error_px':float(np.linalg.norm(pc-tc)*scale),
            'size_error_fraction':float(np.max(np.abs(ps-ts)/np.maximum(ts,1))),
            'partial':view['partial']}


def source_metrics(points, view):
    target_box=fit_box(view)
    if not target_box:return {}
    prefix='annotation' if view.get('annotation_box') else 'detection'
    actual=bounds(points,view);target=np.asarray(target_box);span=np.maximum(target[2:]-target[:2],2.)
    if actual is None:return {prefix+'_edge_error':10.,prefix+'_center_error_px':1e5,prefix+'_size_error_fraction':10.}
    return {prefix+'_edge_error':float(np.max(np.abs(actual-target)/np.tile(span,2))),
            prefix+'_center_error_px':float(np.linalg.norm((actual[:2]+actual[2:]-target[:2]-target[2:])/2)*view['source_scale']),
            prefix+'_size_error_fraction':float(np.max(np.abs(actual[2:]-actual[:2]-span)/span)),
            'projected_box_px':(actual*view['source_scale']).tolist()}


def annotation_metrics(points, view):
    return source_metrics(points,view) if view.get('annotation_box') else {}


def placement_error(view):
    return max(view.get('annotation_edge_error',0),view.get('detection_edge_error',0))


def passes(views):
    return all(v['iou']>=.35 and v['size_error_fraction']<=.6 and v['center_error_px']<=24 and
               placement_error(v)<=.12 for v in views)


@stage('silhouette_verification')
def audit_objects(objects, work, suffix=''):
    rows=[]
    for o in objects:
        if not o.get('_fit') or o.get('type') in ('mesh','patch'):continue
        dims=np.asarray(o['size']);q=np.r_[o['pos'],np.log(dims),np.radians(o.get('yaw',0))]
        views=[]
        for v in o['_observations']:
            points=np.concatenate([p for name,p,_ in _parts(props.norm_type(o['type']),dims) if not v.get('parts') or name in v['parts']])
            views.append(dict(metrics(raster(o,v),v),**source_metrics(_transform(points/dims[[0,2,1]],q),v)))
        passed=passes(views)
        row={'object':o.get('name'),'type':o['type'],'passed':passed,'views':views,'fit':o['_fit']}
        o['_fit']['validation_passed']=passed;rows.append(row)
    jsave(os.path.join(work,'object_alignment%s.json'%suffix),{'version':VERSION,'objects':rows,
           'passed':all(r['passed'] for r in rows),'checked':bool(rows)})
    return rows


@stage('silhouette_verification')
def audit_scene(data, work):
    """Check final exported vertices after building, placement correction and occlusion repair."""
    groups={}
    for mesh in data.get('meshes',[]):
        if mesh.get('_observations'):groups.setdefault(mesh.get('_primitive',mesh['name']),[]).append(mesh)
    rows=[]
    for name,group in groups.items():
        views=mesh_metrics(group)
        passed=passes(views)
        rows.append({'object':name,'type':group[0].get('kind'),'passed':passed,'views':views,
                     'fit':group[0].get('_object_fit'),'final_geometry':True})
    report={'version':VERSION,'objects':rows,'passed':all(r['passed'] for r in rows),'checked':bool(rows)}
    jsave(os.path.join(work,'object_alignment.json'),report)
    data['object_alignment']={'file':'object_alignment.json','checked_objects':len(rows),
                              'failed_objects':sum(not r['passed'] for r in rows),'passed':report['passed']}
    return report


def mesh_metrics(group):
    """Project actual exported faces, including any placement or occlusion repairs."""
    views=[]
    for view in group[0].get('_observations',[]):
        predicted=np.zeros(tuple(view['shape']),np.uint8)
        points=[]
        for mesh in group:
            if view.get('parts') and not any(mesh['name'].endswith('_'+p) for p in view['parts']):continue
            if len(mesh['v']):points.extend(mesh['v'])
            uv,z=_uv(mesh['v'],view)
            for face in mesh['f']:
                if len(face)<3 or (z[face]<=.02).any():continue
                poly=np.clip(np.rint(uv[face]),-10000,10000).astype(np.int32)
                cv2.fillPoly(predicted,[poly],1)
        predicted[unbits(view['occluder'],view['shape'])]=0
        views.append(dict(metrics(predicted>0,view),**source_metrics(points,view)))
    return views

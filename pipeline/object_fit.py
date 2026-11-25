"""Joint, visibility-aware fitting of recognizable prototypes to image/depth observations."""
import base64
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

VERSION = 1
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
    mask=np.asarray(mask,bool)&fr.valid&~fr.person&~fr.bars
    if mask.sum()<16:return None
    h,w=fr.H,fr.W;scale=min(1.,256/w);shape=(max(1,round(h*scale)),max(1,round(w*scale)))
    def resize(m):return cv2.resize(np.asarray(m,np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    p=fr.P[mask];z=p[:,2]*fr.s
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
    mask=np.asarray(mask,bool)&fr.valid&~fr.person&~fr.bars
    if mask.sum()<16:return o
    P=fr.X[mask];P=P[np.isfinite(P).all(1)]
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
    if obs:o.setdefault('_observations',[]).append(obs)
    return o


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
    return obs


@stage('prototype_fitting')
def fit(o, max_evaluations=45):
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
    def residual(q):
        residuals=[]
        for view,points in zip(obs,sets):
            uv,z=_uv(_transform(points,q),view)
            outline=_outline(uv,80)
            # Symmetric contour error prevents shrinking a prototype to a few interior points.
            dist=cKDTree(outline).query(view['target'])[0]
            outside=map_coordinates(view['outside'],[outline[:,1],outline[:,0]],order=1,mode='constant',cval=view['span'])
            occluded=map_coordinates(view['occluder_array'].astype(float),[outline[:,1],outline[:,0]],order=0,mode='constant',cval=0)>.5
            outside[occluded]=0
            residuals.extend((dist/view['span'],outside/view['span']))
            if not view['partial']:
                residuals.append((np.median(uv,axis=0)-np.median(view['target'],axis=0))/view['span'])
            residuals.append(np.array([(np.percentile(z,15)-view['depth'])/view['depth_sigma']])*.15)
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
        for view in obs:
            predicted=raster(trial,view);m=metrics(predicted,view)
            center=m['center_error_px']/max(1.,view['span']*view['source_scale'])
            errors.append(1-m['iou']+.5*min(3.,center)+.12*min(3.,m['size_error_fraction']))
        regularization=np.mean((q[3:6]-q0[3:6])**2)*(.3 if o.get('size_locked') else .025)
        if kind in ROUND:regularization+=(q[5]-q[3])**2*.08
        return float(np.mean(errors)+.25*max(errors)+regularization)
    initial_silhouette=silhouette_objective(q0)
    # Projected rectangles have multiple orientation minima; compare fitted headings.
    seeds=sorted(solutions,key=silhouette_objective)[:(3 if kind in ('table','desk','counter','ceiling_light') else 1)]
    for seed in seeds:
        refined=minimize(silhouette_objective,seed,method='Powell',bounds=list(zip(lower,upper)),
                         options={'maxfev':350,'xtol':.002,'ftol':.002})
        if silhouette_objective(refined.x)<silhouette_objective(best):best=refined.x
    score=float(np.mean(residual(best)**2))
    o.update(pos=best[:3].tolist(),size=np.exp(best[3:6]).tolist(),yaw=float(np.degrees(best[6])),abs_z=True)
    if o.get('size_locked'):o['size']=np.asarray(dims).tolist()
    # Bbox placement is now a joint fit; build_env must not replace it with a second guess.
    if 'bbox' in o:o['_annotation_box']=o.pop('bbox')
    o.pop('face',None)
    o['_fit']={'version':VERSION,'before':initial,'after':score,'observations':len(obs),
               'silhouette_before':initial_silhouette,'silhouette_after':silhouette_objective(best),
               'partial':any(v['partial'] for v in obs),'parameters':['x','y','z','width','height','depth','yaw']}
    return o


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
        if previous.get('type')!=candidate.get('type') or not previous.get('_observations'):continue
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
        if old is None:previous['_observations'].append(view);continue
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
    lo,hi=uv.min(0),uv.max(0);pad=np.maximum(5.,(hi-lo)*.25)
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


@stage('silhouette_verification')
def audit_objects(objects, work, suffix=''):
    rows=[]
    for o in objects:
        if not o.get('_fit') or o.get('type') in ('mesh','patch'):continue
        views=[metrics(raster(o,v),v) for v in o['_observations']]
        passed=all(v['iou']>=.35 and v['size_error_fraction']<=.6 and v['center_error_px']<=24 for v in views)
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
        passed=all(v['iou']>=.35 and v['size_error_fraction']<=.6 and v['center_error_px']<=24 for v in views)
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
        for mesh in group:
            if view.get('parts') and not any(mesh['name'].endswith('_'+p) for p in view['parts']):continue
            uv,z=_uv(mesh['v'],view)
            for face in mesh['f']:
                if len(face)<3 or (z[face]<=.02).any():continue
                poly=np.clip(np.rint(uv[face]),-10000,10000).astype(np.int32)
                cv2.fillPoly(predicted,[poly],1)
        predicted[unbits(view['occluder'],view['shape'])]=0
        views.append(metrics(predicted>0,view))
    return views

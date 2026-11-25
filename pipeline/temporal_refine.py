"""Shot-wide arm refinement after interpolation, preserving fixed geometry and roots."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.ndimage import minimum_filter1d


def arms(rig, values, frames, keys, world, y0, fps=30.):
    from fit_people import ANG, NB, fk_b, coco_b, merge_b, bounds
    original = np.asarray(values, float)
    n = len(original)
    if n > 180:
        # Bound the sparse solve cost on long shots. Overlapping windows share
        # observed motion and taper corrections to avoid visible window seams.
        total=np.zeros_like(original); coverage=np.zeros(n); reports=[]
        class WindowWorld:
            cam=world.cam
            def __init__(self,start):self.start=start
            def R_cw(self,f):return world.R_cw(f+self.start)
            def C_at(self,f):return world.C_at(f+self.start)
            def f_at(self,f):return world.f_at(f+self.start)
        frames=np.asarray(frames,int); keys=np.asarray(keys,float)
        for start in range(0,n,150):
            end=min(n,start+180); selected=(frames>=start)&(frames<end)
            if selected.sum()<4:
                proposal=original[start:end]; info={'applied':False,'reason':'insufficient observed arms'}
            else:
                proposal,info=arms(rig,original[start:end],frames[selected]-start,keys[selected],WindowWorld(start),y0,fps)
            weight=np.ones(end-start)
            if start:weight[:min(30,len(weight))]=np.linspace(.05,1.,min(30,len(weight)))
            if end<n:weight[-30:]=np.linspace(1.,.05,30)
            total[start:end]+=proposal*weight[:,None];coverage[start:end]+=weight
            reports.append(info)
        candidate=total/coverage[:,None]
        baseline=merge_b(fk_b(rig,original),rig.s).max(1)
        unsafe=merge_b(fk_b(rig,candidate),rig.s).max(1)>np.maximum(.7,baseline)+.02
        candidate[unsafe]=original[unsafe]
        return candidate,{'applied':any(r['applied'] for r in reports),'windows':reports,
                          'scope':'Overlapping shot-local windows; roots/cameras/scales fixed'}
    columns = [NB+i for i,(joint,*_) in enumerate(ANG) if joint.startswith(('shoulder_','elbow_'))]
    count = len(columns)
    ids = [7,8,9,10]
    observed = np.zeros((n,17,3))
    observed[np.asarray(frames,int)] = keys
    observed[:,:,:2] -= [0,y0]
    weights = np.sqrt(np.where(observed[:,ids,2] >= .5, observed[:,ids,2], 0.))
    if n < 4 or np.count_nonzero(weights) < 8:
        return original, {'applied':False,'reason':'insufficient observed arms'}
    rotations = np.asarray([world.R_cw(f) for f in range(n)])
    cameras = np.asarray([world.C_at(f) for f in range(n)])
    focal = np.asarray([world.f_at(f) for f in range(n)])
    first,last = min(frames),max(frames)
    weights[:first] = 0.; weights[last+1:] = 0.
    def projection(q):
        points = coco_b(rig, fk_b(rig,q))[:,ids]
        camera_points = np.einsum('nij,nkj->nki',rotations,points-cameras[:,None])
        return camera_points[:,:,:2] / np.maximum(camera_points[:,:,2:3],.05) * focal[:,None,None] + [world.cam.W/2,world.cam.H/2]
    target = observed[:,ids,:2]
    baseline_collision = merge_b(fk_b(rig,original),rig.s)
    collision_count = baseline_collision.shape[1]
    def residual(x):
        q = original.copy(); q[:,columns] = x.reshape(n,count)
        delta = q[:,columns]-original[:,columns]
        reprojection = (projection(q)-target)*weights[:,:,None]/4.
        # Retain the depth/branch solution and fast observed gestures, while
        # penalizing unsupported acceleration rather than all motion velocity.
        acceleration = np.diff(q[:,columns],n=2,axis=0)/max(4.,8.*(30./max(fps,1.))**2)
        collision = np.maximum(0.,merge_b(fk_b(rig,q),rig.s)-np.maximum(.7,baseline_collision))
        return np.r_[reprojection.ravel(),delta.ravel()/18.,acceleration.ravel(),collision.ravel()*2.]
    sizes = [n*8,n*count,max(0,n-2)*count,n*collision_count]
    sparsity = lil_matrix((sum(sizes),n*count),dtype=int)
    offset = 0
    for size,per_frame,shift in [(sizes[0],8,0),(sizes[1],count,0),(sizes[2],count,1),(sizes[3],collision_count,0)]:
        for index in range(size//per_frame):
            f = index+shift
            a,b = (max(0,f-1),min(n,f+2)) if shift else (f,f+1)
            sparsity[offset+index*per_frame:offset+(index+1)*per_frame,a*count:b*count] = 1
        offset += size
    lo,hi = bounds(rig)
    lower = np.maximum(lo[columns],original[:,columns]-25.)
    upper = np.minimum(hi[columns],original[:,columns]+25.)
    initial = np.clip(original[:,columns],lower+1e-7,upper-1e-7)
    fit = least_squares(residual, initial.ravel(), bounds=(lower.ravel(),upper.ravel()),
                        jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=2.,max_nfev=12)
    candidate = original.copy(); candidate[:,columns] = fit.x.reshape(n,count)
    limit = np.maximum(.7,baseline_collision.max(1))+.02
    strength = np.ones(n)
    for fallback in (.5,.25,0.):
        trial = original+(candidate-original)*strength[:,None]
        strength[merge_b(fk_b(rig,trial),rig.s).max(1)>limit] = fallback
    strength = minimum_filter1d(strength,size=5,mode='nearest')
    candidate = original+(candidate-original)*strength[:,None]
    unsafe = merge_b(fk_b(rig,candidate),rig.s).max(1)>limit
    candidate[unsafe] = original[unsafe]
    # Do not animate extrapolated poses outside the track's shot-local span.
    candidate[:first] = original[:first]; candidate[last+1:] = original[last+1:]
    before = float(np.average(np.linalg.norm(projection(original)-target,axis=2),weights=weights))
    after = float(np.average(np.linalg.norm(projection(candidate)-target,axis=2),weights=weights))
    applied = bool(np.isfinite(after) and after < before*.99)
    return (candidate if applied else original), {'applied':applied,'before_px':round(before,3),
        'after_px':round(after if applied else before,3), 'scope':'Observed elbows/wrists, one continuous shot; root/camera/scale fixed'}

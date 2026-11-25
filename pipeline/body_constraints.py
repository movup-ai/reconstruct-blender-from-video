"""Visibility-aware SAM 3D Body priors. Image evidence remains the alignment authority."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

VERSION = 2


def stabilize_arms(parameters, confidence, angles, fps=30.):
    """Damp ambiguous joint-depth flips while retaining well-observed fast gestures."""
    result=parameters.copy()
    for i,(joint,*_) in enumerate(angles):
        if joint.startswith('shoulder_'):
            ids=(5,7,9) if joint.endswith('_L') else (6,8,10)
        elif joint.startswith('elbow_'):
            ids=(7,9) if joint.endswith('_L') else (8,10)
        else: continue
        visible=np.min(confidence[:,ids],axis=1)
        weight=np.clip((.65-visible)/.65,0,1)
        source=parameters[:,4+i]
        clean=gaussian_filter1d(median_filter(source,size=3,mode='nearest'),max(1.,.13*fps),mode='nearest')
        result[:,4+i]=source*(1-weight)+clean*weight
    return result


def full_body_support(keys):
    keys = np.asarray(keys, float)
    if not len(keys): return 0.
    # An isolated hallucinated ankle is not a ground contact observation.
    return float(((keys[:, 15:17, 2] >= .5).all(1) &
                  (keys[:, 11:13, 2] >= .4).all(1)).mean())


def confidence_h36m(coco):
    result = np.zeros((len(coco), 17), float)
    for target, source in ((1,12),(2,14),(3,16),(4,11),(5,13),(6,15),
                           (11,5),(12,7),(13,9),(14,6),(15,8),(16,10)):
        result[:,target] = coco[:,source]
    result[:,0] = coco[:,[11,12]].min(1)
    result[:,8] = coco[:,[5,6]].min(1)
    result[:,7] = np.minimum(result[:,0],result[:,8])
    result[:,9] = coco[:,0]
    result[:,10] = coco[:,:5].max(1)
    return result


def prepare(world, y0, frames, keys, sample_frames, joints, fps=30.):
    """Dense camera-space observations, per-joint confidence, and placement reliability.

    Disagreement with sure 2D joints reduces model influence. Hidden extremities get a
    weak, temporally stabilized prior rather than the former constant 0.8 confidence.
    """
    frames=np.asarray(frames,int); keys=np.asarray(keys,float)
    sf=np.asarray(sample_frames,int); joints=np.asarray(joints,float)
    dense=np.arange(int(sf[0]),int(sf[-1])+1)
    xyz=np.stack([np.interp(dense,sf,joints[:,j,a]) for j in range(17) for a in range(3)],1).reshape(-1,17,3)
    observed=np.stack([np.interp(dense,frames,keys[:,j,a]) for j in range(17) for a in range(3)],1).reshape(-1,17,3)
    observed[:,:,1]-=y0
    # Never invent observation confidence through a missing tracked interval.
    observed[~np.isin(dense,frames),:,2]=0
    fx=np.asarray([world.f_at(int(f)) for f in dense])
    uv=xyz[:,:,:2]/np.maximum(xyz[:,:,2:3],.05)*fx[:,None,None]+[world.cam.W/2,world.cam.H/2]
    error=np.linalg.norm(uv-observed[:,:,:2],axis=2)
    span=np.asarray([max(40.,np.ptp(k[k[:,2]>.3,1])) if (k[:,2]>.3).sum()>2 else 100. for k in observed])
    agreement=np.exp(-.5*(error/np.maximum(8.,.06*span[:,None]))**2)
    confidence=(.04+.86*np.clip(observed[:,:,2],0,1))*agreement
    confidence=np.maximum(.025,confidence)
    sure=observed[:,:,2]>.5
    reprojection=float(np.median(error[sure])) if sure.any() else float('inf')
    relative_error=float(np.median((error/span[:,None])[sure])) if sure.any() else 1.
    trust=float(np.clip(1-relative_error/.15,0,1))
    # Filter relative shape independently of translation, so a walking person keeps their path.
    center=xyz[:,[11,12]].mean(1)
    relative=xyz-center[:,None]
    sigma=max(.5,.10*fps)
    filtered=gaussian_filter1d(median_filter(relative,size=(3,1,1),mode='nearest'),sigma,axis=0,mode='nearest')
    blend=np.clip(1-confidence/.8,0,1)[:,:,None]
    stabilized=relative*(1-blend)+filtered*blend+center[:,None]
    return dense,stabilized,confidence,{'reprojection_px':reprojection,'placement_confidence':trust,
                                      'full_body_fraction':full_body_support(keys)}

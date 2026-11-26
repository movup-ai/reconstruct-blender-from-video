"""SAM 3D Body observations, metric surface calibration and MAVE human placement.

Inference stays in the Python 3.12 worker. Camera-space joints use OpenCV axes
and metres; the final scene remains the pipeline's editable mannequin rig.
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from common import AGENT, imwrite_u, jload, jsave
from profiling import stage

VERSION = 1
COCO_NAMES = ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder',
              'left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip',
              'left-knee','right-knee','left-ankle','right-ankle']


def sample_frames(person, stride=2, evidence=()):
    fs = sorted(int(f) for f in person.get('boxes',{}))
    if not fs: return []
    return sorted(set(fs[::max(1,int(stride))]+[fs[-1]]) | (set(fs)&set(evidence)))


def h36m(joints):
    joints = np.asarray(joints,float)
    result = np.zeros((len(joints),17,3),np.float32)
    for target,source in ((1,12),(2,14),(3,16),(4,11),(5,13),(6,15),
                          (11,5),(12,7),(13,9),(14,6),(15,8),(16,10)):
        result[:,target] = joints[:,source]
    result[:,0] = joints[:,[11,12]].mean(1)
    result[:,8] = joints[:,[5,6]].mean(1)
    result[:,7] = (result[:,0]+result[:,8])/2
    result[:,9] = joints[:,0]
    result[:,10] = joints[:,:5].mean(1)*1.5-result[:,8]*.5
    return result


def place(joints, neck, factors):
    """Paper equations 4-5: variable neck translation, constant track body scale."""
    joints,neck,factors = np.asarray(joints,float),np.asarray(neck,float),np.asarray(factors,float)
    if len(joints)!=len(neck) or len(neck)!=len(factors) or not np.isfinite(factors).all() or np.any(factors<=0):
        raise ValueError('Body placement needs positive finite factors at every observed frame')
    scale = float(np.median(factors))
    return factors[:,None,None]*neck[:,None]+scale*(joints-neck[:,None])


def height_of(joints):
    """Pose-independent anatomical chain length, robustly aggregated over a track."""
    joints = np.asarray(joints,float)
    shoulder = joints[:,[5,6]].mean(1); hip = joints[:,[11,12]].mean(1)
    legs = sum(np.linalg.norm(joints[:,a]-joints[:,b],axis=1) for a,b in ((11,13),(13,15),(12,14),(14,16)))/2
    estimates = legs+np.linalg.norm(shoulder-hip,axis=1)+1.25*np.linalg.norm(joints[:,0]-shoulder,axis=1)+.08
    good = estimates[np.isfinite(estimates)&(estimates>.4)&(estimates<2.5)]
    return float(np.median(good)) if len(good) else None


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
    polygons=camera[triangles(faces)]
    finite=np.isfinite(polygons).all((1,2));front=(polygons[:,:,2]>=.01).all(1)
    visible=finite&(polygons[:,:,2].max(1)>=.01)
    ids=np.flatnonzero(visible&front)
    if len(ids):
        uv=polygons[ids]@K.T;uv=uv[:,:,:2]/uv[:,:,2,None]
        low=uv.min(1);high=uv.max(1)
        visible[ids]&=(high>=0).all(1)&(low<np.asarray(shape)[::-1]).all(1)
    for poly in polygons[visible]:
        if (poly[:,2]<.01).any():poly = _clip_near(poly)
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


def surface_depth(vertices, faces, K, shape):
    """Vectorized triangle z-buffer, preserving visible surfaces instead of vertex splats."""
    vertices = np.asarray(vertices,float); faces = np.asarray(faces,int)
    result = np.full(shape,np.inf)
    for start in range(0,len(faces),1024):
        xyz = vertices[faces[start:start+1024]]
        good = np.isfinite(xyz).all((1,2)) & (xyz[:,:,2]>.01).all(1)
        xyz = xyz[good]
        if not len(xyz): continue
        uv = xyz @ K.T; uv = uv[:,:,:2]/uv[:,:,2:]
        low = np.maximum(np.floor(uv.min(1)),[0,0]).astype(int)
        high = np.minimum(np.ceil(uv.max(1)),[shape[1]-1,shape[0]-1]).astype(int)
        widths = high[:,0]-low[:,0]+1; heights = high[:,1]-low[:,1]+1
        good = (widths>0)&(heights>0)
        xyz,uv,low,widths,heights = xyz[good],uv[good],low[good],widths[good],heights[good]
        counts = widths*heights
        # Very near geometry can cover the image: bound each allocation.
        if counts.sum()>2_000_000:
            clipped = rasterize(vertices,faces[start:start+1024],K,np.eye(3),np.zeros(3),shape)
            np.minimum(result,clipped,out=result); continue
        ids = np.repeat(np.arange(len(counts)),counts)
        if not len(ids): continue
        offset = np.arange(len(ids))-np.repeat(np.cumsum(counts)-counts,counts)
        ix = low[ids,0]+offset%widths[ids]; iy = low[ids,1]+offset//widths[ids]
        x,y = ix+.5,iy+.5
        a,b,c = uv[ids,0],uv[ids,1],uv[ids,2]
        den = (b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
        inverse = np.divide(1,den,out=np.zeros_like(den),where=np.abs(den)>1e-9)
        u = ((b[:,1]-c[:,1])*(x-c[:,0])+(c[:,0]-b[:,0])*(y-c[:,1]))*inverse
        v = ((c[:,1]-a[:,1])*(x-c[:,0])+(a[:,0]-c[:,0])*(y-c[:,1]))*inverse
        w = 1-u-v
        inside = (np.abs(den)>1e-9)&(u>=-1e-7)&(v>=-1e-7)&(w>=-1e-7)
        inv_z = u/xyz[ids,0,2]+v/xyz[ids,1,2]+w/xyz[ids,2,2]
        good = inside&(inv_z>0)
        np.minimum.at(result,(iy[good],ix[good]),1/inv_z[good])
    return result


def surface_ratio(body_depth, measured, mask, valid):
    selected = valid & mask & np.isfinite(body_depth)&np.isfinite(measured)&(body_depth>.05)&(measured>.05)&(measured<200)
    if selected.sum()<20: return None
    ratios = body_depth[selected]/measured[selected]
    logs = np.log(ratios); center = np.median(logs)
    keep = np.abs(logs-center)<max(.1,3*1.4826*np.median(np.abs(logs-center)))
    return float(np.exp(np.median(logs[keep]))) if keep.sum()>=20 else None


def align_depth(body, explicit_scale=None):
    from metric_scene import robust_scale
    ratios = np.asarray(body.get('surface_ratios',[]),float)
    scale = float(explicit_scale) if explicit_scale is not None else robust_scale(ratios,minimum=1)
    if scale is not None and (not np.isfinite(scale) or scale<=0): raise ValueError('Metric depth scale must be positive and finite')
    calibrated = scale is not None
    body['meters_per_unit'] = np.asarray(scale if calibrated else 1.)
    heights = {}
    for key in list(body):
        if not key.endswith('_j'): continue
        prefix = key[:-2]; fs,joints,neck = body[prefix+'_f'],body[key],body[prefix+'_n']
        raw = np.asarray(body.get(prefix+'_neck_depth',np.full(len(fs),np.nan)),float)
        good = np.isfinite(raw)&(raw>.05)&(neck[:,2]>.05)
        factors = np.ones(len(fs))
        if calibrated and good.any():
            values = raw[good]*scale/neck[good,2]
            med = np.median(values); reliable = (values>.5*med)&(values<2*med)
            if reliable.any(): factors = np.interp(fs,fs[good][reliable],values[reliable])
        body[prefix+'_placed'] = place(joints,neck,factors)
        body[prefix+'_factors'] = factors
        h = height_of(body[prefix+'_placed'])
        if h is not None: heights[prefix[1:]] = h
    return {'calibrated':calibrated,'meters_per_unit':float(body['meters_per_unit']),
            'surface_instances':int(len(ratios)),'source':'manual scale' if explicit_scale is not None else 'SAM 3D Body visible mesh surfaces',
            'heights_m':heights}


@stage('sam3d_body')
def recover(frames, pose, cam, y0, camj, depth_samples, c):
    checkpoint = c.get('sam3d_checkpoint') or os.path.join(AGENT,'models','sam3d_body','model.ckpt')
    mhr = c.get('sam3d_mhr') or os.path.join(AGENT,'models','sam3d_body','assets','mhr_model.pt')
    python = c.get('sam3d_python') or c.get('sam3_python')
    config = os.path.join(os.path.dirname(checkpoint),'model_config.yaml')
    for path in (checkpoint,mhr,config,python or ''):
        if not os.path.isfile(path): raise RuntimeError('SAM 3D Body required file missing: '+path)
    sampling = {str(p['id']):sample_frames(p,c.get('sam3d_stride',2),depth_samples) for p in pose['people']}
    selected = sorted({f for fs in sampling.values() for f in fs})
    cache = os.path.join(c['_work'],'sam3d_body_raw%s.npz'%c.get('_shot_tag',''))
    digest = hashlib.sha256(json.dumps({'version':VERSION,'pose':pose,'K':cam.K().tolist(),'zoom':camj.get('zoom'),
        'samples':sampling,'assets':[(os.path.abspath(p),os.stat(p).st_size,os.stat(p).st_mtime_ns) for p in (checkpoint,mhr,config)]},sort_keys=True).encode())
    for f in selected: digest.update(frames[f][y0:y0+cam.H].tobytes())
    mask_path = os.path.join(c.get('_marks_dir',''),'sam3_masks.npz')
    if os.path.isfile(mask_path):
        with open(mask_path,'rb') as source:
            for block in iter(lambda:source.read(1024*1024),b''): digest.update(block)
    for f,sample in sorted(depth_samples.items()):
        digest.update(str(f).encode())
        for key in ('depth','K','valid'): digest.update(np.asarray(sample[key]).tobytes())
    key = digest.hexdigest(); body = None
    if os.path.isfile(cache):
        with np.load(cache,allow_pickle=False) as source:
            if str(source['cache_key'])==key: body = {k:source[k].copy() for k in source.files if k!='cache_key'}
    if body is None:
        with tempfile.TemporaryDirectory(dir=c['_work'],prefix='sam3d_') as tmp:
            pictures = os.path.join(tmp,'frames'); os.makedirs(pictures)
            for f in selected: imwrite_u(os.path.join(pictures,'%06d.jpg'%f),frames[f][y0:y0+cam.H],95)
            sample_path = os.path.join(tmp,'depth.npz')
            np.savez_compressed(sample_path,**{'%s_%d'%(k,f):v for f,s in depth_samples.items() for k,v in s.items()})
            job = os.path.join(tmp,'job.json'); result = os.path.join(tmp,'body.npz')
            jsave(job,{'checkpoint':checkpoint,'mhr':mhr,'pose':pose,'sampling':sampling,'pictures':pictures,
                'K':cam.K().tolist(),'zoom':camj.get('zoom',[1.]*len(frames)),'y0':y0,'mask_path':mask_path,
                'shot_start':c.get('_shot_start',0),'depth_path':sample_path,'surface_side':c.get('sam3d_surface_side',160)})
            subprocess.run([python,os.path.abspath(__file__),'--worker',job,result],check=True,
                timeout=max(60,float(c.get('sam3d_timeout_min',40))*60),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
                env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8'))
            with np.load(result,allow_pickle=False) as source: body = {k:source[k].copy() for k in source.files}
            np.savez_compressed(cache,cache_key=key,**body)
    if selected and not any(k.endswith('_j') for k in body):
        raise RuntimeError('SAM 3D Body produced no valid body estimates for the tracked people')
    summary = align_depth(body)
    summary['peak_vram_mb'] = float(body.get('peak_vram_mb',0.))
    summary['tracks'] = sum(k.endswith('_j') for k in body)
    summary['skipped_estimates'] = int(body.get('rejected_estimates',0))      # failed the per-frame camera check
    summary['checked_estimates'] = int(body.get('checked_estimates',0))
    path = os.path.join(c['_work'],'sam3d_body%s.npz'%c.get('_shot_tag',''))
    np.savez_compressed(path,**body)
    jsave(os.path.join(c['_work'],'sam3d_body_report%s.json'%c.get('_shot_tag','')),summary)
    return body,summary


def worker(job_path, output):
    sys.path.insert(0,os.path.join(AGENT,'third_party','sam-3d-body'))
    import torch
    from PIL import Image
    from sam_3d_body import load_sam_3d_body,SAM3DBodyEstimator
    from sam_3d_body.metadata.mhr70 import mhr_names
    from sam3_tracking import read_masks,mask_at
    if not torch.cuda.is_available(): raise RuntimeError('SAM 3D Body needs an NVIDIA GPU')
    job = jload(job_path)
    model,config = load_sam_3d_body(job['checkpoint'],device='cuda',mhr_path=job['mhr'])
    estimator = SAM3DBodyEstimator(model,config)
    bundle = read_masks(job['mask_path']) if os.path.isfile(job['mask_path']) else None
    indices = {(int(f),int(pid)):i for i,(f,pid) in enumerate(zip(bundle['frames'],bundle['ids']))} if bundle else {}
    mapping = [mhr_names.index(n) for n in COCO_NAMES]; neck_index = mhr_names.index('neck')
    with np.load(job['depth_path'],allow_pickle=False) as source:
        samples = {int(k.split('_')[-1]):{} for k in source.files}
        for key in source.files:
            name,f = key.rsplit('_',1); samples[int(f)][name] = source[key].copy()
    data,ratios,count = {},[],0
    checked,rejected = 0,[]
    with torch.inference_mode():
        for person in job['pose']['people']:
            pid,native = str(person['id']),person.get('sam3_id')
            fs,joints,necks,neck_depths,errors = [],[],[],[],[]
            for f in job['sampling'][pid]:
                with Image.open(os.path.join(job['pictures'],'%06d.jpg'%f)) as source: image = np.asarray(source.convert('RGB')).copy()
                h,w = image.shape[:2]
                box = np.asarray(person['boxes'][str(f)][:4],float); box[[1,3]] -= job['y0']
                mask_i = indices.get((f+job['shot_start'],native))
                mask = mask_at(bundle,mask_i,w,h) if mask_i is not None else None
                K = np.asarray(job['K'],float).copy(); K[0,0] *= job['zoom'][f]; K[1,1] *= job['zoom'][f]
                with contextlib.redirect_stdout(io.StringIO()):
                    pred = estimator.process_one_image(image,bboxes=box[None],masks=mask[None] if mask is not None else None,
                        cam_int=torch.tensor(K[None],dtype=torch.float32),inference_type='body')
                if not pred: continue
                pred = pred[0]
                camera = np.asarray(pred['pred_keypoints_3d'],float)+np.asarray(pred['pred_cam_t'],float)
                vertices = np.asarray(pred['pred_vertices'],float)+np.asarray(pred['pred_cam_t'],float)
                if not np.isfinite(camera).all() or np.any(camera[mapping,2]<=.01): continue
                projected = camera @ K.T; projected = projected[:,:2]/projected[:,2:]
                error = float(np.median(np.linalg.norm(projected-np.asarray(pred['pred_keypoints_2d']),axis=1)))
                checked += 1
                if error>max(5.,.02*max(w,h)):           # one bad estimate is skipped; the shot fails only when most
                    rejected.append([pid,int(f),round(error,1)])   # frames fail (then the camera set-up is wrong)
                    continue
                fs.append(f); joints.append(camera[mapping]); necks.append(camera[neck_index]); errors.append(error)
                if len(fs)==1:
                    data['p'+pid+'_mesh_frame'] = np.asarray(f,np.int32)
                    data['p'+pid+'_mesh_v'] = vertices.astype(np.float32)
                    data['p'+pid+'_mesh_faces'] = np.asarray(estimator.faces,np.int32)
                observed_neck = np.nan
                if f in samples:
                    sample = samples[f]; depth = sample['depth']; dh,dw = depth.shape
                    scale = min(1.,job['surface_side']/max(dh,dw)); shape = (max(1,round(dh*scale)),max(1,round(dw*scale)))
                    small_K = sample['K'].copy(); small_K[0]*=shape[1]/dw; small_K[1]*=shape[0]/dh
                    measured = cv2.resize(depth,shape[::-1],interpolation=cv2.INTER_NEAREST)
                    valid = cv2.resize(sample['valid'].astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
                    source_mask = mask.astype(np.uint8) if mask is not None else np.zeros((h,w),np.uint8)
                    if mask is None:
                        x0,y0,x1,y1 = np.rint(box).astype(int); source_mask[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=1
                    small_mask = cv2.warpPerspective(source_mask,small_K @ np.linalg.inv(K),shape[::-1],flags=cv2.INTER_NEAREST)>0
                    rendered = surface_depth(vertices,estimator.faces,small_K,shape)
                    ratio = surface_ratio(rendered,measured,small_mask,valid)
                    if ratio is not None: ratios.append(ratio)
                    point = camera[neck_index] @ small_K.T; uv = point[:2]/point[2]
                    yy,xx = np.indices(shape); radius = max(2.,.04*(box[3]-box[1])*shape[0]/h)
                    region = ((xx-uv[0])**2+(yy-uv[1])**2<radius**2)&small_mask&valid&np.isfinite(measured)&(measured>.05)
                    if region.sum()>=3: observed_neck=float(np.median(measured[region]))
                neck_depths.append(observed_neck); count+=1
                if count%10==0: print('SAM 3D Body: %d person/frame estimates'%count,flush=True)
            if fs:
                prefix = 'p'+pid
                data.update({prefix+'_f':np.asarray(fs,np.int32),prefix+'_j':np.asarray(joints,np.float32),
                    prefix+'_n':np.asarray(necks,np.float32),prefix+'_neck_depth':np.asarray(neck_depths,np.float32),
                    prefix+'_reprojection_px':np.asarray(errors,np.float32)})
    if checked and len(rejected) > .5*checked:
        raise RuntimeError('SAM 3D Body camera convention check failed on %d of %d frames (the camera set-up is wrong, '
                           'not one bad frame)'%(len(rejected),checked))
    if rejected:
        print('SAM 3D Body: skipped %d of %d estimates that did not match their picture (person, frame, px): %s'%
              (len(rejected),checked,rejected[:8]),flush=True)
    np.savez_compressed(output,**data,surface_ratios=np.asarray(ratios,np.float32),
                        checked_estimates=np.asarray(checked,np.int32),rejected_estimates=np.asarray(len(rejected),np.int32),
                        peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2)
    print('SAM 3D Body finished: %d estimates, %d surface instances, %.0f MB peak VRAM'%
          (count,len(ratios),torch.cuda.max_memory_allocated()/1024**2),flush=True)


if __name__=='__main__':
    if len(sys.argv)!=4 or sys.argv[1]!='--worker': raise SystemExit('Use through the SAM 3D Body pipeline backend')
    worker(sys.argv[2],sys.argv[3])

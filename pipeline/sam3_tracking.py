"""Persistent SAM 3 video IDs with mask-conditioned measurements from the current pose model.

SAM 3 runs in its isolated Python process before the ONNX pose model is loaded.
Masks are packed CPU arrays; identity never depends on the ordering of detections.
"""
import collections
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

import cv2
import numpy as np

from common import AGENT, imwrite_u, jload, jsave

VERSION = 1


def signature(c):
    checkpoint = c.get('sam3_checkpoint') or os.path.join(AGENT,'models','sam3','sam3.pt')
    try:
        stat = os.stat(checkpoint); stamp = [stat.st_size,stat.st_mtime_ns]
    except OSError:
        stamp = None
    return {'version':VERSION,'checkpoint':os.path.abspath(checkpoint),'stamp':stamp,
            'mask_side':c.get('sam3_mask_side',640),'max_objects':c.get('sam3_max_objects',16),
            'probability':c.get('sam3_probability',.35),'occlusion_seconds':c.get('track_occlusion_seconds',2.),
            'pose_backend':'rtmpose','pose_mode':c.get('pose_mode','performance')}


def needs_tracking(pose, c):
    if pose.get('tracker_backend') != 'sam3': return True
    return pose.get('tracking_signature') != signature(c)


def read_masks(path):
    with np.load(path, allow_pickle=False) as source:
        bundle = {key: source[key].copy() for key in source.files}
    shape = bundle['shape']
    if len(shape) != 2 or np.any(shape <= 0): raise ValueError('Invalid SAM 3 mask dimensions')
    n = len(bundle['frames']); size = (int(np.prod(shape))+7)//8
    if bundle['packed'].shape != (n,size) or len(bundle['ids']) != n:
        raise ValueError('Invalid SAM 3 packed mask records')
    pairs = list(zip(bundle['frames'].tolist(),bundle['ids'].tolist()))
    if len(set(pairs)) != n: raise ValueError('Duplicate SAM 3 frame/identity records')
    return bundle


def mask_at(bundle, index, width, height):
    shape = tuple(bundle['shape'])
    mask = np.unpackbits(bundle['packed'][index], count=int(np.prod(shape))).reshape(shape)
    return cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST).astype(bool)


def union_at(bundle, frame, width, height, ids=None):
    mask = np.zeros((height,width),bool)
    for index in np.flatnonzero(bundle['frames'] == frame):
        if ids is None or int(bundle['ids'][index]) in ids:
            mask |= mask_at(bundle,int(index),width,height)
    return mask


def measure_poses(bundle, frames, y0, y1, marker, batch=8):
    tracks = {}; width, height = frames[0].shape[1], y1-y0
    for start in range(0,len(bundle['frames']),batch):
        items, records = [], []
        for index in range(start,min(start+batch,len(bundle['frames']))):
            f, pid = int(bundle['frames'][index]), int(bundle['ids'][index])
            if not 0 <= f < len(frames): raise ValueError('SAM 3 returned a frame outside the clip')
            mask = mask_at(bundle,index,width,height)
            yy,xx = np.nonzero(mask)
            if len(xx) < 8: continue
            box = [float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)]
            # Mask the entire image, not just the requested crop: pose preprocessing
            # expands boxes and could otherwise expose a neighbouring person.
            image = np.where(mask[...,None],frames[f][y0:y1],128).astype(np.uint8)
            items.append((image,[box])); records.append((f,pid,mask,box))
        measured = marker.poses(items)
        if len(measured) != len(records): raise RuntimeError('Pose model returned incomplete SAM 3 measurements')
        for (f,pid,mask,box),(keys,scores) in zip(records,measured):
            if len(keys) != 1 or len(scores) != 1: raise RuntimeError('Expected one pose per SAM 3 mask')
            k,s = np.asarray(keys[0],float).copy(), np.asarray(scores[0],float).copy()
            if k.shape != (17,2) or s.shape != (17,): raise RuntimeError('Pose model must return COCO-17 joints')
            finite = np.isfinite(k).all(1) & np.isfinite(s)
            k[~finite] = 0; s[~finite] = 0
            xy = np.rint(k).astype(int)
            inside = finite & (xy[:,0]>=0) & (xy[:,0]<width) & (xy[:,1]>=0) & (xy[:,1]<height)
            visible = cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))
            supported = np.zeros(17,bool)
            supported[inside] = visible[xy[inside,1],xy[inside,0]]>0
            s[~supported] *= .15
            k[:,1] += y0
            tracks.setdefault(pid,{})[f] = (k,s,[box[0],box[1]+y0,box[2],box[3]+y0])
    return tracks


def run_masks(frames, y0, y1, shots, out, c, fps=30.):
    checkpoint = c.get('sam3_checkpoint') or os.path.join(AGENT,'models','sam3','sam3.pt')
    python = c.get('sam3_python') or os.path.join(AGENT,'downloads','da3-venv','Scripts','python.exe')
    if not os.path.isfile(checkpoint): raise RuntimeError('SAM 3 checkpoint missing: '+checkpoint)
    if not os.path.isfile(python): raise RuntimeError('SAM 3 Python environment missing: '+python)
    destination = os.path.join(out,'sam3_masks.npz')
    with tempfile.TemporaryDirectory(dir=out,prefix='sam3_') as work:
        paths = []
        for start,end in shots:
            folder = os.path.join(work,str(start)); os.makedirs(folder)
            for f in range(start,end):
                imwrite_u(os.path.join(folder,'%06d.jpg'%(f-start)),frames[f][y0:y1],95)
            paths.append({'folder':folder,'start':start,'end':end})
        job, result = os.path.join(work,'job.json'),os.path.join(work,'masks.npz')
        jsave(job,{'shots':paths,'checkpoint':os.path.abspath(checkpoint),'mask_side':int(c.get('sam3_mask_side',640)),
                   'max_objects':int(c.get('sam3_max_objects',16)), 'probability':float(c.get('sam3_probability',.35)),
                   'keep_alive':max(30,round(fps*float(c.get('track_occlusion_seconds',2.))))})
        timeout = max(60,float(c.get('sam3_timeout_min',40))*60)
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        subprocess.run([python,os.path.abspath(__file__),'--worker',job,result],check=True,timeout=timeout,
                       creationflags=flags,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8'))
        bundle = read_masks(result)
        os.replace(result,destination)
    return bundle


def archive_previous(out):
    """Preserve old ID-specific edits before publishing a newly tracked clip."""
    if not os.path.isfile(os.path.join(out,'pose.json')): return None
    backup = os.path.join(out,'tracking_backups',datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    os.makedirs(backup)
    paths = [os.path.join(out,name) for name in ('pose.json','pose_raw.json','check.html','review.json','ai.json',
                                               'hints.json','sam3_masks.npz','tracking_report.json')]
    paths += glob.glob(os.path.join(out,'checked*.json'))
    for path in paths:
        if os.path.isfile(path): shutil.copy2(path,backup)
    if os.path.isdir(os.path.join(out,'review')):
        shutil.copytree(os.path.join(out,'review'),os.path.join(backup,'review'))
    return backup


def publish(staged, out):
    if not os.path.isfile(os.path.join(staged,'pose.json')):
        raise RuntimeError('Incomplete tracking output; previous marks preserved')
    new_pose = jload(os.path.join(staged,'pose.json'))
    if new_pose.get('tracker_backend')=='sam3' and not os.path.isfile(os.path.join(staged,'sam3_masks.npz')):
        raise RuntimeError('SAM 3 masks missing; previous marks preserved')
    backup = archive_previous(out)
    # Remove only annotations referring to the previous IDs, after successful inference.
    review = os.path.join(out,'review.json')
    if os.path.isfile(review):
        values = jload(review)
        values.update(people={},confirmed=False,auto=False,source='tracking changed; review new person IDs')
        jsave(review,values)
    for path in glob.glob(os.path.join(out,'checked*.json')) + [os.path.join(out,n) for n in ('ai.json','pose_raw.json')]:
        if os.path.isfile(path): os.remove(path)
    hints = os.path.join(out,'hints.json')
    if os.path.isfile(hints):
        values = jload(hints)
        for key in ('people','tracks','scene_justifications'): values.pop(key,None)
        jsave(hints,values)
    # the joint editor's files of earlier builds (kept in the backup): Stage 1 has no editor
    if backup and os.path.isfile(os.path.join(out,'check.html')): os.remove(os.path.join(out,'check.html'))
    if backup and os.path.isdir(os.path.join(out,'review')): shutil.rmtree(os.path.join(out,'review'))
    for name in ('sam3_masks.npz','tracking_report.json','pose.json'):
        source,target = os.path.join(staged,name),os.path.join(out,name)
        if os.path.isfile(source): os.replace(source,target)
        elif os.path.isfile(target) and name in ('sam3_masks.npz','tracking_report.json'): os.remove(target)
    if backup:
        report = os.path.join(out,'tracking_report.json')
        values = jload(report) if os.path.isfile(report) else {}
        values['previous_tracking_backup'] = backup
        jsave(report,values)
    return backup


class FrameFolder:
    """Four-frame CPU cache: retain SAM identity across the whole shot without loading all images."""
    def __init__(self, folder, image_size, mean, std):
        self.paths = sorted(glob.glob(os.path.join(folder,'*.jpg')),key=lambda p:int(os.path.splitext(os.path.basename(p))[0]))
        self.image_size,self.mean,self.std = image_size,mean,std
        self.cache = collections.OrderedDict()
        from PIL import Image
        with Image.open(self.paths[0]) as image: self.width,self.height = image.size
    def __len__(self): return len(self.paths)
    def __getitem__(self,index):
        index = int(index)
        if index not in self.cache:
            import torch
            from PIL import Image
            with Image.open(self.paths[index]) as image:
                array = np.asarray(image.convert('RGB').resize((self.image_size,self.image_size))).copy()
            tensor = torch.from_numpy(array).permute(2,0,1).to(torch.float16)/255.
            mean = torch.as_tensor(self.mean,dtype=torch.float16)[:,None,None]
            std = torch.as_tensor(self.std,dtype=torch.float16)[:,None,None]
            self.cache[index] = (tensor-mean)/std
            while len(self.cache)>4: self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def worker(job_path, output):
    sys.path.insert(0,os.path.join(AGENT,'third_party','sam3'))
    import torch
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    import sam3.model.sam3_video_inference as inference
    if not torch.cuda.is_available(): raise RuntimeError('SAM 3 requires a working NVIDIA GPU')
    job = jload(job_path)
    # The SDK accepts indexable CPU frame sources. Adapt its loader within this
    # isolated worker; no installed source files or model internals are rewritten.
    original_loader = inference.load_resource_as_video_frames
    def load_folder(resource_path,image_size,img_mean,img_std,**kwargs):
        images = FrameFolder(resource_path,image_size,img_mean,img_std)
        return images,images.height,images.width
    inference.load_resource_as_video_frames = load_folder
    predictor = None
    packed,frames,ids,probabilities = [],[],[],[]
    shape = None; next_id = 1
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor = Sam3VideoPredictor(checkpoint_path=job['checkpoint'],compile=False,
                                          apply_temporal_disambiguation=True,video_loader_type='cv2')
            predictor.model.max_num_objects = int(job.get('max_objects',16))
            predictor.model.max_trk_keep_alive = int(job.get('keep_alive',60))
            predictor.model.init_trk_keep_alive = int(job.get('keep_alive',60))
            for shot in job['shots']:
                local_ids = {}
                session = predictor.handle_request({'type':'start_session','resource_path':shot['folder'],
                           'offload_video_to_cpu':True,'offload_state_to_cpu':True})['session_id']
                try:
                    predictor.handle_request({'type':'add_prompt','session_id':session,'frame_index':0,'text':'person',
                                              'output_prob_thresh':job.get('probability',.35)})
                    request = {'type':'propagate_in_video','session_id':session,'propagation_direction':'forward',
                               'start_frame_index':0,'evict_cached_frame_outputs':True,
                               'output_prob_thresh':job.get('probability',.35)}
                    for result in predictor.handle_stream_request(request):
                        f = int(result['frame_index'])+shot['start']; out = result['outputs']
                        for pid,mask,prob in zip(out['out_obj_ids'],out['out_binary_masks'],out['out_probs']):
                            mask = np.asarray(mask).squeeze().astype(np.uint8)
                            h,w = mask.shape; scale = min(1.,job.get('mask_side',640)/max(h,w))
                            shape = (max(1,round(h*scale)),max(1,round(w*scale)))
                            mask = cv2.resize(mask,shape[::-1],interpolation=cv2.INTER_NEAREST)
                            if mask.sum()<8: continue
                            pid = int(pid)
                            if pid not in local_ids: local_ids[pid]=next_id; next_id+=1
                            packed.append(np.packbits(mask.ravel())); frames.append(f); ids.append(local_ids[pid]); probabilities.append(float(prob))
                        if (f-shot['start'])%30 == 0:
                            print('SAM 3 frame %d/%d: %d active people'%(f+1,shot['end'],len(out['out_obj_ids'])),flush=True)
                finally:
                    predictor.handle_request({'type':'close_session','session_id':session})
        if shape is None:
            from PIL import Image
            with Image.open(os.path.join(job['shots'][0]['folder'],'000000.jpg')) as image: w,h = image.size
            scale = min(1.,job.get('mask_side',640)/max(h,w))
            shape = (max(1,round(h*scale)),max(1,round(w*scale)))
        size = (int(np.prod(shape))+7)//8
        order = sorted(range(len(frames)),key=lambda i:(frames[i],ids[i]))
        np.savez_compressed(output,frames=np.asarray(frames,np.int32)[order],ids=np.asarray(ids,np.int64)[order],
            packed=np.asarray(packed,np.uint8).reshape(-1,size)[order],shape=np.asarray(shape),version=VERSION,
            probabilities=np.asarray(probabilities,np.float32)[order],peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2)
        print('SAM 3 finished: %d identities, peak allocated VRAM %.0f MB'%(next_id-1,torch.cuda.max_memory_allocated()/1024**2),flush=True)
    finally:
        inference.load_resource_as_video_frames = original_loader
        if predictor is not None: predictor.shutdown()


if __name__ == '__main__':
    if len(sys.argv)!=4 or sys.argv[1]!='--worker': raise SystemExit('Use through automark with tracker_backend=sam3')
    worker(sys.argv[2],sys.argv[3])

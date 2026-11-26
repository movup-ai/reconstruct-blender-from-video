# -*- coding: utf-8 -*-
"""Stage 2: SAM 3 finds and follows every object of each asset type through the clip, from a text prompt.

Runs in SAM 3's own Python (like sam3_tracking.py).  Objects stand still, so SAM 3 sees a few frames per second
(config "bg_detect_fps", default 5), every shot from its first frame: an object that only comes into view later (the
camera zooms or pans) is found where it appears.  One session per (shot, type); one model load for all of them.

  detect(frames_at, picture, shots, prompts, out, c) -> bundle {frames, types, ids, probabilities, packed, shape}
     frames_at: {clip frame: BGR picture area}   ids are unique over the whole clip"""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from common import AGENT, imwrite_u, jload, jsave

VERSION = 1


def sample_frames(shots, fps, c):
    """the clip frames SAM 3 looks at: every shot's first and last frame and a few per second in between"""
    step = max(1, int(round(float(fps) / max(0.5, float(c.get("bg_detect_fps", 5))))))
    out = []
    for a, b in shots:
        out += list(range(a, b, step))
        if b - 1 not in out:
            out.append(b - 1)
    return sorted(set(out))


def detect(frames_at, shots, prompts, out, c):
    checkpoint = c.get('sam3_checkpoint') or os.path.join(AGENT, 'models', 'sam3', 'sam3.pt')
    python = c.get('sam3_python') or os.path.join(AGENT, 'downloads', 'da3-venv', 'Scripts', 'python.exe')
    if not os.path.isfile(checkpoint): raise RuntimeError('SAM 3 checkpoint missing: ' + checkpoint)
    if not os.path.isfile(python): raise RuntimeError('SAM 3 Python environment missing: ' + python)
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out, prefix='sam3_') as work:
        jobs = []
        for si, (a, b) in enumerate(shots):
            fs = [f for f in sorted(frames_at) if a <= f < b]
            if not fs:
                continue
            folder = os.path.join(work, str(si)); os.makedirs(folder)
            for k, f in enumerate(fs):
                imwrite_u(os.path.join(folder, '%06d.jpg' % k), frames_at[f], 95)
            jobs.append({'folder': folder, 'frames': fs})
        job, result = os.path.join(work, 'job.json'), os.path.join(work, 'objects.npz')
        jsave(job, {'shots': jobs, 'prompts': list(prompts), 'checkpoint': os.path.abspath(checkpoint),
                    'mask_side': int(c.get('sam3_mask_side', 640)), 'max_objects': int(c.get('bg_max_objects', 12)),
                    'probability': float(c.get('bg_probability', .5))})
        timeout = max(60, float(c.get('sam3_timeout_min', 40)) * 60)
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        subprocess.run([python, os.path.abspath(__file__), '--worker', job, result], check=True, timeout=timeout,
                       creationflags=flags, env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8'))
        with np.load(result, allow_pickle=False) as src:
            bundle = {k: src[k].copy() for k in src.files}
    return bundle


def mask_of(bundle, index, width, height):
    shape = tuple(int(x) for x in bundle['shape'])
    m = np.unpackbits(bundle['packed'][index], count=int(np.prod(shape))).reshape(shape)
    return cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def worker(job_path, output):
    sys.path.insert(0, os.path.join(AGENT, 'third_party', 'sam3'))
    import torch
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    import sam3.model.sam3_video_inference as inference
    from sam3_tracking import FrameFolder
    if not torch.cuda.is_available(): raise RuntimeError('SAM 3 requires a working NVIDIA GPU')
    job = jload(job_path)
    original_loader = inference.load_resource_as_video_frames

    def load_folder(resource_path, image_size, img_mean, img_std, **kwargs):
        images = FrameFolder(resource_path, image_size, img_mean, img_std)
        return images, images.height, images.width
    inference.load_resource_as_video_frames = load_folder
    predictor = None
    packed, frames, types, ids, probabilities = [], [], [], [], []
    shape = None; next_id = 1
    thresh = float(job.get('probability', .5))
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            predictor = Sam3VideoPredictor(checkpoint_path=job['checkpoint'], compile=False,
                                           apply_temporal_disambiguation=True, video_loader_type='cv2')
            predictor.model.max_num_objects = int(job.get('max_objects', 12))
            n = max(len(s['frames']) for s in job['shots'])
            predictor.model.max_trk_keep_alive = n          # an object hidden behind a person is still the same object
            predictor.model.init_trk_keep_alive = n
            for shot in job['shots']:
                fs = shot['frames']
                for ti, prompt in enumerate(job['prompts']):
                    local = {}
                    session = predictor.handle_request({'type': 'start_session', 'resource_path': shot['folder'],
                                                        'offload_video_to_cpu': True, 'offload_state_to_cpu': True})['session_id']
                    try:
                        predictor.handle_request({'type': 'add_prompt', 'session_id': session, 'frame_index': 0,
                                                  'text': prompt, 'output_prob_thresh': thresh})
                        request = {'type': 'propagate_in_video', 'session_id': session, 'propagation_direction': 'forward',
                                   'start_frame_index': 0, 'evict_cached_frame_outputs': True, 'output_prob_thresh': thresh}
                        for result in predictor.handle_stream_request(request):
                            f = fs[int(result['frame_index'])]; out = result['outputs']
                            for oid, mask, prob in zip(out['out_obj_ids'], out['out_binary_masks'], out['out_probs']):
                                mask = np.asarray(mask).squeeze().astype(np.uint8)
                                h, w = mask.shape; scale = min(1., job.get('mask_side', 640) / max(h, w))
                                shape = (max(1, round(h * scale)), max(1, round(w * scale)))
                                mask = cv2.resize(mask, shape[::-1], interpolation=cv2.INTER_NEAREST)
                                if mask.sum() < 8: continue
                                oid = int(oid)
                                if oid not in local: local[oid] = next_id; next_id += 1
                                packed.append(np.packbits(mask.ravel())); frames.append(f); types.append(ti)
                                ids.append(local[oid]); probabilities.append(float(prob))
                    finally:
                        predictor.handle_request({'type': 'close_session', 'session_id': session})
                    print('SAM 3 "%s", frames %d-%d: %d found' % (prompt, fs[0], fs[-1], len(local)), flush=True)
        if shape is None:
            from PIL import Image
            with Image.open(os.path.join(job['shots'][0]['folder'], '000000.jpg')) as image: w, h = image.size
            scale = min(1., job.get('mask_side', 640) / max(h, w))
            shape = (max(1, round(h * scale)), max(1, round(w * scale)))
        size = (int(np.prod(shape)) + 7) // 8
        order = sorted(range(len(frames)), key=lambda i: (frames[i], ids[i]))
        np.savez_compressed(output, frames=np.asarray(frames, np.int32)[order], types=np.asarray(types, np.int32)[order],
                            ids=np.asarray(ids, np.int64)[order], packed=np.asarray(packed, np.uint8).reshape(-1, size)[order],
                            probabilities=np.asarray(probabilities, np.float32)[order], shape=np.asarray(shape), version=VERSION)
    finally:
        inference.load_resource_as_video_frames = original_loader
        if predictor is not None: predictor.shutdown()


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker': raise SystemExit('Use through background.py')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    worker(sys.argv[2], sys.argv[3])

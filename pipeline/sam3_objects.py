# -*- coding: utf-8 -*-
"""Stage 2: SAM 3 finds and follows every object of each asset type through the clip, from a text prompt.

Runs in SAM 3's own Python (like sam3_tracking.py).  Objects stand still, so SAM 3 sees a few frames per second
(config "bg_detect_fps", default 5), every shot from its first frame: an object that only comes into view later (the
camera zooms or pans) is found where it appears.

Two passes, so a library of hundreds of types stays fast:
  1. scan   SAM 3's image detector scores every type on a few frames spread over each shot (config "bg_scan_frames",
            default 4, first and last frame included), many types per pass: about 6 s per frame for 400 types.  A type
            scoring at least "bg_scan_threshold" (default 0.8 x the detection threshold) on any of them is in the shot.
  2. track  SAM 3's video predictor follows only those types through the shot: one session per (shot, type found).
            (Following all 392 types one by one took about an hour per clip.)

  detect(frames_at, shots, prompts, out, c) -> bundle {frames, types, ids, probabilities, packed, shape, scan}
     frames_at: {clip frame: BGR picture area}   ids are unique over the whole clip
     scan: (shots, types) the best scan score of each type in each shot"""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from common import AGENT, imwrite_u, jload, jsave

VERSION = 2                    # 2: scan every type first, track only the ones found


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
                    'probability': float(c.get('bg_probability', .5)), 'scan_frames': int(c.get('bg_scan_frames', 4)),
                    'scan_threshold': float(c.get('bg_scan_threshold', .8 * float(c.get('bg_probability', .5)))),
                    'scan_batch': int(c.get('bg_scan_batch', 16))})
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


def scan_indices(n, k):
    """k frame positions spread over n frames, the first and the last included"""
    if n <= 0:
        return []
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, max(1, min(k, n)))})


def scan(job):
    """pass 1: SAM 3's image detector scores every type on a few frames of each shot -> (shots, types) best score.
    The image is encoded once per frame, the type names once in all, and many types are scored per pass."""
    import time
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.data_misc import FindStage
    from torchvision.transforms import v2
    t0 = time.time()
    prompts = job['prompts']; batch = max(1, int(job.get('scan_batch', 16)))
    model = build_sam3_image_model(checkpoint_path=job['checkpoint'], load_from_HF=False, enable_segmentation=False)
    tf = v2.Compose([v2.ToDtype(torch.uint8, scale=True), v2.Resize(size=(1008, 1008)), v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize(mean=[.5] * 3, std=[.5] * 3)])
    best = np.zeros((len(job['shots']), len(prompts)), np.float32)
    nframes = 0
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            text = model.backbone.forward_text(prompts, device='cuda')
            for si, shot in enumerate(job['shots']):
                for k in scan_indices(len(shot['frames']), int(job.get('scan_frames', 4))):
                    with Image.open(os.path.join(shot['folder'], '%06d.jpg' % k)) as im:
                        x = tf(v2.functional.to_image(im.convert('RGB')).cuda()).unsqueeze(0)
                    image = model.backbone.forward_image(x)
                    nframes += 1
                    for a in range(0, len(prompts), batch):
                        idx = list(range(a, min(a + batch, len(prompts))))
                        find = FindStage(img_ids=torch.zeros(len(idx), dtype=torch.long, device='cuda'),
                                         text_ids=torch.tensor(idx, dtype=torch.long, device='cuda'), input_boxes=None,
                                         input_boxes_mask=None, input_boxes_label=None, input_points=None, input_points_mask=None)
                        bo = dict(image); bo.update(text)
                        out = model.forward_grounding(backbone_out=bo, find_input=find, find_target=None,
                                                      geometric_prompt=model._get_dummy_prompt(len(idx)))
                        p = out['pred_logits'].sigmoid().squeeze(-1) * out['presence_logit_dec'].sigmoid()
                        best[si, idx] = np.maximum(best[si, idx], p.max(-1).values.float().cpu().numpy())
    finally:
        del model
        torch.cuda.empty_cache()
    print('SAM 3 scan: %d types on %d frame(s) in %.0fs' % (len(prompts), nframes, time.time() - t0), flush=True)
    return best


def worker(job_path, output):
    sys.path.insert(0, os.path.join(AGENT, 'third_party', 'sam3'))
    import torch
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    import sam3.model.sam3_video_inference as inference
    from sam3_tracking import FrameFolder
    if not torch.cuda.is_available(): raise RuntimeError('SAM 3 requires a working NVIDIA GPU')
    job = jload(job_path)
    # pass 1: which types are in each shot at all (the image model is unloaded before the video model loads)
    best = scan(job) if int(job.get('scan_frames', 4)) > 0 else np.ones((len(job['shots']), len(job['prompts'])), np.float32)
    keep = best >= float(job.get('scan_threshold', .4))
    for si, shot in enumerate(job['shots']):
        found = [job['prompts'][i] for i in np.flatnonzero(keep[si])]
        print('SAM 3 scan, frames %d-%d: %d of %d types to follow%s' % (shot['frames'][0], shot['frames'][-1], len(found),
              len(job['prompts']), (': ' + ', '.join(found)) if found else ''), flush=True)
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
            if keep.any():                                    # nothing found anywhere: no video model at all
                predictor = Sam3VideoPredictor(checkpoint_path=job['checkpoint'], compile=False,
                                               apply_temporal_disambiguation=True, video_loader_type='cv2')
                predictor.model.max_num_objects = int(job.get('max_objects', 12))
                n = max(len(s['frames']) for s in job['shots'])
                predictor.model.max_trk_keep_alive = n      # an object hidden behind a person is still the same object
                predictor.model.init_trk_keep_alive = n
            for si, shot in enumerate(job['shots']):
                fs = shot['frames']
                for ti, prompt in enumerate(job['prompts']):
                    if not keep[si, ti]:
                        continue                              # pass 2 follows only the types the scan found in this shot
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
                            probabilities=np.asarray(probabilities, np.float32)[order], shape=np.asarray(shape), version=VERSION,
                            scan=best)
    finally:
        inference.load_resource_as_video_frames = original_loader
        if predictor is not None: predictor.shutdown()


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker': raise SystemExit('Use through background.py')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    worker(sys.argv[2], sys.argv[3])

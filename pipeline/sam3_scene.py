"""SAM 3 image masks for scene objects, isolated from the scene models' CUDA runtime."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from common import AGENT, DEFAULT_CFG, jload, jsave

VERSION = 1


def valid_boxes(boxes, width, height):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not np.isfinite(boxes).all():
        raise ValueError("SAM 3 scene boxes must be finite")
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("SAM 3 scene boxes must have positive width and height")
    return boxes


def grid_candidates(logits, scores, iou_threshold, stability_threshold, min_area):
    """Filter low-resolution instance logits without allocating full-size masks."""
    logits = np.asarray(logits, np.float32).reshape(-1, *logits.shape[-2:])
    scores = np.asarray(scores, np.float32).reshape(-1)
    if len(logits) != len(scores):
        raise ValueError("SAM 3 mask and score counts differ")
    positive = logits > 0
    stability = (logits > 1).sum((1, 2)) / np.maximum((logits > -1).sum((1, 2)), 1)
    area = positive.mean((1, 2))
    keep = (np.isfinite(logits).all((1, 2)) & np.isfinite(scores) &
            (scores > iou_threshold) & (stability > stability_threshold) &
            (area > min_area) & (area < .98))
    return positive[keep], (scores + .2 * stability)[keep]


def suppress_masks(masks, scores, shape, threshold=.6, limit=256):
    """Greedy mask-IoU suppression, bounded before upsampling and safe from FP16 overflow."""
    if not len(masks):
        return np.zeros((0, *shape), bool)
    order = np.argsort(-np.asarray(scores), kind='stable')[:max(limit * 2, limit)]
    masks = np.asarray(masks, bool)[order]
    small = np.stack([cv2.resize(m.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)
                      for m in masks]).reshape(len(masks), -1).astype(np.float32)
    areas = small.sum(1)
    keep = []
    for i in range(len(masks)):
        if not areas[i]:
            continue
        if keep:
            intersection = small[keep] @ small[i]
            overlap = intersection / np.maximum(areas[keep] + areas[i] - intersection, 1)
            if np.any(overlap > threshold):
                continue
        keep.append(i)
        if len(keep) >= limit:
            break
    return np.asarray([cv2.resize(masks[i].astype(np.float32), (shape[1], shape[0]),
                                 interpolation=cv2.INTER_LINEAR) > .5 for i in keep], bool).reshape(-1, *shape)


def segment(rgb, c, automatic=True, boxes=(), cache_dir=None, before_inference=None, **options):
    """Return automatic masks and one mask per requested XYXY box; failures never use a retired model."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.shape[0] or not rgb.shape[1]:
        raise ValueError("SAM 3 scene input must be a nonempty RGB image")
    boxes = valid_boxes(boxes, rgb.shape[1], rgb.shape[0])
    checkpoint = c.get('sam3_checkpoint') or DEFAULT_CFG['sam3_checkpoint']
    python = c.get('sam3_python') or DEFAULT_CFG['sam3_python']
    for path in (checkpoint, python):
        if not os.path.isfile(path):
            raise RuntimeError('SAM 3 scene required file missing: ' + path)
    params = {'version': VERSION, 'automatic': bool(automatic), 'boxes': boxes.tolist(),
              'points': int(options.get('points', c.get('sam_points', 32))),
              'iou': float(options.get('iou', .7)), 'stability': float(options.get('stability', .86)),
              'nms': float(options.get('nms', .6)), 'min_area': float(options.get('min_area', .002)),
              'batch': int(c.get('sam3_image_batch', 16)), 'limit': int(c.get('sam3_image_max_masks', 256))}
    if not 1 <= params['points'] <= 64 or not 1 <= params['batch'] <= 32 or not 1 <= params['limit'] <= 512:
        raise ValueError('SAM 3 scene grid, batch or mask limit is outside the supported range')
    stamp = os.stat(checkpoint)
    digest = hashlib.sha256(json.dumps([params, os.path.abspath(checkpoint), stamp.st_size, stamp.st_mtime_ns],
                                       sort_keys=True).encode())
    digest.update(str(rgb.shape).encode()); digest.update(rgb.tobytes())
    key = digest.hexdigest()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, key + '.npz') if cache_dir else None
    def read(path):
        with np.load(path, allow_pickle=False) as result:
            masks, prompted = result['masks'].copy(), result['box_masks'].copy()
            if masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2] or prompted.shape != (len(boxes), *rgb.shape[:2]):
                raise ValueError('Invalid SAM 3 scene mask dimensions')
            return {'masks': masks.astype(bool), 'box_masks': prompted.astype(bool),
                    'peak_vram_mb': float(result['peak_vram_mb'])}
    if cache and os.path.isfile(cache):
        return read(cache)
    if before_inference:
        before_inference()
    with tempfile.TemporaryDirectory(prefix='sam3_scene_') as tmp:
        image = os.path.join(tmp, 'image.png')
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError('Could not encode the SAM 3 scene image')
        with open(image, 'wb') as target:
            target.write(encoded.tobytes())
        job, output = os.path.join(tmp, 'job.json'), os.path.join(tmp, 'masks.npz')
        jsave(job, {**params, 'image': image, 'checkpoint': checkpoint})
        subprocess.run([python, os.path.abspath(__file__), '--worker', job, output], check=True,
                       timeout=max(60, float(c.get('sam3_timeout_min', 40)) * 60),
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                       env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8'))
        result = read(output)
        if cache:
            # Write only validated results, atomically, so interrupted jobs cannot poison the cache.
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.npz', delete=False) as target:
                temp = target.name
            try:
                np.savez_compressed(temp, **result)
                os.replace(temp, cache)
            finally:
                if os.path.exists(temp):
                    os.remove(temp)
        return result


def worker(job_path, output):
    sys.path.insert(0, os.path.join(AGENT, 'third_party', 'sam3'))
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not torch.cuda.is_available():
        raise RuntimeError('SAM 3 scene segmentation needs an NVIDIA GPU')
    job = jload(job_path)
    torch.cuda.reset_peak_memory_stats()
    model = build_sam3_image_model(checkpoint_path=job['checkpoint'], load_from_HF=False,
                                   enable_inst_interactivity=True, device='cuda')
    processor = Sam3Processor(model)
    image = Image.open(job['image']).convert('RGB')
    shape = (image.height, image.width)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = processor.set_image(image)
        selected, quality = [], []
        if job['automatic']:
            grid = (np.arange(job['points']) + .5) / job['points']
            points = np.asarray([(x * image.width, y * image.height) for y in grid for x in grid], np.float32)
            for start in range(0, len(points), job['batch']):
                prompts = points[start:start + job['batch'], None]
                _, scores, low = model.predict_inst(state, point_coords=prompts,
                    point_labels=np.ones(prompts.shape[:2], np.int32), multimask_output=True)
                masks, scores = grid_candidates(low, scores, job['iou'], job['stability'], job['min_area'])
                selected.extend(masks); quality.extend(scores)
        automatic = suppress_masks(selected, quality, shape, job['nms'], job['limit'])
        prompted = []
        for box in job['boxes']:
            masks, scores, _ = model.predict_inst(state, box=np.asarray(box, np.float32), multimask_output=True)
            masks = np.asarray(masks).reshape(-1, *shape)
            scores = np.asarray(scores).reshape(-1)
            if not len(scores) or not np.isfinite(scores).all():
                raise RuntimeError('SAM 3 returned no valid mask for an annotated object')
            prompted.append(masks[int(np.argmax(scores))] > 0)
    np.savez_compressed(output, masks=automatic, box_masks=np.asarray(prompted, bool).reshape(-1, *shape),
                        peak_vram_mb=torch.cuda.max_memory_allocated() / 1024**2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', nargs=2, metavar=('JOB', 'OUTPUT'), required=True)
    args = parser.parse_args()
    worker(*args.worker)

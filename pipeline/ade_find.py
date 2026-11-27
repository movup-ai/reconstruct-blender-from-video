# -*- coding: utf-8 -*-
"""The scene finder: Mask2Former panoptic (ADE20K, 150 names) finds, outlines and names every object, and names
every pixel (floor, wall, ceiling, sky, person, ...) for the depth step's gravity and floor (metric_scene.py).

  models\\mask2former_ade_panoptic   facebook/mask2former-swin-large-ade-panoptic (config.json,
                                     preprocessor_config.json, model.safetensors)

On a few frames per second of every shot (config "bg_detect_fps", default 5) - an object that only comes into view
later (zoom, pan) is found where it appears - Mask2Former gives each object its own outline and one of its 150 names.
ade_objects.ASSET turns the name into the asset placed for it (people and scenery are not placed).  Objects stand
still, so one object in the next sampled frame is the outline of the same asset that overlaps it most (mask overlap at
least "bg_link_iou", default 0.3); it keeps its id.

  find(frames_at, shots, types, c) -> (bundle, info)
     bundle {frames, types, ids, probabilities, packed, shape}: one row per object per frame, its outline packed;
            types index the given asset list; ids are unique over the clip
            {label_frames, labels, names}: every looked-at frame's name map (labels[k] at shape, an index into
            names; the same forward pass, read as one name per pixel)
     info   {seconds, frames, objects, missing {ade name: asset with no .blend}}"""
import os
import time

import cv2
import numpy as np

import ade_objects

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "models", "mask2former_ade_panoptic")
VERSION = 2


def available():
    return os.path.isfile(os.path.join(MODEL_DIR, "model.safetensors")) and os.path.isfile(os.path.join(MODEL_DIR, "config.json"))


def sample_frames(shots, fps, c):
    """the clip frames Mask2Former looks at: every shot's first and last frame and a few per second in between"""
    step = max(1, int(round(float(fps) / max(0.5, float(c.get("bg_detect_fps", 5))))))
    out = []
    for a, b in shots:
        out += list(range(a, b, step))
        if b - 1 not in out:
            out.append(b - 1)
    return sorted(set(out))


def labels_at(bundle, f, width, height):
    """frame f's name map (index into bundle["names"]) at width x height, or None when f was not looked at"""
    k = np.flatnonzero(np.asarray(bundle["label_frames"]) == int(f))
    if not len(k):
        return None
    return cv2.resize(np.asarray(bundle["labels"][k[0]]), (width, height), interpolation=cv2.INTER_NEAREST)


def mask_of(bundle, index, width, height):
    """row index's outline at width x height"""
    shape = tuple(int(x) for x in bundle["shape"])
    m = np.unpackbits(bundle["packed"][index], count=int(np.prod(shape))).reshape(shape)
    return cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def iou(a, b):
    i = np.logical_and(a, b).sum()
    return float(i) / max(1, np.logical_or(a, b).sum())


def link(active, found, threshold, next_id):
    """one frame's objects -> the same ids as the objects they continue.  active {id: (type, last outline)} of the
    shot so far, found [(type, outline, score)].  Biggest first, each takes the unused id of the same type whose last
    outline overlaps it most (at least threshold), else a new id.  -> ([(type, id, score, outline)], next_id);
    active is updated (an object hidden for a while is still matched when it comes back)"""
    taken, out = set(), []
    for ti, m, score in sorted(found, key=lambda x: -x[1].sum()):
        best, oid = threshold, None
        for k, (tj, prev) in active.items():
            if tj == ti and k not in taken:
                v = iou(m, prev)
                if v >= best:
                    best, oid = v, k
        if oid is None:
            oid = next_id; next_id += 1
        taken.add(oid)
        out.append((ti, oid, score, m))
    for ti, oid, score, m in out:
        active[oid] = (ti, m)
    return out, next_id


def find(frames_at, shots, types, c):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    if not available():
        raise RuntimeError("Mask2Former panoptic is missing: put config.json, preprocessor_config.json and "
                           "model.safetensors of facebook/mask2former-swin-large-ade-panoptic in %s" % MODEL_DIR)
    t0 = time.time()
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    proc = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_DIR, dtype=torch.float16).to("cuda").eval()
    names = [model.config.id2label[i].strip() for i in range(len(model.config.id2label))]
    index = {t: i for i, t in enumerate(types)}
    width = int(c.get("bg_find_width", 1024)); thr = float(c.get("bg_ade_threshold", 0.5))
    min_area = float(c.get("bg_min_area", 0.002)); link_iou = float(c.get("bg_link_iou", 0.3))
    side = int(c.get("sam3_mask_side", 640))
    rows = []                                       # (frame, type index, id, score, mask at mask size)
    label_frames, labels = [], []                   # every frame's name per pixel
    missing, next_id, shape = {}, 1, None
    try:
        for a, b in shots:
            active = {}                             # id -> (type index, last outline)
            for f in [f for f in sorted(frames_at) if a <= f < b]:
                rgb = cv2.cvtColor(frames_at[f], cv2.COLOR_BGR2RGB)
                H, W = rgb.shape[:2]
                w = int(round(min(width, W) / 32.0)) * 32; h = max(32, int(round(w * H / float(W) / 32.0)) * 32)
                inp = proc(images=rgb, size={"height": h, "width": w}, return_tensors="pt").to("cuda")
                inp["pixel_values"] = inp["pixel_values"].half()
                with torch.inference_mode():
                    out = model(**inp)
                s = min(1.0, side / float(max(H, W)))
                shape = (max(1, round(H * s)), max(1, round(W * s)))
                res = proc.post_process_panoptic_segmentation(out, threshold=thr, target_sizes=[shape],
                                                               label_ids_to_fuse=set())[0]
                seg = res["segmentation"].cpu().numpy()
                sem = proc.post_process_semantic_segmentation(out, target_sizes=[shape])[0]
                label_frames.append(int(f)); labels.append(sem.to(torch.int16).cpu().numpy())
                found = []
                for info in res["segments_info"]:
                    name = names[int(info["label_id"])]
                    asset = ade_objects.asset_of(name)
                    if not asset:
                        continue                    # people and scenery
                    if asset not in index:
                        missing[name] = asset; continue
                    m = seg == info["id"]
                    if m.mean() < min_area:
                        continue
                    found.append((index[asset], m, float(info["score"])))
                # the same object as in the frames before: same asset, the outline that overlaps it most
                linked, next_id = link(active, found, link_iou, next_id)
                rows += [(f, ti, oid, score, m) for ti, oid, score, m in linked]
    finally:
        del model
        torch.cuda.empty_cache()
    if shape is None:
        shape = (1, 1)
    size = (int(np.prod(shape)) + 7) // 8
    bundle = {"frames": np.asarray([r[0] for r in rows], np.int32), "types": np.asarray([r[1] for r in rows], np.int32),
              "ids": np.asarray([r[2] for r in rows], np.int64),
              "probabilities": np.asarray([r[3] for r in rows], np.float32),
              "packed": np.asarray([np.packbits(r[4].ravel()) for r in rows], np.uint8).reshape(-1, size),
              "shape": np.asarray(shape), "version": VERSION,
              "label_frames": np.asarray(label_frames, np.int32),
              "labels": np.asarray(labels, np.int16).reshape(-1, *shape), "names": np.asarray(names)}
    info = {"seconds": round(time.time() - t0, 1), "frames": len(frames_at), "objects": len({r[2] for r in rows}),
            "missing": missing}
    return bundle, info

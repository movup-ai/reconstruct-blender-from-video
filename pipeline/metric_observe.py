"""Bounded MoGe/Mask2Former observation worker. No gated checkpoints required."""
import argparse
import os
import sys

import cv2
import numpy as np

from common import AGENT, torch_device


def run(source, destination):
    import torch
    sys.path.insert(0, os.path.join(AGENT, "third_party", "MoGe"))
    from moge.model.v2 import MoGeModel
    with np.load(source, allow_pickle=False) as src:
        images, fovs = src["images"], src["fov"]
    model = MoGeModel.from_pretrained(os.path.join(AGENT, "models", "moge2", "model.pt")).to(torch_device()).eval()
    result = {"images": images}
    depth, valid = [], []
    for rgb, fov in zip(images, fovs):
        tensor = torch.as_tensor(rgb.copy(), device="cuda", dtype=torch.float32).permute(2, 0, 1) / 255
        with torch.inference_mode():
            pred = model.infer(tensor, fov_x=float(fov), use_fp16=True)
        points = pred["points"].float().cpu().numpy()
        depth.append(points[..., 2])
        valid.append(pred["mask"].cpu().numpy().astype(bool) & np.isfinite(points).all(-1))
    del pred, tensor, model
    torch.cuda.empty_cache()
    # Load the two large models sequentially to stay within the 4070's VRAM.
    import semseg
    if semseg.available():
        segmenter = semseg.Segmenter()
        labels = [segmenter(rgb, width=384) for rgb in images]
        result.update(labels=np.asarray(labels, np.int16), names=np.asarray(segmenter.names))
    else:
        result.update(labels=np.full(images.shape[:3], -1, np.int16), names=np.asarray([], dtype="U1"))
    result.update(depth=np.asarray(depth, np.float32), valid=np.asarray(valid, bool))
    np.savez_compressed(destination, **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)

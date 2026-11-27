"""Bounded MoGe depth worker (its own process, so the model's memory is freed when it ends).  What each pixel is
comes from Mask2Former panoptic (ade_find.py), not from here."""
import argparse

import numpy as np

from common import torch_device


def run(source, destination, version="v2", refine=3):
    import torch
    import moge_model
    with np.load(source, allow_pickle=False) as src:
        images, fovs = src["images"], src["fov"]
    model = moge_model.load(version, torch_device())
    kw = moge_model.infer_kwargs(version, {"moge_refine_steps": refine})
    result = {"images": images, "model": np.asarray({"v2": "moge-2-vitl-normal", "v3": "moge-3-vitl"}.get(version, version))}
    depth, valid = [], []
    for rgb, fov in zip(images, fovs):
        tensor = torch.as_tensor(rgb.copy(), device="cuda", dtype=torch.float32).permute(2, 0, 1) / 255
        with torch.inference_mode():
            pred = model.infer(tensor, fov_x=float(fov), use_fp16=True, **kw)
        points = pred["points"].float().cpu().numpy()
        depth.append(points[..., 2])
        valid.append(pred["mask"].cpu().numpy().astype(bool) & np.isfinite(points).all(-1))
    del pred, tensor, model
    torch.cuda.empty_cache()
    result.update(depth=np.asarray(depth, np.float32), valid=np.asarray(valid, bool))
    np.savez_compressed(destination, **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--moge", default="v2")
    parser.add_argument("--refine", type=int, default=3)
    args = parser.parse_args()
    run(args.input, args.output, args.moge, args.refine)

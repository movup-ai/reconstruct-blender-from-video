"""DA3 worker, executed with an isolated Python environment via da3_python."""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "da3"))
cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "downloads", "da3-cache")
os.makedirs(cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", cache)
os.environ.setdefault("TORCH_HOME", os.path.join(cache, "torch"))
os.environ.setdefault("HF_HOME", os.path.join(cache, "hf"))
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resolution", type=int, default=504)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DA3 requires a working NVIDIA GPU")
    with np.load(args.input, allow_pickle=False) as bundle:
        images = list(bundle["images"])
        intrinsics = bundle["intrinsics"]
    model = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    prediction = model.inference(image=images, intrinsics=intrinsics, process_res=args.resolution,
                                 ref_view_strategy="middle")
    np.savez_compressed(args.output, extrinsics=prediction.extrinsics, depth=prediction.depth, intrinsics=prediction.intrinsics)


if __name__ == "__main__":
    main()

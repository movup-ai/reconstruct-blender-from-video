"""Query saved metric observations from the same cameras used by the final blockout.

python pipeline/scene_evidence.py CLIP --frame 30 --pixel 640 360
Pixels are in the cropped picture area (as shown by the set editor).
"""
import argparse
import json
import os

import numpy as np

from common import cfg, jload
from metric_scene import plane_proposals, points
from scene_verify import load_evidence


def query(data, work, frame=None, pixel=None):
    bundles, skipped = load_evidence(data, work)
    if not bundles: return {"available": False, "stale_files": skipped}
    available = [(abs(int(f)-(frame or 0)), b, i) for b in bundles for i, f in enumerate(b["frames"])]
    _, bundle, index = min(available, key=lambda item: item[0])
    f = int(bundle["frames"][index]); depth = bundle["depth"][index]
    K, R, C = bundle["K"][index], bundle["R"][index], bundle["C"][index]
    xyz = points(depth, K) @ R + C
    valid = bundle["valid"][index] & ~bundle["person"][index] & np.isfinite(xyz).all(-1)
    result = {"available": True, "frame": f, "requested_frame": frame, "coordinates": "z-up world metres",
              "static_samples": int(valid.sum()), "camera_center": C.tolist()}
    if pixel is not None:
        x, y = pixel; W, H = data["res"]
        if not (0 <= x < W and 0 <= y < H): raise ValueError("pixel is outside the picture area")
        x, y = min(depth.shape[1]-1, int(x/W*depth.shape[1])), min(depth.shape[0]-1, int(y/H*depth.shape[0]))
        label = int(bundle["labels"][index,y,x])
        result["pixel"] = {"valid_static": bool(valid[y,x]), "person": bool(bundle["person"][index,y,x]),
            "depth_m": float(depth[y,x]) if np.isfinite(depth[y,x]) else None,
            "world": xyz[y,x].tolist() if valid[y,x] else None,
            "semantic": str(bundle["names"][label]) if 0 <= label < len(bundle["names"]) else None}
    result["surfaces"] = [{"name": p["name"], "semantic": p["_evidence"]["semantic"],
        "center": np.mean(p["v"],axis=0).tolist(), "extent_xyz": np.ptp(p["v"],axis=0).tolist(),
        "frame": p["_evidence"]["frame"]} for p in plane_proposals(bundle)[:20]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip"); parser.add_argument("--frame", type=int)
    parser.add_argument("--pixel", type=float, nargs=2)
    args = parser.parse_args(); c = cfg(); work = os.path.join(c["work_root"], args.clip)
    print(json.dumps(query(jload(os.path.join(work,"scene_data.json")), work, args.frame, args.pixel), indent=2))


if __name__ == "__main__": main()

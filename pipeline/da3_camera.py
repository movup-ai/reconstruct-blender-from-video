"""Optional multi-view camera recovery with Depth Anything 3."""
import os
import subprocess
import tempfile
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from common import AGENT, pick_device

_models = {}


def solve(frames, cam, y0, height, c):
    path = c.get("da3_model") or os.path.join(AGENT, "models", "da3")
    if not os.path.isdir(path):
        raise RuntimeError("Depth Anything 3 checkpoint missing: %s" % path)
    count = min(len(frames), max(2, int(c.get("da3_max_views", 12))))
    indices = np.unique(np.linspace(0, len(frames) - 1, count, dtype=int))
    images = [cv2.cvtColor(frames[f][y0:y0 + height], cv2.COLOR_BGR2RGB) for f in indices]
    intrinsics = np.repeat(cam.K()[None], len(images), axis=0)
    python = c.get("da3_python")
    if python:
        with tempfile.TemporaryDirectory(dir=c.get("_work")) as work:
            source, result = os.path.join(work, "input.npz"), os.path.join(work, "result.npz")
            np.savez_compressed(source, images=np.asarray(images), intrinsics=intrinsics)
            subprocess.run([python, os.path.join(os.path.dirname(__file__), "da3_infer.py"), "--input", source,
                            "--output", result, "--model", path, "--resolution", str(c.get("da3_resolution", 504))],
                           check=True, timeout=600)
            with np.load(result, allow_pickle=False) as prediction:
                ext, depths = prediction["extrinsics"].copy(), prediction["depth"].copy()
    else:
        from depth_anything_3.api import DepthAnything3
        if path not in _models:
            _models[path] = DepthAnything3.from_pretrained(path).to(pick_device("cuda")).eval()
        prediction = _models[path].inference(image=images, intrinsics=intrinsics,
                                             process_res=int(c.get("da3_resolution", 504)), ref_view_strategy="middle")
        ext, depths = prediction.extrinsics, prediction.depth
    ext = np.asarray(ext, float)
    if len(ext) != len(indices) or not np.isfinite(ext).all():
        raise RuntimeError("DA3 returned invalid camera extrinsics")
    rotation = ext[:, :3, :3]
    relative = rotation @ rotation[0].T
    if len(indices) > 1:
        rotations = Slerp(indices, Rotation.from_matrix(relative))(np.arange(len(frames))).as_matrix()
    else:
        rotations = relative
    centers = -np.einsum("nji,nj->ni", rotation, ext[:, :3, 3])
    travel = (centers - centers[0]) @ rotation[0].T
    travel = np.column_stack([np.interp(np.arange(len(frames)), indices, travel[:, axis]) for axis in range(3)])
    angle = np.degrees(np.linalg.norm(Rotation.from_matrix(rotations).as_rotvec(), axis=1))
    result = {"R": rotations.tolist(), "zoom": [1.0] * len(frames), "max_rotation_deg": float(angle.max()),
              "parallax_px": 0.0, "warnings": [], "cuts": [], "inlier_share": 0.0, "backend": "da3",
              "sample_frames": indices.tolist(), "travel_cam0": travel.tolist()}
    return result, {int(f): np.asarray(d, float) for f, d in zip(indices, depths)}


def align_scale(camj, depth, people, cam, y0, up, height, c):
    """Align model units with the solved floor using visible ankles across sampled views."""
    scale = c.get("da3_meters_per_unit")
    ratios = []
    samples = depth if isinstance(depth, dict) else {0: depth}
    for frame, sample in samples.items():
        rotation = np.asarray(camj.get("R", [np.eye(3)])[frame])
        center = np.asarray(camj["travel_cam0"][frame])
        for fs, keys in people.values():
            if frame not in fs:
                continue
            k = np.asarray(keys[fs.index(frame)], float)
            for joint in (15, 16):
                x, y, confidence = k[joint]
                y -= y0
                if confidence < 0.6 or not (0 <= x < cam.W and 0 <= y < cam.H):
                    continue
                ray = np.array([(x - cam.W / 2) / cam.f, (y - cam.H / 2) / cam.f, 1.0])
                measured = sample[min(sample.shape[0] - 1, int(y / cam.H * sample.shape[0])),
                                  min(sample.shape[1] - 1, int(x / cam.W * sample.shape[1]))]
                denominator = float(up @ (center + rotation.T @ (ray * measured)))
                if denominator < -0.05 and np.isfinite(measured) and measured > 0:
                    ratios.append(-height / denominator)
    if scale is None and len(ratios) >= 2:
        scale = float(np.median(ratios))
    if scale is None:
        camj.pop("travel_cam0")
        camj["warnings"].append("DA3 translation scale unavailable: set da3_meters_per_unit or provide visible floor contacts")
        return
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("da3_meters_per_unit must be positive and finite")
    camj["travel_cam0"] = (np.asarray(camj["travel_cam0"]) * scale).tolist()
    camj["meters_per_unit"] = scale

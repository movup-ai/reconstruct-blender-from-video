# -*- coding: utf-8 -*-
"""Depth Anything V2 Small (the camera's depth cues, camera_estimate.py) and Stage 1's floor (flat_set).

Stage 1 builds no set from depth: the floor is a plain ground disc under the people.  The background (Stage 2,
planned) will be built from the saved depth (metric_observation*.npz) and prebuilt assets."""
import os
import numpy as np
import cv2
from common import AGENT, gpu_dlls, log

MODEL_DIR = os.path.join(AGENT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "depth_anything_v2_small.onnx")

class DepthModel:
    def __init__(self, device="cuda"):
        import onnxruntime as ort
        if device != "cuda":
            raise RuntimeError("the depth model runs on the NVIDIA card only (device %r)" % device)
        gpu_dlls()                                       # CUDA / cuDNN installed by pip (onnxruntime-gpu[cuda,cudnn])
        self.s = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in self.s.get_providers():
            raise RuntimeError("depth model did not load on the NVIDIA card - run Setup.bat (installs onnxruntime-gpu)")
        self.s.disable_fallback()                        # a GPU error must stop the job, not quietly use the CPU
        log("    depth model on %s" % self.s.get_providers()[0])
        self.inp = self.s.get_inputs()[0].name
    def __call__(self, bgr):
        H, W = bgr.shape[:2]
        h = 518; w = int(round(W / H * h / 14.0)) * 14
        w = min(max(w, 14 * 20), 14 * 120)
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        d = self.s.run(None, {self.inp: np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)})[0]
        d = np.squeeze(d).astype(np.float32)
        return cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)      # relative inverse depth (bigger = nearer)

_MODELS = {}
def depth_model(device):
    """the depth model of this process (loaded once)"""
    if device not in _MODELS:
        _MODELS[device] = DepthModel(device)
    return _MODELS[device]

def flat_set(world, N, far):
    """Stage 1's floor: one flat ground disc (radius 0.6 x far) at z = 0 around the shot's camera; no backdrop"""
    R = far * 0.6; seg = 72
    v = [[0, 0, 0]] + [[R * np.cos(2 * np.pi * i / seg), R * np.sin(2 * np.pi * i / seg), 0] for i in range(seg)]
    return [{"name": "ground", "v": v, "f": [[0, 1 + i, 1 + (i + 1) % seg] for i in range(seg)], "colour": "ground", "layer": 1}]

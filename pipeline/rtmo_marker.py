"""RTMO ONNX adapter using the same COCO-17 output as automark.Marker."""
import os
import numpy as np
from common import gpu_session, gpu_dlls


class RTMOMarker:
    def __init__(self, model_path, engine="tensorrt", tiles=True):
        from rtmlib import RTMO
        gpu_dlls()
        if not os.path.isfile(model_path):
            raise RuntimeError("RTMO checkpoint missing: %s (set rtmo_model to a COCO-17 RTMO ONNX file)" % model_path)
        model = RTMO.__new__(RTMO)
        model.model_input_size = (640, 640)
        model.mean = model.std = None
        model.backend, model.device = "onnxruntime", "cuda"
        model.nms_thr, model.score_thr, model.to_openpose = 0.45, 0.5, False
        model.session, self.engine = gpu_session(model_path, engine)
        model.session.disable_fallback()
        self.model, self.tiles = model, tiles
        if not any(p in model.session.get_providers() for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")):
            raise RuntimeError("RTMO requires an NVIDIA GPU execution provider")
        self.n_det = self.n_pose_calls = self.n_crops = 0

    def __call__(self, image):
        from automark import drop_same_skeletons
        h, w = image.shape[:2]
        tw, th = int(w * 0.6), int(h * 0.6)
        views = [(0, 0, image)]
        if self.tiles:
            views += [(x, y, image[y:y + th, x:x + tw]) for y in (0, h - th) for x in (0, w - tw)]
        people = []
        for x, y, view in views:
            keys, scores = self.model(view)
            self.n_det += 1
            for k, s in zip(keys, scores):
                good = s > 0.3
                if good.sum() < 4:
                    continue
                k = np.asarray(k, float) + [x, y]
                lo, hi = k[good].min(0), k[good].max(0)
                if hi[1] - lo[1] > 18:
                    people.append((k, s, [*lo, *hi]))
        return drop_same_skeletons(people)

    def run_many(self, images):
        return [self(image) for image in images]

    def poses(self, items):
        """Re-detect inside predicted crops when filling an occlusion gap."""
        results = []
        for image, boxes in items:
            keys, scores = [], []
            for x0, y0, x1, y1 in boxes:
                x0, y0 = max(0, int(x0)), max(0, int(y0))
                x1, y1 = min(image.shape[1], int(x1)), min(image.shape[0], int(y1))
                k, s = np.zeros((17, 2)), np.zeros(17)
                if x1 > x0 and y1 > y0:
                    kk, ss = self.model(image[y0:y1, x0:x1])
                    self.n_pose_calls += 1
                    self.n_crops += 1
                    if len(ss):
                        best = int(np.argmax(ss.mean(axis=1)))
                        k, s = kk[best] + [x0, y0], ss[best]
                keys.append(k); scores.append(s)
            results.append((np.asarray(keys).reshape(-1, 17, 2), np.asarray(scores).reshape(-1, 17)))
        return results

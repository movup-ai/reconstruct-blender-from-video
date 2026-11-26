# -*- coding: utf-8 -*-
"""What every pixel of a set frame is: Mask2Former (Swin-L) trained on ADE20K's 150 indoor and outdoor classes
(models\\mask2former_ade, facebook/mask2former-swin-large-ade-semantic), on the NVIDIA card in half precision.
Stage 1 (metric_observe.py / metric_scene.py) uses its floor, wall and ceiling labels for gravity and the camera
height.  ROLE (class -> what a piece is) is kept for the planned Stage 2 background."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models", "mask2former_ade")

# ADE20K class -> role (Stage 2).  Surfaces and big things ("floor", "ground", "ceiling", "wall",
# "building", "far", "sky", "rock", "tree"); "relief" = built from its measured shape; any other value is a
# library model type sized from the measurement.  "person": not part of the set.
ROLE = {
    "floor": "floor", "rug": "floor", "stage": "floor",
    "road": "ground", "sidewalk": "ground", "earth": "ground", "field": "ground", "sand": "ground", "path": "ground",
    "land": "ground", "dirt track": "ground", "runway": "ground", "grass": "ground", "water": "ground", "sea": "ground",
    "lake": "ground", "river": "ground", "swimming pool": "ground",
    "ceiling": "ceiling", "sky": "sky", "person": "person",
    "wall": "wall", "windowpane": "window", "blind": "curtain", "painting": "picture", "poster": "picture", "bulletin board": "wall",
    "mirror": "mirror", "door": "door", "screen door": "door", "signboard": "wall", "clock": "wall", "sconce": "wall",
    "building": "building", "house": "building", "skyscraper": "building", "tower": "building", "hovel": "building",
    "mountain": "far", "hill": "far", "rock": "rock", "tree": "tree", "palm": "tree", "forest": "tree",
    "plant": "relief", "flower": "relief", "curtain": "curtain", "sculpture": "relief", "vase": "vase", "pot": "pot",
    "apparel": "relief", "towel": "relief", "blanket": "relief", "cushion": "relief", "pillow": "relief", "animal": "relief",
    "basket": "relief", "plaything": "relief", "food": "relief", "flag": "relief", "fountain": "relief",
    "chandelier": "relief", "bag": "relief", "canopy": "relief", "bicycle": "relief", "minibike": "relief", "boat": "relief",
    "bed": "bed", "cabinet": "cabinet", "table": "table", "chair": "chair", "sofa": "sofa", "shelf": "shelf",
    "armchair": "sofa", "seat": "chair", "desk": "desk", "wardrobe": "cabinet", "lamp": "lamp", "bathtub": "box",
    "counter": "counter", "sink": "sink", "refrigerator": "fridge", "bookcase": "bookshelf", "coffee table": "table",
    "bench": "bench", "countertop": "counter", "stove": "stove", "kitchen island": "counter", "computer": "tv",
    "swivel chair": "office_chair", "television receiver": "tv", "monitor": "tv", "crt screen": "tv", "screen": "tv",
    "stool": "chair", "ottoman": "sofa", "chest of drawers": "cabinet", "box": "box", "barrel": "bin", "ashcan": "bin",
    "oven": "stove", "microwave": "box", "column": "pillar", "railing": "railing", "bannister": "railing",
    "fence": "fence", "stairs": "stairs", "stairway": "stairs", "step": "stairs", "car": "car", "truck": "car",
    "van": "car", "bus": "car", "fireplace": "cabinet", "pool table": "table", "radiator": "box", "case": "cabinet",
    "buffet": "cabinet", "bar": "counter", "booth": "box", "streetlight": "pillar", "pole": "pillar",
    "traffic light": "pillar", "awning": "awning", "tent": "building", "book": "box", "bottle": "relief",
    "tray": "box", "plate": "box", "glass": "relief", "fan": "relief", "hood": "box", "washer": "box",
    "dishwasher": "box", "toilet": "relief", "shower": "box", "tank": "box", "arcade machine": "cabinet",
    "conveyer belt": "box", "cradle": "relief", "ball": "relief", "light": "lamp", "base": "box",
    "kitchen": "counter", "trade name": "wall", "pier": "floor", "bridge": "building", "grandstand": "building",
    "escalator": "stairs", "airplane": "relief", "ship": "relief", "waterfall": "far", "crt": "tv",
}


def available():
    import importlib.util
    return os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")) and importlib.util.find_spec("transformers") is not None


def role(name):
    return ROLE.get((name or "").strip(), "object")


class Segmenter:
    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the segmentation model runs on the NVIDIA card only")
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)      # (the backbone's unused last norm is "missing")
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(MODEL_DIR)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_DIR, dtype=torch.float16).to(device).eval()
        lab = self.model.config.id2label
        self.names = [lab[i].strip() for i in range(len(lab))]

    def __call__(self, rgb, width=1024):
        """RGB picture -> (H, W) class index map at the picture's size.  Run at <width> wide, the frame's own shape
        (the model's default square input squeezes a wide frame)"""
        torch = self.torch
        H, W = rgb.shape[:2]
        w = int(round(width / 32.0)) * 32; h = max(32, int(round(width * H / float(W) / 32.0)) * 32)
        inp = self.proc(images=rgb, size={"height": h, "width": w}, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.inference_mode():
            out = self.model(**inp)
            seg = self.proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0]
        return seg.to(torch.int16).cpu().numpy()

    def label(self, seg, mask, min_share=0.0):
        """the class most of <mask> has -> (name, share) or (None, 0)"""
        v = seg[mask]
        if not len(v):
            return None, 0.0
        c = np.bincount(v.astype(np.int64), minlength=len(self.names))
        k = int(c.argmax()); s = float(c[k]) / len(v)
        return (self.names[k], s) if s >= min_share else (None, s)

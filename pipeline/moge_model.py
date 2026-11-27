# -*- coding: utf-8 -*-
"""The MoGe the lens (lens_auto.py), metric scene and object depth (metric_observe.py) steps load:
MoGe-2 ViT-L, models\\moge2\\model.pt, code in third_party\\MoGe."""
import os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MOGE_DIR = os.path.join(ROOT, "third_party", "MoGe")
WEIGHTS = os.path.join(ROOT, "models", "moge2", "model.pt")
NAME = "MoGe-2"


def missing():
    """why MoGe-2 cannot run, or None"""
    if not os.path.exists(WEIGHTS):
        return "%s weights missing: %s" % (NAME, WEIGHTS)
    if not os.path.isfile(os.path.join(MOGE_DIR, "moge", "model", "v2.py")):
        return "third_party\\MoGe has no %s code" % NAME
    if importlib.util.find_spec("torch") is None:
        return "%s needs the python package torch" % NAME
    if not any(importlib.util.find_spec(n) is not None for n in ("utils3d_moge", "utils3d")):
        return "%s needs utils3d" % NAME
    return None


def load(device):
    """MoGe-2 on device, in eval mode"""
    if MOGE_DIR not in sys.path:
        sys.path.insert(0, MOGE_DIR)
    from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained(WEIGHTS).to(device).eval()

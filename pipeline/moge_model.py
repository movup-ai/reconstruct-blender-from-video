# -*- coding: utf-8 -*-
"""Which MoGe the lens (lens_auto.py) and metric scene (metric_observe.py) steps load.

config.json "moge_version": "v2" = MoGe-2 ViT-L (models\\moge2\\model.pt)
                            "v3" = MoGe-3 ViT-L (models\\moge3\\model.pt; needs flex_gemm + triton-windows)
"moge_refine_steps" (MoGe-3 only, default 3): its sparse refinement passes."""
import os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MOGE_DIR = os.path.join(ROOT, "third_party", "MoGe")
WEIGHTS = {"v2": os.path.join(ROOT, "models", "moge2", "model.pt"),
           "v3": os.path.join(ROOT, "models", "moge3", "model.pt")}
NAMES = {"v2": "MoGe-2", "v3": "MoGe-3"}
DEFAULT = "v2"


def version(c=None):
    v = str((c or {}).get("moge_version", DEFAULT)).lower()
    return v if v in WEIGHTS else DEFAULT


def name(c=None):
    return NAMES[version(c)]


def missing(c=None):
    """why the configured MoGe cannot run, or None"""
    v = version(c)
    if not os.path.exists(WEIGHTS[v]):
        return "%s weights missing: %s" % (NAMES[v], WEIGHTS[v])
    if not os.path.isfile(os.path.join(MOGE_DIR, "moge", "model", v + ".py")):
        return "third_party\\MoGe has no %s code" % NAMES[v]
    needs = ["torch"] + (["flex_gemm", "triton"] if v == "v3" else [])
    for n in needs:
        if importlib.util.find_spec(n) is None:
            return "%s needs the python package %s" % (NAMES[v], n)
    if not any(importlib.util.find_spec(n) is not None for n in ("utils3d_moge", "utils3d")):
        return "%s needs utils3d" % NAMES[v]
    return None


def load(v, device):
    """the MoGe model of version v ("v2" / "v3") on device, in eval mode"""
    if MOGE_DIR not in sys.path:
        sys.path.insert(0, MOGE_DIR)
    if v == "v3":
        from moge.model.v3 import MoGeModel
    else:
        from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained(WEIGHTS[v]).to(device).eval()


def infer_kwargs(v, c=None):
    """extra infer() arguments for version v"""
    return {"refine_steps": int((c or {}).get("moge_refine_steps", 3))} if v == "v3" else {}

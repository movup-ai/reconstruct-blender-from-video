# -*- coding: utf-8 -*-
"""download the Depth Anything V2 Small depth model (Apache-2.0, ~99 MB) into models\\"""
import os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cfg
from scene_depth import MODEL_PATH, MODEL_DIR
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 50e6:
    print("depth model already there"); sys.exit(0)
os.makedirs(MODEL_DIR, exist_ok=True)
url = cfg()["depth_model_url"]; tmp = MODEL_PATH + ".part"
print("downloading depth model (~99 MB) ...")
def hook(n, bs, total):
    if total > 0 and n % 200 == 0:
        print("  %d%%" % min(100, n * bs * 100 // total), flush=True)
urllib.request.urlretrieve(url, tmp, hook)
os.replace(tmp, MODEL_PATH)
print("saved", MODEL_PATH)

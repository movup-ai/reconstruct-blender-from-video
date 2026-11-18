# -*- coding: utf-8 -*-
"""Render a clip in the earlier sample layout (package_v1: <name>/comparison|meta|model_3d|target|white_model,
meta.jsonl, <name>blend.blend, GBK zip) - for the approved trial samples (0529 / b5b6 / ec34), so a corrected
version looks exactly like the one the client reviewed except for the fix.

  python pipeline\\render_sample_v1.py <video> <name> <out_root> [--width 2560] [--marker 52,190,200]"""
import argparse, json, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload, jsave
import run_pipeline as rp
import package as pkg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("name"); ap.add_argument("out_root")
    ap.add_argument("--width", type=int, default=2560); ap.add_argument("--marker", default="52,190,200")
    a = ap.parse_args()
    c = dict(cfg()); c["out_width"] = a.width; c["submit_root"] = a.out_root
    work = os.path.join(c["work_root"], a.name)
    dp = os.path.join(work, "scene_data.json"); data = jload(dp)
    data["submit_dir"] = os.path.join(a.out_root, a.name); data["work_dir"] = work
    data["render_scale"] = a.width / float(data["res"][0]); data.pop("package_name", None)
    pal = data.setdefault("palette", {})
    for k, v in rp.PALETTE.items():
        pal.setdefault(k, v)
    pal["marker"] = [int(x) for x in a.marker.split(",")]
    rp.retime_pulldown(data, a.video, c)                 # even motion after the pulldown frames are dropped
    jsave(dp, data)
    rp.blender_render(c, work, dp, data["frames"])
    rec = os.path.join(data["submit_dir"], "model_3d", "reconstruction")      # v2.1 extra, not part of the v1 layout
    shutil.rmtree(rec, ignore_errors=True)
    ci = jload(os.path.join(work, "clip_info.json"))
    sub, zp = pkg.package_v1(a.name, a.video, os.path.join(work, "render"), a.out_root, ci, c, "")
    print("package", sub, zp)


if __name__ == "__main__":
    main()

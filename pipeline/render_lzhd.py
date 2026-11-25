# -*- coding: utf-8 -*-
"""Render + package one clip to the LZHD V1.0 spec (White-model Video Reconstruction and Acceptance Standard V1.0, 2026-11-15):
official human_master.blend characters (style "master"), palette colours by meaning, Eevee, white model at the
source's own width / height / frame rate / bit rate, comparison original-over-white-model at twice the height.

  python pipeline\\render_lzhd.py <video> <clip name> <out_root> --pkg dragon-22 [--frames 1-24] [--still 40]

Colours: the colour card (colour_card.py, docs/COLOUR_CARD.md), as for every render.
Light: hints.json "light": {"az": deg from the camera view (0 behind the camera, -90 screen left, 180 back light),
"el": deg, "sun": W/m2, "ambient": world strength}."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload, jsave, output_dir
import run_pipeline as rp
import package as pkg

import colour_card


def spec_palette(data):
    """Apply semantic colors while preserving the geometry used to fit this scene."""
    colour_card.apply(data, log=None)
    # Geometry and fitted joint definitions are a pair. Never swap only the
    # render style on an already fitted scene (it changes projected proportions).


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("name"); ap.add_argument("out_root")
    ap.add_argument("--pkg", required=True); ap.add_argument("--still", type=int, default=0)
    ap.add_argument("--no-package", action="store_true")
    a = ap.parse_args()
    if not pkg.PKG_RE.match(a.pkg):
        raise SystemExit("package name must look like dragon-22")
    c = dict(cfg())
    work = os.path.join(c["work_root"], a.name)
    dp = os.path.join(work, "scene_data.json"); data = jload(dp)
    ci = jload(os.path.join(work, "clip_info.json"))
    hp = os.path.join(rp.marks_dir(a.video, c), "hints.json")
    if not os.path.exists(hp):
        hp = os.path.join(output_dir(), a.name, "hints.json")
    hints = jload(hp) if os.path.exists(hp) else {}
    data["submit_dir"] = os.path.join(a.out_root, a.name); data["work_dir"] = work
    data["package_name"] = a.pkg; data["spec"] = "lzhd1"
    data["render_scale"] = 1.0                                  # exactly the picture area of the source, no resampling
    if hints.get("light"):
        data["light"] = hints["light"]
    spec_palette(data)
    if hints.get("plant_feet"):                                 # feet on the floor (+ stepping gait), as render_only does
        import ground_feet
        av = {}
        pp = os.path.join(rp.marks_dir(a.video, c), "pose.json")
        if os.path.exists(pp):
            for t in jload(pp)["people"]:
                av[str(t["id"])] = {int(f) for f, k in t["frames"].items() if min(k[15][2], k[16][2]) >= 0.4}
        data.pop("feet_planted", None)
        ground_feet.plant(data, log=rp.log, hints=hints, ankle_vis=av)
    rp_dp = os.path.join(work, "scene_lzhd.json")
    jsave(rp_dp, data)
    if a.still:
        import subprocess
        with open(os.path.join(work, "blender_log.txt"), "w", encoding="utf-8", errors="ignore") as lf:
            subprocess.run([rp.blender_exe(c), "-b", "-P", os.path.join(HERE, "blender_build.py"), "--", rp_dp, "still", str(a.still)],
                           stdout=lf, stderr=subprocess.STDOUT)
        print("still", os.path.join(work, "still.jpg"))
        return
    rp.blender_render(c, work, rp_dp, data["frames"])
    if a.no_package:
        return
    desc_p = ci.get("desc_path") or ""
    desc = ""
    if desc_p and os.path.exists(desc_p):
        with open(desc_p, encoding="utf-8") as fh:
            desc = fh.read().strip()
    sub, zp, meta = pkg.package_lzhd(a.name, a.pkg, a.video, os.path.join(work, "render"), a.out_root, ci, c, desc,
                                     notes=hints.get("notes"))
    print("package", sub, zp)
    print(json.dumps(meta["output_checks"], ensure_ascii=False), meta["qualification_flags"])


if __name__ == "__main__":
    main()

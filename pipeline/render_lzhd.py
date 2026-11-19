# -*- coding: utf-8 -*-
"""Render + package one clip to the LZHD V1.0 spec (白模视频复刻需求及验收标准 V1.0, 2026-11-15):
official human_master.blend characters (style "master"), palette colours by meaning, Eevee, white model at the
source's own width / height / frame rate / bit rate, comparison original-over-white-model at twice the height.

  python pipeline\\render_lzhd.py <video> <clip name> <out_root> --pkg dragon-22 [--frames 1-24] [--still 40]

Set colours (env_claude.json "colour"): bg (背景 #B8B8B8), fg (前景物体 #9A6B3B), near (近处交互物 #DF872F),
far_i (远处交互物 #D4689E), lamp (光源 #F5F5F5), special (特殊对象 #7A4EC2).  The older grey names (white, grey,
dark, light, set, ground ...) become bg, brown -> fg, orange -> near, pink -> far_i, purple -> special.
Light: hints.json "light": {"az": deg from the camera view (0 behind the camera, -90 screen left, 180 back light),
"el": deg, "sun": W/m2, "ambient": world strength}."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload, jsave, output_dir
import run_pipeline as rp
import package as pkg

SPEC = {"red": [214, 62, 61], "blue": [45, 107, 216], "green": [46, 157, 90], "yellow": [217, 160, 20],
        "bg": [184, 184, 184], "fg": [154, 107, 59], "near": [223, 135, 47], "far_i": [212, 104, 158],
        "lamp": [245, 245, 245], "special": [122, 78, 194], "marker": [255, 255, 255],
        "cyan": [32, 168, 184], "magenta": [176, 40, 120]}                 # 5th / 6th person (spec: assign and keep)
LEGACY = {"brown": "fg", "orange": "near", "pink": "far_i", "purple": "special"}
PEOPLE = ("red", "blue", "green", "yellow")


def spec_palette(data):
    pal = data.setdefault("palette", {})
    used = {m["colour"] for m in data.get("meshes", [])} | {p["colour"] for p in data.get("props", [])}
    for k in list(pal) + list(used):
        if k in SPEC:
            continue
        pal[k] = SPEC[LEGACY.get(k, "bg")]                      # every other set colour is background grey
    pal.update(SPEC)
    order = list(PEOPLE) + ["cyan", "magenta"]
    for i, ch in enumerate(sorted(data.get("chars", {}).values(), key=lambda c: order.index(c["colour"]) if c["colour"] in order else 9)):
        if ch["colour"] not in order:                          # purple (5th) -> cyan: purple is the special-object colour
            ch["colour"] = "cyan"
        ch["style"] = "master"


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

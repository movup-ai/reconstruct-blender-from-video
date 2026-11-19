# -*- coding: utf-8 -*-
"""Redo one clip by package name (avoids passing Chinese clip names through PowerShell):

  python pipeline\\redo_clip.py the_hidden_blade-12 [--stages fit,build_env,render] [--zoom 24] [--frames 40,60,80]
  python pipeline\\redo_clip.py the_hidden_blade-12 --stages none --zoom 24          (only new close-up sheets)
  python pipeline\\redo_clip.py the_hidden_blade-12 --angles 40,90 --chars P1         (print fitted joint angles)

fit       -> refit the people from marks + review.json + output\\<clip>\\hints.json
build_env -> put the set (output\\<clip>\\env_claude.json) into the scene
render    -> Blender render + v2.1 package (SUBMIT\\<clip>\\, SUBMIT\\<clip>.zip)
--zoom N  -> Work_auto\\review_v21\\<package>_zoom_K.jpg, N frames over the clip (or --frames), 6 frames per sheet:
             per character  film | film + detected joints (yellow) + fitted skeleton | render"""
import argparse, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from agent import config
from agent.clips import Library

PY = sys.executable


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pkg")
    ap.add_argument("--stages", default="fit,build_env,render"); ap.add_argument("--zoom", type=int, default=0)
    ap.add_argument("--frames", default=""); ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--angles", default=""); ap.add_argument("--chars", default="")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    with open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8") as fh:
        names = json.load(fh)
    inv = {v: k for k, v in names.items()}
    clip = inv.get(a.pkg, a.pkg)
    c = config.load(); cl = next(x for x in Library(c).clips if x.name == clip)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    st = [s for s in a.stages.split(",") if s and s != "none"]
    if "fit" in st:
        p = subprocess.run([PY, os.path.join(HERE, "run_pipeline.py"), cl.video, "--stage", "fit"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        print("fit:", "ok" if p.returncode == 0 else p.stdout[-1500:] + p.stderr[-1500:])
        for line in p.stdout.splitlines():
            if "error" in line and "frame" in line or "overlap" in line or "person" in line:
                print("   ", line.strip())
    if "build_env" in st:
        p = subprocess.run([PY, os.path.join(HERE, "build_env.py"), clip], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        print("build_env:", "ok" if p.returncode == 0 else p.stdout[-800:] + p.stderr[-800:])
    if "render" in st:
        p = subprocess.run([PY, os.path.join(HERE, "run_pipeline.py"), cl.video, "--stage", "render"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        print("render:", "ok" if p.returncode == 0 and "1 ok" in p.stdout else p.stdout[-1500:] + p.stderr[-800:])
    with open(os.path.join(c["work_root"], clip, "scene_data.json"), encoding="utf-8") as fh:
        sd = json.load(fh)
    N = sd["frames"]
    if a.angles:
        for kind, ch in sd["chars"].items():
            if a.chars and kind not in a.chars.split(","):
                continue
            for f in [int(x) for x in a.angles.split(",")]:
                r = ch["root"][f]; cm = sd["cam"][f]
                yc = np.degrees(np.arctan2(-(cm[0] - r[0]), cm[1] - r[1])) % 360          # heading that faces the camera
                print("%s %s f%d vis=%s yaw=%.0f (faces camera at %.0f, back to camera at %.0f) pelvis_z=%.2f " % (
                      kind, ch["colour"], f, ch["visible"][f], r[3] % 360, yc, (yc + 180) % 360, ch["pelvis_z"][f])
                      + " ".join("%s=%s" % (j, [round(v) for v in ch["rot"][j][f]]) for j in
                                 ("spine", "chest", "head", "shoulder_L", "elbow_L", "shoulder_R", "elbow_R", "hip_L", "knee_L", "hip_R", "knee_R")))
    if a.zoom or a.frames:
        fr = [int(x) for x in a.frames.split(",")] if a.frames else sorted(set(np.linspace(0, N - 1, a.zoom).astype(int).tolist()))
        out_dir = os.path.join(c["work_root"], "review_v21"); os.makedirs(out_dir, exist_ok=True)
        groups = [fr[i:i + 6] for i in range(0, len(fr), 6)]
        for gi, g in enumerate(groups):
            out = os.path.join(out_dir, "%s_zoom_%d.jpg" % (a.pkg, gi + 1))
            p = subprocess.run([PY, os.path.join(HERE, "compare.py"), clip, "--zoom", "--size", str(a.size), "--frames", ",".join(map(str, g)), "--out", out],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
            print(out if p.returncode == 0 else p.stderr[-500:])


if __name__ == "__main__":
    main()

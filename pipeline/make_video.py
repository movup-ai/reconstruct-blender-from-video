# -*- coding: utf-8 -*-
"""White-model video of a rendered clip, without the client package (no package name, meta or zip needed).

  python pipeline\\make_video.py <clip name>        clip name = its folder in Work_auto (after --stage blend)

Writes into SUBMIT\\<clip>\\:
  white_model\\<clip>.mp4    the Blender render, same frame rate, length and audio as the source
  comparison\\<clip>.mp4     source on top, white model below"""
import os, sys
from common import cfg, ffmpeg_exe, jload, log
import package as pkg


def main():
    if len(sys.argv) != 2:
        print(__doc__); return 2
    name = sys.argv[1]; c = cfg()
    work = os.path.join(c["work_root"], name)
    ci = jload(os.path.join(work, "clip_info.json"))
    video, W, Hfull, y0, Hp, N, fps = ci["video"], ci["W"], ci["Hfull"], ci["y0"], ci["Hpic"], ci["frames"], ci["fps"]
    frames = os.path.join(work, "render", "wm_%04d.jpg")
    if not os.path.exists(frames % 1):
        frames = os.path.join(work, "render", "wm_%04d.png")
    if not os.path.exists(frames % 1):
        log("no rendered frames in %s - run:  run_pipeline.py <video> --stage blend" % os.path.join(work, "render")); return 1
    OW, OH = pkg.out_size(W, Hfull, c.get("out_width", 1920))
    sy = OH / float(Hfull)                                 # picture area inside the full frame (letterbox)
    PH = int(round(Hp * sy)); PH -= PH % 2; PY = int(round(y0 * sy)); PY -= PY % 2
    ff = ffmpeg_exe(); D = N / float(fps)
    inp = ["-framerate", "%.6f" % fps, "-start_number", "1", "-i", frames, "-i", video]
    tail = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setsar=1"
    WM = "[0:v]scale=%d:%d:flags=lanczos,pad=%d:%d:0:%d:black,%s" % (OW, PH, OW, OH, PY, tail)
    sub = os.path.join(c["submit_root"], name)
    wm = os.path.join(sub, "white_model", name + ".mp4"); cmp_ = os.path.join(sub, "comparison", name + ".mp4")
    for p in (wm, cmp_):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    pkg.encode(ff, c, OW, OH, inp, WM + "[v]", video, D, wm)
    pkg.encode(ff, c, OW, 2 * OH, inp, WM + "[b];[1:v]scale=%d:%d:flags=lanczos,%s[a];[a][b]vstack=inputs=2[v]" % (OW, OH, tail),
               video, D, cmp_)
    log("white model: %s\ncomparison:  %s" % (wm, cmp_))
    return 0


if __name__ == "__main__":
    sys.exit(main())

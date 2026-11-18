# -*- coding: utf-8 -*-
"""11月17日 batch -> client portal packages (spec v2.1 as shown on the portal, 2026-11-17):
ZIP name = the portal task's name pef3d2593_tNNN-1 (11月17日.mp4 = t001, 11月17日(n).mp4 = t(n+1)),
all three videos in the 1080P layout (white model letterboxed into 1920x1080, comparison = original over white model
inside one 1920x1080 frame, target/original.mp4 = the source byte for byte), model_3d/<zip name>.blend + scene.json,
meta/meta.json + README, checked with the client's tools\\v21\\check_package.py.

Built from the finished LZHD render of each clip (Work_auto\\<clip>\\render, the frames the LZHD package was made
from) and its .blend (DELIVERY\\lzhd_batch\\<clip>\\model_3d\\<lzhd pkg>.blend, copied and renamed).

  python pipeline\\portal_pack.py [--clips Dragon_11月17日(22),...] [--out DELIVERY\\portal]"""
import argparse, glob, json, os, re, shutil, subprocess, sys, zipfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import cfg, jload
import package as pkg

SRC = os.path.join(ROOT, "SOURCE", "11月17日原始素材191秒")
LZ = os.path.join(ROOT, "DELIVERY", "lzhd_batch")


def clips():
    out = []
    for film in ("Dragon", "Swan"):
        for v in sorted(glob.glob(os.path.join(SRC, film, "*.mp4"))):
            stem = os.path.splitext(os.path.basename(v))[0]
            m = re.search(r"\((\d+)\)$", stem)
            n = int(m.group(1)) if m else 0
            out.append({"clip": "%s_%s" % (film, stem), "video": v, "n": n, "lzhd": "%s-%d" % (film.lower(), n),
                        "portal": "pef3d2593_t%03d-1" % (n + 1)})
    return out


def pad_target(sub_dir, video, ci, c):
    """portal rule: target/original.mp4 must be 1920x1080 -> the source padded with black bars (centred, the
    picture keeps its pixels and position = the white model's letterbox), same frames, source audio copied"""
    ff = pkg.ffmpeg_exe()
    si = pkg.source_video_info(ff, video)
    W, H = si["width"], si["height"]
    CW, CH = pkg.canvas(W, H)
    if (W, H) == (CW, CH):
        return None
    out = os.path.join(sub_dir, "target", "original.mp4")
    D = ci["frames"] / float(ci["fps"])
    tail = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setsar=1"
    pkg.encode(ff, dict(c, out_fps=int(round(si["fps"]))), CW, CH, ["-i", video, "-i", video],
               "[0:v]pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black,%s[v]" % (CW, CH, tail), video, D, out)
    n = pkg.count_frames(ff, out)
    if n != ci["frames"]:
        raise RuntimeError("padded original has %d frames, source %d" % (n, ci["frames"]))
    note = ("原片为 %dx%d；按平台要求，`target/original.mp4` 上下各加 %d 像素黑边成为 %dx%d（画面未拉伸、未裁切，"
            "帧数与音轨不变，与白模的黑边位置一致），因此重新编码过一次。" % (W, H, (CH - H) // 2, CW, CH))
    with open(os.path.join(sub_dir, "meta", "README.md"), "a", encoding="utf-8") as fh:
        fh.write(chr(10) + "## 原片尺寸说明" + chr(10) + chr(10) + "- " + note + chr(10))
    return out


def finish(sub_dir, P, zp, out, checker, clip, bad):
    os.makedirs(out, exist_ok=True)
    final = os.path.join(out, P + ".zip")
    shutil.copyfile(zp, final)
    cmp_ = os.path.join(out, P + ".comparison.mp4")
    with zipfile.ZipFile(final) as z, z.open("comparison/comparison.mp4") as s_, open(cmp_, "wb") as d:
        shutil.copyfileobj(s_, d, 1 << 20)
    p = subprocess.run([sys.executable, checker, final, cmp_], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    open(os.path.join(out, P + ".check.txt"), "w", encoding="utf-8").write(p.stdout + p.stderr)
    os.remove(cmp_)
    flagged = "分辨率不符" in p.stdout
    print("%-24s %-20s %s%s" % (clip, P, "check ok" if p.returncode == 0 else "CHECK FAILED", "  (size note!)" if flagged else ""), flush=True)
    if p.returncode != 0 or flagged:
        bad.append("%s: check_package %s (see %s.check.txt)" % (clip, "failed" if p.returncode else "size note", P))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clips", default=""); ap.add_argument("--out", default=os.path.join(ROOT, "DELIVERY", "portal"))
    ap.add_argument("--retarget", action="store_true", help="only pad target/original.mp4 of already staged packages, re-zip, re-check")
    a = ap.parse_args()
    c = dict(cfg())
    todo = [x for x in clips() if not a.clips or x["clip"] in a.clips.split(",")]
    names_p = os.path.join(ROOT, "data", "package_names.json")
    names = json.load(open(names_p, encoding="utf-8"))
    for x in clips():
        names[x["clip"]] = x["portal"]
    json.dump(names, open(names_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    stage = os.path.join(a.out, "_stage"); os.makedirs(stage, exist_ok=True)
    checker = os.path.join(ROOT, "tools", "v21", "check_package.py")
    bad = []
    if a.retarget:
        for x in todo:
            clip, P = x["clip"], x["portal"]
            sub_dir = os.path.join(stage, clip)
            if not os.path.isdir(sub_dir):
                continue
            shutil.copyfile(x["video"], os.path.join(sub_dir, "target", "original.mp4"))
            ci = jload(os.path.join(c["work_root"], clip, "clip_info.json"))
            pad_target(sub_dir, x["video"], ci, c)
            zp = os.path.join(stage, clip + ".zip")
            pkg.zip_v21(sub_dir, P, zp)
            finish(sub_dir, P, zp, a.out, checker, clip, bad)
        print("retarget done, %d problems" % len(bad))
        for b in bad:
            print("  " + b)
        return
    for x in todo:
        clip, P = x["clip"], x["portal"]
        work = os.path.join(c["work_root"], clip)
        rdir = os.path.join(work, "render")
        ci = jload(os.path.join(work, "clip_info.json"))
        nfr = len([f for f in os.listdir(rdir) if f.lower().endswith(".jpg")])
        if nfr != ci["frames"]:
            bad.append("%s: render has %d frames, clip %d" % (clip, nfr, ci["frames"])); continue
        src_md = os.path.join(LZ, clip, "model_3d")
        if not os.path.isdir(src_md):                                        # the approved trial (dragon-22)
            src_md = os.path.join(ROOT, "DELIVERY", "lzhd_trial", clip, "model_3d")
        blend = os.path.join(src_md, x["lzhd"] + ".blend")
        if not os.path.exists(blend):
            bad.append("%s: no LZHD blend %s" % (clip, blend)); continue
        lz_zip = os.path.join(LZ, x["lzhd"] + ".zip")
        newest = max(os.path.getmtime(os.path.join(rdir, f)) for f in os.listdir(rdir))
        if os.path.getmtime(blend) + 5 < newest - 600:                       # blend much older than the frames
            bad.append("%s: blend older than its render" % clip); continue
        sub = os.path.join(stage, clip); md = os.path.join(sub, "model_3d")
        shutil.rmtree(sub, ignore_errors=True); os.makedirs(os.path.join(md, "reconstruction"))
        shutil.copyfile(blend, os.path.join(md, P + ".blend"))
        sj = open(os.path.join(src_md, "reconstruction", "scene.json"), encoding="utf-8").read()
        sj = sj.replace("model_3d/%s.blend" % x["lzhd"], "model_3d/%s.blend" % P).replace('"%s.blend"' % x["lzhd"], '"%s.blend"' % P)
        open(os.path.join(md, "reconstruction", "scene.json"), "w", encoding="utf-8").write(sj)
        desc_p = ci.get("desc_path") or ""
        desc = open(desc_p, encoding="utf-8").read().strip() if desc_p and os.path.exists(desc_p) else ""
        sub_dir, zp = pkg.package(clip, x["video"], rdir, stage, ci, c, desc)
        # meta/meta.json exactly as the portal's template (same keys, same order, this clip's real values);
        # the extra details package() wrote go into meta/README.md
        mp = os.path.join(sub_dir, "meta", "meta.json")
        full = json.load(open(mp, encoding="utf-8"))
        meta = {"id": P, "blender_version": full["blender_version"], "source": "target/original.mp4",
                "white_model": "white_model/blendervid.mp4", "comparison": "comparison/comparison.mp4",
                "model_3d": "model_3d/%s.blend" % P, "scene_json_path": "model_3d/reconstruction/scene.json",
                "frame_start": 1, "frame_end": int(full["frames"]), "frames": int(full["frames"]),
                "fps": int(full["fps"]), "fps_base": 1, "duration": round(full["frames"] / float(full["fps"]), 6)}
        json.dump(meta, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        rp = os.path.join(sub_dir, "meta", "README.md")
        extra = {k: v for k, v in full.items() if k not in meta}
        with open(rp, "a", encoding="utf-8") as fh:
            fh.write("\n## 附加信息（原 meta 扩展字段）\n\n```json\n%s\n```\n" % json.dumps(extra, ensure_ascii=False, indent=2))
        pad_target(sub_dir, x["video"], ci, c)
        pkg.zip_v21(sub_dir, P, zp)
        finish(sub_dir, P, zp, a.out, checker, clip, bad)
    print("%d done, %d problems" % (len(todo) - len(bad), len(bad)))
    for b in bad:
        print("  " + b)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Rendered frames -> a submission package.  package() follows the v2.1 rules (2026-11-12, see its docstring);
package_v1() below is the earlier layout of the client's sample (11月2日试稿):

  <name>/comparison/<name>.mp4     target on top, white model below (3840 x 2*H)
  <name>/meta/meta.jsonl           same keys, same order as the sample
  <name>/model_3d/<name>blend.blend
  <name>/target/<name>.mp4         the source clip, unchanged
  <name>/white_model/<name>.mp4    3840 wide, the source's aspect ratio, 24 fps
  <name>.zip                       one top folder, folder entries for the five sub-folders, GBK names (as the sample)

Video: H.264 High, yuv420p, Rec.709 (tagged), 24 fps, ~30 Mbit/s constant; audio AAC stereo 48 kHz 320 kbit/s.
Encoder: h264_nvenc (GPU) when it works and the frame fits NVENC (<= 4096 per side: the white model), else libx264
(always for the ~3840x5600 comparison). Config "encoder" ("auto" | "libx264") and optional "nvenc_ffmpeg" (see nvenc_ffmpeg)."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from math import gcd

from common import ffmpeg_exe

SUBDIRS = ("comparison", "meta", "model_3d", "target", "white_model")
META_KEYS = ["source_video_id", "source_video_name", "source_video_desc", "source_video_duration", "source_video_resolution",
             "source_video_fps", "source_video_format", "source_video_video_path", "source_video_origin", "pattern_video_id",
             "pattern_video_name", "pattern_video_desc", "pattern_video_duration", "pattern_video_resolution", "pattern_video_fps",
             "pattern_video_format", "pattern_video_video_path", "pattern_video_origin", "comparison_video_path", "production_type",
             "scenario", "video_type", "ext_json"]


def probe(ff, path):
    """duration (s), width, height, fps of the first video stream"""
    p = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    out = {}
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", p.stderr)
    if m:
        out["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5}).*?([\d.]+) fps", p.stderr)
    if m:
        out.update(width=int(m.group(1)), height=int(m.group(2)), fps=float(m.group(3)))
    out["has_audio"] = "Audio:" in p.stderr
    return out


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + p.stderr[-2000:])


def out_size(W, Hfull, out_w):
    OW = int(out_w)
    OH = int(round(out_w * Hfull / float(W)))
    return OW - OW % 2, OH - OH % 2


COLOUR_ARGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
AUDIO_ARGS = ["-c:a", "copy", "-movflags", "+faststart"]   # v2.1: the source's own audio stream, identical in all three videos
COMMON_TAIL = COLOUR_ARGS + AUDIO_ARGS
NVENC_MAX = 4096          # NVENC H.264 maximum width and height (RTX 40); taller frames (the comparison) go to libx264


def encode_args(c, encoder="libx264"):
    br = str(c.get("video_bitrate", "30M"))
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-profile:v", "high", "-pix_fmt", "yuv420p",
                "-rc", "cbr", "-b:v", br, "-maxrate", br, "-bufsize", br, "-rc-lookahead", "20", "-spatial-aq", "1",
                "-bf", "3", "-g", "250"] + COMMON_TAIL
    return ["-c:v", "libx264", "-preset", str(c.get("x264_preset", "faster")), "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-b:v", br, "-minrate", br, "-maxrate", br, "-bufsize", br,
            "-threads", "8", "-x264-params",
            "nal-hrd=cbr:force-cfr=1:rc-lookahead=20:colorprim=bt709:transfer=bt709:colormatrix=bt709",   # lean memory at 3840x4320
            ] + COMMON_TAIL


_NVENC_OK = {}


def nvenc_works(ff):
    """True when this ffmpeg can open h264_nvenc on this machine (driver new enough, GPU present); cached per exe"""
    if ff not in _NVENC_OK:
        try:
            p = subprocess.run([ff, "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:r=24", "-t", "0.25",
                                "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=60)
            _NVENC_OK[ff] = p.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _NVENC_OK[ff] = False
    return _NVENC_OK[ff]


def nvenc_ffmpeg(c, ff, w, h):
    """the ffmpeg to encode a w x h video with NVENC, or None -> libx264.
    config "encoder": "auto" (default: NVENC when it works), "nvenc" (same, NVENC preferred) or "libx264" (CPU only);
    "nvenc_ffmpeg": optional path of another ffmpeg.exe to use for NVENC when the bundled one cannot
    (the bundled imageio ffmpeg needs a newer NVIDIA driver than an older ffmpeg build does)."""
    if str(c.get("encoder", "auto")).lower() in ("libx264", "cpu", "x264") or max(w, h) > NVENC_MAX:
        return None
    for exe in (ff, c.get("nvenc_ffmpeg") or ""):
        if exe and os.path.isfile(exe) and nvenc_works(exe):
            return exe
    return None


def pipe_encode(ff_src, ff_enc, src_args, enc_args):
    """ff_src <src_args> | ff_enc <enc_args>; raises RuntimeError like run()"""
    with tempfile.TemporaryFile() as err:
        p1 = subprocess.Popen([ff_src] + src_args, stdout=subprocess.PIPE, stderr=err)
        try:
            p2 = subprocess.run([ff_enc] + enc_args, stdin=p1.stdout, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        finally:
            p1.stdout.close()
            p1.wait()
        if p2.returncode != 0 or p1.returncode != 0:
            err.seek(0)
            raise RuntimeError("ffmpeg failed:\n" + (err.read().decode("utf-8", "ignore") + p2.stderr)[-2000:])


def encode(ff, c, w, h, inp, filt, audio_src, D, out):
    """filter graph -> [v] -> out (+ audio of input 1, cut to D seconds): NVENC when available, libx264 otherwise
    or if NVENC fails. With a separate NVENC ffmpeg the video is encoded there (video only) and the bundled ffmpeg adds
    the audio (AAC as in the libx264 path) and +faststart, so that ffmpeg needs no audio codecs. -> encoder used"""
    head = ["-y", "-v", "error"] + inp + ["-filter_complex", filt, "-map", "[v]"]
    t = ["-t", "%.6f" % D]
    exe = nvenc_ffmpeg(c, ff, w, h)
    if exe:
        tmp = out[:-4] + ".nvenc.mp4"
        try:
            if exe == ff:
                run([ff] + head + ["-map", "1:a?"] + t + encode_args(c, "h264_nvenc") + [out])
            else:      # bundled ffmpeg decodes + filters (faster than an old build) -> raw yuv420p pipe -> NVENC ffmpeg
                pipe_encode(ff, exe, head + t + ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                            ["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "%dx%d" % (w, h),
                             "-r", str(int(c.get("out_fps", 24))), "-i", "-"] + encode_args(c, "h264_nvenc")[:-len(AUDIO_ARGS)] + [tmp])
                run([ff, "-y", "-v", "error", "-i", tmp, "-i", audio_src, "-map", "0:v", "-map", "1:a?", "-c:v", "copy"]
                    + t + COMMON_TAIL + [out])
            return "h264_nvenc"
        except RuntimeError as e:
            print("  NVENC failed, encoding with libx264 instead: %s" % str(e).strip().splitlines()[-1][:300])
            if os.path.exists(out):
                os.remove(out)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    run([ff] + head + ["-map", "1:a?"] + t + encode_args(c) + [out])
    return "libx264"


class GBKInfo(zipfile.ZipInfo):
    """file names stored as GBK bytes with the UTF-8 flag off, like zips made on Chinese Windows (the sample);
    names GBK cannot hold fall back to UTF-8 with the flag on"""
    def _encodeFilenameFlags(self):
        try:
            return self.filename.encode("gbk"), self.flag_bits & ~0x800
        except UnicodeEncodeError:
            return self.filename.encode("utf-8"), self.flag_bits | 0x800


def zip_gbk(src_dir, name, zp):
    """zip like the sample: <name>/<sub>/ folder entries + files, deflate, names in GBK when possible"""
    tmp = zp + ".part"

    def info(arc, is_dir=False):
        zi = GBKInfo(arc)
        zi.compress_type = zipfile.ZIP_STORED if is_dir else zipfile.ZIP_DEFLATED
        if is_dir:
            zi.external_attr = 0o40775 << 16 | 0x10
        return zi

    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for d in SUBDIRS:
            z.writestr(info("%s/%s/" % (name, d), True), b"")
            for fn in sorted(os.listdir(os.path.join(src_dir, d))):
                if fn.endswith((".blend1", ".tmp")):
                    continue
                zi = info("%s/%s/%s" % (name, d, fn))
                with open(os.path.join(src_dir, d, fn), "rb") as fh, z.open(zi, "w", force_zip64=True) as out:
                    shutil.copyfileobj(fh, out, 1 << 20)
    os.replace(tmp, zp)


def pulldown_drops(video, N, fps, ofps):
    """frames to drop so a 30 fps source made from 24 fps film (every 5th frame repeated) plays back as its original
    24 frames: the repeated frames, detected from the picture. [] when the source has no such pattern (then the
    usual fps conversion is used). Dropping the same frames from the film and the render keeps them frame-matched."""
    if abs(fps * ofps * 5 / 4.0 - fps * fps) > fps * 1.5 or N < 20:          # only 30 fps (29.97) -> 24
        return []
    try:
        import cv2, numpy as np
        cap = cv2.VideoCapture(video); prev = None; d = []
        for _ in range(N):
            ok, f = cap.read()
            if not ok:
                break
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120)).astype(np.float32)
            d.append(float(np.abs(g - prev).mean()) if prev is not None else 1e9)
            prev = g
        cap.release()
    except Exception:
        return []
    d = np.array(d); med = float(np.median(d[1:])) if len(d) > 1 else 0.0
    if med <= 0:
        return []
    dup = [i for i in range(1, len(d)) if d[i] < 0.35 * med]
    if len(dup) < 3:
        return []
    phase = max(range(5), key=lambda p: sum(1 for i in dup if i % 5 == p))
    hits = [i for i in dup if i % 5 == phase]
    if len(hits) < 0.6 * (len(d) // 5) or len(hits) < 0.8 * len(dup):    # not a steady 1-in-5 repeat
        return []
    return hits


PKG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*-[0-9]+$")
V21_FILES = ("comparison/comparison.mp4", "target/original.mp4", "white_model/blendervid.mp4",
             "model_3d/{p}.blend", "model_3d/reconstruction/scene.json", "meta/meta.json")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def package_name(clip, c=None):
    """v2.1 permanent package name for a clip: data/package_names.json {clip name: "english_title-N"}"""
    p = os.path.join(ROOT, "data", "package_names.json")
    try:
        with open(p, encoding="utf-8") as fh:
            names = json.load(fh)
    except (OSError, ValueError):
        names = {}
    return names.get(clip)


def canvas(W, H):
    """v2.1 1080P canvas: 1920x1080, or 1080x1920 for a vertical source"""
    return (1080, 1920) if H > W else (1920, 1080)


def fit_even(w, h, bw, bh):
    """w x h scaled to fit inside bw x bh, even sizes"""
    f = min(bw / float(w), bh / float(h))
    a, b = int(round(w * f)), int(round(h * f))
    return a - a % 2, b - b % 2


def package(name, video, render_dir, submit_root, ci, c, desc=""):
    """v2.1 (2026-11-12): one clip = one package, five folders at the zip root, six fixed files:
      comparison/comparison.mp4      original on top, white model below, both fitted into one 1080P frame
      target/original.mp4            the source clip, byte for byte
      white_model/blendervid.mp4     1920x1080 (1080x1920 vertical), the source framing letterboxed
      model_3d/<package>.blend       master project (+ reconstruction/scene.json, reconstruct.py, README.md)
      meta/meta.json                 id, paths, blender version, frames, fps, duration (+ meta/README.md)
    The white model and the comparison carry the source's own audio stream, copied (identical in all three).
    Staged in <submit_root>/<clip>/, zipped to <submit_root>/<clip>.zip; deliver.py names it <package>.zip."""
    pkg = package_name(name, c)
    if not pkg or not PKG_RE.match(pkg):
        raise RuntimeError("no v2.1 package name for %s in data/package_names.json (got %r)" % (name, pkg))
    ff = ffmpeg_exe()
    sub = os.path.join(submit_root, name)
    if os.path.isdir(sub):                                   # drop files of the old layout; keep model_3d (the .blend just saved)
        for d in ("comparison", "meta", "target", "white_model"):
            shutil.rmtree(os.path.join(sub, d), ignore_errors=True)
    for d in SUBDIRS:
        os.makedirs(os.path.join(sub, d), exist_ok=True)
    tgt = os.path.join(sub, "target", "original.mp4")
    shutil.copyfile(video, tgt)
    W, Hfull, y0, Hp, N, fps = ci["W"], ci["Hfull"], ci["y0"], ci["Hpic"], ci["frames"], ci["fps"]
    CW, CH = canvas(W, Hfull)
    FW, FH = fit_even(W, Hfull, CW, CH)                      # the source frame inside the canvas
    f = FH / float(Hfull)
    PH = int(round(Hp * f)); PH -= PH % 2; PY = int(round(y0 * f)); PY -= PY % 2
    ofps = int(c.get("out_fps", 24))
    D = N / float(fps)
    frames = os.path.join(render_dir, "wm_%04d.jpg")
    if not os.path.exists(frames % 1):
        frames = os.path.join(render_dir, "wm_%04d.png")
    inp = ["-framerate", "%.6f" % fps, "-start_number", "1", "-i", frames, "-i", video]
    drops = pulldown_drops(video, N, fps, ofps)
    if drops:            # repeated film frames: drop exactly those from both streams -> smooth, frame-matched 24 fps
        retime = "settb=1/%d,select='not(%s)',setpts=N/(%d*TB),fps=%d," % (
            ofps * 1000, "+".join("eq(n,%d)" % i for i in drops), ofps, ofps)     # fine time base: exact 1/24 s steps
        print("  pulldown: dropping %d repeated frames (%s ...)" % (len(drops), ", ".join(map(str, drops[:6]))))
    else:
        retime = "fps=%d," % ofps
    tail = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setsar=1"
    # white model in the source's full frame (FW x FH, picture area where the source has it), then onto the canvas
    WMF = "[0:v]%sscale=%d:%d:flags=lanczos,pad=%d:%d:0:%d:black" % (retime, FW, PH, FW, FH, PY)
    wm = os.path.join(sub, "white_model", "blendervid.mp4")
    cmp_ = os.path.join(sub, "comparison", "comparison.mp4")
    encode(ff, c, CW, CH, inp, WMF + ",pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black,%s[v]" % (CW, CH, tail), video, D, wm)
    # comparison: original on top, white model below, each fitted into half of the same canvas
    if CW >= CH:
        hw, hh = fit_even(W, Hfull, CW, CH // 2); stack = "vstack"
    else:
        hw, hh = fit_even(W, Hfull, CW, CH // 2); stack = "vstack"
    encode(ff, c, CW, CH, inp,
           WMF + ",scale=%d:%d:flags=lanczos[b];[1:v]%sscale=%d:%d:flags=lanczos[a];[a][b]%s=inputs=2,"
           "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black,%s[v]" % (hw, hh, retime, hw, hh, stack, CW, CH, tail), video, D, cmp_)
    # the master project and its reconstruction data (written by blender_build.py)
    md = os.path.join(sub, "model_3d")
    blend = os.path.join(md, pkg + ".blend")
    if not os.path.exists(blend):
        raise RuntimeError("missing %s (render again)" % blend)
    for fn in os.listdir(md):
        if fn not in (pkg + ".blend", "reconstruction"):
            p = os.path.join(md, fn)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    rec = os.path.join(md, "reconstruction")
    if not os.path.exists(os.path.join(rec, "scene.json")):
        raise RuntimeError("missing model_3d/reconstruction/scene.json")
    for fn in ("reconstruct.py", "README.md"):
        shutil.copyfile(os.path.join(ROOT, "assets", "reconstruction", fn), os.path.join(rec, fn))
    pw = probe(ff, wm)
    nout = count_frames(ff, wm)
    bver = blender_version(render_dir)
    meta = {"id": pkg, "paths_base": ".", "source": "target/original.mp4", "white_model": "white_model/blendervid.mp4",
            "comparison": "comparison/comparison.mp4", "model_3d": "model_3d/%s.blend" % pkg,
            "scene_json_path": "model_3d/reconstruction/scene.json", "blender_version": bver,
            "frame_start": 1, "frame_end": nout, "frames": nout, "fps": ofps, "fps_base": 1.0,
            "duration": round(nout / float(ofps), 6), "duration_basis": "video stream (frames / fps)",
            "project_frames": {"frame_start": 1, "frame_end": N, "fps": int(round(fps)), "fps_base": 1.0,
                               "note": "the .blend timeline follows the source clip frame for frame (%d frames at %g fps)" % (N, fps)},
            "source_video": {"resolution": [W, Hfull], "fps": round(fps, 3), "frames": N, "duration": round(D, 6)},
            "output_video": {"resolution": [CW, CH], "fps": ofps, "frames": nout,
                             "retime": ("3:2 pulldown removed: the %d repeated frames of the 30 fps source dropped from both the "
                                        "original and the white model (24 fps film frames, same duration, no speed change)" % len(drops))
                                       if drops else "none",
                             "audio": "source audio stream copied unchanged into white_model and comparison" if pt_audio(ff, video) else "source has no audio"},
            "checks": {"structure": "check_package.py", "reconstruction": "未验证", "visual_review": "未验证"}}
    with open(os.path.join(sub, "meta", "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(sub, "meta", "README.md"), "w", encoding="utf-8") as fh:
        fh.write(meta_readme(pkg, meta, desc, drops, W, Hfull, CW, CH))
    zp = os.path.join(submit_root, name + ".zip")
    zip_v21(sub, pkg, zp)
    return sub, zp


def pt_audio(ff, path):
    return probe(ff, path).get("has_audio", False)


def count_frames(ff, path):
    """decoded video frames (decodes the whole file, so it also proves it plays to the end)"""
    p = subprocess.run([ff, "-v", "error", "-i", path, "-map", "0:v:0", "-f", "null", "-"], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore")
    q = subprocess.run([ff, "-hide_banner", "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"], capture_output=True,
                       text=True, encoding="utf-8", errors="ignore")
    m = re.findall(r"frame=\s*(\d+)", q.stderr)
    if p.returncode != 0 or p.stderr.strip():
        raise RuntimeError("video does not decode cleanly: %s\n%s" % (path, p.stderr[-800:]))
    return int(m[-1]) if m else 0


def blender_version(render_dir):
    try:
        with open(os.path.join(os.path.dirname(render_dir), "render_done.txt"), encoding="utf-8") as fh:
            done_lines = fh.read().splitlines()
        for line in done_lines:
            if line.startswith("blender "):
                return line.split(" ", 1)[1].strip()
    except OSError:
        pass
    return "未记录"


def meta_readme(pkg, meta, desc, drops, W, H, CW, CH):
    ov = meta["output_video"]
    return "\n".join([
        "# %s — 工程说明" % pkg, "",
        "- 主工程：`model_3d/%s.blend`（Blender %s 保存）。打开即可，活动相机为 `Camera`。" % (pkg, meta["blender_version"]),
        "- 场景数据：`model_3d/reconstruction/scene.json`，读取 / 重建入口 `reconstruct.py`，说明见同目录 `README.md`。",
        "- 依赖：无额外依赖（无插件、无外部贴图、无外链文件；所有网格与材质都在工程内）。",
        "",
        "## 工程内容", "",
        "- 集合 `Set_背景`：灰白场景（地面、墙、道具等简单形体）；多镜头时每个镜头的场景只在该镜头帧段内显示（hide_render 关键帧）。",
        "- 角色：官方 `human_master.blend` 的骨架与 10 块网格（每人一次性均匀缩放），头部与胸部识别标识面为白色（材质 `WM_marker`），"
        "颜色：主角红、第二蓝、第三绿、第四黄、第五起青。",
        "- 角色控制：`P<n>_root`（位置 + 朝向关键帧）→ `P<n>_pelvis`（高度关键帧）→ 各关节空物体（XYZ 欧拉旋转关键帧，每帧一键）；"
        "网格块挂在对应关节下，改关节旋转即可继续编辑动作。",
        "- 相机：`Camera`，每帧位置 / 旋转关键帧（随原片运镜）；焦距与传感器见 scene.json `camera`。",
        "- 帧段：工程 1–%d 帧，%d fps，与原片逐帧对应。" % (meta["project_frames"]["frame_end"], meta["project_frames"]["fps"]),
        "",
        "## 视频", "",
        "- `target/original.mp4`：原片，未改动（%dx%d）。%s" % (W, H, "" if (W, H) in ((1920, 1080), (1080, 1920)) else
                                                         "原片尺寸不是 1920x1080，按规范不拉伸、不重编码，原样提交。"),
        "- `white_model/blendervid.mp4`：%dx%d，%d fps，%d 帧；原片画面按原比例放入 1080P 画布，多余部分为黑边。" % (CW, CH, ov["fps"], ov["frames"]),
        "- `comparison/comparison.mp4`：%dx%d，上为原片、下为白模，同一时间基准，按原比例放入同一 1080P 画布。" % (CW, CH),
        "- 帧率：%s" % ov["retime"],
        "- 音轨：%s。" % ov["audio"],
        "",
        "## 核对状态", "",
        "- 结构：check_package.py 检查（见交付记录）。",
        "- 重建：未验证（每包未单独运行 reconstruct.py；入口已在同类包上试运行）。",
        "- 视觉审核：未验证。",
        "" if not desc else "\n## 镜头说明\n\n" + desc + "\n"])


def zip_v21(src_dir, pkg, zp):
    """v2.1 zip: the five folders directly at the root, '/' paths, UTF-8 names, deflate, no wrapper folder"""
    tmp = zp + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for d in SUBDIRS:
            base = os.path.join(src_dir, d)
            for dirpath, dirs, files in os.walk(base):
                dirs.sort()
                for fn in sorted(files):
                    if fn.endswith((".blend1", ".tmp", ".part")) or fn in ("Thumbs.db", "desktop.ini", ".DS_Store"):
                        continue
                    full = os.path.join(dirpath, fn)
                    arc = os.path.relpath(full, src_dir).replace(os.sep, "/")
                    z.write(full, arc, compress_type=zipfile.ZIP_STORED if fn.endswith(".mp4") else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(tmp) as z:
        names = set(z.namelist())
    need = [f.format(p=pkg) for f in V21_FILES]
    miss = [n for n in need if n not in names]
    if miss:
        os.remove(tmp)
        raise RuntimeError("package incomplete, missing " + ", ".join(miss))
    os.replace(tmp, zp)


def package_v1(name, video, render_dir, submit_root, ci, c, desc=""):
    """the earlier (v1.0 sample) layout, kept for reference"""
    ff = ffmpeg_exe()
    sub = os.path.join(submit_root, name)
    for d in SUBDIRS:
        os.makedirs(os.path.join(sub, d), exist_ok=True)
    tgt = os.path.join(sub, "target", name + ".mp4")
    shutil.copyfile(video, tgt)
    W, Hfull, y0, Hp, N, fps = ci["W"], ci["Hfull"], ci["y0"], ci["Hpic"], ci["frames"], ci["fps"]
    OW, OH = out_size(W, Hfull, c.get("out_width", 3840))
    sy = OH / float(Hfull)                                           # picture area inside the full frame (letterbox)
    PH = int(round(Hp * sy)); PH -= PH % 2; PY = int(round(y0 * sy)); PY -= PY % 2
    ofps = int(c.get("out_fps", 24))
    D = N / float(fps)
    frames = os.path.join(render_dir, "wm_%04d.jpg")
    if not os.path.exists(frames % 1):
        frames = os.path.join(render_dir, "wm_%04d.png")
    inp = ["-framerate", "%.6f" % fps, "-start_number", "1", "-i", frames, "-i", video]
    drops = pulldown_drops(video, N, fps, ofps)
    if drops:            # repeated film frames: drop exactly those from both streams -> smooth, frame-matched 24 fps
        retime = "settb=1/%d,select='not(%s)',setpts=N/(%d*TB),fps=%d," % (
            ofps * 1000, "+".join("eq(n,%d)" % i for i in drops), ofps, ofps)     # fine time base: exact 1/24 s steps
        print("  pulldown: dropping %d repeated frames (%s ...)" % (len(drops), ", ".join(map(str, drops[:6]))))
    else:
        retime = "fps=%d," % ofps
    WM = ("[0:v]%sscale=%d:%d:flags=lanczos,pad=%d:%d:0:%d:black,scale=out_color_matrix=bt709:out_range=tv,"
          "format=yuv420p,setsar=1" % (retime, OW, PH, OW, OH, PY))
    wm = os.path.join(sub, "white_model", name + ".mp4")
    cmp_ = os.path.join(sub, "comparison", name + ".mp4")
    encode(ff, c, OW, OH, inp, WM + "[v]", video, D, wm)
    encode(ff, c, OW, 2 * OH, inp, WM + "[b];[1:v]%sscale=%d:%d:flags=lanczos,scale=out_color_matrix=bt709:out_range=tv,"
           "format=yuv420p,setsar=1[a];[a][b]vstack=inputs=2[v]" % (retime, OW, OH), video, D, cmp_)   # 3840 x ~5600: libx264 (> NVENC_MAX)
    # the .blend: model_3d/<name>blend.blend as in the sample
    md = os.path.join(sub, "model_3d")
    plain = os.path.join(md, name + ".blend")
    final = os.path.join(md, name + "blend.blend")
    if os.path.exists(plain):
        os.replace(plain, final)
    for fn in os.listdir(md):
        if fn != os.path.basename(final):
            os.remove(os.path.join(md, fn))
    pw, pt = probe(ff, wm), probe(ff, tgt)
    g = gcd(OW, 2 * OH)
    vals = {"source_video_id": name + ".mp4", "source_video_name": name, "source_video_desc": desc,
            "source_video_duration": round(pw.get("duration", D), 3), "source_video_resolution": [OW, OH],
            "source_video_fps": ofps, "source_video_format": "mp4", "source_video_video_path": "white_model/%s.mp4" % name,
            "source_video_origin": "", "pattern_video_id": name + ".mp4", "pattern_video_name": name, "pattern_video_desc": desc,
            "pattern_video_duration": round(pt.get("duration", D), 3), "pattern_video_resolution": [W, Hfull],
            "pattern_video_fps": int(round(fps)), "pattern_video_format": "mp4", "pattern_video_video_path": "target/%s.mp4" % name,
            "pattern_video_origin": "", "comparison_video_path": "comparison/%s.mp4" % name,
            "production_type": c.get("production_type", "manual_rebuild"), "scenario": c.get("scenario", "film"),
            "video_type": c.get("video_type", "人物"),
            "ext_json": {"code_type": "H.264", "aspect_ratio": "%d:%d" % (OW // g, 2 * OH // g),
                         "model_3d_path": "model_3d/%s" % os.path.basename(final)}}
    meta = {k: vals[k] for k in META_KEYS}
    with open(os.path.join(sub, "meta", "meta.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
    zp = os.path.join(submit_root, name + ".zip")
    zip_gbk(sub, name, zp)
    return sub, zp


# ---------------------------------------------------------------------------------------------- LZHD V1.0 (2026-11-15)
def source_video_info(ff, path):
    """width, height, fps (rational string), video bitrate (bit/s), frames, duration, audio? of a video"""
    fp = os.path.join(os.path.dirname(ff), "ffprobe.exe")
    exe = fp if os.path.exists(fp) else (shutil.which("ffprobe") or "ffprobe")
    p = subprocess.run([exe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path], capture_output=True,
                       text=True, encoding="utf-8", errors="ignore")
    d = json.loads(p.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    a = [s for s in d["streams"] if s["codec_type"] == "audio"]
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    br = int(v.get("bit_rate") or 0) or int(float(d["format"].get("bit_rate", 0)) * 0.95)
    return {"width": int(v["width"]), "height": int(v["height"]), "fps_rational": "%d/%d" % (num, den), "fps": num / float(den),
            "video_bitrate": br, "frames": int(v.get("nb_frames") or 0), "duration": float(d["format"]["duration"]),
            "codec": v.get("codec_name"), "has_audio": bool(a), "audio_codec": a[0]["codec_name"] if a else None}


def lzhd_flags(si):
    """qualification flags of the spec (section 7): recorded, never 'fixed' by resampling"""
    f = []
    if min(si["width"], si["height"]) < 1080:
        f.append("原片不足1080P（%dx%d）" % (si["width"], si["height"]))
    if si["fps"] < 24 - 1e-3:
        f.append("原片低于24fps（%.3f）" % si["fps"])
    if si["duration"] < 5.0:
        f.append("原片短于5秒（%.3fs）" % si["duration"])
    if si["video_bitrate"] < 18000000:
        f.append("原片视频码率低于18Mbps（%.2f Mbps）" % (si["video_bitrate"] / 1e6))
    return f


def package_lzhd(name, pkg, video, render_dir, submit_root, ci, c, desc="", notes=None):
    """LZHD 白模视频复刻需求及验收标准 V1.0 (2026-11-15):
      white_model/blendervid.mp4   same width, height, frame rate and video bit rate as the source; the source's audio
      comparison/comparison.mp4    original on top, white model below: W x 2H, same fps, audio copied
      target/original.mp4          the source clip, byte for byte
      model_3d/<pkg>.blend         editable project (official human_master rig, Eevee, active camera, timeline 1..N)
      model_3d/reconstruction/     scene.json (+ reconstruct.py, README.md)
      meta/meta.json + README.md   specs, qualification flags, honest check record (未验证 where not done)
    No resampling, padding, cropping or speed change: frame N of the source = frame N of the white model."""
    ff = ffmpeg_exe()
    si = source_video_info(ff, video)
    sub = os.path.join(submit_root, name)
    if os.path.isdir(sub):
        for d in ("comparison", "meta", "target", "white_model"):
            shutil.rmtree(os.path.join(sub, d), ignore_errors=True)
    for d in SUBDIRS:
        os.makedirs(os.path.join(sub, d), exist_ok=True)
    shutil.copyfile(video, os.path.join(sub, "target", "original.mp4"))
    W, H, y0, Hp, N = si["width"], si["height"], ci["y0"], ci["Hpic"], ci["frames"]
    fr = si["fps_rational"]; D = N / si["fps"]
    frames = os.path.join(render_dir, "wm_%04d.jpg")
    if not os.path.exists(frames % 1):
        frames = os.path.join(render_dir, "wm_%04d.png")
    inp = ["-framerate", fr, "-start_number", "1", "-i", frames, "-i", video]
    tail = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setsar=1"
    # the render is the picture area at exactly W x Hp; where the source has black bars the white model has them too
    WMF = "[0:v]scale=%d:%d:flags=lanczos,pad=%d:%d:0:%d:black" % (W, Hp, W, H, y0)
    br = si["video_bitrate"]
    cw = dict(c, video_bitrate="%dk" % int(round(br / 1000.0)), out_fps=int(round(si["fps"])))
    cc = dict(c, video_bitrate="%dk" % int(round(2 * br / 1000.0)), out_fps=int(round(si["fps"])))
    wm = os.path.join(sub, "white_model", "blendervid.mp4")
    cmp_ = os.path.join(sub, "comparison", "comparison.mp4")
    encode(ff, cw, W, H, inp, WMF + "," + tail + "[v]", video, D, wm)
    encode(ff, cc, W, 2 * H, inp, WMF + "[b];[1:v]setpts=PTS-STARTPTS[a];[a][b]vstack=inputs=2," + tail + "[v]", video, D, cmp_)
    md = os.path.join(sub, "model_3d")
    blend = os.path.join(md, pkg + ".blend")
    if not os.path.exists(blend):
        raise RuntimeError("missing %s (render again)" % blend)
    for fn in os.listdir(md):
        if fn not in (pkg + ".blend", "reconstruction"):
            p = os.path.join(md, fn)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    rec = os.path.join(md, "reconstruction")
    if not os.path.exists(os.path.join(rec, "scene.json")):
        raise RuntimeError("missing model_3d/reconstruction/scene.json")
    for fn in ("reconstruct.py", "README.md"):
        shutil.copyfile(os.path.join(ROOT, "assets", "reconstruction", fn), os.path.join(rec, fn))
    wi, ci2 = source_video_info(ff, wm), source_video_info(ff, cmp_)
    nw, nc = count_frames(ff, wm), count_frames(ff, cmp_)
    spec_ok = {"white_model_size": [wi["width"], wi["height"]] == [W, H], "white_model_fps": wi["fps_rational"] == fr,
               "white_model_frames": nw == N, "comparison_size": [ci2["width"], ci2["height"]] == [W, 2 * H],
               "comparison_fps": ci2["fps_rational"] == fr, "comparison_frames": nc == N,
               "white_model_bitrate_ratio": round(wi["video_bitrate"] / float(br), 3)}
    ok = all(v for k, v in spec_ok.items() if k != "white_model_bitrate_ratio")
    meta = {"id": pkg, "clip": name, "spec": "LZHD 白模视频复刻需求及验收标准 V1.0 (2026-11-15)",
            "paths_base": ".", "source": "target/original.mp4", "white_model": "white_model/blendervid.mp4",
            "comparison": "comparison/comparison.mp4", "model_3d": "model_3d/%s.blend" % pkg,
            "scene_json_path": "model_3d/reconstruction/scene.json", "blender_version": blender_version(render_dir),
            "render_engine": "Eevee",
            "character_model": "human_master.blend (sha256 cbc13dbfce8cb5f5e6c51203ff29126b009f9f15643a4c35750fec6f1864e5ec), "
                               "uniform model_scale per character",
            "frame_start": 1, "frame_end": N, "frames": N, "fps": fr, "duration": round(D, 6),
            "source_video": si, "white_model_video": wi, "comparison_video": ci2,
            "frame_map": "参考视频第 N 帧 = 三维工程第 N 帧 = 白模第 N 帧（1..%d）" % N,
            "qualification_flags": lzhd_flags(si), "output_checks": spec_ok,
            "checks": {"技术结构通过": "通过（解码、帧数、尺寸、帧率已核对）" if ok else "未通过",
                       "重建已验证": "未验证", "视觉验收": "未验证", "客户验收": "未验证"},
            "notes": notes or []}
    with open(os.path.join(sub, "meta", "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(sub, "meta", "README.md"), "w", encoding="utf-8") as fh:
        fh.write(lzhd_readme(pkg, meta, desc))
    zp = os.path.join(submit_root, pkg + ".zip")
    zip_v21(sub, pkg, zp)
    return sub, zp, meta


def lzhd_readme(pkg, meta, desc):
    s, w = meta["source_video"], meta["white_model_video"]
    lines = [
        "# %s — 工程说明（LZHD V1.0）" % pkg, "",
        "- 主工程：`model_3d/%s.blend`（Blender %s，Eevee）。活动相机 `Camera`，时间轴 1–%d 帧，%s fps，与原片逐帧对应。"
        % (pkg, meta["blender_version"], meta["frames"], meta["fps"]),
        "- 人形：官方 `human_master.blend` 的骨架与 10 块网格（未换骨架、未改比例，仅导入时一次性均匀 model_scale）。"
        "身体按色卡（主角 #D63E3D、第二人 #2D6BD8、第三人 #2E9D5A、第四人 #D9A014），头部与胸部识别标识为源文件自带的标识面，改为白色。",
        "- 动作：每个角色的关节空物体（`<n>_pelvis`、`<n>_shoulder_L` …）逐帧旋转关键帧；对应的 DEF 骨骼用 Copy Transforms 跟随"
        "（`<n>_<关节>_bone`）。改关节空物体即可继续编辑动作。",
        "- 场景：集合 `Set_背景`，按色卡语义上色（背景 #B8B8B8、前景物体 #9A6B3B、近处交互物 #DF872F、远处交互物 #D4689E、"
        "光源 #F5F5F5、特殊对象 #7A4EC2）。",
        "- 光照：集合 `Lights`（主光 `Key_sun` + 环境光），方向按原片主光。", "",
        "## 视频", "",
        "- `target/original.mp4`：原片，未改动（%dx%d，%s fps，%.2f Mbps）。" % (s["width"], s["height"], s["fps_rational"], s["video_bitrate"] / 1e6),
        "- `white_model/blendervid.mp4`：%dx%d，%s fps，%.2f Mbps，音轨为原片音轨拷贝。" % (w["width"], w["height"], w["fps_rational"], w["video_bitrate"] / 1e6),
        "- `comparison/comparison.mp4`：%dx%d，上原片、下白模，同帧对齐，音轨为原片音轨拷贝。"
        % (meta["comparison_video"]["width"], meta["comparison_video"]["height"]),
        "- 资格旗标：%s。" % ("；".join(meta["qualification_flags"]) or "无"), "",
        "## 核对记录（诚实记录）", ""]
    lines += ["- %s：%s" % kv for kv in meta["checks"].items()]
    if meta["notes"]:
        lines += ["", "## 备注", ""] + ["- " + n for n in meta["notes"]]
    if desc:
        lines += ["", "## 镜头说明", "", desc]
    return "\n".join(lines) + "\n"

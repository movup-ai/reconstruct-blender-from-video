# -*- coding: utf-8 -*-
"""Rendered frames -> a submission package.  package() follows the v2.1 rules (2026-11-12, see its docstring);
package_lzhd() the LZHD V1.0 trial layout (render_lzhd.py).

Video: H.264 High, yuv420p, Rec.709 (tagged), 24 fps, ~30 Mbit/s constant; audio AAC stereo 48 kHz 320 kbit/s.
Encoder: h264_nvenc (GPU) when it works and the frame fits NVENC (<= 4096 per side: the white model), else libx264
(always for the ~3840x5600 comparison). Config "encoder" ("auto" | "libx264") and optional "nvenc_ffmpeg" (see nvenc_ffmpeg)."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile

from common import ffmpeg_exe
from validation import review_enabled


def clear_optional_review(rec, config):
    """Do not ship a stale audit page after a normal rebuild."""
    from pathlib import Path
    if review_enabled(config):return
    root=Path(rec).resolve();target=root/'review'
    if target.is_symlink():target.unlink();return
    if not target.resolve().is_relative_to(root):raise ValueError('Review path escapes reconstruction folder')
    if target.is_dir():shutil.rmtree(target)

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


NAMES_P = os.path.join(ROOT, "data", "package_names.json")


def _all_names():
    try:
        with open(NAMES_P, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _member(c):
    """the member these names belong to ("" outside the app's logins): their entries are "<member>/<clip name>"
    in the one shared file, so two members' clips with the same name keep their own delivery names"""
    return (c or {}).get("_user") or os.environ.get("BA_USER") or ""


def package_names(c=None):
    """{clip name: "english_title-N"} of this member (c from common.member_cfg, or a job's BA_USER), or the
    entries without a member when there is none ({} when the file is missing)"""
    m = _member(c)
    if not m:
        return {k: v for k, v in _all_names().items() if "/" not in k}
    p = m + "/"
    return {k[len(p):]: v for k, v in _all_names().items() if k.startswith(p)}


def package_name(clip, c=None):
    """v2.1 permanent package name for a clip: data/package_names.json {clip name: "english_title-N"}"""
    return package_names(c).get(clip)


def title_slug(title):
    """English title -> the name part of a v2.1 package name ("The Hidden Blade!" -> "the_hidden_blade")"""
    s = re.sub(r"[^a-z0-9]+", "_", str(title or "").lower()).strip("_")
    return s[:60].strip("_")


def propose_names(clips, title, names=None, c=None):
    """clips: clip names in file-name order.  Clips that already have a package name keep it; the others get
    <slug>-N, numbered after the highest N already used for that title - by any member, so two members naming
    clips of the same film never get the same name.  -> {clip: (package name, is_new)}"""
    if names is None:
        names = package_names(c)
        used = set(names.values()) | set(_all_names().values())
    else:
        used = set(names.values())
    slug = title_slug(title)
    nums = [int(v.rsplit("-", 1)[1]) for v in used if v.rsplit("-", 1)[0] == slug and v.rsplit("-", 1)[-1].isdigit()]
    n = max(nums or [0])
    out = {}
    for cl in clips:
        if names.get(cl):
            out[cl] = (names[cl], False)
            continue
        if not slug:
            out[cl] = (None, False)
            continue
        n += 1
        while "%s-%d" % (slug, n) in used:
            n += 1
        out[cl] = ("%s-%d" % (slug, n), True)
    return out


_NAMES_LOCK = threading.Lock()


def assign_names(clips, title, c=None):
    """write the new names of propose_names() into data/package_names.json (this member's entries); never renames
    a clip"""
    with _NAMES_LOCK:                               # two members naming at once must not both take the same number
        plan = propose_names(clips, title, c=c)
        new = {cl: pkg for cl, (pkg, is_new) in plan.items() if is_new and pkg}
        for pkg in new.values():
            if not PKG_RE.match(pkg):
                raise ValueError("%r is not a valid v2.1 package name" % pkg)
        if new:
            m = _member(c)
            allnames = _all_names()
            allnames.update({(m + "/" if m else "") + cl: pkg for cl, pkg in new.items()})
            tmp = NAMES_P + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(allnames, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, NAMES_P)
    return new


def register_names(names, c=None):
    """{clip: package name} given from outside (a task zip's own name) -> this member's entries in
    data/package_names.json; a clip that already has a name keeps it.  -> the names written"""
    with _NAMES_LOCK:
        have = package_names(c)
        new = {cl: p for cl, p in names.items() if p and PKG_RE.match(p) and not have.get(cl)}
        if new:
            m = _member(c)
            allnames = _all_names()
            allnames.update({(m + "/" if m else "") + cl: p for cl, p in new.items()})
            tmp = NAMES_P + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(allnames, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, NAMES_P)
    return new


def canvas(W, H):
    """v2.1 1080P canvas: 1920x1080, or 1080x1920 for a vertical source"""
    return (1080, 1920) if H > W else (1920, 1080)


def fit_even(w, h, bw, bh):
    """w x h scaled to fit inside bw x bh, even sizes"""
    f = min(bw / float(w), bh / float(h))
    a, b = int(round(w * f)), int(round(h * f))
    return a - a % 2, b - b % 2


def cover_even(w, h, bw, bh):
    """w x h scaled to cover all of bw x bh (the overflow is cropped), even sizes"""
    f = max(bw / float(w), bh / float(h))
    a, b = max(bw, int(round(w * f))), max(bh, int(round(h * f)))
    return a + a % 2, b + b % 2


def package(name, video, render_dir, submit_root, ci, c, desc=""):
    """v2.1 (2026-11-12): one clip = one package, five folders at the zip root, six fixed files:
      comparison/comparison.mp4      original on top, white model below, each 1920x1080 framed as the white model:
                                     1920x2160, no bars (a vertical source: side by side, 2160x1920)
      target/original.mp4            the source clip, byte for byte
      white_model/blendervid.mp4     1920x1080 (1080x1920 vertical), the source framing scaled to fill it, the
                                     overflow cropped evenly (a 2:1 source loses its side edges, no black bars)
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
    FW, FH = cover_even(W, Hfull, CW, CH)                    # the source frame filling the canvas (no black bars)
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
    encode(ff, c, CW, CH, inp, WMF + ",crop=%d:%d:(iw-ow)/2:(ih-oh)/2,%s[v]" % (CW, CH, tail), video, D, wm)
    # comparison: the original and the white model, each exactly as blendervid.mp4 frames it (filled, centre-cropped),
    # stacked at full size with no bars: original on top (1920x2160); a vertical source side by side (2160x1920)
    crop = "crop=%d:%d:(iw-ow)/2:(ih-oh)/2" % (CW, CH)
    stack, SW, SH = ("vstack", CW, 2 * CH) if CW >= CH else ("hstack", 2 * CW, CH)
    encode(ff, c, SW, SH, inp,
           WMF + ",%s[b];[1:v]%sscale=%d:%d:flags=lanczos,%s[a];[a][b]%s=inputs=2,%s[v]" % (crop, retime, FW, FH, crop, stack, tail),
           video, D, cmp_)
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
            "project_frames": {"frame_start": 1, "frame_end": N, "fps": int(round(fps)), "fps_base": int(round(fps))/float(fps),
                               "note": "the .blend timeline follows the source clip frame for frame (%d frames at %g fps)" % (N, fps)},
            "source_video": {"resolution": [W, Hfull], "fps": round(fps, 3), "frames": N, "duration": round(D, 6)},
            "output_video": {"resolution": [CW, CH], "fps": ofps, "frames": nout,
                             "retime": ("3:2 pulldown removed: the %d repeated frames of the 30 fps source dropped from both the "
                                        "original and the white model (24 fps film frames, same duration, no speed change)" % len(drops))
                                       if drops else "none",
                             "audio": "source audio stream copied unchanged into white_model and comparison" if pt_audio(ff, video) else "source has no audio"},
            "checks": {"structure": "check_package.py", "reconstruction": "unverified", "visual_review": "unverified"}}
    geometry_report=os.path.join(os.path.dirname(render_dir),'l3_geometry.json')
    if os.path.isfile(geometry_report):
        shutil.copyfile(geometry_report,os.path.join(rec,'l3_geometry.json'))
        meta['checks']['rigid_geometry']='see model_3d/reconstruction/l3_geometry.json'
    clear_optional_review(rec,c)
    if review_enabled(c):
        from frame_review import build as build_frame_review
        meta['frame_review']=build_frame_review(video,render_dir,os.path.join(rec,'review'),ci)
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
    return "not recorded"


def meta_readme(pkg, meta, desc, drops, W, H, CW, CH):
    ov = meta["output_video"]
    return "\n".join([
        "# %s — Project guide" % pkg, "",
        "- Main project: `model_3d/%s.blend`, saved with Blender %s. Open it directly; the active camera is `Camera`." % (pkg, meta["blender_version"]),
        "- Scene data: `model_3d/reconstruction/scene.json`. Read or rebuild it with `reconstruct.py`; see the adjacent `README.md`.",
        "- Dependencies: no extra dependencies, plugins, external textures or linked files. All meshes and materials are included.",
        "",
        "## Project contents", "",
        "- `Set_Background` collection: gray and white scene primitives (floor, walls and props). In multi-shot clips, hide_render keys limit each set to its shot's frame range.",
        "- Characters: the supplied `human_master.blend` rig and 10 mesh parts, uniformly scaled once per person. Head and chest marker faces use white `WM_marker` material. "
        "Role colors: main red, second blue, third green, fourth yellow, and cyan for additional people.",
        "- Character controls: `P<n>_root` (position and heading keys) → `P<n>_pelvis` (height keys) → joint empties (XYZ Euler rotation keys every frame). "
        "Mesh parts follow their corresponding joints; edit joint rotations to adjust motion.",
        "- Camera: `Camera`, with position and rotation keys every frame following the source camera. Focal length and sensor width are in scene.json `camera`.",
        "- Timeline: frames 1–%d at %d fps, matching the source frame for frame." % (meta["project_frames"]["frame_end"], meta["project_frames"]["fps"]),
        "",
        "## Videos", "",
        "- `target/original.mp4`: unchanged source (%dx%d). %s" % (W, H, "" if (W, H) in ((1920, 1080), (1080, 1920)) else
                                                         "The source is not 1920x1080; it is submitted without resizing or re-encoding, as required."),
        "- `white_model/blendervid.mp4`: %dx%d, %d fps, %d frames. The source framing is scaled proportionally to fill the 1080P canvas and center-cropped without black borders." % (CW, CH, ov["fps"], ov["frames"]),
        ("- `comparison/comparison.mp4`: %dx%d, source above and white model below, synchronized. Both fill %dx%d with the same framing as the white-model video, stacked at full size without black borders." % (CW, 2 * CH, CW, CH)
         if CW >= CH else "- `comparison/comparison.mp4`: %dx%d, source left and white model right, synchronized. Both fill %dx%d with the same framing as the white-model video, placed side by side at full size without black borders." % (2 * CW, CH, CW, CH)),
        "- Frame rate: %s" % ov["retime"],
        "- Audio: %s." % ov["audio"],
        "",
        "## Verification status", "",
        "- Structure: checked by check_package.py; see the delivery log.",
        "- Reconstruction: unverified for this package. The entry point has been tested on similar packages, but reconstruct.py has not been run separately for each package.",
        "- Visual review: unverified.",
        "" if not desc else "\n## Shot notes\n\n" + desc + "\n"])


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
        f.append("Source below 1080P (%dx%d)" % (si["width"], si["height"]))
    if si["fps"] < 24 - 1e-3:
        f.append("Source below 24 fps (%.3f)" % si["fps"])
    if si["duration"] < 5.0:
        f.append("Source shorter than 5 seconds (%.3fs)" % si["duration"])
    if si["video_bitrate"] < 18000000:
        f.append("Source video bitrate below 18 Mbps (%.2f Mbps)" % (si["video_bitrate"] / 1e6))
    return f


def package_lzhd(name, pkg, video, render_dir, submit_root, ci, c, desc="", notes=None):
    """LZHD White-model Video Reconstruction and Acceptance Standard V1.0 (2026-11-15):
      white_model/blendervid.mp4   same width, height, frame rate and video bit rate as the source; the source's audio
      comparison/comparison.mp4    original on top, white model below: W x 2H, same fps, audio copied
      target/original.mp4          the source clip, byte for byte
      model_3d/<pkg>.blend         editable project (official human_master rig, Eevee, active camera, timeline 1..N)
      model_3d/reconstruction/     scene.json (+ reconstruct.py, README.md)
      meta/meta.json + README.md   specs, qualification flags, honest check record (unverified where not done)
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
    meta = {"id": pkg, "clip": name, "spec": "LZHD White-model Video Reconstruction and Acceptance Standard V1.0 (2026-11-15)",
            "paths_base": ".", "source": "target/original.mp4", "white_model": "white_model/blendervid.mp4",
            "comparison": "comparison/comparison.mp4", "model_3d": "model_3d/%s.blend" % pkg,
            "scene_json_path": "model_3d/reconstruction/scene.json", "blender_version": blender_version(render_dir),
            "render_engine": "Eevee",
            "character_model": "See scene.json characters for the fitted model style; fixed uniform scale per actor",
            "frame_start": 1, "frame_end": N, "frames": N, "fps": fr, "duration": round(D, 6),
            "source_video": si, "white_model_video": wi, "comparison_video": ci2,
            "frame_map": "Source frame N = 3D project frame N = white-model frame N (1..%d)" % N,
            "qualification_flags": lzhd_flags(si), "output_checks": spec_ok,
            "checks": {"technical_structure": "passed (decoding, frame count, dimensions and fps checked)" if ok else "failed",
                       "reconstruction": "unverified", "visual_review": "unverified", "client_acceptance": "unverified"},
            "notes": notes or []}
    clear_optional_review(rec,c)
    if review_enabled(c):
        from frame_review import build as build_frame_review
        meta['frame_review']=build_frame_review(video,render_dir,os.path.join(rec,'review'),ci)
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
        "# %s — Project guide (LZHD V1.0)" % pkg, "",
        "- Main project: `model_3d/%s.blend` (Blender %s, Eevee). Active camera: `Camera`. Timeline: frames 1–%d at %s fps, matching the source frame for frame."
        % (pkg, meta["blender_version"], meta["frames"], meta["fps"]),
        "- Characters: the supplied `human_master.blend` rig and 10 mesh parts, with original rig and proportions, uniformly scaled once on import using model_scale. "
        "Body palette: main #D63E3D, second #2D6BD8, third #2E9D5A, fourth #D9A014. The source model's head and chest marker faces are white.",
        "- Motion: joint empties (`<n>_pelvis`, `<n>_shoulder_L`, etc.) have rotation keys every frame. Corresponding DEF bones follow with Copy Transforms "
        "(`<n>_<joint>_bone`). Edit the joint empties to adjust motion.",
        "- Scene: `Set_Background` collection, colored by semantic palette: background #B8B8B8, foreground objects #9A6B3B, nearby interactive objects #DF872F, distant interactive objects #D4689E, "
        "light sources #F5F5F5, special objects #7A4EC2.",
        "- Lighting: `Lights` collection (`Key_sun` plus ambient light), oriented to match the source key light.", "",
        "## Videos", "",
        "- `target/original.mp4`: unchanged source (%dx%d, %s fps, %.2f Mbps)." % (s["width"], s["height"], s["fps_rational"], s["video_bitrate"] / 1e6),
        "- `white_model/blendervid.mp4`: %dx%d, %s fps, %.2f Mbps; source audio copied." % (w["width"], w["height"], w["fps_rational"], w["video_bitrate"] / 1e6),
        "- `comparison/comparison.mp4`: %dx%d, source above and white model below, aligned frame for frame; source audio copied."
        % (meta["comparison_video"]["width"], meta["comparison_video"]["height"]),
        "- Qualification flags: %s." % ("; ".join(meta["qualification_flags"]) or "none"), "",
        "## Verification record", ""]
    lines += ["- %s: %s" % kv for kv in meta["checks"].items()]
    if meta["notes"]:
        lines += ["", "## Notes", ""] + ["- " + n for n in meta["notes"]]
    if desc:
        lines += ["", "## Shot notes", "", desc]
    return "\n".join(lines) + "\n"

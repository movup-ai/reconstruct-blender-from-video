# -*- coding: utf-8 -*-
"""Check v2.1 submission packages against the delivery spec.

  python -m agent.validate [SUBMIT folder]      -> prints a report, writes <work_root>/submit_check.csv
The pipeline runs check_package_v21 + check_zip after packing each clip; any problem fails the job.
Each problem is listed in plain words."""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile

from . import config
from common import ffmpeg_exe                     # pipeline
from package import SUBDIRS                       # pipeline

V21 = ("comparison/comparison.mp4", "target/original.mp4", "white_model/blendervid.mp4",
       "model_3d/{p}.blend", "model_3d/reconstruction/scene.json", "meta/meta.json")
PKG_ID = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*-[0-9]+$")
# the white model is one 1080P frame; the comparison is the source and the white model, each 1080P:
# stacked for a landscape white model, side by side for a vertical one
COMPARISON_FOR = {(1920, 1080): (1920, 2160), (1080, 1920): (2160, 1920)}
SKIPPED = (".blend1", ".tmp", ".part")            # files the packer leaves out of the zip (package.zip_v21)
SKIPPED_NAMES = ("Thumbs.db", "desktop.ini", ".DS_Store")


def probe(ff, path):
    p = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    e = p.stderr
    out = {}
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+).*?bitrate:\s*(\d+)", e)
    if m:
        out["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        out["kbps"] = int(m.group(4))
    m = re.search(r"Video:\s*(\w+)[^\n]*?,\s*(\w+)\(([^)]*)\)[^\n]*?(\d{2,5})x(\d{2,5})[^\n]*?([\d.]+) fps", e)
    if m:
        out.update(vcodec=m.group(1), pix=m.group(2), colour=m.group(3), width=int(m.group(4)), height=int(m.group(5)), fps=float(m.group(6)))
    m = re.search(r"Audio:\s*(\w+)[^\n]*?(\d+) Hz,\s*(\w+)", e)
    if m:
        out.update(acodec=m.group(1), hz=int(m.group(2)), channels=m.group(3))
    return out


def _md5_audio(ff, path):
    p = subprocess.run([ff, "-v", "error", "-i", path, "-map", "0:a:0?", "-c", "copy", "-f", "md5", "-"], capture_output=True,
                       text=True, encoding="utf-8", errors="ignore")
    return p.stdout.strip() or None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_file(a, b):
    """True when two files hold the same bytes (size first, then a streamed hash)"""
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
    except OSError:
        return False
    return _sha256(a) == _sha256(b)


def check_package(sub, ff=None, source=None):
    """-> (problems, facts) for one unpacked submission folder (v2.1 layout when meta/meta.json is there)"""
    ff = ff or ffmpeg_exe()
    if os.path.exists(os.path.join(sub, "meta", "meta.json")):
        return check_package_v21(sub, ff, source)
    return (["not the v2.1 layout (no meta/meta.json) - run the clip again"],
            {"name": os.path.basename(sub.rstrip("\\/")), "white_model": {}, "comparison": {}})


def check_package_v21(sub, ff, source=None):
    """v2.1: five folders, six fixed files, meta.json id and paths, 1080P white model with a comparison of the
    matching orientation, H.264, durations, fps, identical audio, the .blend header and scene.json.
    source: the clip's video; target/original.mp4 must be byte-identical to it.  -> (problems, facts)"""
    P, F = [], {"name": os.path.basename(sub.rstrip("\\/")), "white_model": {}, "comparison": {}}
    if sorted(d for d in os.listdir(sub) if os.path.isdir(os.path.join(sub, d))) != sorted(SUBDIRS):
        P.append("folders must be exactly: " + ", ".join(SUBDIRS))
    try:
        with open(os.path.join(sub, "meta", "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        if not isinstance(meta, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:
        return P + ["meta/meta.json unreadable: %s" % e], F
    pkg = str(meta.get("id", ""))
    F["package"] = pkg
    if not PKG_ID.match(pkg):
        P.append("meta.id %r is not a v2.1 package name" % pkg)
    for f in V21:
        if not os.path.isfile(os.path.join(sub, f.format(p=pkg))):
            P.append("missing " + f.format(p=pkg))
    for k in ("source", "white_model", "comparison", "model_3d", "scene_json_path"):
        if not isinstance(meta.get(k), str) or not os.path.isfile(os.path.join(sub, meta.get(k) or "?")):
            P.append("meta %s points to a missing file: %s" % (k, meta.get(k)))
    for k in ("blender_version", "frame_start", "frame_end", "frames", "fps", "fps_base", "duration"):
        if k not in meta:
            P.append("meta.json has no " + k)
    try:
        with open(os.path.join(sub, "model_3d", "reconstruction", "scene.json"), encoding="utf-8") as fh:
            sj = json.load(fh)
        if not isinstance(sj, dict) or not sj.get("objects") or not sj.get("scene"):
            P.append("scene.json has no scene / objects")
    except (OSError, ValueError) as e:
        P.append("scene.json unreadable: %s" % e)
    target = os.path.join(sub, "target", "original.mp4")
    v = {"white_model": probe(ff, os.path.join(sub, "white_model", "blendervid.mp4")),
         "comparison": probe(ff, os.path.join(sub, "comparison", "comparison.mp4")),
         "target": probe(ff, target)}
    F.update(v)
    wm, cmp_ = v["white_model"], v["comparison"]
    wm_size, cmp_size = (wm.get("width"), wm.get("height")), (cmp_.get("width"), cmp_.get("height"))
    if wm_size not in COMPARISON_FOR:
        P.append("white_model is %sx%s, must be 1920x1080 or 1080x1920" % wm_size)
        if cmp_size not in COMPARISON_FOR.values():
            P.append("comparison is %sx%s, must be 1920x2160 or 2160x1920" % cmp_size)
    elif cmp_size != COMPARISON_FOR[wm_size]:
        P.append("comparison is %sx%s, must be %dx%d for a %dx%d white model" % (cmp_size + COMPARISON_FOR[wm_size] + wm_size))
    for d in ("white_model", "comparison"):
        x = v[d]
        if x.get("vcodec") != "h264":
            P.append("%s is %s, expected H.264" % (d, x.get("vcodec")))
        if v["target"].get("duration") and abs(x.get("duration", 0) - v["target"]["duration"]) > 0.15:
            P.append("%s lasts %.2fs, target %.2fs" % (d, x.get("duration", 0), v["target"]["duration"]))
    a = [_md5_audio(ff, os.path.join(sub, f)) for f in ("target/original.mp4", "white_model/blendervid.mp4", "comparison/comparison.mp4")]
    if len(set(a)) != 1:
        P.append("audio differs between the three videos")
    try:
        meta_fps = float(meta.get("fps", 0)) / float(meta.get("fps_base", 1) or 1)
    except (TypeError, ValueError):
        meta_fps = None
        P.append("meta fps / fps_base are not numbers")
    if wm.get("fps") and meta_fps is not None and abs(wm["fps"] - meta_fps) > 0.01:
        P.append("meta fps does not match the white model")
    if wm.get("fps") and cmp_.get("fps") and abs(wm["fps"] - cmp_["fps"]) > 0.01:
        P.append("comparison fps %.3f differs from the white model %.3f" % (cmp_["fps"], wm["fps"]))
    if source is not None:
        if not os.path.isfile(source):
            P.append("the source video is missing: %s" % source)
        elif os.path.isfile(target) and not same_file(source, target):
            P.append("target/original.mp4 is not byte-identical to the source video")
    bp = os.path.join(sub, "model_3d", pkg + ".blend")
    if os.path.exists(bp):
        with open(bp, "rb") as fh:
            head = fh.read(12)
        if not (head.startswith(b"BLENDER") or head[:2] == b"\x1f\x8b" or head[:4] == b"\x28\xb5\x2f\xfd"):
            P.append("model_3d file is not a Blender file")
        F["blend_mb"] = round(os.path.getsize(bp) / 1e6, 1)
    return P, F


def check_zip(zp, sub):
    """-> problems of a v2.1 zip: readable, undamaged, '/' paths without spaces, and exactly the folder's files"""
    P = []
    try:
        with zipfile.ZipFile(zp) as z:
            if z.testzip() is not None:
                P.append("zip is damaged")
            names = [n for n in z.namelist() if not n.endswith("/")]
    except (OSError, zipfile.BadZipFile) as e:
        return ["zip unreadable: %s" % e]
    want = sorted(os.path.relpath(os.path.join(dp, fn), sub).replace(os.sep, "/")
                  for dp, _, fs in os.walk(sub) for fn in fs
                  if not fn.endswith(SKIPPED) and fn not in SKIPPED_NAMES)
    if sorted(names) != want:
        extra, miss = set(names) - set(want), set(want) - set(names)
        P.append("zip contents differ from the folder (extra: %s; missing: %s)" % (", ".join(sorted(extra))[:200] or "none",
                                                                                ", ".join(sorted(miss))[:200] or "none"))
    if any("\\" in n or " " in n for n in names):
        P.append("zip paths with backslashes or spaces")
    return P


def check_all(root):
    ff = ffmpeg_exe()
    rows = []
    for d in sorted(os.listdir(root)):
        sub = os.path.join(root, d)
        if not os.path.isdir(sub) or d.startswith("_"):
            continue
        P, F = check_package(sub, ff)
        zp = sub + ".zip"
        P += check_zip(zp, sub) if os.path.exists(zp) else ["no zip"]
        wm, cm = F.get("white_model") or {}, F.get("comparison") or {}
        rows.append({"package": d, "ok": not P, "problems": P, "white_model": "%sx%s" % (wm.get("width"), wm.get("height")),
                     "comparison": "%sx%s" % (cm.get("width"), cm.get("height")),
                     "kbps": wm.get("kbps"), "zip_mb": round(os.path.getsize(zp) / 1e6, 1) if os.path.exists(zp) else 0})
    return rows


def main(argv):
    c = config.load()
    root = argv[0] if argv else c["submit_root"]
    rows = check_all(root)
    for r in rows:
        print("%s  %-40s %s %s %s kb/s  zip %s MB" % ("OK  " if r["ok"] else "FAIL", r["package"], r["white_model"], r["comparison"], r["kbps"], r["zip_mb"]))
        for p in r["problems"]:
            print("        - " + p)
    out = os.path.join(c["work_root"], "submit_check.csv")
    os.makedirs(c["work_root"], exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["package", "ok", "white_model", "comparison", "kbps", "zip_mb", "problems"])
        for r in rows:
            w.writerow([r["package"], r["ok"], r["white_model"], r["comparison"], r["kbps"], r["zip_mb"], " | ".join(r["problems"])])
    print("\n%d package(s), %d OK.  Report: %s" % (len(rows), sum(r["ok"] for r in rows), out))
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

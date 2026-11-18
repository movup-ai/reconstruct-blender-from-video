# -*- coding: utf-8 -*-
"""Check submission packages against the client sample (11月2日试稿) and the delivery spec.

  python -m agent.validate [SUBMIT folder]      -> prints a report, writes <work_root>/submit_check.csv
Each package must look exactly like the sample; every problem is listed in plain words."""
import csv
import json
import os
import re
import subprocess
import sys
import zipfile

from . import config
from common import ffmpeg_exe                     # pipeline
from package import SUBDIRS, META_KEYS            # pipeline

MIN_KBPS = 18000


def _out_width():
    """the configured white model width (config.json "out_width"; 2560 = client spec 2K, 3840 = like the sample)"""
    try:
        return config.load().get("out_width", 3840)
    except Exception:
        return 3840


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


V21 = ("comparison/comparison.mp4", "target/original.mp4", "white_model/blendervid.mp4",
       "model_3d/{p}.blend", "model_3d/reconstruction/scene.json", "meta/meta.json")


def _md5_audio(ff, path):
    p = subprocess.run([ff, "-v", "error", "-i", path, "-map", "0:a:0?", "-c", "copy", "-f", "md5", "-"], capture_output=True,
                       text=True, encoding="utf-8", errors="ignore")
    return p.stdout.strip() or None


def check_package(sub, ff=None):
    """-> (problems, facts) for one unpacked submission folder (v2.1 layout when meta/meta.json is there)"""
    ff = ff or ffmpeg_exe()
    if os.path.exists(os.path.join(sub, "meta", "meta.json")):
        return check_package_v21(sub, ff)
    return check_package_v1(sub, ff)


def check_package_v21(sub, ff):
    """v2.1 (2026-11-12): five folders, six fixed files, meta.json paths, 1080P videos, identical audio"""
    P, F = [], {"name": os.path.basename(sub.rstrip("\\/"))}
    if sorted(d for d in os.listdir(sub) if os.path.isdir(os.path.join(sub, d))) != sorted(SUBDIRS):
        P.append("folders must be exactly: " + ", ".join(SUBDIRS))
    try:
        meta = json.load(open(os.path.join(sub, "meta", "meta.json"), encoding="utf-8"))
    except (OSError, ValueError) as e:
        return ["meta/meta.json unreadable: %s" % e], dict(F, white_model={}, comparison={})
    pkg = meta.get("id", "")
    F["package"] = pkg
    if not re.match(r"^[a-z0-9]+(?:_[a-z0-9]+)*-[0-9]+$", pkg):
        P.append("meta.id %r is not a v2.1 package name" % pkg)
    for f in V21:
        if not os.path.isfile(os.path.join(sub, f.format(p=pkg))):
            P.append("missing " + f.format(p=pkg))
    for k in ("source", "white_model", "comparison", "model_3d", "scene_json_path"):
        if not os.path.isfile(os.path.join(sub, meta.get(k) or "?")):
            P.append("meta %s points to a missing file: %s" % (k, meta.get(k)))
    for k in ("blender_version", "frame_start", "frame_end", "frames", "fps", "fps_base", "duration"):
        if k not in meta:
            P.append("meta.json has no " + k)
    try:
        sj = json.load(open(os.path.join(sub, "model_3d", "reconstruction", "scene.json"), encoding="utf-8"))
        if not sj.get("objects") or not sj.get("scene"):
            P.append("scene.json has no scene / objects")
    except (OSError, ValueError) as e:
        P.append("scene.json unreadable: %s" % e)
    v = {"white_model": probe(ff, os.path.join(sub, "white_model", "blendervid.mp4")),
         "comparison": probe(ff, os.path.join(sub, "comparison", "comparison.mp4")),
         "target": probe(ff, os.path.join(sub, "target", "original.mp4"))}
    F.update(v)
    for d in ("white_model", "comparison"):
        x = v[d]
        if (x.get("width"), x.get("height")) not in ((1920, 1080), (1080, 1920)):
            P.append("%s is %sx%s, must be 1920x1080 (1080x1920 vertical)" % (d, x.get("width"), x.get("height")))
        if x.get("vcodec") != "h264":
            P.append("%s is %s, expected H.264" % (d, x.get("vcodec")))
        if v["target"].get("duration") and abs(x.get("duration", 0) - v["target"]["duration"]) > 0.15:
            P.append("%s lasts %.2fs, target %.2fs" % (d, x.get("duration", 0), v["target"]["duration"]))
    a = [_md5_audio(ff, os.path.join(sub, f)) for f in ("target/original.mp4", "white_model/blendervid.mp4", "comparison/comparison.mp4")]
    if len(set(a)) != 1:
        P.append("audio differs between the three videos")
    if v["white_model"].get("fps") and abs(v["white_model"]["fps"] - float(meta.get("fps", 0)) / float(meta.get("fps_base", 1) or 1)) > 0.01:
        P.append("meta fps does not match the white model")
    bp = os.path.join(sub, "model_3d", pkg + ".blend")
    if os.path.exists(bp):
        with open(bp, "rb") as fh:
            head = fh.read(12)
        if not (head.startswith(b"BLENDER") or head[:2] == b"\x1f\x8b" or head[:4] == b"\x28\xb5\x2f\xfd"):
            P.append("model_3d file is not a Blender file")
        F["blend_mb"] = round(os.path.getsize(bp) / 1e6, 1)
    return P, F


def check_package_v1(sub, ff=None):
    """-> (problems, facts) for one unpacked submission folder (earlier sample layout)"""
    ff = ff or ffmpeg_exe()
    name = os.path.basename(sub.rstrip("\\/"))
    P, F = [], {"name": name}
    if sorted(d for d in os.listdir(sub) if os.path.isdir(os.path.join(sub, d))) != sorted(SUBDIRS):
        P.append("folders must be exactly: " + ", ".join(SUBDIRS))
    want = {"comparison": name + ".mp4", "meta": "meta.jsonl", "model_3d": name + "blend.blend",
            "target": name + ".mp4", "white_model": name + ".mp4"}
    for d, fn in want.items():
        have = sorted(os.listdir(os.path.join(sub, d))) if os.path.isdir(os.path.join(sub, d)) else []
        if have != [fn]:
            P.append("%s/ should contain only %s (has: %s)" % (d, fn, ", ".join(have) or "nothing"))
    # meta
    meta = None
    mp = os.path.join(sub, "meta", "meta.jsonl")
    try:
        lines = [l for l in open(mp, encoding="utf-8").read().splitlines() if l.strip()]
        if len(lines) != 1:
            P.append("meta.jsonl must have exactly one line")
        meta = json.loads(lines[0])
        if list(meta.keys()) != META_KEYS:
            P.append("meta.jsonl keys differ from the sample (order or names)")
        for k in ("source_video_video_path", "pattern_video_video_path", "comparison_video_path"):
            if not os.path.isfile(os.path.join(sub, meta.get(k, ""))):
                P.append("meta %s points to a missing file: %s" % (k, meta.get(k)))
        if not os.path.isfile(os.path.join(sub, (meta.get("ext_json") or {}).get("model_3d_path", ""))):
            P.append("meta ext_json.model_3d_path points to a missing file")
        if (meta.get("ext_json") or {}).get("code_type") != "H.264":
            P.append("ext_json.code_type must be H.264")
    except (OSError, ValueError, IndexError) as e:
        P.append("meta.jsonl unreadable: %s" % e)
    # videos
    v = {d: probe(ff, os.path.join(sub, d, name + ".mp4")) for d in ("white_model", "comparison", "target")}
    F.update({d: v[d] for d in v})
    wm, cm, tg = v["white_model"], v["comparison"], v["target"]
    for d in ("white_model", "comparison"):
        x = v[d]
        if x.get("vcodec") != "h264":
            P.append("%s is %s, must be H.264" % (d, x.get("vcodec")))
        if abs(x.get("fps", 0) - 24) > 0.01:
            P.append("%s is %s fps, must be 24" % (d, x.get("fps")))
        if x.get("kbps", 0) < MIN_KBPS:
            P.append("%s bitrate %s kb/s is below 18 Mb/s" % (d, x.get("kbps")))
        if "bt709" not in x.get("colour", ""):
            P.append("%s is not tagged Rec.709 (%s)" % (d, x.get("colour")))
        if tg.get("acodec") and (x.get("acodec") != "aac" or x.get("hz") != 48000 or x.get("channels") != "stereo"):
            P.append("%s audio must be AAC 48 kHz stereo (is %s %s %s)" % (d, x.get("acodec"), x.get("hz"), x.get("channels")))
        if tg.get("duration") and abs(x.get("duration", 0) - tg["duration"]) > 0.15:
            P.append("%s lasts %.2fs, target %.2fs" % (d, x.get("duration", 0), tg["duration"]))
    want_w = int(_out_width())
    if wm.get("width") != want_w:
        P.append("white_model is %s wide, the configured output width is %d" % (wm.get("width"), want_w))
    if tg.get("width") and wm.get("width") and abs(wm["height"] - wm["width"] * tg["height"] / float(tg["width"])) > 2.5:
        P.append("white_model aspect %sx%s does not match the target %sx%s" % (wm["width"], wm["height"], tg["width"], tg["height"]))
    if cm.get("width") != wm.get("width") or cm.get("height") != 2 * (wm.get("height") or 0):
        P.append("comparison must be the target over the white model (%sx%s, is %sx%s)" % (
            wm.get("width"), 2 * (wm.get("height") or 0), cm.get("width"), cm.get("height")))
    if meta:
        if meta.get("source_video_resolution") != [wm.get("width"), wm.get("height")]:
            P.append("meta source_video_resolution does not match the white model")
        if meta.get("pattern_video_resolution") != [tg.get("width"), tg.get("height")]:
            P.append("meta pattern_video_resolution does not match the target")
        if cm.get("width"):
            from math import gcd
            g = gcd(cm["width"], cm["height"])
            if (meta.get("ext_json") or {}).get("aspect_ratio") != "%d:%d" % (cm["width"] // g, cm["height"] // g):
                P.append("meta aspect_ratio must be the comparison's (%d:%d)" % (cm["width"] // g, cm["height"] // g))
    # blend
    bp = os.path.join(sub, "model_3d", name + "blend.blend")
    if os.path.exists(bp):
        with open(bp, "rb") as fh:
            head = fh.read(12)
        if not (head.startswith(b"BLENDER") or head[:2] == b"\x1f\x8b" or head[:4] == b"\x28\xb5\x2f\xfd"):
            P.append("model_3d file is not a Blender file")
        F["blend_mb"] = round(os.path.getsize(bp) / 1e6, 1)
    return P, F


def check_zip(zp, sub):
    if os.path.exists(os.path.join(sub, "meta", "meta.json")):          # v2.1: five folders at the root, '/' paths
        P = []
        try:
            with zipfile.ZipFile(zp) as z:
                if z.testzip() is not None:
                    P.append("zip is damaged")
                names = z.namelist()
        except (OSError, zipfile.BadZipFile) as e:
            return ["zip unreadable: %s" % e]
        want = sorted(os.path.relpath(os.path.join(dp, fn), sub).replace(os.sep, "/")
                      for dp, _, fs in os.walk(sub) for fn in fs)
        if sorted(names) != want:
            P.append("zip contents differ from the folder")
        if any("\\" in n or " " in n for n in names):
            P.append("zip paths with backslashes or spaces")
        return P
    P = []
    name = os.path.basename(sub)
    try:
        with zipfile.ZipFile(zp) as z:
            if z.testzip() is not None:
                P.append("zip is damaged")
            names = []
            for i in z.infolist():
                n = i.filename
                if not i.flag_bits & 0x800:
                    try:
                        n = n.encode("cp437").decode("gbk")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                names.append(n)
    except (OSError, zipfile.BadZipFile) as e:
        return ["zip unreadable: %s" % e]
    want = []
    for d in SUBDIRS:
        want.append("%s/%s/" % (name, d))
        want += ["%s/%s/%s" % (name, d, fn) for fn in sorted(os.listdir(os.path.join(sub, d)))]
    if sorted(names) != sorted(want):
        extra = set(names) - set(want); miss = set(want) - set(names)
        P.append("zip contents differ from the folder (extra: %s; missing: %s)" % (", ".join(sorted(extra))[:200], ", ".join(sorted(miss))[:200]))
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
        rows.append({"package": d, "ok": not P, "problems": P, "white_model": "%sx%s" % (F["white_model"].get("width"), F["white_model"].get("height")),
                     "comparison": "%sx%s" % (F["comparison"].get("width"), F["comparison"].get("height")),
                     "kbps": F["white_model"].get("kbps"), "zip_mb": round(os.path.getsize(zp) / 1e6, 1) if os.path.exists(zp) else 0})
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
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["package", "ok", "white_model", "comparison", "kbps", "zip_mb", "problems"])
        for r in rows:
            w.writerow([r["package"], r["ok"], r["white_model"], r["comparison"], r["kbps"], r["zip_mb"], " | ".join(r["problems"])])
    print("\n%d package(s), %d OK.  Report: %s" % (len(rows), sum(r["ok"] for r in rows), out))
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

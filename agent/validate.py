# -*- coding: utf-8 -*-
"""Check submission packages against the client sample (November 2 trial) and the delivery spec.

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
from package import SUBDIRS                       # pipeline

MIN_KBPS = 18000



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
    return (["not the v2.1 layout (no meta/meta.json) - finish the clip again"], {"name": os.path.basename(sub.rstrip("\\/"))})


def check_package_v21(sub, ff):
    """v2.1 (2026-11-12): five folders, six fixed files, meta.json paths, 1080P videos, identical audio"""
    P, F = [], {"name": os.path.basename(sub.rstrip("\\/"))}
    if sorted(d for d in os.listdir(sub) if os.path.isdir(os.path.join(sub, d))) != sorted(SUBDIRS):
        P.append("folders must be exactly: " + ", ".join(SUBDIRS))
    try:
        with open(os.path.join(sub, "meta", "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
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
        with open(os.path.join(sub, "model_3d", "reconstruction", "scene.json"), encoding="utf-8") as fh:
            sj = json.load(fh)
        if not sj.get("objects") or not sj.get("scene"):
            P.append("scene.json has no scene / objects")
    except (OSError, ValueError) as e:
        P.append("scene.json unreadable: %s" % e)
    v = {"white_model": probe(ff, os.path.join(sub, "white_model", "blendervid.mp4")),
         "comparison": probe(ff, os.path.join(sub, "comparison", "comparison.mp4")),
         "target": probe(ff, os.path.join(sub, "target", "original.mp4"))}
    F.update(v)
    sizes = {"white_model": ((1920, 1080), (1080, 1920)),            # the 1080P white model
             "comparison": ((1920, 2160), (2160, 1920))}             # two 1080P frames stacked (vertical: side by side)
    for d in ("white_model", "comparison"):
        x = v[d]
        if (x.get("width"), x.get("height")) not in sizes[d]:
            P.append("%s is %sx%s, must be %s" % (d, x.get("width"), x.get("height"), " or ".join("%dx%d" % s for s in sizes[d])))
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

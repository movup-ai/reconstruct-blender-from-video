# -*- coding: utf-8 -*-
"""v2.1 delivery: our packages -> the batch folder the client expects, checked with the client's own check_package.py.

  python pipeline\\deliver.py [--films 隐剑] [--batch batch20261116] [--out DELIVERY] [--movie "隐剑鬼爪|The Hidden Blade|2004"]

For every clip with a package name (data/package_names.json) and a finished SUBMIT\\<clip>.zip:
  DELIVERY\\<batch>\\data\\<package>\\v1\\<package>.zip             the package (copied byte for byte)
                                         comparison.mp4            extracted from that zip (the same bytes)
                                         <package>.zip.sha256      written by tools\\v21\\check_package.py when it passes
  DELIVERY\\<batch>\\manifest_v1.csv       11 columns of the template, UTF-8 with BOM, every package of the batch
  DELIVERY\\<batch>\\movie_titles_v1.csv   Chinese / English title, year, type, year basis, source, R2 path per package
  DELIVERY\\<batch>\\check_log.txt         the checker's output per package
  DELIVERY\\UPLOAD\\<package>.zip           the same zips in one flat folder: these are the files to upload
                                         (never upload SUBMIT\\<clip>.zip: those are named after the clip)
lens / num_characters / camera_movement are estimated from the fitted scene (lens from the focal length, characters
counted, movement from the solved camera); theme (选片题材) is left empty for the user to fill in."""
import argparse, csv, datetime, hashlib, json, os, shutil, subprocess, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from agent import config                      # noqa: E402
from agent.clips import Library, folders      # noqa: E402

MANIFEST_COLS = ["package_name", "version", "zip_sha256", "zip_size", "comparison_sha256", "movie_title", "theme", "lens",
                 "num_characters", "camera_movement", "submitted_at"]
TITLE_COLS = ["package_name", "title_zh", "title_en", "year", "work_type", "year_basis", "source", "r2_path"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lens_class(mm, sensor=36.0):
    eq = mm * 36.0 / sensor
    return "广角镜头" if eq < 28 else ("标准/中焦镜头" if eq <= 70 else "长焦镜头")


def people_class(n):
    return {0: "单人", 1: "单人", 2: "双人", 3: "三人"}.get(n, "三人以上")


def movement(cam, lens_note=""):
    """camera keys [x, y, z, rx, ry, rz] per frame -> allowed movement words"""
    C = np.array(cam, float)
    if len(C) < 2:
        return "固定镜头"
    pos = C[:, :3]; rot = C[:, 3:]
    travel = float(np.linalg.norm(pos[-1] - pos[0])); span = float(np.ptp(pos, axis=0).max())
    turn = float(np.abs((rot - rot[0] + 180) % 360 - 180).max())
    out = []
    if span > 0.3:
        # along the view direction -> 推 / 拉, sideways -> 移
        rx, rz = np.radians(rot[0, 0]), np.radians(rot[0, 2])
        view = np.array([np.sin(rz) * np.sin(rx), -np.cos(rz) * np.sin(rx), -np.cos(rx)])
        along = float((pos[-1] - pos[0]) @ view)
        if abs(along) > 0.6 * travel and travel > 0.3:
            out.append("推镜头" if along > 0 else "拉镜头")
        else:
            out.append("移镜头")
    if turn > 3.0:
        out.append("摇镜头")
    return "; ".join(out) or "固定镜头"


def run_check(checker, zp, cmp_):
    p = subprocess.run([sys.executable, checker, zp, cmp_], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return p.returncode, p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", default="")
    bj = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))   # batch date: Beijing time
    ap.add_argument("--batch", default="batch" + bj.strftime("%Y%m%d"))
    ap.add_argument("--out", default=os.path.join(ROOT, "DELIVERY"))
    ap.add_argument("--movie", action="append", default=[],
                    help="film folder keyword|中文名|English title|year (repeatable); default for 隐剑 built in")
    a = ap.parse_args()
    c = config.load(); lib = Library(c)
    with open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8") as fh:
        names = json.load(fh)
    titles = {"隐剑": ("隐剑鬼爪", "The Hidden Blade", "2004")}
    for m in a.movie:
        k, zh, en, yr = m.split("|"); titles[k] = (zh, en, yr)
    checker = os.path.join(ROOT, "tools", "v21", "check_package.py")
    bdir = os.path.join(a.out, a.batch)
    rows, trows, log, bad = [], [], [], []
    clips = [cl for cl in lib.clips if cl.name in names and (not a.films or any(k in cl.film for k in a.films.split(",")))]
    clips.sort(key=lambda cl: int(names[cl.name].rsplit("-", 1)[1]))
    for cl in clips:
        pkg = names[cl.name]; f = folders(c, cl)
        if not os.path.exists(f["zip"]):
            bad.append("%s (%s): no package yet" % (pkg, cl.name)); continue
        vdir = os.path.join(bdir, "data", pkg, "v1")
        os.makedirs(vdir, exist_ok=True)
        zp = os.path.join(vdir, pkg + ".zip"); cp = os.path.join(vdir, "comparison.mp4")
        shutil.copyfile(f["zip"], zp)
        with zipfile.ZipFile(zp) as z, z.open("comparison/comparison.mp4") as src, open(cp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        code, out = run_check(checker, zp, cp)
        log.append("=" * 70 + "\n" + out)
        if code != 0:
            bad.append("%s (%s): check_package.py failed" % (pkg, cl.name)); continue
        with open(f["scene"], encoding="utf-8") as fh:
            sd = json.load(fh)
        title = next((v for k, v in titles.items() if k in cl.film), ("", "", ""))
        rows.append({"package_name": pkg, "version": "v1", "zip_sha256": sha256(zp), "zip_size": os.path.getsize(zp),
                     "comparison_sha256": sha256(cp), "movie_title": title[0], "theme": "",
                     "lens": lens_class(sd.get("lens", 35), sd.get("sensor", 36)), "num_characters": people_class(len(sd.get("chars") or {})),
                     "camera_movement": movement(sd.get("cam") or []),
                     "submitted_at": bj.strftime("%Y-%m-%dT%H:%M:%S+08:00")})
        trows.append({"package_name": pkg, "title_zh": title[0], "title_en": title[1], "year": title[2], "work_type": "电影",
                      "year_basis": "首映年份", "source": cl.film, "r2_path": "raw/%s/data/%s/v1/" % (a.batch, pkg)})
        up = os.path.join(a.out, "UPLOAD"); os.makedirs(up, exist_ok=True)     # flat folder: only the zips to upload
        shutil.copyfile(zp, os.path.join(up, pkg + ".zip"))
        print("%-22s ok  %s" % (pkg, cl.name))
    os.makedirs(bdir, exist_ok=True)
    for fn, cols, rr in (("manifest_v1.csv", MANIFEST_COLS, rows), ("movie_titles_v1.csv", TITLE_COLS, trows)):
        with open(os.path.join(bdir, fn), "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rr)
    with open(os.path.join(bdir, "check_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))
    print("%d packages delivered to %s, %d problems" % (len(rows), bdir, len(bad)))
    for b in bad:
        print("  " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Fill the client's task zips (e.g. SOURCE\\zacha-T001.zip) from our finished packages.

  python pipeline\\task_pack.py [--src SOURCE] [--out SOURCE_DONE]

Each task zip holds  T001/meta/task.json, T001/target/T001.mp4 and empty white_model / comparison / model_3d.
task.json "sourceVideo.name" is the clip's file name in the film folder; our package for that clip
(SUBMIT\\<clip>\\) gives white_model, comparison and the .blend, written under the task's own names
(package.white_model / comparison / model_3d in task.json).  task.json and the target are copied unchanged.
The filled zip goes to --out with the same file name; the task zips in --src are not modified."""
import argparse, hashlib, json, os, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from agent import config                      # noqa: E402
from agent.clips import Library, folders      # noqa: E402


def md5(data):
    return hashlib.md5(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "SOURCE"))
    ap.add_argument("--out", default=os.path.join(ROOT, "SOURCE_DONE"))
    a = ap.parse_args()
    c = config.load()
    lib = Library(c)
    os.makedirs(a.out, exist_ok=True)
    bad = 0
    for zn in sorted(f for f in os.listdir(a.src) if f.lower().endswith(".zip")):
        zin = zipfile.ZipFile(os.path.join(a.src, zn))
        tj = [n for n in zin.namelist() if n.endswith("meta/task.json")][0]
        task = json.loads(zin.read(tj).decode("utf-8"))
        pk, vid = task["package"], task["sourceVideo"]["name"]
        target = zin.read(pk["target"])
        hits = [cl for cl in lib.clips if os.path.basename(cl.video) == vid and os.path.getsize(cl.video) == len(target)]
        if len(hits) != 1:
            print("%s: %d clips match %s" % (zn, len(hits), vid)); bad += 1; continue
        cl = hits[0]
        with open(cl.video, "rb") as fh:
            same = md5(fh.read()) == md5(target)
        sub = folders(c, cl)["submit"]
        mine = {"white_model": os.path.join(sub, "white_model", cl.name + ".mp4"),
                "comparison": os.path.join(sub, "comparison", cl.name + ".mp4"),
                "model_3d": os.path.join(sub, "model_3d", cl.name + "blend.blend")}
        missing = [k for k, p in mine.items() if not os.path.exists(p)]
        if missing or not same:
            print("%s (%s): %s" % (zn, cl.name, "missing " + ",".join(missing) if missing else "target differs from the clip"))
            bad += 1; continue
        tmp = os.path.join(a.out, zn + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zo:
            for info in zin.infolist():                       # keep the task's folders, task.json and target as they are
                if info.is_dir() or info.filename in (tj, pk["target"]):
                    zo.writestr(info, zin.read(info.filename))
            for k, p in mine.items():
                zo.write(p, pk[k], compress_type=zipfile.ZIP_STORED if k != "model_3d" else zipfile.ZIP_DEFLATED)
        os.replace(tmp, os.path.join(a.out, zn))
        print("%s <- %s" % (zn, cl.name))
    print("done, %d problems" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

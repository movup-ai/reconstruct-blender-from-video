# -*- coding: utf-8 -*-
"""Edit a clip's detected tracks (marks pose.json); a backup pose.before_trackedit.json is kept once.

  python pipeline\\track_edit.py <package|clip> show A [B]            x-centre / height of the boxes every 5 frames
  python pipeline\\track_edit.py <package|clip> swap A B FROM [TO]    exchange tracks A and B for frames FROM..TO
  python pipeline\\track_edit.py <package|clip> move A B FROM [TO]    frames FROM..TO of track A become track B
  python pipeline\\track_edit.py <package|clip> drop A FROM [TO]      delete frames FROM..TO of track A
  python pipeline\\track_edit.py <package|clip> hold A FROM TO        repeat A's last frame before FROM over FROM..TO
  python pipeline\\track_edit.py <package|clip> fill A FROM TO        interpolate A's marks over the gap FROM..TO
After editing, refit the clip (run_pipeline --stage fit) and rebuild its set."""
import json, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from agent import config
from agent.clips import Library, folders

sys.stdout.reconfigure(encoding="utf-8")
c = config.load(); lib = Library(c)
with open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8") as fh:
    names = json.load(fh)
inv = {v: k for k, v in names.items()}
a = sys.argv[1:]
clip = inv.get(a[0], a[0]); cl = next(x for x in lib.clips if x.name == clip)
pp = folders(c, cl)["pose"]
with open(pp, encoding="utf-8") as fh:
    P = json.load(fh)
T = {str(p["id"]): p for p in P["people"]}


def rng(i):
    f0 = int(a[i]); f1 = int(a[i + 1]) if len(a) > i + 1 else 10 ** 6
    return f0, f1


def get(t):
    if t not in T:
        T[t] = {"id": int(t), "frames": {}, "boxes": {}}; P["people"].append(T[t])
    return T[t]


op = a[1]
if op == "show":
    ids = a[2:]
    allf = sorted({int(f) for t in ids for f in T[t]["boxes"]})
    for f in allf[::5]:
        row = []
        for t in ids:
            b = T[t]["boxes"].get(str(f))
            row.append("%s: x %5.0f h %4.0f" % (t, (b[0] + b[2]) / 2, b[3] - b[1]) if b else "%s: -" % t)
        print("f%-4d  " % f + "   ".join(row))
    sys.exit(0)
bk = pp.replace("pose.json", "pose.before_trackedit.json")
if not os.path.exists(bk):
    shutil.copyfile(pp, bk)
if op in ("swap", "move"):
    A, B = get(a[2]), get(a[3]); f0, f1 = rng(4)
    for key in ("frames", "boxes"):
        fa = {k: v for k, v in A[key].items() if f0 <= int(k) <= f1}
        fb = {k: v for k, v in B[key].items() if f0 <= int(k) <= f1}
        for k in fa: A[key].pop(k)
        if op == "swap":
            for k in fb: B[key].pop(k)
            A[key].update(fb)
        B[key].update(fa)
        A[key] = dict(sorted(A[key].items(), key=lambda t: int(t[0]))); B[key] = dict(sorted(B[key].items(), key=lambda t: int(t[0])))
elif op == "hold":                                   # hold A F0 F1: repeat A's last frame before F0 up to F1 (lies / stands still)
    A = get(a[2]); f0, f1 = rng(3)
    last = max(int(k) for k in A["boxes"] if int(k) < f0)
    for f in range(f0, f1 + 1):
        for key in ("frames", "boxes"):
            if str(last) in A[key]:
                A[key][str(f)] = A[key][str(last)]
elif op == "fill":                                   # fill A F0 F1: interpolate A's marks / box over the gap F0..F1
    A = get(a[2]); f0, f1 = rng(3)
    ks = sorted(int(k) for k in A["boxes"])
    lo = max(k for k in ks if k < f0); hi = min(k for k in ks if k > f1)
    import numpy as np
    for key in ("frames", "boxes"):
        va, vb = np.array(A[key][str(lo)], float), np.array(A[key][str(hi)], float)
        for f in range(f0, f1 + 1):
            t = (f - lo) / float(hi - lo)
            v = (1 - t) * va + t * vb
            if key == "frames":
                v[:, 2] = np.minimum(va[:, 2], vb[:, 2]) * 0.8
            A[key][str(f)] = np.round(v, 2).tolist()
    for key in ("frames", "boxes"):
        A[key] = dict(sorted(A[key].items(), key=lambda t: int(t[0])))
elif op == "drop":
    A = get(a[2]); f0, f1 = rng(3)
    for key in ("frames", "boxes"):
        for k in [k for k in A[key] if f0 <= int(k) <= f1]:
            A[key].pop(k)
P["people"] = [p for p in P["people"] if p["boxes"]]
with open(pp, "w", encoding="utf-8") as fh:
    json.dump(P, fh, ensure_ascii=False)
print("ok:", " ".join(a[1:]), "->", {str(p["id"]): len(p["boxes"]) for p in P["people"]})

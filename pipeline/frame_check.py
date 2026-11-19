# -*- coding: utf-8 -*-
"""Check EVERY frame of a clip: each fitted mannequin projected through the solved camera against the detected body
points of its track, plus people without a mannequin and mannequins without a person.

  python pipeline\\frame_check.py [package ...] [--films 隐剑] [--err 0.12] [--csv out.csv]

Per character and frame: error = median distance of the 17 projected rig points to the confident detected points,
divided by the person's box height.  Frames above --err (default 12% of the body height) are mismatched; they are
grouped into stretches.  Also per body part (head / arms / legs / torso) so the report says what is off.
Uncovered: a detected track with many confident frames that has no fitted character (role ignore / not fitted),
listed with its frames and box, so it can be checked (missing person) or left (false detection)."""
import argparse, csv, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from common import cfg, jload
from build_env import Cam
from compare import rig_uv
from run_pipeline import marks_dir

PARTS = {"head": [0, 1, 2, 3, 4], "arms": [7, 8, 9, 10], "torso": [5, 6, 11, 12], "legs": [13, 14, 15, 16]}


def stretches(fr, gap=3):
    out = []
    for f in fr:
        if out and f <= out[-1][1] + gap:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def check_clip(clip, pkg, a, c):
    work = os.path.join(c["work_root"], clip)
    data = jload(os.path.join(work, "scene_data.json")); ci = jload(os.path.join(work, "clip_info.json"))
    pose = jload(os.path.join(marks_dir(ci["video"], c), "pose.json"))
    y0 = ci["y0"]; cam = Cam(data)
    tracks = {p["id"]: p for p in pose["people"]}
    N = data["frames"]; rows = []; lines = []
    fitted = set()
    for kind, ch in data["chars"].items():
        p = tracks.get(ch["track_id"]); fitted.add(ch["track_id"])
        bad, n, errs, partbad = [], 0, [], {k: 0 for k in PARTS}
        for f in range(N):
            if not ch["visible"][f] or p is None or str(f) not in p["frames"]:
                continue
            k = np.array(p["frames"][str(f)], float); k[:, 1] -= y0
            ok = k[:, 2] > 0.35
            if ok.sum() < 5:
                continue
            uv = rig_uv(ch, cam, f)
            b = p["boxes"].get(str(f)); H = max(40.0, (b[3] - b[1]) if b else np.ptp(k[ok, 1]))
            d = np.linalg.norm(uv - k[:, :2], axis=1) / H
            e = float(np.median(d[ok])); n += 1; errs.append(e)
            if e > a.err:
                bad.append(f)
                for nm, idx in PARTS.items():
                    m = [i for i in idx if ok[i]]
                    if m and float(np.median(d[m])) > a.err * 1.3:
                        partbad[nm] += 1
        st = stretches(bad)
        worst = sorted(st, key=lambda s: -(s[1] - s[0]))[:6]
        rows.append({"package": pkg, "char": kind, "colour": ch["colour"], "track": ch["track_id"], "frames_checked": n,
                     "bad_frames": len(bad), "bad_pct": round(100.0 * len(bad) / max(1, n), 1),
                     "median_err": round(float(np.median(errs)), 3) if errs else None,
                     "parts": ",".join("%s:%d" % (k, v) for k, v in partbad.items() if v),
                     "stretches": " ".join("%d-%d" % tuple(s) for s in sorted(worst))})
        lines.append("  %-7s %-6s track %-3s checked %3d frames, mismatched %3d (%4.1f%%) median err %s  parts[%s]  %s" % (
            kind, ch["colour"], ch["track_id"], n, len(bad), rows[-1]["bad_pct"], rows[-1]["median_err"], rows[-1]["parts"], rows[-1]["stretches"]))
    for tid, p in tracks.items():
        if tid in fitted:
            continue
        good = [int(f) for f, k in p["frames"].items() if (np.array(k)[:, 2] > 0.35).sum() >= 8]
        if len(good) >= max(8, 0.1 * N):
            B = np.median(np.array([p["boxes"][str(f)] for f in good if str(f) in p["boxes"]]), 0)
            lines.append("  UNFITTED track %s: %d confident frames %d-%d, box x %.0f-%.0f h %.0f  (missing person?)" % (
                tid, len(good), min(good), max(good), B[0], B[2], B[3] - B[1]))
            rows.append({"package": pkg, "char": "-", "colour": "-", "track": tid, "frames_checked": len(good), "bad_frames": len(good),
                         "bad_pct": 100.0, "median_err": None, "parts": "unfitted", "stretches": "%d-%d" % (min(good), max(good))})
    print("=== %s  (%s)  %d frames" % (pkg, clip, N)); print("\n".join(lines))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pkgs", nargs="*"); ap.add_argument("--films", default="隐剑")
    ap.add_argument("--err", type=float, default=0.12); ap.add_argument("--csv", default=os.path.join(ROOT, "Work_auto", "frame_check.csv"))
    a = ap.parse_args(); c = cfg()
    sys.stdout.reconfigure(encoding="utf-8")
    with open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8") as fh:
        names = json.load(fh)
    items = sorted(((k, v) for k, v in names.items() if (not a.pkgs or v in a.pkgs) and a.films in k), key=lambda t: int(t[1].rsplit("-", 1)[1]))
    rows = []
    for clip, pkg in items:
        rows += check_clip(clip, pkg, a, c)
    with open(a.csv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("csv:", a.csv)


if __name__ == "__main__":
    main()

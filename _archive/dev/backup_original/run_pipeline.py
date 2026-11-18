# -*- coding: utf-8 -*-
"""
Automatic white-model pipeline:  video  ->  finished submission folder + zip, no manual steps.

  python run_pipeline.py VIDEO_OR_FOLDER [...] [--stage fit|blend|all] [--force] [--device cpu|cuda]

  stage fit   : marks (automark), camera, ground, 3D people, grey set, preview.jpg  (no Blender)
  stage blend : fit + Blender build/render              (no videos)
  stage all   : everything, then white_model / comparison videos, meta.jsonl and the zip
Uses output\\<clip>\\checked.json when a checker saved one, otherwise the automatic marks.
"""
import os, sys, re, glob, json, time, subprocess, argparse, traceback
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import *
from camera_solve import solve as solve_camera
from ground_solve import World, solve_ground
from fit_people import fit_all
from scene_depth import build_set
from rig import Rig, JOINTS
import package as pkg

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
PALETTE = {"ground": [170, 170, 170], "set": [140, 140, 140], "far": [165, 165, 167], "backdrop": [198, 198, 200],
           "red": [205, 62, 60], "blue": [58, 104, 201], "green": [70, 170, 90], "yellow": [225, 200, 60],
           "purple": [150, 90, 190], "pink": [220, 110, 170], "orange": [221, 135, 50], "marker": [52, 190, 200]}
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

def clip_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)\s*月\s*(\d+)\s*日\s*[\(（]\s*(\d+)\s*[\)）]", stem)
    if m:
        return "%s_%s_%s" % m.groups()
    return re.sub(r'[\\/:*?"<>|\s]+', "_", stem).strip("_") or "clip"

def find_marks(video, c, device):
    stem = os.path.splitext(os.path.basename(video))[0]
    out = os.path.join(AGENT, "output")
    hits = [p for p in glob.glob(os.path.join(out, "**", "pose.json"), recursive=True)
            if os.path.basename(os.path.dirname(p)) == stem]
    if not hits:
        log("  no marks yet - running automark")
        cmd = [sys.executable, os.path.join(AGENT, "automark.py"), video, "--out", out, "--device", device]
        if c.get("source_root") and os.path.abspath(video).lower().startswith(os.path.abspath(c["source_root"]).lower()):
            cmd += ["--base", c["source_root"]]
        subprocess.run(cmd, check=True)
        hits = [p for p in glob.glob(os.path.join(out, "**", "pose.json"), recursive=True)
                if os.path.basename(os.path.dirname(p)) == stem]
    pose_p = hits[0]
    d = os.path.dirname(pose_p)
    chk = sorted(glob.glob(os.path.join(d, "checked*.json")), key=os.path.getmtime)
    return pose_p, (chk[-1] if chk else None)

def keys_at(ch, f):
    r = ch["root"][f]
    return (r[0], r[1], r[2]), r[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f]

def preview(frames, y0, world, chars, people, meshes, path, n=6):
    N = len(frames); Hp = world.cam.H
    fr = np.linspace(0, N - 1, n).astype(int)
    tiles = []
    track_of = {ch["track_id"]: kind for kind, ch in chars.items()}
    for f in fr:
        img = frames[f][y0:y0 + Hp].copy()
        ov = img.copy()
        for m in meshes:
            V = np.array(m["v"], float)
            if len(V) == 0:
                continue
            uv, z = world.project(V, f)
            col = (200, 200, 200) if m["colour"] != "ground" else (120, 220, 120)
            for face in m["f"][:: max(1, len(m["f"]) // 1500)]:
                if all(z[i] > 0.2 for i in face):
                    cv2.polylines(ov, [uv[face].astype(np.int32)], True, col, 1, cv2.LINE_AA)
        img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
        for pid, (fs, K) in people.items():
            if f in fs:
                k = K[fs.index(f)]
                for x, y, s in k:
                    if s > 0.3:
                        cv2.circle(img, (int(x), int(y - y0)), 4, (0, 0, 255), -1)
        for kind, ch in chars.items():
            if not ch["visible"][f]:
                continue
            rig = Rig(ch["height"], ch.get("style", "classic"))
            P = rig.coco_points(rig.fk(*keys_at(ch, f)))
            uv, _ = world.project(P, f)
            colr = tuple(int(v) for v in PALETTE[ch["colour"]][::-1])
            for a, b in BONES:
                cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), colr, 3, cv2.LINE_AA)
            cv2.putText(img, kind, tuple(uv[0].astype(int) + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colr, 2)
        cv2.putText(img, "frame %d" % f, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    imwrite_u(path, np.vstack(rows), 85)

def run_clip(video, c, args):
    name = clip_name(video)
    work = os.path.join(c["work_root"], name); c["_work"] = work
    os.makedirs(work, exist_ok=True)
    zp = os.path.join(c["submit_root"], name + ".zip")
    if args.stage == "all" and os.path.exists(zp) and not args.force:
        log("skip (zip exists):", name); return {"clip": name, "status": "skipped"}
    t0 = time.time()
    log("=== %s  (%s)" % (name, video))
    if args.stage == "render":                             # re-render an edited scene (no new fitting)
        return render_only(name, video, work, c, args, t0)
    pose_p, chk_p = find_marks(video, c, args.device)
    pose = jload(pose_p); checked = jload(chk_p) if chk_p else None
    log("  marks: %s%s" % (pose_p, "  + checked: " + chk_p if chk_p else "  (not checked by a person)"))
    frames, fps = read_frames(video)
    N = len(frames); Hfull, W = frames[0].shape[:2]
    _, y0, _, Hp = pose["picture_area"]
    log("  camera ...")
    camj = solve_camera(frames, pose, c); jsave(os.path.join(work, "camera.json"), camj)
    people = coco_people(pose, checked)
    people = {pid: (list(fs), K) for pid, (fs, K) in people.items()}
    cam = Camera(W, Hp, c["lens_mm"])
    u, hc, ginfo = solve_ground(people, cam, y0, Hp, c)
    world = World(cam, camj["R"], u, hc)
    log("  ground: %s" % json.dumps(ginfo))
    log("  people ...")
    chars, prep = fit_all(world, y0, people, N, c)
    boxes = {}
    for p in pose["people"]:
        for f, b in p.get("boxes", {}).items():
            boxes.setdefault(int(f), []).append(b)
    log("  set ...")
    meshes, sinfo = build_set(frames, y0, world, chars, boxes, c, fake=c.get("_fake_depth", False))
    data = {"name": name, "submit_dir": os.path.join(c["submit_root"], name), "work_dir": work,
            "res": [W, Hp], "fps": int(round(fps)), "frames": N, "render_scale": c["render_scale"],
            "lens": c["lens_mm"], "sensor": 36.0, "cam": [world.blender_cam(i) for i in range(N)],
            "palette": PALETTE, "sky": [228, 229, 232], "exposure": 1.2, "gain": {"ground": 2.2},
            "chars": chars, "meshes": meshes, "props": []}
    dp = os.path.join(work, "scene_data.json"); jsave(dp, data)
    preview(frames, y0, world, chars, people, meshes, os.path.join(work, "preview.jpg"))
    warns = list(camj["warnings"]) + [x for x in (ginfo.get("warning"), sinfo.get("warning")) if x]
    if not chars:
        warns.append("no people found")
    rep = {"clip": name, "video": video, "checked": bool(chk_p), "frames": N, "people": prep, "ground": ginfo,
           "camera_rotation_deg": round(camj["max_rotation_deg"], 2), "warnings": warns}
    info = {"W": W, "Hfull": Hfull, "y0": y0, "Hpic": Hp, "frames": N, "fps": fps}
    jsave(os.path.join(work, "clip_info.json"), dict(info, video=video, desc_path=os.path.join(os.path.dirname(pose_p), "desc.txt")))
    if args.stage in ("blend", "all"):
        blender_render(c, work, dp, N)
    if args.stage == "all":
        rep.update(make_zip(name, video, work, c))
    rep["seconds"] = round(time.time() - t0, 1); rep["status"] = "ok"
    jsave(os.path.join(work, "report.json"), rep)
    log("  done in %.0fs%s" % (rep["seconds"], ("   WARNINGS: " + " | ".join(warns)) if warns else ""))
    return rep

def blender_render(c, work, dp, N, extra=("render",)):
    be = blender_exe(c)
    if not be:
        raise RuntimeError("Blender not found - set the Blender path in Settings (config.json)")
    log("  blender build + render (%d frames) ..." % N)
    done = os.path.join(work, "render_done.txt")
    if os.path.exists(done): os.remove(done)
    with open(os.path.join(work, "blender_log.txt"), "w", encoding="utf-8", errors="ignore") as lf:
        subprocess.run([be, "-b", "-P", os.path.join(HERE, "blender_build.py"), "--", dp] + list(extra), stdout=lf, stderr=subprocess.STDOUT)
    if not os.path.exists(done):
        raise RuntimeError("Blender did not finish - see %s" % os.path.join(work, "blender_log.txt"))

def make_zip(name, video, work, c):
    log("  videos + zip ...")
    ci = jload(os.path.join(work, "clip_info.json"))
    desc = open(ci["desc_path"], encoding="utf-8").read().strip() if os.path.exists(ci["desc_path"]) else ""
    sub, zp = pkg.package(name, video, os.path.join(work, "render"), c["submit_root"], ci, c, desc)
    return {"submission": sub, "zip": zp}

def render_only(name, video, work, c, args, t0):
    dp = os.path.join(work, "scene_data.json")
    if not os.path.exists(dp):
        raise RuntimeError("no scene yet - build this clip first")
    data = jload(dp)
    data["submit_dir"] = os.path.join(c["submit_root"], name); data["work_dir"] = work
    jsave(dp, data)
    blender_render(c, work, dp, data["frames"])
    rep = {"clip": name, "video": video, "status": "ok", "warnings": []}
    rep.update(make_zip(name, video, work, c))
    rep["seconds"] = round(time.time() - t0, 1)
    log("  done in %.0fs" % rep["seconds"])
    return rep

def collect(inputs):
    vids = []
    for p in inputs:
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                vids += [os.path.join(r, f) for f in sorted(fs) if f.lower().endswith(VIDEO_EXT)]
        elif p.lower().endswith(VIDEO_EXT):
            vids.append(p)
    return vids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--stage", default="all", choices=["fit", "blend", "all", "render"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    c = cfg(); c["device"] = args.device
    vids = collect(args.inputs or [c.get("source_root", r"F:\Blender task\Source")])
    log("%d clips" % len(vids))
    os.makedirs(c["work_root"], exist_ok=True)
    rows = []
    for v in vids:
        try:
            rows.append(run_clip(v, dict(c), args))
        except Exception as e:
            traceback.print_exc()
            rows.append({"clip": clip_name(v), "video": v, "status": "FAILED", "warnings": [str(e)[:300]]})
    rp = os.path.join(c["work_root"], "report_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))
    with open(rp, "w", encoding="utf-8-sig") as fh:
        fh.write("clip,status,checked,people,median_error_px,seconds,warnings,video\n")
        for r in rows:
            pe = r.get("people", [])
            fh.write('%s,%s,%s,%d,%s,%s,"%s","%s"\n' % (r["clip"], r["status"], r.get("checked", ""), len(pe),
                     "/".join(str(p["median_error_px"]) for p in pe), r.get("seconds", ""), " | ".join(r.get("warnings", [])).replace('"', "'"), r.get("video", "")))
    log("\nreport: " + rp)
    bad = [r for r in rows if r["status"] == "FAILED"]
    log("%d ok, %d skipped, %d failed" % (sum(r["status"] == "ok" for r in rows), sum(r["status"] == "skipped" for r in rows), len(bad)))

if __name__ == "__main__":
    main()

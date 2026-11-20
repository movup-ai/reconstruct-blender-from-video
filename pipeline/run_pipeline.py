# -*- coding: utf-8 -*-
"""
Automatic white-model pipeline:  video  ->  finished submission folder + zip, no manual steps.

  python run_pipeline.py VIDEO_OR_FOLDER [...] [--stage fit|blend|all] [--force] [--device cpu|cuda]

  stage fit   : marks (automark), camera, ground, 3D people, grey set, preview.jpg  (no Blender)
  stage blend : fit + Blender build/render              (no videos)
  stage all   : everything, then white_model / comparison videos, meta.jsonl and the zip
Uses output\\<clip>\\checked.json when a checker saved one, otherwise the automatic marks.
"""
import os, sys, re, glob, hashlib, json, time, subprocess, argparse, traceback
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import *
from camera_solve import solve as solve_camera
from ground_solve import World, solve_ground
from camera_estimate import estimate_shot
from shots import split_shots as find_shots
from fit_people import fit_all
from rig import Rig, JOINTS
import package as pkg
import quality

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
PALETTE = {"ground": [242, 242, 242], "set": [178, 178, 180], "far": [197, 197, 199], "backdrop": [205, 205, 207],
           "ridge": [160, 160, 163], "rock": [138, 138, 138], "tree": [172, 172, 170], "dark": [112, 112, 112], "light": [218, 218, 218],   # tree (vegetation) lighter than rock
           "cabin": [98, 98, 98], "roof": [133, 133, 133], "fence": [146, 146, 146], "mount": [197, 197, 199],
           "red": [205, 62, 60], "blue": [58, 104, 201], "green": [70, 170, 90], "yellow": [225, 200, 60],
           "purple": [150, 90, 190], "pink": [220, 110, 170], "orange": [221, 135, 50], "marker": [255, 255, 255],   # v2.1: eye + chest bands pure white "cyan": [80, 205, 235],
           "brown": [130, 85, 50], "white": [245, 245, 245], "grey": [150, 150, 150]}
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

def film_short(path, c=None):
    """short film name from the folder under source_root ("人生一世￥13" -> "人生一世")"""
    root = os.path.abspath((c or cfg())["source_root"])
    rel = os.path.relpath(os.path.abspath(path), root)
    top = rel.split(os.sep)[0] if not rel.startswith("..") and os.sep in rel else os.path.basename(os.path.dirname(path))
    if not rel.startswith("..") and rel.count(os.sep) >= 2:      # batch folder with one sub-folder per film:
        top = rel.split(os.sep)[-2]                               # 11月17日原始素材191秒\Dragon\x.mp4 -> Dragon
    m = re.match(r"[\u4e00-\u9fff]+", top)
    return m.group(0) if m else re.sub(r"[^\w]+", "_", top)[:20]

def clip_name(path, c=None):
    """unique per clip: <film>_<file stem>  (several films have a clip called 11月12日(1))"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'[\\/:*?"<>|\s]+', "_", film_short(path, c) + "_" + stem).strip("_") or "clip"

def marks_base(c):
    """where automark writes: output\\ for the project's SOURCE folder; output\\_sources\\<folder>_<id>\\ for any
    other source folder picked in the app, so two folders with the same file names never share joints"""
    out = c.get("marks_root") or os.path.join(AGENT, "output")
    root = c.get("source_root")
    if root and os.path.normcase(os.path.abspath(root)) != os.path.normcase(os.path.join(AGENT, "SOURCE")):
        full = os.path.abspath(root)
        key = re.sub(r'[\\/:*?"<>|\s]+', "_", os.path.basename(full.rstrip("\\/")))[:30] or "folder"
        out = os.path.join(out, "_sources", "%s_%s" % (key, hashlib.sha1(os.path.normcase(full).encode("utf-8")).hexdigest()[:6]))
    return out

def marks_dir(video, c):
    """automark mirrors the Source layout under marks_base() (--base source_root)"""
    out = marks_base(c)
    stem = os.path.splitext(os.path.basename(video))[0]
    root = c.get("source_root")
    rel = os.path.relpath(os.path.dirname(os.path.abspath(video)), root) if root else ""
    rel = "" if rel.startswith("..") or rel == "." else rel
    return os.path.join(out, rel, stem)

def load_review(video, c):
    p = os.path.join(marks_dir(video, c), "review.json")
    return jload(p) if os.path.exists(p) else None

def find_marks(video, c, device):
    out = marks_base(c)
    d = marks_dir(video, c)
    if not os.path.exists(os.path.join(d, "pose.json")):
        log("  no marks yet - running automark")
        cmd = [sys.executable, os.path.join(AGENT, "automark.py"), video, "--out", out, "--device", device]
        if c.get("source_root") and os.path.abspath(video).lower().startswith(os.path.abspath(c["source_root"]).lower()):
            cmd += ["--base", c["source_root"]]
        subprocess.run(cmd, check=True)
    pose_p = os.path.join(d, "pose.json")
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

def stitch_tracks(pose, shots, max_gap=30):
    """one person = one track: the detector loses people when they bend, turn or pass behind something and finds them
    again as 'new' people, sometimes with two overlapping detections.  Tracks are joined when one picks up where the
    other stopped (same place and size, within max_gap frames) or when both follow the same body at the same time.
    Never across a cut.  Returns {absorbed id: kept id}"""
    shot_of = {}
    for si, (a, b) in enumerate(shots):
        for f in range(a, b):
            shot_of[f] = si
    T = {p["id"]: p for p in pose["people"]}
    mapping = {}

    def span(p):
        fs = sorted(int(f) for f in p["boxes"])
        return fs[0], fs[-1]

    def box(p, f):
        b = p["boxes"].get(str(f)); return None if b is None else np.array(b[:4], float)

    def overlap(a, b):
        x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        return inter / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))

    while True:
        best = None
        ids = sorted(T)
        for i in ids:
            for j in ids:
                if i >= j:
                    continue
                A, B = T[i], T[j]
                (a0, a1), (b0, b1) = span(A), span(B)
                if a0 > b0:
                    A, B, (a0, a1), (b0, b1) = B, A, (b0, b1), (a0, a1)
                common = sorted(set(A["boxes"]) & set(B["boxes"]), key=int)
                if common:                                             # both at once: the same body twice?
                    ov = np.mean([overlap(box(A, int(f)), box(B, int(f))) for f in common])
                    if ov < 0.45 or len(common) > 0.8 * min(len(A["boxes"]), len(B["boxes"])):
                        continue
                    from dedupe_roles import same_skeleton           # overlapping boxes, but two bodies (one walking
                    if np.mean([same_skeleton(A["frames"].get(f), B["frames"].get(f), A["boxes"][f])   # behind the
                                for f in common]) < 0.5:              # other): not the same person
                        continue
                    cost = 1.0 - ov
                else:                                                  # one after the other
                    gap = b0 - a1
                    if gap <= 0 or gap > max_gap or shot_of.get(a1) != shot_of.get(b0):
                        continue
                    ea, sb = box(A, a1), box(B, b0)
                    ha, hb = ea[3] - ea[1], sb[3] - sb[1]
                    if not 0.5 < hb / max(ha, 1e-6) < 2.0:
                        continue
                    dist = np.hypot((ea[0] + ea[2] - sb[0] - sb[2]) / 2, (ea[1] + ea[3] - sb[1] - sb[3]) / 2)
                    if dist > 0.8 * max(ha, hb):
                        continue
                    cost = dist / max(ha, hb) + gap / float(max_gap)
                if best is None or cost < best[0]:
                    best = (cost, A["id"], B["id"])
        if best is None:
            break
        _, ia, ib = best
        keep, drop = min(ia, ib), max(ia, ib)                          # the lower number = the bigger, main person
        K, Dr = T[keep], T.pop(drop)
        for f, kp in Dr["frames"].items():                             # where both exist, keep the surer marks
            if f not in K["frames"] or np.mean(np.array(kp)[:, 2]) > np.mean(np.array(K["frames"][f])[:, 2]):
                K["frames"][f] = kp
                if f in Dr["boxes"]:
                    K["boxes"][f] = Dr["boxes"][f]
        for k_, v_ in list(mapping.items()):
            if v_ == drop:
                mapping[k_] = keep
        mapping[drop] = keep
    pose["people"] = [T[i] for i in sorted(T)]
    return mapping


def edit_tracks(pose, spec):
    """hints.json "tracks": {"assign": [[from, first, last, to], ...], "drop": [[track, first, last], ...]}.
    Every move reads the tracks as they were before any of them (so two tracks can swap a stretch).  In memory only:
    pose.json is not changed.  Returns log lines."""
    T = {p["id"]: p for p in pose["people"]}
    snap = {i: {"frames": dict(p["frames"]), "boxes": dict(p.get("boxes", {}))} for i, p in T.items()}
    msgs = []
    moved = []
    for a in spec.get("assign") or []:
        src, f0, f1, dst = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        if src not in snap:
            msgs.append("assign: no track %d" % src); continue
        fr = [f for f in snap[src]["frames"] if f0 <= int(f) <= f1]
        moved.append((src, dst, fr))
    for src, dst, fr in moved:                                       # take the frames away first ...
        for f in fr:
            T[src]["frames"].pop(f, None); T[src].get("boxes", {}).pop(f, None)
    for src, dst, fr in moved:                                       # ... then give them to their new person
        if dst not in T:
            T[dst] = {"id": dst, "frames": {}, "boxes": {}}
        for f in fr:
            T[dst]["frames"][f] = snap[src]["frames"][f]
            if f in snap[src]["boxes"]:
                T[dst].setdefault("boxes", {})[f] = snap[src]["boxes"][f]
        msgs.append("track %d frames %s-%s -> track %d (%d frames)" % (src, min(map(int, fr)) if fr else "-",
                                                                     max(map(int, fr)) if fr else "-", dst, len(fr)))
    for d in spec.get("drop") or []:
        t, f0, f1 = int(d[0]), int(d[1]), int(d[2])
        if t in T:
            fr = [f for f in T[t]["frames"] if f0 <= int(f) <= f1]
            for f in fr:
                T[t]["frames"].pop(f, None); T[t].get("boxes", {}).pop(f, None)
            msgs.append("track %d frames %d-%d dropped (%d frames)" % (t, f0, f1, len(fr)))
    pose["people"] = [T[i] for i in sorted(T) if T[i]["frames"]]
    return msgs


def apply_stitch(pose_p, pose, mapping, checked_p, review_p):
    """save the joined tracks so the review page shows each person once; move roles and joint fixes along"""
    if not mapping:
        return
    raw = pose_p.replace("pose.json", "pose_raw.json")
    if not os.path.exists(raw):
        shutil.copyfile(pose_p, raw)
    pose["stitched"] = {str(k): v for k, v in mapping.items()}
    jsave(pose_p, pose)
    if checked_p and os.path.exists(checked_p):
        chk = jload(checked_p)
        for f, ppl in chk.get("frames", {}).items():
            seen = set()
            for p in ppl:
                p["id"] = mapping.get(p["id"], p["id"])
            chk["frames"][f] = [p for p in ppl if not (p["id"] in seen or seen.add(p["id"]))]
        jsave(checked_p, chk)
    if review_p and os.path.exists(review_p):
        rv = jload(review_p)
        ppl = rv.get("people", {})
        rank = {"main": 0, "second": 1, "third": 2, "fourth": 3, "extra": 4, "ignore": 5}
        for old, new in mapping.items():
            a = ppl.pop(str(old), None)
            if a and (str(new) not in ppl or rank.get(a.get("role"), 9) < rank.get(ppl[str(new)].get("role"), 9)):
                ppl[str(new)] = a                                     # the joined person keeps the more important role
        jsave(review_p, rv)


def steady_camera(camj, c):
    """camera rotations from frame-to-frame tracking carry small random errors (shake that is not in the film):
    a camera that hardly turns is locked still, a moving one is smoothed over a few frames (real pans keep their shape)"""
    from scipy.spatial.transform import Rotation
    from scipy.ndimage import gaussian_filter1d
    R = np.array(camj["R"], float)
    if len(R) < 3:
        return camj
    rv = Rotation.from_matrix(R).as_rotvec()                          # frame-0 camera -> frame-i camera
    total = float(np.degrees(np.max(np.linalg.norm(rv, axis=1))))
    sm4 = gaussian_filter1d(rv, 4.0, axis=0, mode="nearest")
    path = float(np.degrees(np.sum(np.linalg.norm(np.diff(sm4, axis=0), axis=1))))
    net = float(np.degrees(np.linalg.norm(sm4[-1] - sm4[0])))
    direct = net / max(path, 1e-6)                                    # 1 = turns one way (a pan), ~0 = wanders back
    if total < c.get("camera_still_deg", 0.6) or (total < 4.0 and direct < 0.5):
        camj["R"] = [np.eye(3).tolist() for _ in R]; camj["max_rotation_deg"] = 0.0
        camj["steady"] = "locked (only tracking noise: %.2f deg, wandering back and forth)" % total
        return camj
    sig = float(c.get("camera_smooth_frames", 4.0 if direct < 0.8 else 2.0))
    sm = gaussian_filter1d(rv, sig, axis=0, mode="nearest")
    sm -= sm[0]                                                       # frame 0 stays the reference
    camj["R"] = Rotation.from_rotvec(sm).as_matrix().tolist()
    jit = float(np.degrees(np.median(np.linalg.norm(np.diff(rv, 2, axis=0), axis=1)))) if len(rv) > 3 else 0.0
    camj["steady"] = "smoothed (%.2f deg typical jitter removed)" % jit
    return camj


def split_shots(frames, y0, Hp, min_len=6):
    """hard cuts from colour-histogram jumps between neighbouring frames -> [(start, end)]"""
    def hist(f):
        hsv = cv2.cvtColor(cv2.resize(f[y0:y0 + Hp], (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(h, h).flatten()
    H = [hist(f) for f in frames]
    G = [cv2.cvtColor(cv2.resize(f[y0:y0 + Hp], (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    d = np.array([cv2.compareHist(H[i - 1], H[i], cv2.HISTCMP_BHATTACHARYYA) for i in range(1, len(H))])
    if not len(d):
        return [(0, len(frames))]
    mad = np.array([float(np.mean(np.abs(G[i] - G[i - 1]))) for i in range(1, len(G))])   # the picture really changes
    med = float(np.median(d))
    cuts = [i + 1 for i in range(len(d)) if d[i] > max(0.35, 4 * med) and mad[i] > max(12.0, 3 * float(np.median(mad)))
            and d[i] >= d[max(0, i - 1)] and d[i] >= d[min(len(d) - 1, i + 1)]]
    bounds = [0]
    for ct in cuts:
        if ct - bounds[-1] >= min_len and len(frames) - ct >= min_len:
            bounds.append(ct)
    bounds.append(len(frames))
    return list(zip(bounds[:-1], bounds[1:]))

def slice_pose(pose, s0, s1):
    out = dict(pose); ppl = []
    for p in pose["people"]:
        fr = {str(int(f) - s0): v for f, v in p["frames"].items() if s0 <= int(f) < s1}
        if not fr:
            continue
        bx = {str(int(f) - s0): v for f, v in p.get("boxes", {}).items() if s0 <= int(f) < s1}
        ppl.append(dict(p, frames=fr, boxes=bx))
    out["people"] = ppl; out["frames"] = s1 - s0
    return out

def slice_checked(chk, s0, s1):
    if not chk:
        return None
    out = dict(chk)
    out["frames"] = {str(int(f) - s0): v for f, v in chk.get("frames", {}).items() if s0 <= int(f) < s1}
    out["checked_frames"] = [f - s0 for f in chk.get("checked_frames", []) if s0 <= f < s1]
    return out

def widen(ch, s0, s1, N):
    """per-shot character keys (length s1-s0) -> whole-clip keys, hidden outside its shot"""
    Ns = s1 - s0
    def ext(lst):
        return [lst[min(max(f - s0, 0), Ns - 1)] for f in range(N)]
    out = dict(ch)
    out["root"] = ext(ch["root"]); out["pelvis_z"] = ext(ch["pelvis_z"])
    out["rot"] = {j: ext(v) for j, v in ch["rot"].items()}
    out["visible"] = [bool(s0 <= f < s1 and ch["visible"][f - s0]) for f in range(N)]
    return out

def stack_previews(paths, out):
    ims = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in paths if os.path.exists(p)]
    if not ims:
        return
    w = min(i.shape[1] for i in ims)
    ims = [cv2.resize(i, (w, int(i.shape[0] * w / i.shape[1]))) for i in ims]
    imwrite_u(out, np.vstack(ims), 85)

def hand_set_path(video, name, c):
    """the clip's hand-built set (env_claude.json, where build_env.py looks for it) or None"""
    for d in (os.path.join(output_dir(), name), marks_dir(video, c)):
        p = os.path.join(d, "env_claude.json")
        if os.path.exists(p):
            return p
    return None

def carry_over(work, name, c, chars, N, sky):
    """without the automatic depth set the fit cannot tell when someone stands on raised ground; build_env.py uses the
    fit's heights (root_z_fit) to put people onto big hand-built surfaces (terraces, steps).  Keep the heights of the
    previous fit of this clip for the same people, and its sky colour, so a re-fit does not change the finished set."""
    prev_p = os.path.join(work, "scene_data.json")
    if not os.path.exists(prev_p) and c.get("_sandbox"):           # sandbox: the real clip's last fit (read only)
        real = dict(DEFAULT_CFG)
        if os.path.exists(CFG_PATH):
            real.update(jload(CFG_PATH))
        prev_p = os.path.join(real["work_root"], name, "scene_data.json")
    if not os.path.exists(prev_p):
        return sky
    try:
        prev = jload(prev_p)
    except Exception:
        return sky
    if prev.get("frames") != N:
        return sky
    kept = []
    for kind, ch in chars.items():
        pc = (prev.get("chars") or {}).get(kind)
        if not pc or pc.get("track_id") != ch.get("track_id") or len(pc.get("root", [])) != N:
            continue
        zs = pc.get("root_z_fit") or [r[2] for r in pc["root"]]
        if len(zs) == N and any(abs(z) > 1e-4 for z in zs):
            ch["root"] = [[r[0], r[1], z, r[3]] for r, z in zip(ch["root"], zs)]   # new lists: widen() shares them
            kept.append(kind)
    if kept:
        log("  heights above the floor kept from the previous fit: %s" % ", ".join(kept))
    if sky is None and not c.get("_set_grey") and prev.get("sky"):
        sky = prev["sky"]
    return sky

def run_clip(video, c, args):
    name = getattr(args, "name", None) or clip_name(video, c)
    work = os.path.join(c["work_root"], name); c["_work"] = work
    os.makedirs(work, exist_ok=True)
    zp = os.path.join(c["submit_root"], name + ".zip")
    if args.stage == "all" and os.path.exists(zp) and not args.force:
        log("skip (zip exists):", name); return {"clip": name, "status": "skipped"}
    review = load_review(video, c)
    if args.require_confirm and args.stage in ("blend", "all", "render") and not (review and review.get("confirmed")):
        log("skip (key points not confirmed yet):", name); return {"clip": name, "status": "waiting"}
    c["_review"] = review or {}
    envp = os.path.join(marks_dir(video, c), "env_ai.json")       # AI environment corrections (agent.ai)
    if os.path.exists(envp) and (review or {}).get("environment", {}).get("use_ai", True):
        c["_env_ai"] = jload(envp).get("shots", {})
    if c.get("_sandbox"):                                          # sandbox: start from copies of the real marks and set files
        real = dict(c); real.pop("marks_root", None)
        real_md = marks_dir(video, dict(real, marks_root=os.path.join(AGENT, "output")))
        for src, dst in ((real_md, marks_dir(video, c)), (os.path.join(AGENT, "output", name), os.path.join(output_dir(), name))):
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
    hp = os.path.join(marks_dir(video, c), "hints.json")          # what the picture cannot show (written by Claude)
    if not os.path.exists(hp):                                     # also accepted next to the set: output\<clip>\hints.json
        hp = os.path.join(output_dir(), name, "hints.json")
    hints = jload(hp) if os.path.exists(hp) else {}
    c["_hints"] = hints
    for k_ in ("lens_mm", "person_height", "smooth_frames", "fit_step"):
        if k_ in hints:
            c[k_] = hints[k_]
    if (hints.get("camera") or {}).get("type") == "static":
        c["_static_camera"] = True
    if review and review.get("camera", {}).get("type") == "static":
        c["_static_camera"] = True
    if review and review.get("camera", {}).get("lens"):
        c["lens_mm"] = float(review["camera"]["lens"])
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
    env = (review or {}).get("environment", {})
    if env.get("set_colours", c.get("set_colours", "grey")) == "grey":   # client style (both samples): light grey set
        c["_set_grey"] = True
    shots = find_shots(frames, y0, Hp) if c.get("camera_estimator", "new") == "new" else split_shots(frames, y0, Hp)
    log("  shots: %s" % ", ".join("%d-%d" % (a, b - 1) for a, b in shots))
    stitched = stitch_tracks(pose, shots)
    if stitched:
        log("  joined broken person tracks: %s" % ", ".join("P%s -> P%s" % kv for kv in sorted(stitched.items())))
        apply_stitch(pose_p, pose, stitched, chk_p, os.path.join(marks_dir(video, c), "review.json"))
        checked = jload(chk_p) if chk_p else None
        review = load_review(video, c); c["_review"] = review or {}
    if (c.get("_hints") or {}).get("tracks"):                     # hand fixes of identity swaps (hints.json "tracks")
        for msg in edit_tracks(pose, c["_hints"]["tracks"]):
            log("  tracks (hints): " + msg)
    # a hand-built set (build_env.py) replaces the automatic depth set anyway: skip building it (and loading the depth model)
    hand_set = hand_set_path(video, name, c) is not None and not getattr(args, "auto_set", False)
    if hand_set:
        log("  hand-built set: %s  (automatic depth set skipped)" % hand_set_path(video, name, c))
    palette = dict(PALETTE); sky = None
    cam_all, chars_all, meshes_all, prep_all, warns, shot_rep, tiles = [None] * N, {}, [], [], [], [], []
    # the clip's lens from the picture (lens_auto.py, MoGe-2) unless it was set by hand (hints.json / review)
    lens_info = None
    rl = (review or {}).get("camera", {}).get("lens")          # the review form is pre-filled with config.json's lens:
    by_hand = "lens_mm" in hints or bool(rl and abs(float(rl) - float(cfg()["lens_mm"])) > 0.05)   # only another value is "by hand"
    if c.get("lens_auto", True) and not by_hand and not c.get("_fake_depth"):
        try:
            import lens_auto
            tl = time.time()
            lens, lens_info = lens_auto.clip_lens(frames, y0, Hp, shots, work, c)
            if lens is None:
                log("  lens: %.1f mm from config.json (%s)" % (c["lens_mm"], lens_info.get("why")))
            else:
                log("  lens: %.1f mm from the picture (config.json has %.1f mm)%s  %.0fs" % (
                    lens, c["lens_mm"], "  [kept from the last fit]" if lens_info.get("cached") else "", time.time() - tl))
                lens_info["config_mm"] = c["lens_mm"]; c["lens_mm"] = round(lens, 1)
                if lens_info.get("warning"):
                    warns.append(lens_info["warning"])
        except Exception as e:                                 # never let the lens stop a clip
            log("  lens: %.1f mm from config.json (measuring failed: %s)" % (c["lens_mm"], str(e)[:200]))
    for si, (s0, s1) in enumerate(shots):
        tag = "_s%d" % (si + 1) if len(shots) > 1 else ""
        c["_shot_tag"] = tag
        sf = frames[s0:s1]; Ns = s1 - s0
        spose = slice_pose(pose, s0, s1); schk = slice_checked(checked, s0, s1)
        log("  [shot %d: frames %d-%d]" % (si + 1, s0, s1 - 1))
        log("  camera ...")
        tp0 = time.time()
        people = coco_people(spose, schk)
        people = {pid: (list(fs), K) for pid, (fs, K) in people.items()}
        people = quality.drop_junk(people, c, fps, log)        # short + unclear tracks: not a person worth fitting
        rvp = (c.get("_review") or {}).get("people", {})       # "Not a person" (statue, poster): not for the camera / floor either
        people = {pid: v for pid, v in people.items() if rvp.get(str(pid), {}).get("role") != "ignore"}
        cam = Camera(W, Hp, c["lens_mm"])
        if c.get("camera_estimator", "new") == "new":      # dense tracking + depth / horizon / people ground (camera_estimate.py)
            camj, u, hc, ginfo = estimate_shot(sf, spose, people, cam, y0, Hp, c)
            log("    camera %.1fs  %s" % (time.time() - tp0, camj.get("steady", "")))
        else:                                              # the earlier solver
            camj = solve_camera(sf, spose, c)
            log("    camera %.1fs" % (time.time() - tp0))
            camj = steady_camera(camj, c)
            log("    camera %s" % camj.get("steady", ""))
            u, hc, ginfo = solve_ground(people, cam, y0, Hp, c)
        if c.get("_static_camera"):                        # reviewer said the camera does not move
            I3 = np.eye(3).tolist()
            camj["R"] = [I3 for _ in camj["R"]]; camj["max_rotation_deg"] = 0.0
            camj["warnings"] = [w for w in camj["warnings"] if "travel" not in w.lower()]
        hcam = (c.get("_hints") or {}).get("camera") or {}
        hcam = dict(hcam, **(hcam.get("shots") or {}).get(str(si + 1), {}))
        if "height" in hcam or "pitch_down_deg" in hcam:       # set by hand: camera height and tilt at the shot's first frame
            from scipy.spatial.transform import Rotation as _Rot
            hc = float(hcam.get("height", hc))
            u = _Rot.from_euler("xz", [np.radians(float(hcam.get("pitch_down_deg", ginfo.get("pitch_deg", 0.0)))),
                                       np.radians(float(hcam.get("roll_deg", 0.0)))]).apply([0, -1.0, 0])
            ginfo = {"source": "hints.json", "camera_height": hc, "pitch_deg": float(hcam.get("pitch_down_deg", 0.0))}
        world = World(cam, camj["R"], u, hc)
        if hcam.get("path"):                                   # camera move set by hand (turn + travel)
            world.set_path(hcam["path"], Ns)
            ginfo["camera_path"] = "hints.json (%d keys)" % len(hcam["path"])
            camj["warnings"] = [w for w in camj["warnings"] if "MOVE" not in w]
        log("  ground: %s" % json.dumps(ginfo))
        log("  people ...")
        tp0 = time.time()
        c["_fps"] = float(fps)
        chars, prep = fit_all(world, y0, people, Ns, c)
        # half-body shot (no feet in the picture): the floor was a guess - put it under the people and fit them again
        dz = None if ("height" in hcam or c.get("floor_from_people", True) is False) else quality.floor_shift(chars, people)
        if dz:
            hc2 = float(np.clip(hc + dz, 0.3, 12.0))
            log("    nobody's feet are in the picture: floor moved %+.2f m to stand the people on it (camera height %.2f -> %.2f m)" % (-dz, hc, hc2))
            world = World(cam, camj["R"], u, hc2)
            if hcam.get("path"):
                world.set_path(hcam["path"], Ns)
            ginfo = dict(ginfo, camera_height=hc2, floor="under the people (no feet in the picture)", camera_height_before=hc)
            hc = hc2
            chars, prep = fit_all(world, y0, people, Ns, c)
        log("    people %.1fs" % (time.time() - tp0))
        boxes = {}
        for p in spose["people"]:
            for f, b in p.get("boxes", {}).items():
                boxes.setdefault(int(f), []).append(b)
        log("  set ...")
        tp0 = time.time()
        if env.get("depth_set") is False or hand_set:
            from scene_depth import flat_set           # no depth model loaded (~2 GB): placeholder ground + backdrop
            meshes, sinfo = flat_set(world, Ns, float(c.get("far_distance", 1500.0))), {}
            if hand_set:
                sinfo["source"] = "hand-built set (env_claude.json); automatic depth set skipped"
        else:
            from scene_depth import build_set
            meshes, sinfo = build_set(sf, y0, world, chars, boxes, c, fake=c.get("_fake_depth", False))
        log("    set %.1fs" % (time.time() - tp0))
        # people whose feet are never in the picture stay on the solved floor: the depth terrain under a half-body
        # person is whatever is behind or below the frame (a wall base, a pedestal) and lifted one by a metre
        def feet_seen(ch):
            fs_, K_ = people.get(ch.get("track_id"), ([], []))
            return not len(fs_) or quality.track_stats(fs_, K_)["feet_seen"] >= quality.LIMITS["feet_seen"]
        grounded = {k: ch for k, ch in chars.items() if feet_seen(ch)}
        if len(grounded) < len(chars) and (sinfo.get("terrain_xyz") or sinfo.get("plane")):
            log("    feet not in the picture: %s kept on the solved floor (not moved onto the set's terrain)" %
                ", ".join(sorted(set(chars) - set(grounded))))
        T = np.array(sinfo.pop("terrain_xyz"), float) if sinfo.get("terrain_xyz") else None
        if T is not None or sinfo.get("plane"):            # uneven / sloping ground: feet on the set's ground
            from scipy.spatial import cKDTree
            from build_env import Cam
            tree = cKDTree(T[:, :2]) if T is not None else None
            pl = sinfo.get("plane")
            def ground_z(r):
                z = r[2]
                if tree is not None:
                    dd, ii = tree.query([r[0], r[1]], k=min(8, len(T)))
                    if np.min(dd) < 3.0:
                        z = float(np.median(T[np.atleast_1d(ii), 2]))
                if pl:
                    z = (pl[3] - pl[0] * r[0] - pl[1] * r[1]) / pl[2]
                return round(float(z), 4)
            # moving a person up or down without refitting shifts them in the picture by about the same share of
            # their height; keep the move only when the character still lines up with the person in the film
            # (a pedestal, a step or a wall base under a statue / a far person read as "ground" otherwise)
            scam = Cam({"res": [W, Hp], "lens": c["lens_mm"], "sensor": 36.0, "cam": [world.blender_cam(i) for i in range(Ns)]})
            stracks = {p["id"]: p for p in spose["people"]}
            for k, ch in grounded.items():
                new = [ground_z(r) for r in ch["root"]]
                tr = stracks.get(ch.get("track_id"))
                if tr is None or max(abs(z - r[2]) for z, r in zip(new, ch["root"])) < 0.02:
                    for z, r in zip(new, ch["root"]):
                        r[2] = z
                    continue
                e0 = quality.fit_error(ch, tr, scam, y0, Ns)
                old = [r[2] for r in ch["root"]]
                for z, r in zip(new, ch["root"]):
                    r[2] = z
                e1 = quality.fit_error(ch, tr, scam, y0, Ns)
                if e0 is not None and e1 is not None and e1 > max(1.25 * e0, e0 + 1.5):
                    for z, r in zip(old, ch["root"]):
                        r[2] = z
                    log("    %s kept on the solved floor: on the set's ground it would be %.0f%% of its height off the "
                        "picture (now %.0f%%)" % (k, e1, e0))
        palette.update(sinfo.get("palette", {}))
        if sky is None and sinfo.get("sky_rgb") and not c.get("_set_grey"):
            sky = sinfo["sky_rgb"]
        tp = os.path.join(work, "preview%s.jpg" % tag)
        preview(sf, y0, world, chars, people, meshes, tp, n=4 if len(shots) > 1 else 6)
        tiles.append(tp)
        # every shot gets its own spot in the world (200 m apart) so sets never overlap; the camera cuts between them
        dx = 200.0 * si
        for i in range(Ns):
            bc = world.blender_cam(i); bc[0] += dx; cam_all[s0 + i] = bc
        for m in meshes:
            m["v"] = [[v[0] + dx, v[1], v[2]] for v in m["v"]]
            if tag and not m["name"].endswith(tag) and tag not in m["name"]:
                m["name"] += tag
        meshes_all += meshes
        for kind, ch in chars.items():
            for r in ch["root"]:
                r[0] = round(r[0] + dx, 4)
            chars_all[(tag.strip("_") + "_" if tag else "") + kind] = widen(ch, s0, s1, N)
        for p in prep:
            p["shot"] = si + 1
        prep_all += prep
        warns += [("shot %d: " % (si + 1) if tag else "") + w for w in
                  [w for w in camj["warnings"] if not w.startswith("camera lost")] + [x for x in (ginfo.get("warning"), sinfo.get("warning")) if x]]
        shot_rep.append({"shot": si + 1, "frames": [s0, s1 - 1], "max_rotation_deg": round(camj["max_rotation_deg"], 2),
                         "parallax_px": round(camj["parallax_px"], 2), "ground": ginfo, "set": {k: v for k, v in sinfo.items() if k != "palette"}})
    c["_shot_tag"] = ""
    if hand_set:
        sky = carry_over(work, name, c, chars_all, N, sky)
    cuts = [a for a, _ in shots[1:]]
    jsave(os.path.join(work, "camera.json"), {"shots": shot_rep, "cuts": cuts, "warnings": warns,
                                               "max_rotation_deg": max(s["max_rotation_deg"] for s in shot_rep),
                                               "parallax_px": max(s["parallax_px"] for s in shot_rep)})
    stack_previews(tiles, os.path.join(work, "preview.jpg"))
    data = {"name": name, "package_name": pkg.package_name(name, c), "submit_dir": os.path.join(c["submit_root"], name), "work_dir": work,
            "res": [W, Hp], "fps": int(round(fps)), "frames": N, "render_scale": float(c.get("out_width", 3840)) / W,
            "lens": c["lens_mm"], "sensor": 36.0, "cam": cam_all,
            "palette": palette, "sky": sky or [228, 229, 232], "exposure": 1.2, "gain": {"_video": c.get("video_gain", 2.2), "_ground": c.get("ground_gain", 3.2)},
            "chars": chars_all, "meshes": meshes_all, "props": [], "shots": shot_rep}
    dp = os.path.join(work, "scene_data.json"); jsave(dp, data)
    chars = chars_all
    if not chars:
        warns.append("no people found")
    rep = {"clip": name, "video": video, "checked": bool(chk_p), "frames": N, "people": prep_all, "shots": shot_rep,
           "ground": shot_rep[0]["ground"], "camera_rotation_deg": max(s["max_rotation_deg"] for s in shot_rep), "warnings": warns}
    if c.get("_junk"):
        rep["left_out"] = c["_junk"]
    rep["lens"] = dict(lens_info or {}, used_mm=c["lens_mm"])
    try:                                                   # the clip's quality in numbers (quality.py); never stops a clip
        rep["quality"] = quality.measure(data, pose, y0)
        log("  quality:"); quality.show(rep["quality"], log)
        warns += ["quality - " + w for w in rep["quality"]["flags"]]
    except Exception as e:
        log("  quality numbers failed: %s" % str(e)[:200])
    info = {"W": W, "Hfull": Hfull, "y0": y0, "Hpic": Hp, "frames": N, "fps": fps}
    jsave(os.path.join(work, "clip_info.json"), dict(info, video=video, desc_path=os.path.join(os.path.dirname(pose_p), "desc.txt")))
    frames = sf = None                                     # ~1 GB per clip: free it before Blender and the encoder
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
    desc = ""
    if os.path.exists(ci["desc_path"]):
        with open(ci["desc_path"], encoding="utf-8") as fh:
            desc = fh.read().strip()
    sub, zp = pkg.package(name, video, os.path.join(work, "render"), c["submit_root"], ci, c, desc)
    return {"submission": sub, "zip": zp}

def retime_pulldown(data, video, c):
    """Even motion after the 3:2 pulldown removal.  A 30 fps clip made from 24 fps film repeats every 5th frame; the
    package drops those repeats from the film AND the render.  The fit is smoothed on the 30 fps timeline, so the
    render moves on the repeated frames too, and dropping them left a double step every 4th output frame (a 6 Hz
    twitch).  Here the animation (people, camera, props) is resampled so the kept frames are evenly spaced in time and
    each dropped frame repeats its predecessor.  Marks data["pulldown_retimed"]; runs once per fit."""
    if data.get("pulldown_retimed"):
        return 0
    N = int(data["frames"]); fps = float(data["fps"]); ofps = int(c.get("out_fps", 24))
    drops = pkg.pulldown_drops(video, N, fps, ofps)
    if not drops:
        return 0
    dset = set(drops)
    keep = [f for f in range(N) if f not in dset]
    pos = np.linspace(0, N - 1, len(keep))                # kept frame j sits at an even fraction of the clip

    def rs(arr):
        A = np.asarray(arr, float)
        if A.ndim == 0 or len(A) != N:
            return arr
        src = np.arange(N)
        flat = A.reshape(N, -1)
        out = flat.copy()
        for j, f in enumerate(keep):
            for k in range(flat.shape[1]):
                out[f, k] = np.interp(pos[j], src, flat[:, k])
        for f in sorted(dset):
            if f > 0:
                out[f] = out[f - 1]
        return out.reshape(A.shape).round(5).tolist()

    for ch in data.get("chars", {}).values():
        ch["root"] = rs(ch["root"]); ch["pelvis_z"] = rs(ch["pelvis_z"])
        ch["rot"] = {j: rs(v) for j, v in ch["rot"].items()}
    if data.get("cam"):
        data["cam"] = rs(data["cam"])
    for p in data.get("props", []):
        if p.get("keys"):
            p["keys"] = rs(p["keys"])
    data["pulldown_retimed"] = True
    log("  pulldown: animation re-timed onto the %d film frames (%d repeats)" % (len(keep), len(drops)))
    return len(drops)

def render_only(name, video, work, c, args, t0):
    dp = os.path.join(work, "scene_data.json")
    if not os.path.exists(dp):
        raise RuntimeError("no scene yet - build this clip first")
    data = jload(dp)
    data["submit_dir"] = os.path.join(c["submit_root"], name); data["work_dir"] = work
    data["render_scale"] = float(c.get("out_width", 3840)) / data["res"][0]     # the current output width, also for older fits
    try:                                                                        # v2.1: render at the size the 1080P canvas shows
        ci = jload(os.path.join(work, "clip_info.json"))
        CW, CH = pkg.canvas(ci["W"], ci["Hfull"])
        data["render_scale"] = pkg.fit_even(ci["W"], ci["Hfull"], CW, CH)[0] / float(data["res"][0])
    except (OSError, KeyError, ValueError):
        pass
    data["package_name"] = pkg.package_name(name, c)
    if not data["package_name"]:
        raise RuntimeError("no v2.1 package name for %s - add it to data/package_names.json" % name)
    pal = data.setdefault("palette", {})
    for k, v in PALETTE.items():                                                # any colour missing from an older fit
        pal.setdefault(k, v)
    for k in ("tree", "rock", "marker"):                                        # current set greys + white bands, also for older fits
        pal[k] = PALETTE[k]
    hp = os.path.join(marks_dir(video, c), "hints.json")                       # per-clip opt-ins in hints.json
    if not os.path.exists(hp):
        hp = os.path.join(output_dir(), name, "hints.json")
    hints = jload(hp) if os.path.exists(hp) else {}
    if hints.get("plant_feet") or hints.get("pulldown_retime"):                # the fit stays in scene_data.json; the
        data.pop("feet_planted", None); data.pop("pulldown_retimed", None)      # render-time changes go to a copy
        jsave(dp, {k: v for k, v in data.items()})                              # (palette / paths for the checkers)
        dp = os.path.join(work, "scene_render.json")
    if hints.get("plant_feet"):                                                 # feet on the floor + stepping gait
        import ground_feet
        av = {}
        pp = os.path.join(marks_dir(video, c), "pose.json")
        if os.path.exists(pp):                                                  # frames whose ankles were detected
            for t in jload(pp)["people"]:
                av[str(t["id"])] = {int(f) for f, k in t["frames"].items() if min(k[15][2], k[16][2]) >= 0.4}
        ground_feet.plant(data, log=log, hints=hints, ankle_vis=av)
    if c.get("pulldown_retime", False) or hints.get("pulldown_retime"):         # opt-in: even motion after pulldown removal
        retime_pulldown(data, video, c)
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default=None, help="submission name for a single clip (default: <film>_<file name>)")
    ap.add_argument("--require-confirm", action="store_true", help="only build/render clips whose key points were confirmed in the review app")
    ap.add_argument("--auto-set", action="store_true", help="build the automatic depth set even when the clip has a hand-built set (env_claude.json)")
    args = ap.parse_args()
    args.device = pick_device(args.device)                 # "cpu" on a PC without an NVIDIA card
    c = cfg(); c["device"] = args.device
    vids = collect(args.inputs or [c["source_root"]])
    if not args.inputs:                                    # whole Source folder: leave the client's sample folder out
        skip = {s.lower() for s in c.get("source_exclude", ["3D"])}
        root = os.path.abspath(c["source_root"])
        vids = [v for v in vids if os.path.relpath(os.path.abspath(v), root).split(os.sep)[0].lower() not in skip]
    log("%d clips" % len(vids))
    os.makedirs(c["work_root"], exist_ok=True)
    rows = []
    for v in vids:
        try:
            rows.append(run_clip(v, dict(c), args))
        except Exception as e:
            traceback.print_exc()
            rows.append({"clip": clip_name(v, c), "video": v, "status": "FAILED", "warnings": [str(e)[:300]]})
    if len(rows) > 1:                                      # a batch from the command line: one CSV for the whole run
        rp = os.path.join(c["work_root"], "report_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))   # (the app's one-clip jobs have their log)
        with open(rp, "w", encoding="utf-8-sig") as fh:
            fh.write("clip,status,checked,people,median_error_px,seconds,warnings,video\n")
            for r in rows:
                pe = r.get("people", [])
                fh.write('%s,%s,%s,%d,%s,%s,"%s","%s"\n' % (r["clip"], r["status"], r.get("checked", ""), len(pe),
                         "/".join(str(p["median_error_px"]) for p in pe), r.get("seconds", ""), " | ".join(r.get("warnings", [])).replace('"', "'"), r.get("video", "")))
        log("\nreport: " + rp)
    bad = [r for r in rows if r["status"] == "FAILED"]
    log("%d ok, %d skipped, %d failed" % (sum(r["status"] == "ok" for r in rows), sum(r["status"] == "skipped" for r in rows), len(bad)))
    # the exit code is the result the app trusts:  0 done · 1 a clip failed · 3 the clip's review is not confirmed yet
    if bad:
        sys.exit(1)
    if any(r["status"] == "waiting" for r in rows):
        sys.exit(3)

if __name__ == "__main__":
    main()

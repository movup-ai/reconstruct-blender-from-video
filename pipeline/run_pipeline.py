# -*- coding: utf-8 -*-
"""
Stage 1 "Movement": raw video -> checked v2.1 zip, one automatic run, no manual steps.

  python run_pipeline.py VIDEO_OR_FOLDER [...] [--stage movement|render] [--name CLIP] [--force] [--device cuda]

  stage movement : marks (SAM 3 tracking + RTMPose) -> shots -> lens (MoGe-2) -> per shot: camera (flow + depth cues,
                   zoom), metric scene (MoGe-2 + Mask2Former), SAM 3D Body, floor + camera height, 3D people ->
                   floor disc -> scene_data.json / camera.json / clip_info.json / quality -> Blender render (colour
                   card) -> v2.1 zip -> v2.1 package check.  Any failure (package check included) fails the clip.
  stage render   : internal re-render of a saved scene_data.json: Blender render -> zip -> package check.
  stage background: Stage 2 (background.py): Stage 1's scene + the prebuilt objects of assets\\objects found by SAM 3
                   -> Work\\<clip>\\background\\ -> a second zip in SUBMIT\\_background\\ (Stage 1's zip is kept).
Exit code: 0 every clip done (or skipped: zip there, no --force), 1 a clip failed.
"""
import os, sys, re, glob, hashlib, json, time, shutil, subprocess, argparse, traceback
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import *
from ground_solve import World
from camera_estimate import estimate_shot
from shots import split_shots as find_shots
from fit_people import fit_all
from rig import Rig, JOINTS
import package as pkg
import quality
import gpu_lease
from profiling import stage, profile_clip

fit_all = stage("people_fit")(fit_all)
estimate_shot = stage("camera")(estimate_shot)

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
STAGES = ("movement", "render", "background")
PALETTE = {"ground": [242, 242, 242], "set": [178, 178, 180], "far": [197, 197, 199], "backdrop": [205, 205, 207],
           "ridge": [160, 160, 163], "rock": [138, 138, 138], "tree": [172, 172, 170], "dark": [112, 112, 112], "light": [218, 218, 218],   # tree (vegetation) lighter than rock
           "cabin": [98, 98, 98], "roof": [133, 133, 133], "fence": [146, 146, 146], "mount": [197, 197, 199],
           "red": [205, 62, 60], "blue": [58, 104, 201], "green": [70, 170, 90], "yellow": [225, 200, 60],
           "purple": [150, 90, 190], "pink": [220, 110, 170], "orange": [221, 135, 50], "marker": [255, 255, 255],   # v2.1: eye + chest bands pure white
           "cyan": [80, 205, 235], "magenta": [205, 70, 180],
           "brown": [130, 85, 50], "white": [245, 245, 245], "grey": [150, 150, 150]}
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
# files an earlier (Prepare / Finish, background, L3, audit) build left in the clip's work folder: never reused
STALE = ("scene_verification.json", "scene_proposals.json", "scene_overview.jpg", "build_state.json", "fitted_scene.json",
         "scene_render.json", "held_check.jpg", "env_check.jpg", "report_background.json", "background_revision.json",
         "l3_geometry.json", "verification_profile.json", "environment_profile.json")

def film_short(path, c=None):
    """Short film name from the folder under source_root, excluding supplier suffixes."""
    root = os.path.abspath((c or cfg())["source_root"])
    rel = os.path.relpath(os.path.abspath(path), root)
    top = rel.split(os.sep)[0] if not rel.startswith("..") and os.sep in rel else os.path.basename(os.path.dirname(path))
    if not rel.startswith("..") and rel.count(os.sep) >= 2:      # batch folder with one sub-folder per film:
        top = rel.split(os.sep)[-2]                               # batch\Dragon\x.mp4 -> Dragon
    m = re.match(r"[一-鿿]+", top)
    return m.group(0) if m else re.sub(r"[^\w]+", "_", top)[:20]

def clip_name(path, c=None):
    """unique per clip: <film>_<file stem>  (several films can share the same clip filename)"""
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

@stage("marks")
def find_marks(video, c, device):
    """SAM 3 person tracks + RTMPose joints (automark.py) -> pose.json (and sam3_masks.npz) in the marks folder,
    measured again when the tracking settings changed or the masks are missing -> path of pose.json"""
    out = marks_base(c)
    d = marks_dir(video, c)
    from sam3_tracking import needs_tracking
    old = jload(os.path.join(d,'pose.json')) if os.path.isfile(os.path.join(d,'pose.json')) else None
    changed = old is not None and needs_tracking(old,c)
    missing_masks = old is not None and old.get('tracker_backend')=='sam3' and not os.path.isfile(os.path.join(d,'sam3_masks.npz'))
    if old is None or changed or missing_masks:
        log("  measuring SAM 3 person tracks ...")
        cmd = [sys.executable, os.path.join(AGENT, "automark.py"), video, "--out", out, "--device", device]
        if changed or missing_masks: cmd.append('--force')
        if c.get("source_root") and os.path.abspath(video).lower().startswith(os.path.abspath(c["source_root"]).lower()):
            cmd += ["--base", c["source_root"]]
        subprocess.run(cmd, check=True)
    return os.path.join(d, "pose.json")

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

def alignment_warnings(alignment):
    """quality.alignment_audit numbers -> report warnings (an automatic run never stops on one badly fitted person)"""
    out = []
    for r in alignment.get("people", []):
        if r.get("passed", True):
            continue
        if r.get("reason"):
            out.append("human alignment - %s: %s" % (r["char"], r["reason"]))
        else:
            med = r.get("median_error_px")
            out.append("human alignment - %s: %d of %d checked frames off the detected person (median %s px, %d frames "
                       "not shown); see human_alignment.json" % (r["char"], r.get("bad_frames", 0), r.get("checked_frames", 0),
                       "%.0f" % med if med is not None else "-", r.get("missing_frames", 0)))
    return out

@profile_clip
@gpu_lease.job
def run_clip(video, c, args):
    # Actor IDs are local to a clip. Share scale across its shots, never across
    # unrelated videos processed by the same command.
    c['_actor_heights'] = {}
    c['_required_tracks'] = set()
    name = getattr(args, "name", None) or clip_name(video, c)
    work = os.path.join(c["work_root"], name); c["_work"] = work
    os.makedirs(work, exist_ok=True)
    zp = os.path.join(c["submit_root"], name + ".zip")
    if args.stage == "movement" and os.path.exists(zp) and not args.force:
        log("skip (zip exists):", name); return {"clip": name, "status": "skipped"}
    if c.get("_sandbox"):                                          # sandbox: start from a copy of the real marks
        real = dict(c); real.pop("marks_root", None)
        real_md = marks_dir(video, dict(real, marks_root=os.path.join(AGENT, "output")))
        if os.path.isdir(real_md) and not os.path.exists(marks_dir(video, c)):
            shutil.copytree(real_md, marks_dir(video, c))
    if not c.get("_sandbox") and not pkg.package_name(name, c):    # a missing delivery name must not fail a run after its GPU work
        made = pkg.auto_name([name], c)
        if made:
            log("  no delivery name for %s: gave it %s (made up - Name videos... in the app replaces it)" % (name, made[name]))
    t0 = time.time()
    log("=== %s  (%s)" % (name, video))
    if args.stage == "render":                             # re-render the saved scene (no new fitting)
        return render_only(name, video, work, c, args, t0)
    if args.stage == "background":                         # Stage 2: Stage 1's scene + prebuilt objects -> a second zip
        import background
        return background.run(name, video, work, marks_dir(video, c), c, render_and_pack)
    gpu_lease.acquire()
    for path in glob.glob(os.path.join(work, "metric_evidence*.npz")) + [os.path.join(work, fn) for fn in STALE]:
        if os.path.isfile(path):                           # an earlier build's files: never shipped or reused
            os.remove(path)
    pose_p = find_marks(video, c, args.device)
    pose = jload(pose_p)
    c['_persistent_tracks'] = True
    log("  marks: %s" % pose_p)
    frames, fps = read_frames(video)
    N = len(frames); Hfull, W = frames[0].shape[:2]
    _, y0, _, Hp = pose["picture_area"]
    shots = find_shots(frames, y0, Hp)
    log("  shots: %s" % ", ".join("%d-%d" % (a, b - 1) for a, b in shots))
    palette = dict(PALETTE)
    cam_all, chars_all, meshes_all, prep_all, warns, shot_rep, tiles = [None] * N, {}, [], [], [], [], []
    zoom_all = [1.0] * N                                   # focal scale per frame (the lens zooms): f = zoom * lens
    # the clip's lens from the picture (lens_auto.py, MoGe-2); set by hand only in config.json (lens_auto false + lens_mm)
    lens_info = None
    if c.get("lens_auto", True):
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
    else:
        log("  lens: %.1f mm set by hand in config.json (lens_auto off)" % c["lens_mm"])
    for si, (s0, s1) in enumerate(shots):
        tag = "_s%d" % (si + 1) if len(shots) > 1 else ""
        c["_shot_tag"] = tag
        c.pop('_sam3d_body',None)
        c['_shot_start'] = s0
        c['_marks_dir'] = os.path.dirname(pose_p)
        sf = frames[s0:s1]; Ns = s1 - s0
        spose = slice_pose(pose, s0, s1)
        log("  [shot %d: frames %d-%d]" % (si + 1, s0, s1 - 1))
        log("  camera ...")
        tp0 = time.time()
        people = coco_people(spose, None)
        people = {pid: (list(fs), K) for pid, (fs, K) in people.items()}
        people = quality.drop_junk(people, c, fps, log)        # short + unclear tracks: not a person worth fitting
        cam = Camera(W, Hp, c["lens_mm"])
        camj, u, hc, ginfo = estimate_shot(sf, spose, people, cam, y0, Hp, c)   # dense tracking + depth / horizon / people ground
        log("    camera %.1fs  %s" % (time.time() - tp0, camj.get("steady", "")))
        z = camj.get("zoom")
        if z and lens_info and lens_info.get("frames_mm") and not lens_info.get("why"):
            # the lens was measured a quarter, half and three quarters into the shot, the zoom is from its first frame:
            # each measurement divided by the zoom there is the first frame's lens -> folded into this shot's zoom
            import lens_auto
            idx = lens_auto.sample_frames(shots)
            if si < len(idx) and si < len(lens_info["frames_mm"]):
                l0 = [mm / z[f - s0] for f, mm in zip(idx[si], lens_info["frames_mm"][si]) if 0 <= f - s0 < len(z) and z[f - s0] > 0]
                if l0:
                    k0 = float(np.clip(np.median(l0), 12.0, 300.0)) / c["lens_mm"]
                    if abs(k0 - 1.0) > 0.03:
                        camj["zoom"] = [float(v) * k0 for v in z]
                        log("    zoom x%.2f over the shot: the first frame's lens is %.1f mm (%.1f -> %.1f mm)" % (
                            z[-1] / z[0], c["lens_mm"] * k0, c["lens_mm"] * k0 * z[0], c["lens_mm"] * k0 * z[-1]))
        observation = None
        if c.get("metric_scene", True):
            try:
                import metric_scene
                log("  metric scene: MoGe-2 geometry and semantic surfaces ...")
                observation = metric_scene.observe(sf, spose, cam, y0, camj, c)
            except Exception as e:
                observation = None
                camj["warnings"].append("metric evidence unavailable: %s" % str(e)[-240:])
                log("    metric evidence unavailable: %s" % str(e)[-240:])
        body_summary = None
        import sam3d_body
        log('  SAM 3D Body: recovering metric people ...')
        samples = {}
        if observation is not None:
            samples = {int(f):{k:observation[k][i] for k in ('depth','K','valid')} for i,f in enumerate(observation['frames'])}
        c['_sam3d_body'],body_summary = sam3d_body.recover(sf,spose,cam,y0,camj,samples,c)
        ginfo['sam3d_body'] = body_summary
        if observation is not None:
            u,metric_info = metric_scene.align(observation,camj,u,hc,c,people,cam,y0)
            ginfo['metric_alignment'] = metric_info
            if body_summary and body_summary['calibrated']:
                observation['scale'] = body_summary['meters_per_unit']
                observation['scale_source'] = 'SAM 3D Body visible mesh surfaces'
                measured_height = metric_scene.camera_height(observation,camj,u)
                if measured_height is not None:
                    hc = measured_height; ginfo['camera_height'] = hc
                    ginfo['height_source'] = 'body-scaled semantic ground'
        world = World(cam, camj["R"], u, hc, zoom=camj.get("zoom"), yaw=camj.get("world_yaw", 0.))
        if camj.get("travel_cam0"):
            world.T = np.asarray(camj["travel_cam0"]) @ world.M
        log("  ground: %s" % json.dumps(ginfo))
        log("  people ...")
        tp0 = time.time()
        c["_fps"] = float(fps)
        chars, prep = fit_all(world, y0, people, Ns, c)
        # half-body shot (no feet in the picture): the floor was a guess - put it under the people and fit them again.
        # Not when SAM 3D Body calibrated the metric scale (its body-scaled ground already gave the camera height).
        dz = None if (c.get("floor_from_people", True) is False or (body_summary and body_summary.get("calibrated"))) \
            else quality.floor_shift(chars, people)
        if dz:
            hc2 = float(np.clip(hc + dz, 0.3, 12.0))
            log("    nobody's feet are in the picture: floor moved %+.2f m to stand the people on it (camera height %.2f -> %.2f m)" % (-dz, hc, hc2))
            world = World(cam, camj["R"], u, hc2, zoom=camj.get("zoom"), yaw=camj.get("world_yaw", 0.))
            if camj.get("travel_cam0"):
                world.T = np.asarray(camj["travel_cam0"]) @ world.M
            ginfo = dict(ginfo, camera_height=hc2, floor="under the people (no feet in the picture)", camera_height_before=hc)
            hc = hc2
            chars, prep = fit_all(world, y0, people, Ns, c)
        log("    people %.1fs" % (time.time() - tp0))
        if observation is not None:
            if body_summary and body_summary['calibrated']:
                ginfo['depth_scale'] = body_summary
            else:
                ginfo["depth_scale"] = metric_scene.body_scale(observation, world, chars)
        log("  set ...  (Stage 1: the floor only)")
        from scene_depth import flat_set               # one flat ground disc under the people; no set, no props
        meshes = flat_set(world, Ns, float(c.get("far_distance", 1500.0)))
        sinfo = {"source": "flat ground disc (Stage 1: floor only)"}
        tp = os.path.join(work, "preview%s.jpg" % tag)
        preview(sf, y0, world, chars, people, meshes, tp, n=4 if len(shots) > 1 else 6)
        tiles.append(tp)
        # every shot gets its own spot in the world (200 m apart) so floors never overlap; the camera cuts between them
        dx = 200.0 * si
        for i in range(Ns):
            bc = world.blender_cam(i); bc[0] += dx; cam_all[s0 + i] = bc; zoom_all[s0 + i] = round(world.f_at(i) / cam.f, 5)
        for m in meshes:
            m["v"] = [[v[0] + dx, v[1], v[2]] for v in m["v"]]
            if tag and not m["name"].endswith(tag) and tag not in m["name"]:
                m["name"] += tag
        meshes_all += meshes
        for kind, ch in chars.items():
            for r in ch["root"]:
                r[0] = round(r[0] + dx, 4)
            widened = widen(ch, s0, s1, N)
            widened['source_range'] = [s0, s1-1]
            chars_all[(tag.strip("_") + "_" if tag else "") + kind] = widened
        for p in prep:
            p["shot"] = si + 1
        prep_all += prep
        warns += [("shot %d: " % (si + 1) if tag else "") + w for w in
                  [w for w in camj["warnings"] if not w.startswith("camera lost")] + [x for x in (ginfo.get("warning"),) if x]]
        shot_rep.append({"shot": si + 1, "frames": [s0, s1 - 1], "max_rotation_deg": round(camj["max_rotation_deg"], 2),
                         "parallax_px": round(camj["parallax_px"], 2), "ground": ginfo, "set": sinfo})
        shot_rep[-1]["camera_backend"] = camj.get("backend", "flow")
        if "meters_per_unit" in camj:
            shot_rep[-1]["meters_per_unit"] = camj["meters_per_unit"]
    c["_shot_tag"] = ""
    cuts = [a for a, _ in shots[1:]]
    jsave(os.path.join(work, "camera.json"), {"shots": shot_rep, "cuts": cuts, "warnings": warns,
                                               "max_rotation_deg": max(s["max_rotation_deg"] for s in shot_rep),
                                               "parallax_px": max(s["parallax_px"] for s in shot_rep)})
    stack_previews(tiles, os.path.join(work, "preview.jpg"))
    data = {"name": name, "package_name": pkg.package_name(name, c), "submit_dir": os.path.join(c["submit_root"], name), "work_dir": work,
            "res": [W, Hp], "fps": float(fps), "frames": N, "render_scale": float(c.get("out_width", 3840)) / W,
            "lens": c["lens_mm"], "sensor": 36.0, "cam": cam_all, "zoom": zoom_all,
            "palette": palette, "sky": [228, 229, 232], "exposure": 1.2, "gain": {"_video": c.get("video_gain", 2.2), "_ground": c.get("ground_gain", 3.2)},
            "chars": chars_all, "meshes": meshes_all, "props": [], "shots": shot_rep}
    data['human_backend'] = 'sam3d_body'
    data['required_track_ids'] = sorted(c.get('_required_tracks', set()), key=str)
    import source_exposure                                 # opt-in (config source_exposure), off by default
    source_exposure.attach(data,video,work,c,pose,{'y0':y0,'Hpic':Hp,'fps':fps},frames)
    dp = os.path.join(work, "scene_data.json"); jsave(dp, data)
    if not chars_all:
        warns.append("no people found")
    rep = {"clip": name, "video": video, "stage": "movement", "frames": N, "people": prep_all, "shots": shot_rep,
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
    gpu_lease.release()                                    # the rest uses no model: Blender, the encoder, the checks
    alignment = quality.cached_alignment(data, pose, y0, work)     # numbers + warnings, never a stop
    rep["human_alignment"] = alignment
    warns += alignment_warnings(alignment)
    jsave(os.path.join(work, "report.json"), dict(rep, status="rendering"))
    rep.update(render_and_pack(name, video, work, c))
    rep["seconds"] = round(time.time() - t0, 1)
    rep["status"] = "ok"
    jsave(os.path.join(work, "report.json"), rep)
    log("  done in %.0fs%s" % (rep["seconds"], ("   WARNINGS: " + " | ".join(warns)) if warns else ""))
    return rep

@stage("blender")
def blender_render(c, work, dp, N, extra=("render",)):
    gpu_lease.release()
    be = blender_exe(c)
    if not be:
        raise RuntimeError("Blender not found - set the Blender path in Settings (config.json)")
    render_data=jload(dp)
    expected={'work_dir':work,'submit_dir':os.path.join(c['submit_root'],render_data['name'])}
    for key,path in expected.items():
        if os.path.normcase(os.path.abspath(render_data.get(key,'')))!=os.path.normcase(os.path.abspath(path)):
            raise RuntimeError('Scene %s does not match this render workspace; refresh its output paths before rendering.'%key)
    d_ = jload(dp)
    from render_settings import engine as render_engine
    d_['render_engine']=render_engine(c,d_)
    if not c.get('source_exposure',False):d_.pop('source_exposure',None)
    elif not d_.get('source_exposure'):
        import source_exposure
        source_exposure.attach(d_,jload(os.path.join(work,'clip_info.json'))['video'],work,c)
    for key in ('background_policy', 'previs_standard', 'validation_mode', 'light'):   # keys of earlier builds
        d_.pop(key, None)
    import colour_card                                     # every video follows the colour card (docs/COLOUR_CARD.md);
    colour_card.apply(d_, log=log)                         # a failure fails the clip (never a render in other colours)
    jsave(dp,d_)
    log("  blender build + render (%d frames) ..." % N)
    done = os.path.join(work, "render_done.txt")
    if os.path.exists(done): os.remove(done)
    with open(os.path.join(work, "blender_log.txt"), "w", encoding="utf-8", errors="ignore") as lf:
        subprocess.run([be, "-b", "-P", os.path.join(HERE, "blender_build.py"), "--", dp] + list(extra), stdout=lf, stderr=subprocess.STDOUT)
    if not os.path.exists(done):
        raise RuntimeError("Blender did not finish - see %s" % os.path.join(work, "blender_log.txt"))

@stage("encoding_and_package")
def make_zip(name, video, work, c):
    log("  videos + zip ...")
    ci = jload(os.path.join(work, "clip_info.json"))
    desc = ""
    if os.path.exists(ci["desc_path"]):
        with open(ci["desc_path"], encoding="utf-8") as fh:
            desc = fh.read().strip()
    sub, zp = pkg.package(name, video, os.path.join(work, "render"), c["submit_root"], ci, c, desc)
    return {"submission": sub, "zip": zp}

@stage("package_check")
def check_package(sub, zp, video):
    """the v2.1 package check (agent/validate.py): folders and the six files, meta, video sizes / codec / durations /
    audio, target byte-identical to the source, the .blend header, zip = folder.  Any problem fails the clip."""
    root = os.path.dirname(HERE)
    if root not in sys.path:
        sys.path.insert(0, root)
    from agent.validate import check_package_v21, check_zip
    log("  v2.1 package check ...")
    problems, _ = check_package_v21(sub, ffmpeg_exe(), source=video)
    problems = list(problems) + list(check_zip(zp, sub))
    if problems:
        raise RuntimeError("v2.1 package check failed: " + "; ".join(problems))
    log("  v2.1 package check: ok")
    return {"package_check": "ok"}

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
    if data.get("zoom"):
        data["zoom"] = rs(data["zoom"])
    if data.get('source_exposure'):
        data['source_exposure']['stops'] = rs(data['source_exposure']['stops'])
    for p in data.get("props", []):
        if p.get("keys"):
            p["keys"] = rs(p["keys"])
    data["pulldown_retimed"] = True
    log("  pulldown: animation re-timed onto the %d film frames (%d repeats)" % (len(keep), len(drops)))
    return len(drops)

def render_and_pack(name, video, work, c):
    """scene_data.json -> Blender render (colour card) -> v2.1 zip -> v2.1 package check -> {submission, zip, package_check}"""
    dp = os.path.join(work, "scene_data.json")
    if not os.path.exists(dp):
        raise RuntimeError("no scene yet - run the movement stage for this clip first")
    data = jload(dp)
    old = sorted(k for k, ch in data.get("chars", {}).items() if ch.get("style", "sample") not in ("sample", "classic"))
    if old:
        raise RuntimeError("this scene was built with a removed character rig (%s) - run the movement stage again" % ", ".join(old))
    data["submit_dir"] = os.path.join(c["submit_root"], name); data["work_dir"] = work
    data["render_scale"] = float(c.get("out_width", 3840)) / data["res"][0]     # the current output width, also for older fits
    try:                                                                        # v2.1: render at the size the 1080P canvas shows
        ci = jload(os.path.join(work, "clip_info.json"))
        CW, CH = pkg.canvas(ci["W"], ci["Hfull"])
        data["render_scale"] = pkg.cover_even(ci["W"], ci["Hfull"], CW, CH)[0] / float(data["res"][0])
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
    import source_exposure
    source_exposure.attach(data,video,work,c)
    if c.get("pulldown_retime", False):                                         # opt-in: even motion after pulldown removal
        retime_pulldown(data, video, c)
    jsave(dp, data)
    blender_render(c, work, dp, data["frames"])
    out = make_zip(name, video, work, c)
    out.update(check_package(out["submission"], out["zip"], video))
    return out

def render_only(name, video, work, c, args, t0):
    """--stage render: re-render the saved scene, pack and check it (people and camera as fitted)"""
    dp = os.path.join(work, "scene_data.json")
    if not os.path.exists(dp):
        raise RuntimeError("no scene yet - run the movement stage for this clip first")
    rep = {"clip": name, "video": video, "stage": "render", "status": "ok", "warnings": []}
    pp = os.path.join(marks_dir(video, c), 'pose.json')
    if os.path.exists(pp):
        source_pose = jload(pp)
        alignment = quality.cached_alignment(jload(dp), source_pose, (source_pose.get('picture_area') or [0, 0])[1], work)
        rep["human_alignment"] = alignment
        rep["warnings"] += alignment_warnings(alignment)
    rep.update(render_and_pack(name, video, work, c))
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
    ap.add_argument("--stage", default="movement", choices=list(STAGES),
                    help="movement: the whole Stage 1 run (default); render: re-render a saved scene, zip and check it; "
                         "background: Stage 2, objects from assets\\objects added to Stage 1's scene -> a second zip")
    ap.add_argument("--force", action="store_true", help="run again even when the clip's zip exists")
    ap.add_argument("--device", default="cuda", choices=["cuda"], help="the models run on the NVIDIA card only")
    ap.add_argument("--name", default=None, help="submission name for a single clip (default: <film>_<file name>)")
    args = ap.parse_args()
    if args.stage != "render":                             # (rendering in Blender uses no model)
        args.device = pick_device(args.device)             # stops here without a working NVIDIA card
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
            fh.write("clip,status,people,median_error_px,seconds,warnings,video\n")
            for r in rows:
                pe = r.get("people", [])
                fh.write('%s,%s,%d,%s,%s,"%s","%s"\n' % (r["clip"], r["status"], len(pe),
                         "/".join(str(p["median_error_px"]) for p in pe), r.get("seconds", ""), " | ".join(r.get("warnings", [])).replace('"', "'"), r.get("video", "")))
        log("\nreport: " + rp)
    bad = [r for r in rows if r["status"] == "FAILED"]
    log("%d ok, %d skipped, %d failed" % (sum(r["status"] == "ok" for r in rows), sum(r["status"] == "skipped" for r in rows), len(bad)))
    # the exit code is the result the app trusts:  0 done · 1 a clip failed
    if bad:
        sys.exit(1)

if __name__ == "__main__":
    main()

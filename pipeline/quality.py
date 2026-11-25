# -*- coding: utf-8 -*-
"""Quality of a fitted clip in numbers, and the filter for tracks that are not worth fitting.

  python pipeline\\quality.py <clip name>        print the numbers of a fitted clip (Work_auto\\<clip>)

Every fit writes the same numbers to report.json ("quality") and adds the flags to its warnings:
  fit_error_pct   how far the character's joints are from the detected joints, % of the body height (audit.py)
  jumps_per_100   frames with a sudden movement per 100 frames (a joint turning > 6 deg in one frame, a jerk > 5 deg,
                  or the body's step changing by > 4 cm from one frame to the next)
  feet_seen       share of the frames in which an ankle was detected
  foot_gap_m      lowest foot above (+) / below (-) the floor, median (foot_gaps.py)
  pose_change     how much the pose itself changes (0 = rigid: a statue, a poster, a picture)

Junk tracks (junk_reason): a track that is BOTH short (under junk_seconds, default 1 s) and weak (fewer than
junk_joints sure joints a frame, default 8) is left out before the camera, the ground and the fit use it.  A track
the reviewer gave a role is never left out.  config.json "junk_filter": false switches the filter off."""
import os, sys
import numpy as np

SURE = 0.5                                     # a joint the pose model was sure of
LIMITS = {"fit_error_pct": 8.0, "jumps_per_100": 10.0, "foot_gap_m": 0.10, "feet_seen": 0.5, "pose_change": 0.03}


def track_stats(fs, K, fps=30.0):
    """one track: frames fs, K (n, 17, 3) COCO px + score -> its numbers"""
    K = np.asarray(K, float).reshape(len(fs), 17, 3)
    conf = K[:, :, 2]
    out = {"frames": int(len(fs)), "seconds": round(len(fs) / float(fps or 30.0), 2),
           "sure_joints": round(float((conf > SURE).sum(1).mean()), 1) if len(fs) else 0.0,
           "feet_seen": round(float((conf[:, 15:17] > SURE).any(1).mean()), 2) if len(fs) else 0.0,
           "pose_change": None}
    use = (conf > SURE).mean(0) >= 0.8                     # joints seen in most frames
    if len(fs) >= 4 and use.sum() >= 6:
        Y = K[:, use, :2] - K[:, use, :2].mean(1, keepdims=True)          # position and size taken out
        Y = Y / np.maximum(np.sqrt((Y ** 2).sum((1, 2), keepdims=True)), 1e-9)
        out["pose_change"] = round(float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(2)).mean()), 4)
    return out


def junk_reason(st, c=None):
    """why a track is not fitted (None = it is)"""
    c = c or {}
    if st["seconds"] < float(c.get("junk_seconds", 1.0)) and st["sure_joints"] < float(c.get("junk_joints", 8)):
        return "short and unclear (%.1f s, %.0f sure joints a frame)" % (st["seconds"], st["sure_joints"])
    return None


def drop_junk(people, c, fps, log=None):
    """people {id: (frames, K)} without the junk tracks; what was left out is added to c["_junk"]"""
    if c.get("junk_filter", True) is False or c.get('_persistent_tracks'):
        return people
    rv = (c.get("_review") or {}).get("people", {})
    keep = {}
    for pid, (fs, K) in people.items():
        why = junk_reason(track_stats(fs, K, fps), c) if len(fs) else None
        if why and rv.get(str(pid), {}).get("role") in (None, "", "ignore"):
            c.setdefault("_junk", []).append({"track": pid, "frames": [int(min(fs)), int(max(fs))], "why": why})
            if log:
                log("    track %s left out: %s" % (pid, why))
            continue
        keep[pid] = (fs, K)
    return keep


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def jumps(ch, deg=6.0, acc=5.0, move=0.04):
    """frames with a sudden movement (jerk_check.py's test), any joint"""
    root = np.array(ch["root"], float)
    vis = np.array(ch.get("visible") or [1] * len(root), bool)
    series = [root[:, 3]] if root.shape[1] > 3 else []
    for r in ch["rot"].values():
        r = np.array(r, float)
        series += [r[:, ax] for ax in range(3) if np.ptp(r[:, ax]) > 0.5]
    bad = set()
    for s in series:
        v = _wrap(np.diff(s))
        bad |= set(np.where(np.abs(v) > deg)[0] + 1) | set(np.where(np.abs(np.diff(v)) > acc)[0] + 1)
    # position: a sudden CHANGE of the step (jerk_check.py flags every step over 4 cm, which is plain walking at 30 fps)
    step = np.diff(root[:, :3], axis=0)
    bad |= set(np.where(np.linalg.norm(np.diff(step, axis=0), axis=1) > move)[0] + 1)
    return sorted(i for i in bad if vis[i] and vis[i - 1])


def foot_gap(ch):
    """median height of the lowest foot over the floor (m), visible frames"""
    from rig import Rig, JOINTS
    rig = Rig(ch["height"], ch.get("style", "master")); gaps = []
    for f, r in enumerate(ch["root"]):
        if ch.get("visible") and not ch["visible"][f]:
            continue
        W = rig.fk(r[:3], r[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f])
        gaps.append(min(W["ankle_L"][1][2], W["ankle_R"][1][2]) - rig.sole - r[2])
    return float(np.median(gaps)) if gaps else None


def floor_shift(chars, people, min_shift=0.05, max_shift=1.5):
    """Half-body shots: nobody's feet are in the picture, so nothing in the picture says where the floor is, and the
    people come out standing above or sinking into a guessed floor.  Returns how far to RAISE the camera over the
    floor (m) so that they stand on it: minus the median foot gap of the fitted people.  None when any fitted person
    shows their feet (those feet are the floor) or the gap is small.
      chars: fit_all's characters of one shot; people: {track id: (frames, K)} of the same shot"""
    gaps = []
    for ch in chars.values():
        fs, K = people.get(ch.get("track_id"), ([], []))
        if len(fs) and track_stats(fs, K)["feet_seen"] >= LIMITS["feet_seen"]:
            return None
        g = foot_gap(ch)
        if g is not None:
            gaps.append(g)
    if not gaps:
        return None
    g = float(np.median(gaps))
    return None if abs(g) < min_shift else float(np.clip(-g, -max_shift, max_shift))


def fit_error(ch, track, cam, y0, frames):
    """mean distance character joint - detected joint, % of the body height on screen (sure joints only)"""
    from compare import rig_uv
    vals = []
    for f in range(frames):
        k = track["frames"].get(str(f))
        if k is None or not ch["visible"][f]:
            continue
        k = np.array(k, float); ok = np.where(k[:, 2] > SURE)[0]
        if len(ok) < 3:
            continue
        h = max(40.0, float(np.ptp(k[k[:, 2] > 0.3, 1])))
        uv = rig_uv(ch, cam, f)
        vals.append(float(np.mean([np.linalg.norm(uv[j] - (k[j, :2] - [0, y0])) for j in ok])) / h * 100)
    return float(np.mean(vals)) if vals else None


def measure(data, pose, y0):
    """scene_data + pose.json -> {"people": [...], "flags": [...], "ok": bool}; never raises for one bad character"""
    tracks = {p["id"]: p for p in pose["people"]}
    fps = float(pose.get("fps") or data.get("fps") or 30.0)
    cam = None
    try:
        from build_env import Cam
        cam = Cam(data)
    except Exception:
        pass
    people, flags = [], []
    for kind, ch in data["chars"].items():
        row = {"char": kind, "track": ch.get("track_id"), "colour": ch.get("colour")}
        vis = int(np.sum(ch.get("visible") or [1] * len(ch["root"])))
        row["frames"] = vis
        t = tracks.get(ch.get("track_id"))
        try:
            jf = jumps(ch)
            row["jumps_per_100"] = round(100.0 * len(jf) / max(vis, 1), 1)
            row["jump_frames"] = [int(f) for f in jf[:500]]     # the joint editor opens on these first
            g = foot_gap(ch)
            row["foot_gap_m"] = None if g is None else round(g, 3)
            if t is not None:
                fs = sorted(int(f) for f in t["frames"])
                st = track_stats(fs, [t["frames"][str(f)] for f in fs], fps)
                row.update(feet_seen=st["feet_seen"], pose_change=st["pose_change"], sure_joints=st["sure_joints"])
                if cam is not None:
                    e = fit_error(ch, t, cam, y0, int(data["frames"]))
                    row["fit_error_pct"] = None if e is None else round(e, 1)
        except Exception as e:                               # the numbers must never stop a clip
            row["error"] = str(e)[:200]
        why = []
        if (row.get("fit_error_pct") or 0) > LIMITS["fit_error_pct"]:
            why.append("joints %.0f%% of the body off the detected ones" % row["fit_error_pct"])
        if (row.get("jumps_per_100") or 0) > LIMITS["jumps_per_100"]:
            why.append("jerky (%.0f sudden movements per 100 frames)" % row["jumps_per_100"])
        if row.get("feet_seen") is not None and row["feet_seen"] < LIMITS["feet_seen"]:
            why.append("feet not seen in most frames: size and distance are a guess")
        elif row.get("foot_gap_m") is not None and abs(row["foot_gap_m"]) > LIMITS["foot_gap_m"]:
            why.append("feet %.2f m %s the floor" % (abs(row["foot_gap_m"]), "above" if row["foot_gap_m"] > 0 else "below"))
        if row.get("pose_change") is not None and row["pose_change"] < LIMITS["pose_change"] and vis >= fps:
            why.append("pose never changes: a statue or a picture?")
        row["flags"] = why
        flags += ["%s: %s" % (kind, w) for w in why]
        people.append(row)
    return {"people": people, "flags": flags, "ok": not flags, "limits": LIMITS}


def alignment_audit(data, pose, y0=0):
    """Delivery gate for gross source/rig misalignment, independent of foot or motion guesses."""
    from build_env import Cam
    from rig import Rig, JOINTS
    cam=Cam(data); tracks={p['id']:p for p in (pose or {}).get('people',[])}
    rows=[]
    for name,ch in data.get('chars',{}).items():
        track=tracks.get(ch.get('track_id'))
        if track is None: continue
        rig=Rig(ch['height'],ch.get('style','sample')); errors=[]; bad=0; missing=0
        for f,keys in track.get('frames',{}).items():
            f=int(f)
            if f>=data['frames']: continue
            k=np.asarray(keys,float); sure=k[:,2]>.5
            if sure.sum()<4: continue
            if f>=len(ch['root']) or not ch.get('visible',[True]*data['frames'])[f]:
                missing+=1; bad+=1; errors.append(float('inf')); continue
            r=ch['root'][f]
            points=rig.coco_points(rig.fk(r[:3],r[3],{j:ch['rot'][j][f] for j in JOINTS},ch['pelvis_z'][f]))
            uv,z=cam.project(points,f)
            error=float(np.median(np.linalg.norm(uv[sure]-(k[sure,:2]-[0,y0]),axis=1)))
            span=max(40.,float(np.ptp(k[k[:,2]>.3,1])))
            bad+=int(not np.isfinite(error) or (z[sure]<=.05).any() or error>max(32.,.15*span))
            errors.append(error)
        if len(errors)<4: continue
        failed=bad/len(errors)>.3
        finite=[e for e in errors if np.isfinite(e)]
        rows.append({'char':name,'track':ch['track_id'],'checked_frames':len(errors),
                     'bad_frames':bad,'missing_frames':missing,'median_error_px':float(np.median(finite)) if finite else None,
                     'passed':not failed})
    return {'passed':all(r['passed'] for r in rows),'people':rows,'checked':bool(rows)}


def require_alignment(result):
    failed=[r['char'] for r in result['people'] if not r['passed']]
    if failed:
        raise RuntimeError('Human alignment failed for %s. Re-check people/camera placement before rendering; see human_alignment.json.' % ', '.join(failed))


def show(q, log=print):
    for r in q["people"]:
        log("    %-10s track %-3s %4d frames  fit %s%%  jumps %s /100  feet seen %s  foot gap %s m  pose change %s" % (
            r["char"], r.get("track"), r["frames"], r.get("fit_error_pct"), r.get("jumps_per_100"), r.get("feet_seen"),
            r.get("foot_gap_m"), r.get("pose_change")))
        for w in r["flags"]:
            log("       ! " + w)


def main():
    if len(sys.argv) != 2:
        print(__doc__); return 2
    from common import cfg, jload
    from run_pipeline import marks_dir
    c = cfg(); work = os.path.join(c["work_root"], sys.argv[1])
    data = jload(os.path.join(work, "scene_data.json")); ci = jload(os.path.join(work, "clip_info.json"))
    q = measure(data, jload(os.path.join(marks_dir(ci["video"], c), "pose.json")), ci["y0"])
    show(q)
    print("quality: %s" % ("ok" if q["ok"] else "%d flag(s)" % len(q["flags"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

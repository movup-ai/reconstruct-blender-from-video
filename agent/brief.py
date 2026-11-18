# -*- coding: utf-8 -*-
"""Hand-off files for Claude.  When a person confirms a clip's detection, everything Claude needs to finish the clip
without detecting anything itself is written as one small, plain file:

  <marks>/claude_brief.json      one clip: confirmed people (role, colour, height, where and when they are on screen),
                                 shots, camera, ground, the set objects so far, warnings, and the paths of every file
  <work_root>/CLAUDE_HANDOFF.md  all clips: what is confirmed, what is still waiting  (+ CLAUDE_HANDOFF.json)

    python -m agent.brief            rewrite every brief and the index
"""
import glob
import json
import os
import sys
import time

from . import config
from .clips import Library, folders, status, read_json

SAMPLES = 9                                              # positions written per person per shot


def _box_track(p, s0, s1, W, H):
    """a person's box at a few moments of a shot, as fractions of the picture (x, y of the centre, height)"""
    fs = sorted(int(f) for f in p.get("boxes", {}) if s0 <= int(f) <= s1)
    if not fs:
        return None
    pick = sorted({fs[int(round(i * (len(fs) - 1) / max(1, SAMPLES - 1)))] for i in range(SAMPLES)})
    out = []
    for f in pick:
        b = p["boxes"][str(f)]
        out.append({"frame": f, "cx": round((b[0] + b[2]) / 2 / W, 3), "cy": round((b[1] + b[3]) / 2 / H, 3),
                    "h": round((b[3] - b[1]) / H, 3), "feet_y": round(b[3] / H, 3)})
    return {"first": fs[0], "last": fs[-1], "frames": len(fs), "path": out}


def clip_brief(c, cl):
    """-> dict, or None when the clip has not been prepared yet"""
    f = folders(c, cl)
    pose = read_json(f["pose"])
    if not pose:
        return None
    rv = read_json(f["review"]) or {}
    rep = read_json(os.path.join(f["work"], "report.json"), {}) or {}
    cam = read_json(os.path.join(f["work"], "camera.json"), {}) or {}
    env = (read_json(os.path.join(f["marks"], "env_ai.json"), {}) or {})
    W, H = pose["width"], pose["height"]
    N = pose["frames"]
    shots = cam.get("shots") or [{"shot": 1, "frames": [0, N - 1]}]
    fitted = {}
    for q in rep.get("people", []):
        fitted.setdefault(str(q["track"]), []).append(q)
    people, ignored = [], []
    for p in pose.get("people", []):
        pid = str(p["id"])
        r = (rv.get("people") or {}).get(pid, {})
        role = r.get("role", "unreviewed")
        if role == "ignore":
            ignored.append(int(pid))
            continue
        per_shot = {}
        for s in shots:
            t = _box_track(p, s["frames"][0], s["frames"][1], W, H)
            if t:
                per_shot[str(s["shot"])] = t
        fit = fitted.get(pid, [])
        people.append({"id": int(pid), "role": role, "colour": r.get("colour"), "height_m": r.get("height"),
                       "name": r.get("name", ""), "frames_seen": len(p.get("frames", {})),
                       "fit_error_px": min((q["median_error_px"] for q in fit), default=None),
                       "shots": per_shot})
    order = {"main": 0, "second": 1, "third": 2, "fourth": 3, "extra": 4}
    people.sort(key=lambda x: (order.get(x["role"], 9), x["id"]))
    tags = [""] if len(shots) == 1 else ["_s%d" % s["shot"] for s in shots]
    shot_rows = []
    for s, tag in zip(shots, tags):
        reg = read_json(os.path.join(f["work"], "regions%s.json" % tag), {}) or {}
        esh = (env.get("shots") or {}).get(tag, {})
        shot_rows.append({
            "shot": s["shot"], "tag": tag, "frames": s["frames"],
            "camera_rotation_deg": s.get("max_rotation_deg"), "parallax_px": s.get("parallax_px"),
            "ground": s.get("ground"),
            "scene": esh.get("scene", ""),
            "objects": esh.get("objects", []),               # what the set is asked to contain (boxes in plate pixels)
            "built": reg.get("added", []),                   # what was built from them (metres)
            "plate_size": reg.get("size"),
            "plates": sorted(glob.glob(os.path.join(f["work"], "plate%s_*.jpg" % tag))) or
                      [p for p in [os.path.join(f["work"], "plate%s.jpg" % tag)] if os.path.exists(p)],
            "blocks_picture": os.path.join(f["work"], "blocks%s.jpg" % tag)})
    st = status(c, cl)
    return {
        "brief_version": 1, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "clip": cl.name, "clip_id": cl.id, "film": cl.film, "video": cl.video,
        "confirmed": bool(rv.get("confirmed")), "confirmed_by": "auto" if rv.get("auto") else ("person" if rv.get("confirmed") else None),
        "status": st["label"],
        "video_info": {"frames": N, "fps": round(pose["fps"], 3), "size": [W, H], "picture_area": pose.get("picture_area")},
        "camera": dict(rv.get("camera") or {}, cuts=cam.get("cuts", [])),
        "people": people, "ignored_tracks": ignored,
        "props": (rv.get("environment") or {}).get("props", []),
        "description": rv.get("notes", ""),
        "shots": shot_rows,
        "warnings": rep.get("warnings", []),
        "files": {"marks": f["marks"], "work": f["work"], "pose": f["pose"], "review": f["review"],
                  "objects": os.path.join(f["marks"], "env_ai.json"), "scene": f["scene"], "preview": f["preview"],
                  "submit": f["submit"], "zip": f["zip"]},
        "how_to": {
            "edit_set": "write the shots' objects in files.objects ({'shots': {tag: {'scene': '', 'objects': [...]}}}), "
                        "then rebuild:  .venv\\Scripts\\python pipeline\\run_pipeline.py <video> --stage fit",
            "render": ".venv\\Scripts\\python pipeline\\run_pipeline.py <video> --stage all --force --require-confirm",
            "note": "people come from pose.json + checked.json + review.json - do not detect them again"}}


def write_clip(c, cl):
    b = clip_brief(c, cl)
    if b is None:
        return None
    p = os.path.join(folders(c, cl)["marks"], "claude_brief.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(b, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return p


def write_index(c, lib, store=None):
    """the list of all clips for Claude: CLAUDE_HANDOFF.md (to read) and .json (same content)"""
    latest = store.latest() if store else {}
    rows = []
    for cl in lib.clips:
        st = status(c, cl, latest.get(cl.id))
        f = folders(c, cl)
        bp = os.path.join(f["marks"], "claude_brief.json")
        b = read_json(bp) if os.path.exists(bp) else None
        rows.append({"film": cl.film, "clip": cl.name, "id": cl.id, "status": st["label"], "prepared": st["fit"],
                     "confirmed": st["confirmed"], "rendered": st["done"],
                     "people": [{"id": p["id"], "role": p["role"], "colour": p["colour"]} for p in (b or {}).get("people", [])],
                     "shots": len((b or {}).get("shots", [])) or None,
                     "brief": bp if b else None, "video": cl.video})
    n = len(rows); conf = sum(r["confirmed"] for r in rows); done = sum(r["rendered"] for r in rows)
    head = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "clips": n, "confirmed": conf, "rendered": done,
            "all_confirmed": n > 0 and conf == n, "source_root": c["source_root"], "work_root": c["work_root"],
            "submit_root": c["submit_root"]}
    os.makedirs(c["work_root"], exist_ok=True)
    with open(os.path.join(c["work_root"], "CLAUDE_HANDOFF.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(head, rows=rows), fh, ensure_ascii=False, indent=1)
    L = ["# Hand-off to Claude", "",
         "Written %s.  **%d of %d clips confirmed**, %d rendered.%s" % (
             head["written"], conf, n, done, "  All clips are confirmed." if head["all_confirmed"] else ""), "",
         "Each confirmed clip has a `claude_brief.json` (people, roles, colours, shots, camera, set objects, file paths).",
         "People are already detected and confirmed by a person: build the set and render, do not detect again.", "",
         "| film | clip | status | people | shots | brief |", "|---|---|---|---|---|---|"]
    for r in rows:
        ppl = ", ".join("P%s %s (%s)" % (p["id"], p["role"], p["colour"]) for p in r["people"]) or "-"
        L.append("| %s | %s | %s | %s | %s | %s |" % (r["film"], r["clip"], r["status"], ppl, r["shots"] or "-",
                                                    ("`%s`" % r["brief"]) if r["brief"] and r["confirmed"] else "-"))
    mp = os.path.join(c["work_root"], "CLAUDE_HANDOFF.md")
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return mp, head


def main():
    c = config.load()
    lib = Library(c)
    n = sum(1 for cl in lib.clips if write_clip(c, cl))
    mp, head = write_index(c, lib)
    print("%d briefs written; %d of %d clips confirmed\n%s" % (n, head["confirmed"], head["clips"], mp))
    return 0


if __name__ == "__main__":
    sys.exit(main())

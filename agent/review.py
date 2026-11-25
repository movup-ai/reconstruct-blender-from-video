# -*- coding: utf-8 -*-
"""The review (key points) of a clip: people roles / colours / heights, camera, set, objects, notes.
suggested() builds it from the local AI's ai.json when there is one, otherwise from simple rules."""
import json
import os
import time

from .clips import folders, read_json

ROLE_COLOUR = {"main": "red", "second": "blue", "third": "green", "fourth": "yellow", "extra": "purple", "ignore": None}
ORDER = ["main", "second", "third", "fourth"]


def guess_camera(cam):
    if not cam:
        return "static"
    if any(("MOVE" in w or "travel" in w or "parallax" in w) for w in cam.get("warnings", [])):
        return "moving"
    return "static" if float(cam.get("max_rotation_deg", 0)) < 0.6 else "pan"


def suggested(c, cl, confirmed=False, auto=False):
    f = folders(c, cl)
    pose = read_json(f["pose"], {}) or {}
    ai = read_json(f["ai"], {}) or {}
    cam = read_json(os.path.join(f["work"], "camera.json"), {}) or {}
    scene = read_json(os.path.join(f['work'],'scene_data.json'),{}) or {}
    body_heights = {str(ch.get('track_id')):ch.get('height') for ch in scene.get('chars',{}).values()
                    if ch.get('human_backend')=='sam3d_body'}
    people = {}
    ai_people = {str(k): v for k, v in (ai.get("people") or {}).items()}
    rank = 0
    for p in pose.get("people", []):                     # pose.json numbers people by size on screen (P1 = biggest)
        pid = str(p["id"]); a = ai_people.get(pid, {})
        n = len(p.get("frames", {}))
        if a.get("role") in ROLE_COLOUR:
            role = a["role"]
        elif a.get("is_person") is False or n < (4 if pose.get('tracker_backend')=='sam3' else 10):
            role = "ignore"
        else:
            role = ORDER[rank] if rank < len(ORDER) else "extra"
            rank += 1
        people[pid] = {"role": role, "colour": ROLE_COLOUR.get(role) or "purple",
                       "height": float(a.get("height") or body_heights.get(pid) or c["person_height"]), "name": a.get("name", "")}
    return {"clip": cl.name, "video": cl.video, "confirmed": bool(confirmed), "auto": bool(auto),
            "people": people,
            "camera": {"type": ai.get("camera") if ai.get("camera") in ("static", "pan", "moving", "handheld") else guess_camera(cam),
                       "lens": c["lens_mm"], "note": ""},
            "environment": {"ground": True, "depth_set": True, "set_colours": c.get("set_colours", "video"), "props": ai.get("props", [])},
            "notes": ai.get("description", ""), "source": "ai" if ai else "rules"}


def validate(d):
    """clean a review posted by the form"""
    if not isinstance(d, dict):
        raise ValueError("review must be an object")
    out = {"confirmed": bool(d.get("confirmed")), "auto": False, "people": {}, "camera": {}, "environment": {}, "notes": ""}
    for pid, p in (d.get("people") or {}).items():
        role = p.get("role", "extra")
        if role not in ROLE_COLOUR:
            raise ValueError("bad role %r" % role)
        h = float(p.get("height") or 1.72)
        if not 0.4 <= h <= 2.5:
            raise ValueError("height %.2f m is not plausible" % h)
        out["people"][str(pid)] = {"role": role, "colour": str(p.get("colour") or ROLE_COLOUR[role] or "purple"),
                                   "height": h, "name": str(p.get("name", ""))[:60]}
    cam = d.get("camera") or {}
    ctype = cam.get("type", "static")
    if ctype not in ("static", "pan", "moving", "handheld"):
        raise ValueError("bad camera type")
    lens = cam.get("lens")
    out["camera"] = {"type": ctype, "lens": float(lens) if lens else None, "note": str(cam.get("note", ""))[:300]}
    env = d.get("environment") or {}
    out["environment"] = {"ground": True, "depth_set": env.get("depth_set") is not False, "use_ai": env.get("use_ai") is not False,
                          "set_colours": "video" if env.get("set_colours") == "video" else "grey",
                          "props": [{k: str(p.get(k, ""))[:120] for k in ("name", "kind", "frames", "note")}
                                    for p in (env.get("props") or []) if p.get("name")]}
    out["notes"] = str(d.get("notes", ""))[:2000]
    from prop_timeline import validate as validate_timeline
    out['prop_timeline'] = validate_timeline(d.get('prop_timeline', []))
    mode = d.get('prop_mode', 'manual' if out['prop_timeline'] else 'auto')
    if mode not in ('auto', 'manual', 'off'): raise ValueError('Unknown prop mode')
    out['prop_mode'] = mode
    from limb_depth import validate as validate_depth
    out['limb_depth'] = validate_depth(d.get('limb_depth', []), set(out['people']))
    owners = set(out['people'])
    for obj in out['prop_timeline']:
        for event in obj['events']:
            if event['person'] is not None and str(event['person']) not in owners:
                raise ValueError('Prop owner is not a character in this clip')
    for k in ("clip", "video"):
        if k in d:
            out[k] = d[k]
    return out


def write(path, d):
    d = dict(d, saved=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)

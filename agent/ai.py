# -*- coding: utf-8 -*-
"""AI suggestions through the DeepSeek API: fills in the review form for a clip.

It decides who is a real person (not a driver behind glass, a poster, a reflection), which role each one has -
matched against the film's cast memory so the same character keeps the same colour in every clip - how the camera
moves, objects people hold, and a one-sentence description for meta.jsonl.

The key is entered by the user in the app (Settings) or put in deepseek_key.txt / the DEEPSEEK_API_KEY variable;
it is never shown again by the app.  Frames of the clip are sent to DeepSeek when the model accepts images;
otherwise only the measured track numbers are sent.

  python -m agent.ai <clip id>          (run by the job queue; writes <marks>/ai.json and desc.txt)"""
import base64
import glob
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from . import config  # puts the pipeline on sys.path
from .clips import Library, folders, read_json
from . import media

log = logging.getLogger("agent.ai")
ROLES = ("main", "second", "third", "fourth", "extra")
KEY_FILE = os.path.join(config.ROOT, "deepseek_key.txt")


def api_key(c):
    k = (c.get("deepseek_key") or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not k and os.path.exists(KEY_FILE):
        with open(KEY_FILE, encoding="utf-8") as fh:
            k = fh.read().strip()
    return k


def status(c):
    """no network call: ready when switched on and a key is present"""
    if not c.get("ai_enabled"):
        return {"ready": False, "detail": "switched off"}
    if not api_key(c):
        return {"ready": False, "detail": "no DeepSeek API key yet - add it under AI suggestions"}
    return {"ready": True, "detail": "DeepSeek %s" % c.get("deepseek_model")}


class ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__("DeepSeek API error %s: %s" % (code, msg))
        self.code, self.msg = code, msg


def chat(c, content, timeout=180):
    """one DeepSeek chat completion (OpenAI-compatible API) -> text"""
    body = {"model": c["deepseek_model"], "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "You analyse film shots for a 3D rebuild and answer with JSON only."},
                         {"role": "user", "content": content}]}
    req = urllib.request.Request(c["deepseek_url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key(c)})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ApiError(e.code, e.read().decode("utf-8", "replace")[:400])
    return d["choices"][0]["message"]["content"]


def test_key(c):
    """tiny request to check the key works (started by the user from the app)"""
    if not api_key(c):
        return {"ok": False, "detail": "no key saved"}
    try:
        chat(c, 'Reply with the JSON {"ok": true}', timeout=30)
    except ApiError as e:
        return {"ok": False, "detail": "wrong or expired key" if e.code in (401, 403) else "DeepSeek said: " + e.msg[:160]}
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "detail": "cannot reach DeepSeek: %s" % e}
    img = np.full((32, 32, 3), 200, np.uint8)
    try:                                                          # can this model look at frames?
        chat(c, [{"type": "text", "text": 'What colour is this image? JSON {"colour": "..."}'},
                 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64jpg(img)}}], timeout=60)
        return {"ok": True, "images": True, "detail": "the key works and %s can see images: people + environment checks on" % c["deepseek_model"]}
    except ApiError as e:
        return {"ok": True, "images": False, "detail": "the key works, but %s cannot see images: only role suggestions from the "
                "measurements; the environment check needs a model that accepts images (%s)" % (c["deepseek_model"], e.msg[:100])}
    except (urllib.error.URLError, OSError) as e:
        return {"ok": True, "images": False, "detail": "the key works; the image test could not finish: %s" % e}


def b64jpg(img, width=None):
    if width and img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()).decode()


def cast_path(c, film):
    return os.path.join(c["work_root"], "_cast", re.sub(r'[\\/:*?"<>|]+', "_", film or "film") + ".json")


TASK = """A film shot is being rebuilt as a simple 3D "white model".  {what}

Known characters of this film so far (keep the same role for the same character):
{cast}

For EVERY tracked person decide:
- is_person: false if it is not a person standing/walking/sitting in the scene (a driver seen through a car window,
  a poster, a statue, a reflection, a mannequin, a detection error)
- role: "main" (主角色), "second" (第二角色), "third", "fourth" or "extra". Reuse the known character's role when it is the
  same character. New important characters get the next free role; background people "extra".
- look: 6-12 words describing clothes / hair / build (only if you can see them)
Also give:
- camera: "static", "pan" (turns only), "moving" (dolly / tracking) or "handheld"
- props: objects a person holds or touches, as [{{"name": ..., "kind": "orange", "note": "held by P1"}}]
- description: one short sentence in Chinese describing what happens in the shot

Answer with JSON only:
{{"people": {{"<id>": {{"is_person": true, "role": "main", "look": "..."}}}}, "camera": "static", "props": [], "description": "..."}}"""


def track_facts(c, cl, pose):
    """what the local models measured, as text (all that is sent when the model cannot take images)"""
    f = folders(c, cl)
    rep = read_json(os.path.join(f["work"], "report.json"), {}) or {}
    cam = read_json(os.path.join(f["work"], "camera.json"), {}) or {}
    W, H, N = pose["width"], pose["height"], pose["frames"]
    lines = ["Clip %d frames, %dx%d. Shots: %s. Camera rotation %.1f deg, parallax %.1f px." % (
        N, W, H, ", ".join("%d-%d" % tuple(s["frames"]) for s in cam.get("shots", [])) or "one",
        float(cam.get("max_rotation_deg") or 0), float(cam.get("parallax_px") or 0))]
    fits = {str(p["track"]): p for p in rep.get("people", [])}
    for p in pose["people"]:
        fs = sorted(int(x) for x in p["boxes"])
        bx = np.array([p["boxes"][str(x)][:4] for x in fs], float)
        hgt = float(np.median(bx[:, 3] - bx[:, 1])); cx = float(np.median((bx[:, 0] + bx[:, 2]) / 2))
        move = float(np.hypot(*(bx[-1, :2] - bx[0, :2])))
        vis = float(np.mean([np.mean(np.array(v)[:, 2] > 0.3) for v in p["frames"].values()]))
        lines.append("P%d: frames %d-%d (%d), height %.0f%% of picture, %s, moves %.0f px, %.0f%% of joints visible, 3D fit error %s px" % (
            p["id"], fs[0], fs[-1], len(fs), 100 * hgt / H, "left" if cx < W / 3 else "right" if cx > 2 * W / 3 else "centre",
            move, 100 * vis, (fits.get(str(p["id"])) or {}).get("median_error_px", "?")))
    return "\n".join(lines)


def suggest(c, cl):
    f = folders(c, cl)
    pose = read_json(f["pose"])
    if not pose:
        raise RuntimeError("the clip has not been prepared (no pose.json)")
    crops = media.person_crops(c, cl)
    ids = [p["id"] for p in pose["people"] if p["id"] in crops][:6]
    cast = read_json(cast_path(c, cl.film), {}) or {}
    cast_txt = "\n".join("- %s: %s" % (r, d) for r, d in cast.items()) or "(none yet - this is the first clip)"
    facts = track_facts(c, cl, pose)
    mode, text, t0 = "images", None, time.time()
    if c.get("ai_send_images", True):
        frame = media.labelled_frame(c, cl, width=768)
        what = ("Image 1 is one frame; every tracked person has a yellow box labelled P<number>. The next images are close-ups "
                "of the tracked people in this order: %s.\nMeasured by the tracker:\n%s" % (", ".join("P%d" % i for i in ids), facts))
        parts = [{"type": "text", "text": TASK.format(what=what, cast=cast_txt)}]
        imgs = ([frame] if frame is not None else []) + [cv2.imdecode(np.fromfile(crops[i], np.uint8), cv2.IMREAD_COLOR) for i in ids]
        parts += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64jpg(im, 640)}} for im in imgs if im is not None]
        try:
            text = chat(c, parts)
        except ApiError as e:
            if e.code in (401, 402, 403, 429):
                raise
            print("the model did not accept images - sending the measurements only (%s)" % e.msg[:120], flush=True)
    if text is None:
        mode = "measurements"
        text = chat(c, TASK.format(what="You cannot see the frames; here is what the tracker measured:\n" + facts, cast=cast_txt))
    data = parse_json(text)
    out = {"provider": "deepseek", "model": c["deepseek_model"], "mode": mode, "seconds": round(time.time() - t0, 1),
           "people": {}, "raw": text[:4000]}
    for pid, p in (data.get("people") or {}).items():
        pid = re.sub(r"\D", "", str(pid))
        if not pid or not isinstance(p, dict):
            continue
        role = p.get("role") if p.get("role") in ROLES else "extra"
        if p.get("is_person") is False:
            role = "ignore"
        out["people"][pid] = {"is_person": p.get("is_person", True) is not False, "role": role, "look": str(p.get("look", ""))[:120]}
    out["camera"] = data.get("camera") if data.get("camera") in ("static", "pan", "moving", "handheld") else None
    out["props"] = [{"name": str(p.get("name", ""))[:60], "kind": p.get("kind") if p.get("kind") in
                     ("orange", "pink", "purple", "brown", "white", "grey") else "orange", "frames": "all", "note": str(p.get("note", ""))[:120]}
                    for p in (data.get("props") or []) if isinstance(p, dict) and p.get("name")][:6] if mode == "images" else []
    out["description"] = str(data.get("description", ""))[:200] if mode == "images" else ""
    return out


ENV_TASK = """These images are the empty background of each shot of a film clip (people removed). It will be rebuilt as a
simple 3D "white model" set, like a stage built from plain shapes: every important object becomes a simple model of
its type (a door with frame and handle, shelves, a counter, a table with legs, a car, a tree...).
Our depth tools already outlined some regions: orange boxes "W<n>" are walls / things standing on the ground,
cyan boxes "B<n>" are groups of depth blocks. Shots in order: {shots}.

1) For every outlined region say what it is and whether it is a correct solid part of the set
   (correct=false when it is really sky, floor, a reflection, a blur or noise).
2) List EVERY object a viewer would recognise in the shot (furniture, doors, windows, shelves, appliances, vehicles,
   trees, plants, fences, lamp posts, signs, buildings in the background...), whether or not it was outlined.
   For each give:
   - type: one of {types}
   - box: [x0, y0, x1, y1] in 0-1000 units of that shot's image width / height, tight around the object; the bottom
     edge is where it touches the floor (or its lowest point if it hangs on a wall)
   - attach: "floor" or "wall"
   - facing: which way the object's front points: "camera", "left", "right" or "away" (for vehicles: the bonnet)
   - size: [width, height, depth] in metres, your best estimate
   - interactive: true if a person touches or uses it in this shot
Keep it to the 25 most important objects per shot; big things first.

Answer JSON only:
{{"shots": {{"<shot>": {{"scene": "indoor kitchen / outdoor street / ...",
  "regions": {{"W0": {{"what": "building facade", "correct": true}}}},
  "objects": [{{"what": "kitchen cabinet", "type": "cabinet", "box": [0, 0, 0, 0], "attach": "floor", "facing": "camera",
               "size": [0.8, 0.9, 0.6], "interactive": false}}]}}}}}}"""


def suggest_env(c, cl):
    """AI check of the set the local models built: label regions, drop wrong ones, add missed objects -> env_ai.json"""
    f = folders(c, cl)
    shots = []
    for rp in sorted(glob.glob(os.path.join(f["work"], "regions*.json")))[:4]:
        reg = read_json(rp)
        pp = os.path.join(f["work"], "plate%s.jpg" % reg["tag"])
        if not reg or not os.path.exists(pp):
            continue
        img = cv2.imdecode(np.fromfile(pp, np.uint8), cv2.IMREAD_COLOR)
        sc = 768.0 / img.shape[1]
        img = cv2.resize(img, (768, int(img.shape[0] * sc)))
        for r in reg["regions"]:
            x0, y0, x1, y1 = [int(v * sc) for v in r["bbox"]]
            colr = (0, 140, 255) if r["type"] == "wall" else (255, 220, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), colr, 2)
            cv2.putText(img, r["id"], (x0 + 3, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(img, r["id"], (x0 + 3, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colr, 2)
        shots.append((reg, img))
    if not shots:
        return None
    from props import TYPES                                              # pipeline
    parts = [{"type": "text", "text": ENV_TASK.format(shots=", ".join(r["tag"] for r, _ in shots),
                                                       types=", ".join(sorted(t for t in TYPES if t != "other")))}]
    parts += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64jpg(im)}} for _, im in shots]
    data = parse_json(chat(c, parts))
    out = {"provider": "deepseek", "model": c["deepseek_model"], "shots": {}}
    for reg, im in shots:
        d = (data.get("shots") or {}).get(reg["tag"]) or {}
        W, H = reg["size"]
        byid = {r["id"]: r for r in reg["regions"]}
        sh = {"scene": str(d.get("scene", ""))[:80], "labels": {}, "drop": [], "objects": []}
        for rid, q in (d.get("regions") or {}).items():
            if rid in byid and isinstance(q, dict):
                sh["labels"][rid] = str(q.get("what", ""))[:60]
                if q.get("correct") is False:
                    sh["drop"].append({"id": rid, "what": sh["labels"][rid], "bbox": byid[rid]["bbox"]})
        for o in ((d.get("objects") or []) + (d.get("missing") or []))[:30]:
            try:
                b = [float(v) for v in o["box"]]
            except (KeyError, TypeError, ValueError):
                continue
            if len(b) != 4 or b[2] <= b[0] or b[3] <= b[1]:
                continue
            size = o.get("size")
            try:
                size = [float(v) for v in size] if isinstance(size, list) and len(size) == 3 else None
            except (TypeError, ValueError):
                size = None
            sh["objects"].append({"what": str(o.get("what", "object"))[:40], "type": str(o.get("type") or o.get("kind") or "other")[:20],
                                  "bbox": [b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H],
                                  "attach": o.get("attach") if o.get("attach") in ("floor", "wall") else None,
                                  "facing": str(o.get("facing", "camera"))[:10], "size": size,
                                  "colour_role": "orange" if o.get("interactive") else None})
        out["shots"][reg["tag"]] = sh
    return out


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("the model did not answer with JSON: %r" % text[:200])
        return json.loads(m.group(0))


def remember_cast(c, cl, ai):
    p = cast_path(c, cl.film)
    cast = read_json(p, {}) or {}
    for q in ai["people"].values():
        if q["role"] in ("main", "second", "third", "fourth") and q.get("look") and q["role"] not in cast:
            cast[q["role"]] = q["look"]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(cast, fh, ensure_ascii=False, indent=1)


def main(argv):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    c = config.load()
    st = status(c)
    if not st["ready"]:
        raise SystemExit("AI suggestions are not available: " + st["detail"])
    cl = Library(c).get(argv[0])
    print("AI suggestions for", cl.name, "with DeepSeek", c["deepseek_model"], flush=True)
    ai = suggest(c, cl)
    f = folders(c, cl)
    with open(f["ai"], "w", encoding="utf-8") as fh:
        json.dump(ai, fh, ensure_ascii=False, indent=1)
    if ai["description"]:
        with open(os.path.join(f["marks"], "desc.txt"), "w", encoding="utf-8") as fh:
            fh.write(ai["description"])
    remember_cast(c, cl, ai)
    print("mode:", ai["mode"], " people:", json.dumps(ai["people"], ensure_ascii=False), flush=True)
    print("camera:", ai["camera"], " description:", ai["description"], " (%.0fs)" % ai["seconds"], flush=True)
    if ai["mode"] != "images":
        print("environment check skipped: the DeepSeek model does not accept images", flush=True)
        return
    env = suggest_env(c, cl)
    old = read_json(os.path.join(f["marks"], "env_ai.json"), {}) or {}
    if env and old.get("edited"):                                  # objects a person edited stay as they are
        for tag, sh in env["shots"].items():
            if tag in (old.get("shots") or {}):
                sh["objects"] = old["shots"][tag].get("objects", [])
        env["edited"] = old["edited"]
    if env:
        with open(os.path.join(f["marks"], "env_ai.json"), "w", encoding="utf-8") as fh:
            json.dump(env, fh, ensure_ascii=False, indent=1)
        for tag, sh in env["shots"].items():
            print("environment %s: %s | %d regions labelled, %d wrong, %d objects added" % (
                tag, sh["scene"], len(sh["labels"]), len(sh["drop"]), len(sh["objects"])), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

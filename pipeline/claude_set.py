# -*- coding: utf-8 -*-
"""Claude builds a clip's set, the way a Claude session does it by hand (docs/ENV_AUTHORING.md), but inside the app.

  python pipeline\\claude_set.py <clip name>      ->  output\\<clip>\\env_claude_ai.json  (then build_env.py --env it)

Why: the local models measure (depth, outlines, class names) but do not understand a room: a curtain cut into strips,
a wall seen through a doorway, a pillar half hidden by a person.  Claude looking at the frames does.  So Claude writes
the set and the local measurements are its ruler:
  - the shot's first / middle / last frame with a pixel grid (the same pixels the set file uses),
  - the fitted camera (height, tilt, lens) and, per picture row, how far away the floor is there,
  - where every fitted person stands (feet pixel, distance) - the set must leave them room,
  - what the depth model measured (env_auto.json: walls, pieces, plants... with their pixel box and distance),
  - every object annotated on the clip page (its box and description) - each one must be in the set.
Claude answers with the set in the env_claude.json format, placed by pixel box + distance / depth, so everything
lines up with the film from the fitted camera.  Then build_env renders the check picture (film | set over film |
set) and Claude sees it and corrects the set once more ("set_rounds", default 2 rounds).

Model: config "set_model" (default claude-sonnet-5-5).  Key: Settings (config "anthropic_key") or ANTHROPIC_API_KEY.
The frames of the clip are sent to the Claude API.  A refused request falls back to another model on Anthropic's side
(server-side fallback)."""
import argparse, base64, importlib.util, json, math, os, re, subprocess, sys, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload, jsave, output_dir, AGENT, read_frames_at, imwrite_u

DEFAULT_MODEL = "claude-sonnet-5-5"
MODELS = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}     # $ per million tokens in / out
MAX_SIDE = 1568                                                   # picture size Claude reads at full detail
SKIP_SECTIONS = ("People on horses", "Legs, kneeling", "People on steps", "Wrong person tracks")


def api_key(c):
    return (c.get("anthropic_key") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()


def available(c):
    return bool(api_key(c)) and importlib.util.find_spec("anthropic") is not None


def model_of(c):
    m = str(c.get("set_model") or DEFAULT_MODEL)
    return m if m in MODELS else DEFAULT_MODEL


# ------------------------------------------------------------------ what Claude is told
def spec_text():
    """the set file format from docs/ENV_AUTHORING.md (coordinates, format, placement, types, colours, what good looks
    like), without the parts about fitting people (hints.json)"""
    p = os.path.join(AGENT, "docs", "ENV_AUTHORING.md")
    s = open(p, encoding="utf-8").read()
    s = s[s.index("## Coordinates"):]
    out, keep = [], True
    for line in s.split("\n"):
        if line.startswith("### ") or line.startswith("## "):
            keep = not any(k in line for k in SKIP_SECTIONS)
        if keep:
            out.append(line)
    return "\n".join(out)


SYSTEM = """You build the 3D "white model" set of one shot of a film clip, the way an artist would block it out in
grey shapes, from the frames you are shown and the measurements below them. People are NOT yours: they are fitted
from the film and placed already; leave room where they walk.

Work like this:
1. Look at the frames. Name the space (corridor, office, street, forest...) and its main surfaces: floor, back wall,
   side walls, ceiling line, doorways, windows, curtains, pillars / columns, big furniture, trees, cars, buildings.
2. Place each surface and object with the pixel grid (grid labels are true picture pixels) and the measured numbers:
   the floor-distance table says how far the floor is at each picture row (a thing standing on the floor at its
   bottom edge row y is that far away); the people's distances are reliable anchors; the depth model's pieces give
   distances for walls and objects (their classes and shapes are often wrong - walls cut into strips, a curtain
   called a door - but their distances are measured).
3. Prefer these placements, so everything lines up with the film from the fitted camera:
   - things standing on the floor: {"type": T, "bbox": [x0, y0, x1, y1], "dist": metres};
   - a back wall facing the camera: "bbox" + "depth" (metres forward) + "thick": 0.1, or a "box" in metres;
   - side walls and walls at an angle: {"type": "box", "bbox": [...], "plane": [xa, ya, xb, yb], "z0": 0, "thick": 0.15},
     or a "patch" with "pts3" corners in metres;
   - curtains: type "curtain"; pillars / columns: type "pillar" (bbox + dist, size from the box);
   - plants, statues and other irregular things are built by the app from their measured shape: still list them
     (type "plant_pot", "box"...) with their bbox, dist and "what".
4. Every object annotated by the user (listed below with its box and description) must be in your set, at that box,
   as what the description says (type "other" means: read the description). Skip annotations the description says a
   person carries or holds - the app puts those on the person.  A "distance level" (1 near .. 9 far) orders them:
   a bigger level is always farther from the camera than every smaller one, and the biggest level in a shot is the
   back of the set, behind every person on every frame (nobody may walk through it).
5. Give every object "what": a short English name ("back wall", "left curtain", "white pillar", "potted plant").
   Add "role": "lamp" for light sources, "near" for things people use close by (a cup, a door they open),
   "far_i" for interactive things far from people (a distant door). Nothing else needs a colour.
6. Readable beats detailed: the main surfaces first, then 5-30 objects. Nothing where a person stands or walks.

Answer with ONE JSON object for this shot - {"ground": {...}, "sky": [r, g, b], "objects": [...]} - in a ```json
block, after at most a few lines of reasoning. The set file format:

"""


def grid(img, step=100):
    """the picture with a yellow grid, labelled every 100 true picture pixels (the coordinates of the set file)"""
    im = img.copy(); H, W = im.shape[:2]
    fs = max(0.7, W / 2200.0)
    for x in range(0, W, step):
        cv2.line(im, (x, 0), (x, H - 1), (0, 255, 255), 1)
        cv2.putText(im, str(x), (x + 3, int(26 * fs)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * fs, (0, 255, 255), 2)
    for y in range(0, H, step):
        cv2.line(im, (0, y), (W - 1, y), (0, 255, 255), 1)
        cv2.putText(im, str(y), (3, y + int(26 * fs)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * fs, (0, 255, 255), 2)
    return im


def jpeg_block(img, q=88):
    H, W = img.shape[:2]
    k = min(1.0, MAX_SIDE / float(max(H, W)))
    if k < 1:
        img = cv2.resize(img, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(buf.tobytes()).decode("ascii")}}


def shot_tags(shots):
    """{shot number: env_ai.json tag prefix} ("" for one shot: tags "_000"; "_s2" for shot 2 of several)"""
    return {s["shot"]: ("" if len(shots) == 1 else "_s%d" % s["shot"]) for s in shots}


def facts(name, c, s, data, ci, env_auto, env_ai, n_shots):
    """the measurements of one shot as text"""
    import build_env as BE
    from rig import Rig, JOINTS
    import describe
    cam = BE.Cam(data)
    a, b = s["frames"]; W, Hp = data["res"]
    g = s.get("ground") or {}
    fov = 2 * math.degrees(math.atan(W / 2.0 / cam.f_at(a)))
    zb = cam.f_at(b) / cam.f_at(a)
    L = ["Picture: %d x %d px (grid labels are these pixels). Frames %d-%d of the clip (%.1f s)." % (W, Hp, a, b, (b - a + 1) / float(data.get("fps", 24))),
         "Camera (fitted): %.2f m above the floor, looking down %.1f deg, horizontal field of view %.0f deg, turns %.0f deg "
         "during the shot%s%s." % (float(data["cam"][a][2]), 90.0 - float(data["cam"][a][3]), fov, float(s.get("max_rotation_deg", 0) or 0),
                                  "" if "camera_height" in g else " (height is a guess: no full-body person to measure it)",
                                  "" if abs(zb - 1) < 0.03 else "; the lens ZOOMS %s x%.2f by the last frame, so things at the "
                                  "picture's edges in the first frame leave it" % ("in" if zb > 1 else "out", zb))]
    rows = []                                                     # floor distance at picture rows (first frame, centre)
    for v in range(0, Hp + 1, 50):
        C, d = cam.ray(W / 2.0, min(v, Hp - 1), a)
        rows.append("y %4d: %s" % (v, ("%.1f m" % (math.hypot(*(C + d * (-C[2] / d[2]) - C)[:2]))) if d[2] < -1e-3 and -C[2] / d[2] < 200 else "above the horizon"))
    L.append("Floor distance at picture row (frame %d, middle of the picture; a thing standing on the floor whose bottom "
             "edge is at that row is that far, along the floor):\n  " % a + "\n  ".join(rows))
    ppl = []
    for kind, ch in data["chars"].items():
        vis = ch.get("visible") or []
        fs = [f for f in (a, (a + b) // 2, b) if f < len(vis) and vis[f]]
        if not fs:
            continue
        rig = Rig(ch["height"], ch.get("style", "sample"))
        parts = []
        for f in fs:
            r = ch["root"][f]
            P = rig.coco_points(rig.fk((r[0], r[1], r[2]), r[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f]))
            uv, z = cam.project(P, f)
            ok = z > 0.2
            if not ok.any():
                continue
            box = [int(uv[ok, 0].min()), int(uv[ok, 1].min()), int(uv[ok, 0].max()), int(uv[ok, 1].max())]
            C = np.array(data["cam"][f][:3])
            parts.append("frame %d: body box %s, feet %.1f m from the camera" % (f, box, math.hypot(r[0] - C[0], r[1] - C[1])))
        if parts:
            ppl.append("%s (%.2f m tall): %s" % (kind, ch["height"], "; ".join(parts)))
    L.append("People (fitted, reliable distances; keep their floor free):\n  " + ("\n  ".join(ppl) or "none"))
    se = (env_auto.get("shots") or {}).get(str(s["shot"])) if env_auto.get("shots") else env_auto
    pieces = []
    for o in (se or {}).get("objects", []):
        src = o.get("_src") or {}
        if o.get("off") or not src.get("px"):
            continue
        if src.get("frame", a) != a and src.get("cls") != "drawn":   # later frames: the depth model's scale drifts
            continue                                                 # (their distances would mislead) - first frame only
        if "pos" in o:
            P = np.array(o["pos"][:2], float)
        elif o.get("pts3"):
            P = np.array(o["pts3"], float)[:, :2].mean(0)
        elif o.get("v"):
            P = np.array(o["v"], float)[:, :2].mean(0)
        elif o.get("dist"):
            P = None
        else:
            continue
        C = np.array(data["cam"][src.get("frame", a)][:3]) if src.get("frame", a) < len(data["cam"]) else np.zeros(3)
        dist = o.get("dist") if P is None else math.hypot(P[0] - C[0], P[1] - C[1])
        lab = (src.get("label") or [None])[0] or src.get("cls")
        pieces.append("%s (depth model: %s%s) frame %d box %s, about %.1f m away" % (
            o.get("name", "?"), lab, ", measured shape" if o.get("type") == "mesh" else "", src.get("frame", a), src["px"], float(dist or 0)))
    L.append("What the depth model measured in frame %d (classes / shapes often wrong, distances measured; where they "
             "disagree with the people's distances or the floor table, trust those):\n  " % a + ("\n  ".join(pieces) or "nothing"))
    tagp = "" if n_shots == 1 else "_s%d" % s["shot"]
    ann, held = [], []
    for tag, sh in ((env_ai or {}).get("shots") or {}).items():
        if not re.fullmatch(re.escape(tagp) + r"_(\d+)", tag):
            continue
        f = a + int(tag.rsplit("_", 1)[1])
        for o in (sh or {}).get("objects", []):
            if len(o.get("bbox") or []) != 4:
                continue
            d = describe.parse(o.get("what"), o.get("type"))
            line = "frame %d box %s: \"%s\" (type %s%s)" % (f, [round(v) for v in o["bbox"]], o.get("what") or "", o.get("type") or "other",
                                                         (", on the %s" % o["attach"] if o.get("attach") else "") +
                                                         (", distance level %s" % o["level"] if o.get("level") else "") +
                                                         (", \"role\": \"%s\"" % o["role"] if o.get("role") else ""))
            (held if d["mode"] == "held" else ann).append(line)
    L.append("Objects the user annotated - every one must be in your set at its box:\n  " + ("\n  ".join(ann) or "none"))
    if held:
        L.append("Annotated as carried by a person (NOT in the set):\n  " + "\n  ".join(held))
    return "\n\n".join(L)


# ------------------------------------------------------------------ talking to Claude
def ask(c, system, content, log=print):
    """one request -> (reply text, usage); streamed (a long answer must not time out)"""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key(c), max_retries=3)
    with client.beta.messages.stream(
            model=model_of(c), max_tokens=32000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            output_config={"effort": str(c.get("set_effort") or "high")},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default") as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request (%s)" % getattr(getattr(msg, "stop_details", None), "category", None))
    text = "".join(b.text for b in msg.content if b.type == "text")
    if msg.stop_reason == "max_tokens":
        log("    (the answer hit the length limit)")
    return text, msg.usage


def parse_set(text):
    """the shot's set from Claude's answer: the last ```json block (else the outermost braces) -> dict"""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = blocks[-1] if blocks else text[text.find("{"): text.rfind("}") + 1]
    env = json.loads(cand)
    if not isinstance(env, dict) or not isinstance(env.get("objects"), list):
        raise ValueError("no objects list in the answer")
    keep = []
    for o in env["objects"]:
        if isinstance(o, dict) and o.get("type") and any(k in o for k in ("bbox", "pos", "px", "pts", "pts3")):
            keep.append(o)
    env["objects"] = keep[:80]
    env.setdefault("ground", {"size": 30, "colour": "ground"})
    return env


def cost(usage, model):
    pi, po = MODELS.get(model, (2.0, 10.0))
    tin = (usage.input_tokens or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * 1.25 + (getattr(usage, "cache_read_input_tokens", 0) or 0) * 0.1
    return (tin * pi + (usage.output_tokens or 0) * po) / 1e6


def test_key(c):
    """Settings -> Test: a tiny request -> {ok, detail}"""
    if not available(c):
        return {"ok": False, "detail": "no key saved" if not api_key(c) else "the anthropic package is not installed - run Setup.bat"}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key(c), max_retries=1)
        r = client.messages.create(model=model_of(c), max_tokens=64, output_config={"effort": "low"},
                                   messages=[{"role": "user", "content": "Reply with the single word: ready"}])
        return {"ok": True, "detail": "%s answers: the key works" % r.model}
    except anthropic.AuthenticationError:
        return {"ok": False, "detail": "the key was refused (wrong or revoked)"}
    except anthropic.PermissionDeniedError:
        return {"ok": False, "detail": "the key may not use %s" % model_of(c)}
    except anthropic.RateLimitError:
        return {"ok": False, "detail": "rate limited or out of credit - try again later"}
    except anthropic.APIConnectionError:
        return {"ok": False, "detail": "cannot reach api.anthropic.com from this PC"}
    except anthropic.APIStatusError as e:
        return {"ok": False, "detail": "API error %s: %s" % (e.status_code, str(e.message)[:160])}


SOFT_WORDS = ("plant", "flower", "vase", "pot", "statue", "sculpture", "bush", "shrub", "cushion", "pillow", "blanket",
              "towel", "clothes", "bag", "basket", "animal", "dog", "cat", "bicycle", "bike", "fountain", "chandelier")


def measured_shapes(se_claude, se_auto, frame0):
    """Claude's plants / statues / other irregular things -> the measured shape the depth model built for the same box
    (env_auto "mesh" objects), so they look like the film instead of a library template.  -> number replaced"""
    from common import box_iou
    from env_auto import px_inside
    meshes = [o for o in (se_auto or {}).get("objects", []) if o.get("type") == "mesh" and (o.get("_src") or {}).get("px")]
    if not meshes:
        return 0
    out, n = [], 0
    used = set()
    for o in se_claude["objects"]:
        words = ("%s %s" % (o.get("type", ""), o.get("what", ""))).lower().replace("_", " ")
        soft = any(w in words for w in SOFT_WORDS)
        if not soft or len(o.get("bbox") or []) != 4:
            out.append(o); continue
        f = int(o.get("frame", frame0))
        hit = [i for i, m in enumerate(meshes) if i not in used and int(m["_src"].get("frame", frame0)) == f and
               (box_iou(m["_src"]["px"], o["bbox"]) > 0.3 or px_inside(m["_src"]["px"], o["bbox"]) > 0.6)]
        if not hit:
            out.append(o); continue
        for i in hit:
            used.add(i)
            m = dict(meshes[i]); m["what"] = o.get("what") or m.get("what"); m["name"] = "claude_%s" % m.get("name", "mesh")
            if o.get("role"):
                m["role"] = o["role"]
            out.append(m)
        n += 1
    se_claude["objects"] = out
    return n


# ------------------------------------------------------------------ one clip
def assemble(shot_envs, n_shots):
    if n_shots == 1:
        env = dict(shot_envs[1]); env.setdefault("sky", [205, 207, 210])
        return env
    return {"sky": [205, 207, 210], "shots": {str(k): v for k, v in shot_envs.items()}}


def build(name, env_path, frames=None):
    py = sys.executable
    cmd = [py, os.path.join(HERE, "build_env.py"), name, "--env", env_path] + (["--frames", ",".join(str(f) for f in frames)] if frames else [])
    r = subprocess.run(cmd, cwd=AGENT, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError("build_env failed: %s" % (r.stderr or r.stdout)[-400:])


def run(name, c, log=print):
    import run_pipeline as rp
    work = os.path.join(c["work_root"], name)
    ci = jload(os.path.join(work, "clip_info.json")); data = jload(os.path.join(work, "scene_data.json"))
    camj = jload(os.path.join(work, "camera.json"))
    shots = camj.get("shots") or [{"shot": 1, "frames": [0, data["frames"] - 1]}]
    out_dir = os.path.join(output_dir(), name)
    ea = os.path.join(out_dir, "env_auto.json")
    env_auto = jload(ea) if os.path.exists(ea) else {}
    drawn = os.path.join(rp.marks_dir(ci["video"], c), "env_ai.json")
    env_ai = jload(drawn) if os.path.exists(drawn) else {}
    model = model_of(c)
    system = SYSTEM + spec_text()
    keyf = sorted({f for s in shots for f in (s["frames"][0], (s["frames"][0] + s["frames"][1]) // 2, s["frames"][1])})
    pics = read_frames_at(ci["video"], keyf)
    y0, Hp = ci["y0"], ci["Hpic"]
    shot_envs, spent, notes = {}, 0.0, {}
    t0 = time.time()
    for s in shots:                                               # round 1: the set from the frames and the numbers
        a, b = s["frames"]
        content = []
        for f in sorted({a, (a + b) // 2, b}):
            content.append({"type": "text", "text": "Frame %d:" % f})
            content.append(jpeg_block(grid(pics[f][y0:y0 + Hp])))
        content.append({"type": "text", "text": facts(name, c, s, data, ci, env_auto, env_ai, len(shots)) +
                        "\n\nWrite the set of this shot (shot %d of %d)." % (s["shot"], len(shots))})
        text, usage = ask(c, system, content, log)
        spent += cost(usage, model)
        shot_envs[s["shot"]] = parse_set(text)
        notes[s["shot"]] = text[:text.find("```")].strip()[:600] if "```" in text else ""
        log("  Claude set, shot %d: %d objects" % (s["shot"], len(shot_envs[s["shot"]]["objects"])))
    out = os.path.join(out_dir, "env_claude_ai.json")
    rounds = max(1, int(c.get("set_rounds", 2)))
    for rnd_ in range(2, rounds + 1):                             # check rounds: Claude sees its set over the film
        jsave(out, assemble(shot_envs, len(shots)))
        for s in shots:
            a, b = s["frames"]
            fs = sorted({a, (a + b) // 2, b})
            build(name, out, fs)
            chk = cv2.imdecode(np.fromfile(os.path.join(work, "env_check.jpg"), np.uint8), cv2.IMREAD_COLOR)
            content = [{"type": "text", "text": "Your set for shot %d, built. Each row is one frame: the film | your set over the film | "
                                                "your set alone (people as red / blue / green blocks; a red X means a body is inside an "
                                                "object). Grid pixels of the first picture are picture pixels / the sheet is scaled." % s["shot"]},
                       jpeg_block(chk),
                       {"type": "text", "text": "Your set (JSON):\n```json\n%s\n```\n\n%s\n\nCompare with the film. Fix what is off: walls "
                                                "or pillars at the wrong place or depth, missing main surfaces or annotated objects, things "
                                                "floating, too big, or standing where a person walks. Answer with the whole corrected set "
                                                "for this shot in one ```json block (the same set if it is right)." % (
                                                    json.dumps(shot_envs[s["shot"]], ensure_ascii=False),
                                                    facts(name, c, s, data, ci, env_auto, env_ai, len(shots)))}]
            try:
                text, usage = ask(c, system, content, log)
                spent += cost(usage, model)
                shot_envs[s["shot"]] = parse_set(text)
                log("  Claude check round %d, shot %d: %d objects" % (rnd_, s["shot"], len(shot_envs[s["shot"]]["objects"])))
            except (ValueError, json.JSONDecodeError) as e:
                log("  Claude check round %d, shot %d: answer not usable (%s) - the earlier set is kept" % (rnd_, s["shot"], e))
    for s in shots:                                               # irregular things: their measured shapes
        se_a = (env_auto.get("shots") or {}).get(str(s["shot"])) if env_auto.get("shots") else env_auto
        k = measured_shapes(shot_envs[s["shot"]], se_a, s["frames"][0])
        if k:
            log("  shot %d: %d plant(s) / irregular thing(s) built from their measured shape" % (s["shot"], k))
    env = assemble(shot_envs, len(shots))
    env["_claude"] = {"model": model, "rounds": rounds, "cost_usd": round(spent, 3), "seconds": round(time.time() - t0, 1),
                      "notes": {str(k): v for k, v in notes.items() if v}}
    env["_inputs"] = dict(rp.set_inputs(work, drawn), model=model)
    jsave(out, env)
    log("  Claude set written (%s, %d round(s), %.0f s, about $%.2f): %s" % (model, rounds, time.time() - t0, spent, out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    a = ap.parse_args()
    c = cfg()
    if not available(c):
        raise SystemExit("no Claude key (Settings -> Background set, or ANTHROPIC_API_KEY) or the anthropic package is missing")
    run(a.name, c)


if __name__ == "__main__":
    main()

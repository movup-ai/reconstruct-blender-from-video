# -*- coding: utf-8 -*-
"""AI scene editing: a typed command (+ screenshots) -> a list of safe edit operations on scene_data.json.
The AI never writes Blender code; it can only use the operations below, which this file applies."""
import json, re, base64, math, copy
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from rig import Rig, JOINTS

OPS_DOC = r"""
World: metres, z up, ground at z=0. The camera starts near (0,0,height) looking along +Y.
A person faces +Y at yaw 0 and its RIGHT side is +X. Frames are numbered 1..N (as in Blender).
"from"/"to" are optional (default: all frames), inclusive.
People are P1, P2, ... (P1 = main character). Joints: spine chest neck head shoulder_L elbow_L wrist_L
shoulder_R elbow_R wrist_R hip_L knee_L ankle_L hip_R knee_R ankle_R.
Angle hints (degrees): spine x+ bends forward; head z+ turns the head to the person's left; shoulder x+ raises the arm
forward; shoulder_L y+ / shoulder_R y- raise the arm sideways; elbow x+ bends the elbow; hip x+ moves the leg forward;
knee x- bends the knee.

Operations (JSON objects in "ops"):
 {"op":"move_person","person":"P1","dx":0,"dy":0,"dz":0,"from":1,"to":N}
 {"op":"turn_person","person":"P1","deg":30,"from":..,"to":..}
 {"op":"bend_joint","person":"P1","joint":"elbow_R","axis":"x","deg":20,"from":..,"to":..}
 {"op":"set_person","person":"P2","colour":"blue" or [r,g,b],"height":1.6}
 {"op":"show_person","person":"P2","visible":false,"from":..,"to":..}
 {"op":"delete_person","person":"P3"}
 {"op":"swap_people","a":"P1","b":"P2"}           (swaps colour + role label, e.g. main <-> second)
 {"op":"smooth_person","person":"P1","amount":2}   (smoother, less jittery motion)
 {"op":"set_colour","target":"ground"|"set"|"far"|"backdrop"|"marker"|"sky"|"red"|...|<mesh name or name prefix>,"rgb":[r,g,b]}
 {"op":"delete_mesh","name":"<mesh name or prefix>"}
 {"op":"move_mesh","name":"<mesh name or prefix>","dx":0,"dy":0,"dz":0}
 {"op":"add_box","name":"rock_1","size":[sx,sy,sz],"at":[x,y,z_bottom],"yaw":0,"colour":"set"|"orange"|[r,g,b]}
 {"op":"add_hand_prop","name":"rake","person":"P1","hand":"R","size":[0.04,0.04,1.6],"offset":[0,0,-0.1],"colour":"orange","from":..,"to":..}
      (an interactable object held in the hand; offset/size are in the hand's own frame: z runs down the forearm)
 {"op":"delete_prop","name":"rake"}
 {"op":"camera","dx":0,"dy":0,"dz":0,"tilt":0,"pan":0,"roll":0,"lens":null}   (tilt+ looks up, pan+ turns left)
 {"op":"exposure","value":1.2}
Colours of roles: interactable props = "orange"; main character = "red"; second = "blue"; scene = greys.
"""

SYSTEM = """You edit a 3D white-model (白模) scene that rebuilds a film clip in Blender with grey blocks and
colour-coded mannequins. The user gives a command, often with screenshots (the white-model frame, the original
film frame, or their own marked-up screenshot). Turn the command into edit operations.
Reply with ONE JSON object only, no other text:
{"reply": "<one or two short sentences to the user, in the user's language>", "ops": [ ... ]}
Use only the operations listed. If the command is unclear or impossible, return "ops": [] and ask in "reply".
Estimate distances in metres from the pictures and the scene summary; prefer small, safe changes.
""" + OPS_DOC

def summarize(data):
    N = data["frames"]
    ppl = []
    for k, ch in data["chars"].items():
        vis = [i for i, v in enumerate(ch["visible"]) if v] or [0]
        pick = sorted(set([vis[0], vis[len(vis) // 2], vis[-1]]))
        ppl.append({"id": k, "role": ch["label"], "colour": ch["colour"], "height": ch["height"],
                    "visible_frames": [vis[0] + 1, vis[-1] + 1],
                    "position_xy_at": {str(f + 1): [round(ch["root"][f][0], 2), round(ch["root"][f][1], 2)] for f in pick},
                    "yaw_at": {str(f + 1): round(ch["root"][f][3]) for f in pick}})
    meshes = []
    for m in data["meshes"][:60]:
        V = np.array(m["v"], float)
        if len(V):
            meshes.append({"name": m["name"], "colour": m["colour"], "centre": V.mean(0).round(1).tolist(),
                           "size": np.ptp(V, 0).round(1).tolist(), "faces": len(m["f"])})
    c0 = data["cam"][0]
    return {"frames": N, "fps": data["fps"], "people": ppl, "meshes": meshes,
            "props": [p["name"] for p in data.get("props", [])],
            "camera_frame1": {"position": [round(v, 2) for v in c0[:3]], "euler_deg": [round(v, 1) for v in c0[3:]], "lens_mm": data["lens"]},
            "palette": data["palette"], "sky": data.get("sky"), "exposure": data.get("exposure", 1.2)}

def _img_parts(images, provider):
    out = []
    for label, url in images:
        if not url:
            continue
        m = re.match(r"data:(image/\w+);base64,(.*)", url, re.S)
        if not m:
            continue
        if provider == "anthropic":
            out += [{"type": "text", "text": label}, {"type": "image", "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)}}]
        else:
            out += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": url}}]
    return out

def ollama_status(c):
    """(running, model installed, list of installed models)"""
    import requests
    try:
        r = requests.get(c.get("ollama_url", "http://127.0.0.1:11434").rstrip("/") + "/api/tags", timeout=2)
        names = [m["name"] for m in r.json().get("models", [])]
        want = c.get("ollama_model", "qwen3-vl:8b")
        have = want in names or want + ":latest" in names or (":" not in want and any(n.split(":")[0] == want for n in names))
        return True, have, names
    except Exception:
        return False, False, []

def ask_local(c, text, images, history):
    import requests
    url = c.get("ollama_url", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
    labels, imgs = [], []
    for label, u in images:
        m = re.match(r"data:image/\w+;base64,(.*)", u or "", re.S)
        if m:
            imgs.append(m.group(1)); labels.append("Image %d = %s" % (len(imgs), label.rstrip(":")))
    msgs = [{"role": "system", "content": SYSTEM}] + [{"role": h["role"], "content": h["text"]} for h in history[-6:]]
    msgs.append({"role": "user", "content": ("\n".join(labels) + "\n\n" if labels else "") + text, "images": imgs})
    r = requests.post(url, timeout=600, json={"model": c.get("ollama_model", "qwen3-vl:8b"), "messages": msgs, "stream": False,
                                              "format": "json", "options": {"temperature": 0.2, "num_ctx": 16384}})
    if r.status_code != 200:
        raise RuntimeError("Local model %s: %s" % (r.status_code, r.text[:300]))
    return r.json()["message"]["content"]

def ask_ai(c, prompt, images, data, history):
    import requests
    prov = c.get("ai_provider", "local")
    text0 = "Scene summary:\n" + json.dumps(summarize(data), ensure_ascii=False) + "\n\nUser command:\n" + prompt
    note = ""
    if prov == "local":
        try:
            return _parse(ask_local(c, text0, images, history))
        except requests.exceptions.ConnectionError:
            if (c.get("deepseek_key") or "").strip():
                prov = "deepseek"; note = " (local model not running - used DeepSeek)"
            else:
                raise RuntimeError("The local AI (Ollama) is not running. Install Ollama from ollama.com, then press "
                                   "'Download local model' in Settings - or add a DeepSeek key as a fallback.")
    res = _ask_cloud(c, prov, prompt, images, data, history)
    res["reply"] = res.get("reply", "") + note
    return res

def _parse(out):
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {"reply": out.strip()[:500], "ops": []}
    try:
        res = json.loads(m.group(0))
    except Exception:
        return {"reply": "The AI answer could not be read - please rephrase.", "ops": []}
    if isinstance(res, dict) and "ops" not in res and "op" in res:
        res = {"reply": "", "ops": [res]}
    res.setdefault("reply", ""); res.setdefault("ops", [])
    if not isinstance(res["ops"], list): res["ops"] = []
    return res

def _ask_cloud(c, prov, prompt, images, data, history):
    import requests
    key = (c.get("anthropic_key") if prov == "anthropic" else c.get("deepseek_key")) or ""
    if not key.strip():
        raise RuntimeError("No API key - open Settings and paste your %s key" % ("Claude (Anthropic)" if prov == "anthropic" else "DeepSeek"))
    text = "Scene summary:\n" + json.dumps(summarize(data), ensure_ascii=False) + "\n\nUser command:\n" + prompt
    msgs = []
    for h in history[-6:]:
        msgs.append({"role": h["role"], "content": h["text"]})
    if prov == "anthropic":
        msgs.append({"role": "user", "content": _img_parts(images, prov) + [{"type": "text", "text": text}]})
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=180,
                          headers={"x-api-key": key.strip(), "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": c.get("anthropic_model", "claude-sonnet-5-5"), "max_tokens": 3000, "system": SYSTEM, "messages": msgs})
        if r.status_code != 200:
            raise RuntimeError("Claude API %s: %s" % (r.status_code, r.text[:400]))
        out = "".join(b.get("text", "") for b in r.json()["content"])
    else:
        msgs = [{"role": "system", "content": SYSTEM}] + msgs
        msgs.append({"role": "user", "content": _img_parts(images, prov) + [{"type": "text", "text": text}]})
        r = requests.post("https://api.deepseek.com/chat/completions", timeout=180,
                          headers={"Authorization": "Bearer " + key.strip(), "Content-Type": "application/json"},
                          json={"model": c.get("deepseek_model", "deepseek-flash"), "messages": msgs, "max_tokens": 3000})
        if r.status_code != 200:
            raise RuntimeError("DeepSeek API %s: %s" % (r.status_code, r.text[:400]))
        out = r.json()["choices"][0]["message"]["content"]
    return _parse(out)

# ------------------------------------------------------------------ applying operations
def _rng(op, N):
    a = int(op.get("from") or 1) - 1; b = int(op.get("to") or N) - 1
    return range(max(0, a), min(N - 1, b) + 1)

def _colour(data, col, base="custom"):
    if isinstance(col, (list, tuple)) and len(col) == 3:
        k = "%s_%d_%d_%d" % (base, *[int(v) for v in col])
        data["palette"][k] = [int(v) for v in col]
        return k
    if isinstance(col, str) and col in data["palette"]:
        return col
    raise ValueError("unknown colour %r" % (col,))

def _meshes(data, name):
    ms = [m for m in data["meshes"] if m["name"] == name] or [m for m in data["meshes"] if m["name"].startswith(name)]
    if not ms:
        raise ValueError("no mesh called %r" % name)
    return ms

def _box(sx, sy, sz, c=(0, 0, 0), yaw=0.0, bottom=False):
    x, y, z = sx / 2, sy / 2, sz / 2
    cz = c[2] + (z if bottom else 0)
    ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    v = []
    for i in (-1, 1):
        for j in (-1, 1):
            for k in (-1, 1):
                px, py = i * x, j * y
                v.append([round(c[0] + ca * px - sa * py, 4), round(c[1] + sa * px + ca * py, 4), round(cz + k * z, 4)])
    f = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return v, f

def hand_prop_keys(data, p):
    ch = data["chars"][p["person"]]; rig = Rig(ch["height"], ch.get("style", "classic")); N = data["frames"]
    keys = []
    j = "wrist_" + p.get("hand", "R").upper()
    on = set(p["frames"])
    for f in range(N):
        r = ch["root"][f]
        W = rig.fk((r[0], r[1], r[2]), r[3], {n: ch["rot"][n][f] for n in JOINTS}, ch["pelvis_z"][f])
        R, t = W[j]
        e = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
        s = 1.0 if (f in on and ch["visible"][f]) else 0.0
        keys.append([round(float(v), 4) for v in t] + [round(float(v), 2) for v in e] + [s, s, s])
    return keys

def refresh_props(data):
    for p in data.get("props", []):
        if p.get("person"):
            if p["person"] in data["chars"]:
                p["keys"] = hand_prop_keys(data, p)
            else:
                p["keys"] = [k[:6] + [0, 0, 0] for k in p["keys"]]

def apply_ops(data, ops):
    """returns (new data, list of messages). data is not modified in place."""
    d = copy.deepcopy(data); N = d["frames"]; msgs = []
    for op in ops:
        try:
            o = op.get("op")
            if o in ("move_person", "turn_person", "bend_joint", "set_person", "show_person", "delete_person", "smooth_person"):
                pid = op["person"]
                if pid not in d["chars"]:
                    raise ValueError("no person %s" % pid)
                ch = d["chars"][pid]
            if o == "move_person":
                for f in _rng(op, N):
                    ch["root"][f][0] += float(op.get("dx", 0)); ch["root"][f][1] += float(op.get("dy", 0)); ch["root"][f][2] += float(op.get("dz", 0))
            elif o == "turn_person":
                for f in _rng(op, N):
                    ch["root"][f][3] += float(op["deg"])
            elif o == "bend_joint":
                j = op["joint"]; a = "xyz".index(op.get("axis", "x"))
                if j not in ch["rot"]:
                    raise ValueError("no joint %s" % j)
                for f in _rng(op, N):
                    ch["rot"][j][f][a] += float(op["deg"])
            elif o == "set_person":
                if op.get("colour") is not None:
                    ch["colour"] = _colour(d, op["colour"])
                    ch["label"] = ch["label"].split("_")[0] + "_" + ch["colour"]
                if op.get("height"):
                    r = float(op["height"]) / ch["height"]; ch["height"] = float(op["height"])
                    ch["pelvis_z"] = [round(z * r, 4) for z in ch["pelvis_z"]]
            elif o == "show_person":
                for f in _rng(op, N):
                    ch["visible"][f] = bool(op.get("visible", True))
            elif o == "delete_person":
                del d["chars"][pid]
            elif o == "smooth_person":
                s = max(0.5, min(6.0, float(op.get("amount", 2))))
                R = np.array(ch["root"], float); R[:, 3] = np.degrees(np.unwrap(np.radians(R[:, 3])))
                ch["root"] = gaussian_filter1d(R, s, axis=0, mode="nearest").round(4).tolist()
                ch["pelvis_z"] = gaussian_filter1d(np.array(ch["pelvis_z"]), s, mode="nearest").round(4).tolist()
                for j in ch["rot"]:
                    ch["rot"][j] = gaussian_filter1d(np.array(ch["rot"][j], float), s, axis=0, mode="nearest").round(2).tolist()
            elif o == "swap_people":
                a, b = d["chars"][op["a"]], d["chars"][op["b"]]
                a["colour"], b["colour"] = b["colour"], a["colour"]; a["label"], b["label"] = b["label"], a["label"]
            elif o == "set_colour":
                t = op["target"]; rgb = [int(v) for v in op["rgb"]]
                if t == "sky":
                    d["sky"] = rgb
                elif t in d["palette"]:
                    d["palette"][t] = rgb
                else:
                    k = _colour(d, rgb)
                    for m in _meshes(d, t):
                        m["colour"] = k
            elif o == "delete_mesh":
                gone = {id(m) for m in _meshes(d, op["name"])}
                d["meshes"] = [m for m in d["meshes"] if id(m) not in gone]
            elif o == "move_mesh":
                dv = [float(op.get("dx", 0)), float(op.get("dy", 0)), float(op.get("dz", 0))]
                for m in _meshes(d, op["name"]):
                    m["v"] = [[round(p[0] + dv[0], 4), round(p[1] + dv[1], 4), round(p[2] + dv[2], 4)] for p in m["v"]]
            elif o == "add_box":
                sz = [float(v) for v in op["size"]]; at = [float(v) for v in op.get("at", [0, 5, 0])]
                v, f = _box(*sz, c=at, yaw=float(op.get("yaw", 0)), bottom=True)
                d["meshes"].append({"name": op.get("name", "box_%d" % len(d["meshes"])), "v": v, "f": f,
                                    "colour": _colour(d, op.get("colour", "set")), "layer": 2})
            elif o == "add_hand_prop":
                pid = op["person"]
                if pid not in d["chars"]:
                    raise ValueError("no person %s" % pid)
                sz = [float(v) for v in op.get("size", [0.05, 0.05, 0.3])]
                off = [float(v) for v in op.get("offset", [0, 0, -0.1])]
                v, f = _box(*sz, c=off)
                p = {"name": op.get("name", "prop"), "v": v, "f": f, "colour": _colour(d, op.get("colour", "orange")),
                     "person": pid, "hand": op.get("hand", "R"), "frames": list(_rng(op, N))}
                d.setdefault("props", [])
                d["props"] = [q for q in d["props"] if q["name"] != p["name"]] + [p]
            elif o == "delete_prop":
                d["props"] = [q for q in d.get("props", []) if q["name"] != op["name"]]
            elif o == "camera":
                for k in d["cam"]:
                    k[0] += float(op.get("dx", 0)); k[1] += float(op.get("dy", 0)); k[2] += float(op.get("dz", 0))
                    k[3] += float(op.get("tilt", 0)); k[4] += float(op.get("roll", 0)); k[5] += float(op.get("pan", 0))
                if op.get("lens"):
                    d["lens"] = float(op["lens"])
            elif o == "exposure":
                d["exposure"] = float(op["value"])
            else:
                raise ValueError("unknown operation %r" % o)
            msgs.append("✓ " + json.dumps(op, ensure_ascii=False))
        except Exception as e:
            msgs.append("✗ %s  (%s)" % (json.dumps(op, ensure_ascii=False), e))
    refresh_props(d)
    return d, msgs

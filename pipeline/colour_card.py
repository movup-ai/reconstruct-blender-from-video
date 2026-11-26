# -*- coding: utf-8 -*-
"""The colour card (sRGB) every white-model video follows - see docs/COLOUR_CARD.md.

  Meaning                     colour   hex       used for
  main character              red      #D63E3D   first character's body and clothes
  2nd character               blue     #2D6BD8   second character
  3rd character               green    #2E9D5A   third character
  4th character               yellow   #D9A014   fourth; 5 or more: a stable colour per character, kept the whole clip
  foreground object           brown    #9A6B3B   things in front of people, such as a table
  background                  grey     #B8B8B8   background and surroundings, such as walls
  nearby interactive object   orange   #DF872F   something a person can interact with close by, such as a cup
  distant interactive object  pink     #D4689E   something interactive far from people, such as a distant door
  light source                white    #F5F5F5   marks a light and its position (not the white head / chest marks)
  special object              purple   #7A4EC2   anything the rules don't cover

apply(data) recolours a scene (scene_data.json) to the card, in place: the palette gets the card's values, the eye /
chest marks are white, characters keep their role colour (red, blue, green, yellow; 5th and later: cyan, magenta,
teal, olive - never a set-piece colour), and every mesh gets the colour of its meaning.  Stage 1 has one mesh per
shot, the floor (colour "ground"), so it is always background: grey #B8B8B8.  The role rules below are kept for the
set pieces and props of the planned Stage 2 (a mesh's "role" always wins):
  light source   lamps, chandeliers, sconces, street lights, ceiling lights;
  nearby / distant interactive   things a hand comes within 12 cm of; interactive kinds (door, cup, phone, car,
                 fridge, tv...) within reach (1 m) of a person = nearby, farther = distant;
  foreground     a piece between the camera and a person: closer to the camera than they are, overlapping them in
                 the picture;
  background     everything else (walls, floor, ground, far things, furniture behind people)."""
import numpy as np

HEX = {"red": "#D63E3D", "blue": "#2D6BD8", "green": "#2E9D5A", "yellow": "#D9A014",
       "fg": "#9A6B3B", "bg": "#B8B8B8", "near": "#DF872F", "far_i": "#D4689E", "lamp": "#F5F5F5", "special": "#7A4EC2"}
EXTRA = {"cyan": "#20A8B8", "magenta": "#B02878", "teal": "#1F7A6E", "olive": "#7E8B2A"}   # 5th person on
MARKER = [255, 255, 255]                                          # the white head / chest bands of the characters
PEOPLE = ("red", "blue", "green", "yellow")
ROLES = ("bg", "fg", "near", "far_i", "lamp", "special")
ROLE_NAMES = {"bg": "background", "fg": "foreground object", "near": "nearby interactive object",
              "far_i": "distant interactive object", "lamp": "light source", "special": "special object"}
ALIASES = {"background": "bg", "foreground": "fg", "brown": "fg", "grey": "bg", "gray": "bg", "orange": "near",
           "interactive": "near", "nearby": "near", "pink": "far_i", "distant": "far_i", "light": "lamp",
           "white": "lamp", "purple": "special"}

LIGHTS = ("lamp", "light", "chandelier", "sconce", "streetlight", "street light", "lantern", "candle", "torch",
          "ceiling light", "spotlight", "bulb", "neon")
INTERACTIVE = ("door", "gate", "car", "van", "truck", "bus", "bicycle", "bike", "motorbike", "fridge", "refrigerator",
               "stove", "oven", "microwave", "sink", "tv", "television", "computer", "laptop", "monitor", "chair",
               "office chair", "drawer", "switch", "elevator", "lift", "window", "piano", "telephone")   # near or distant
HANDHELD = ("phone", "cup", "mug", "glass", "bottle", "kettle", "plate", "bowl", "book", "bag", "remote", "keyboard",
            "tray", "umbrella", "basket", "toy", "ball", "weapon", "gun", "knife", "sword", "stick", "pad", "kick pad",
            "helmet", "trophy", "pen", "folder", "papers", "camera", "torch")    # nearby interactive; far: background
SURFACES = ("wall", "building", "floor", "ceiling", "slope", "terrain", "far", "sky", "ground", "patch", "backdrop")


def rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


CARD = {k: rgb(v) for k, v in HEX.items()}
CARD.update({k: rgb(v) for k, v in EXTRA.items()})


def norm_role(r):
    r = str(r or "").strip().lower()
    return r if r in ROLES else ALIASES.get(r)


def _has(text, words):
    t = " %s " % str(text or "").lower().replace("_", " ")
    return any((" %s " % w) in t or (" %ss " % w) in t for w in words)


def word_role(text):
    """a role a description asks for in words ("interactive", "distant door", "light", "special", "foreground")"""
    t = str(text or "").lower()
    if "special" in t:
        return "special"
    if "foreground" in t or "in front of" in t:
        return "fg"
    if "distant" in t and ("interactive" in t or _has(t, INTERACTIVE) or _has(t, HANDHELD)):
        return "far_i"
    if "interactive" in t or "interact" in t or "touch" in t:
        return "near"
    return None


def _people(data, step=2):
    """per sampled frame: [(wrists (2, 3), body points (n, 3))] of every visible person"""
    from rig import Rig, JOINTS
    N = data.get("frames", 0)
    out = {}
    for ch in (data.get("chars") or {}).values():
        try:
            rig = Rig(ch["height"], ch.get("style", "sample"))
        except Exception:
            continue
        vis = ch.get("visible") or [True] * N
        for f in range(0, N, step):
            if not vis[f]:
                continue
            r = ch["root"][f]
            W = rig.fk((r[0], r[1], r[2]), r[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f])
            wr = np.array([W["wrist_L"][1], W["wrist_R"][1]])
            body = np.array([W[j][1] for j in JOINTS])
            out.setdefault(f, []).append((wr, body))
    return out


def _box_dist(P, lo, hi):
    """distance of points P to the box lo..hi (0 inside)"""
    d = np.maximum(np.maximum(lo - P, 0), P - hi)
    return np.linalg.norm(d, axis=1)


def _groups(meshes):
    """meshes -> {object key: [indices]} (the parts of one object share its "obj" tag)"""
    g = {}
    for i, m in enumerate(meshes):
        g.setdefault(m.get("obj") or m.get("name"), []).append(i)
    return g


def role_of(info, V, people, cam, frames):
    """the card role of one set object: info {kind, label, cls, role}, V its vertices (n, 3)"""
    r = norm_role(info.get("role"))
    if r:
        return r
    words = " ".join(str(info.get(k) or "") for k in ("kind", "label", "what"))
    cls = str(info.get("cls") or "")
    if _has(words, LIGHTS):
        return "lamp"
    wr = word_role(info.get("what"))
    if wr:
        return wr
    surface = cls in SURFACES or str(info.get("kind") or "") in ("wall", "patch", "building")
    if surface and not _has(words, ("door", "gate")):
        return "bg"
    lo, hi = V.min(0), V.max(0)
    touch = reach = 9.0
    for f in frames:
        for wr_, body in people.get(f, []):
            touch = min(touch, float(_box_dist(wr_, lo, hi).min()))
            reach = min(reach, float(_box_dist(body, lo, hi).min()))
    if _has(words, INTERACTIVE):
        return "near" if min(touch, reach) <= 1.0 else "far_i"
    if _has(words, HANDHELD) and min(touch, reach) <= 1.0:
        return "near"
    small = float(np.max(hi - lo)) <= 0.6                         # a hand on a desk does not make the desk a cup
    if touch <= 0.12 and small and not surface:
        return "near"
    if cam is not None and not surface:                          # between the camera and a person: foreground
        for f in frames[::2]:
            if not people.get(f):
                continue
            uv, z = cam.project(V, f)
            if (z <= 0).all():
                continue
            ob = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
            zo = float(np.percentile(z[z > 0], 50))
            for _, body in people[f]:
                pv, pz = cam.project(body, f)
                pb = [pv[:, 0].min(), pv[:, 1].min(), pv[:, 0].max(), pv[:, 1].max()]
                ix = max(0.0, min(ob[2], pb[2]) - max(ob[0], pb[0])); iy = max(0.0, min(ob[3], pb[3]) - max(ob[1], pb[1]))
                pa = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
                if ix * iy > 0.05 * pa and zo < float(np.median(pz)) - 0.15:
                    return "fg"
    return "bg"


def apply(data, log=print):
    """recolour a scene to the card (in place) -> {role: count of objects}"""
    pal = data.setdefault("palette", {})
    pal.update({k: v for k, v in CARD.items()})
    pal["marker"] = MARKER
    extra = list(EXTRA)                                           # 5th person on: stable colours, never a set colour
    used = {ch.get("colour") for ch in (data.get("chars") or {}).values()}
    for kind in sorted(data.get("chars") or {}, key=lambda k: int(k[1:]) if k[1:].isdigit() else 99):
        ch = data["chars"][kind]
        if ch.get("colour") not in PEOPLE and ch.get("colour") not in EXTRA:
            free = [e for e in extra if e not in used] or extra
            ch["colour"] = free[0]; used.add(free[0])
            if ch.get("label"):
                ch["label"] = ch["label"].rsplit("_", 1)[0] + "_" + ch["colour"] if "_" in ch["label"] else ch["label"]
    count = {}
    meshes = data.get("meshes") or []
    seen = None                                                    # (people, camera, frames): only for set pieces
    for key, idx in _groups(meshes).items():
        m0 = meshes[idx[0]]
        if m0.get("layer", 2) in (0, 1) or m0.get("cls") == "ground" or m0.get("colour") == "ground":   # backdrop, terrain, floor
            role = "bg"
        else:
            if seen is None:
                try:
                    from camera_model import Cam
                    cam = Cam(data)
                except Exception:
                    cam = None
                people = _people(data)
                seen = (people, cam, sorted(people))
            V = np.vstack([np.asarray(meshes[i]["v"], float) for i in idx])
            role = role_of(m0, V, seen[0], seen[1], seen[2])
        for i in idx:
            meshes[i]['colour'] = role                                 # background (the floor): grey #B8B8B8
            meshes[i]['semantic_role'] = role
        count[role] = count.get(role, 0) + 1
    props = data.get("props") or []
    for key, idx in _groups(props).items():
        role = norm_role(props[idx[0]].get("role")) or "near"           # a moving prop is moved by people
        for i in idx:
            props[i]["colour"] = role
        count[role] = count.get(role, 0) + 1
    data["colour_card"] = True
    if log:
        log("  colour card: " + ", ".join("%s %d" % (ROLE_NAMES.get(k, k), v) for k, v in sorted(count.items())))
    return count

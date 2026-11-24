# -*- coding: utf-8 -*-
"""
White-model L3 rebuild - generic Blender builder of the automatic pipeline.
Run by the pipeline:  blender -b -P blender_build.py -- <scene_data.json> [render]
Builds the grey set, the colour-coded L3 mannequins (teal face / chest markers painted on head and chest),
the solved camera, saves model_3d/<name>.blend and optionally renders JPEG frames.
"""
import math, json, os, re, sys

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
with open(ARGS[0], encoding="utf-8") as fh:
    DATA = json.load(fh)
SUBMIT_DIR, WORK_DIR = DATA["submit_dir"], DATA["work_dir"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # scene_export.py next to this file
BLEND_PATH = os.path.join(SUBMIT_DIR, "model_3d", (DATA.get("package_name") or DATA["name"]) + ".blend")   # v2.1: <package>.blend
RENDER_SCALE = DATA.get("render_scale", 2)

def srgb(r, g, b):
    f = lambda c: (c / 255.0) / 12.92 if c / 255.0 <= 0.04045 else (((c / 255.0) + 0.055) / 1.055) ** 2.4
    return (f(r), f(g), f(b), 1.0)

COLORS = {k: srgb(*v) for k, v in DATA["palette"].items()}
SKY = srgb(*DATA.get("sky", [228, 229, 232]))
EXPOSURE = DATA.get("exposure", 1.2)
GAIN = DATA.get("gain", {"ground": 2.2})

# ------------------------------------------------------------------ math (pure python, shared with the preview tool)
def mmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def T(x, y, z):
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
def Rxyz(rx, ry, rz):  # degrees, Blender XYZ euler: R = Rz*Ry*Rx
    a, b, c = map(math.radians, (rx, ry, rz))
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    Rx = [[1, 0, 0, 0], [0, ca, -sa, 0], [0, sa, ca, 0], [0, 0, 0, 1]]
    Ry = [[cb, 0, sb, 0], [0, 1, 0, 0], [-sb, 0, cb, 0], [0, 0, 0, 1]]
    Rz = [[cc, -sc, 0, 0], [sc, cc, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return mmul(Rz, mmul(Ry, Rx))
def xf(M, p):
    return [M[i][0] * p[0] + M[i][1] * p[1] + M[i][2] * p[2] + M[i][3] for i in range(3)]

# ------------------------------------------------------------------ primitive meshes (verts, faces)
BOX_F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
def box(sx, sy, sz, c=(0, 0, 0)):
    x, y, z = sx / 2, sy / 2, sz / 2
    v = [(c[0] + i * x, c[1] + j * y, c[2] + k * z) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    return v, list(BOX_F)
def sphere(r, c=(0, 0, 0), seg=16, ring=10, sc=(1, 1, 1)):
    v = [(c[0], c[1], c[2] + r * sc[2])]
    for i in range(1, ring):
        th = math.pi * i / ring
        for j in range(seg):
            ph = 2 * math.pi * j / seg
            v.append((c[0] + r * sc[0] * math.sin(th) * math.cos(ph), c[1] + r * sc[1] * math.sin(th) * math.sin(ph), c[2] + r * sc[2] * math.cos(th)))
    v.append((c[0], c[1], c[2] - r * sc[2]))
    f = []
    for j in range(seg):
        f.append((0, 1 + j, 1 + (j + 1) % seg))
    for i in range(ring - 2):
        for j in range(seg):
            a = 1 + i * seg + j; b = 1 + i * seg + (j + 1) % seg
            f.append((a, a + seg, b + seg, b))
    last = len(v) - 1; base = 1 + (ring - 2) * seg
    for j in range(seg):
        f.append((base + (j + 1) % seg, base + j, last))
    return v, f
def frustum(r0, r1, h, c=(0, 0, 0), seg=16):
    v = []
    for z, r in ((0, r0), (h, r1)):
        for j in range(seg):
            a = 2 * math.pi * j / seg
            v.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2] + z))
    f = [(j, (j + 1) % seg, seg + (j + 1) % seg, seg + j) for j in range(seg)]
    f.append(tuple(range(seg - 1, -1, -1))); f.append(tuple(range(seg, 2 * seg)))
    return v, f
def merge(parts):
    V, F = [], []
    for v, f in parts:
        n = len(V); V += list(v); F += [tuple(n + k for k in face) for face in f]
    return V, F

# ------------------------------------------------------------------ L3 rig
JOINT_ORDER = ["pelvis", "spine", "chest", "neck", "head",
               "shoulder_L", "elbow_L", "wrist_L", "shoulder_R", "elbow_R", "wrist_R",
               "hip_L", "knee_L", "ankle_L", "hip_R", "knee_R", "ankle_R"]
def head_mesh(r, c, seg=32, ring=20, rows=(7, 8)):
    """smooth sphere; the face marker is painted on it: the faces of two latitude rows (eye level) within
    about +-34 degrees of +Y get the marker colour.  Returns (verts, faces), {face index: colour}"""
    v, f = sphere(r, c=c, seg=seg, ring=ring)
    marks = {}
    for i in rows:                   # latitude rows of the face band
        for j in range(5, 11):       # longitudes 56..124 degrees (+Y is 90)
            marks[seg + i * seg + j] = "marker"
    return (v, f), marks

def chest_mesh(sx, sy, sz, c, mw, mz0, mz1):
    """box whose front (+Y) face is split 3x3 so the chest marker is a painted rectangle (width mw, from mz0 to mz1)"""
    x, y, z = sx / 2, sy / 2, sz / 2
    v = [(c[0] + i * x, c[1] + j * y, c[2] + k * z) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    f = [fc for n, fc in enumerate(BOX_F) if n != 3]          # all sides except the front
    xs = [c[0] - x, c[0] - mw / 2, c[0] + mw / 2, c[0] + x]; zs = [c[2] - z, mz0, mz1, c[2] + z]
    base = len(v)
    for zz in zs:
        for xx in xs:
            v.append((xx, c[1] + y, zz))
    marks = {}
    for r_ in range(3):
        for q in range(3):
            a = base + r_ * 4 + q
            f.append((a + 1, a, a + 4, a + 5))                 # faces +Y
            if r_ == 1 and q == 1: marks[len(f) - 1] = "marker"
    return (v, f), marks

_ASSET = []
def sample_asset():
    """client sample character pieces, extracted once by extract_sample_character.py (None = use the box mannequin)"""
    if not _ASSET:
        p = DATA.get("character_asset") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                        "assets", "sample_character.json")
        _ASSET.append(None)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                _ASSET[-1] = json.load(fh)
    return _ASSET[0]

def rig(kind, height):
    """Both characters use the same mannequin as the client samples (only overall height and colour differ):
    smooth sphere head on a short neck, box torso, block limbs split at elbow / wrist / knee / ankle, teal face + chest bands.
    The face and chest markers are painted on the head / chest surfaces (same object, second material).
    Parts: (joint, mesh, colour, smooth[, {face index: colour}])"""
    s = height / 1.75
    S = lambda *a: tuple(x * s for x in a)
    col = DATA["chars"][kind]["colour"]
    if DATA["chars"][kind].get("style") == "master":
        # official human_master.blend: joints from assets/human_master.json; the meshes are the rig's own pieces,
        # appended and driven in build_blender (add_master), so no geometry here
        return [(n, par, S(*o)) for n, par, o in master_asset()["joints"]], []
    if DATA["chars"][kind].get("style") == "base21":
        # v2.1 supplied base model (assets/base_character_v21.json): its own joints and blocks, white eye + chest bands
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "base_character_v21.json")
        with open(p, encoding="utf-8") as fh:
            asset = json.load(fh)
        J = [(n, par, S(*o)) for n, par, o in asset["joints"]]
        P = []
        for j in ["chest", "head", "shoulder_L", "elbow_L", "shoulder_R", "elbow_R", "hip_L", "knee_L", "hip_R", "knee_R"]:
            part = asset["parts"][j]
            P.append((j, ([(x * s, y * s, z * s) for x, y, z in part["v"]], [tuple(f) for f in part["f"]]), col,
                      part.get("smooth", False), {i: "marker" for i in part["marker_faces"]}))
        return J, P
    if DATA["chars"][kind].get("style", "classic") == "sample":
        # client's L3 sample (11_2 blend): one torso block with the chest band, sphere head on the torso,
        # upper / lower arm and leg blocks, no neck, hands or feet
        J = [("pelvis", "root", S(0, 0, 0.94)), ("spine", "pelvis", S(0, 0, 0)), ("chest", "spine", S(0, 0, 0)),
             ("neck", "chest", S(0, 0, 0.52)), ("head", "neck", S(0, 0, 0))]
        for side, sx in (("L", -1), ("R", 1)):
            J += [("shoulder_" + side, "chest", S(sx * 0.222, 0, 0.482)), ("elbow_" + side, "shoulder_" + side, S(0, 0, -0.274)),
                  ("wrist_" + side, "elbow_" + side, S(0, 0, -0.335)), ("hip_" + side, "pelvis", S(sx * 0.081, 0, 0)),
                  ("knee_" + side, "hip_" + side, S(0, 0, -0.459)), ("ankle_" + side, "knee_" + side, S(0, 0, -0.42))]
        asset = sample_asset()
        if asset:                                          # the client's own character mesh (assets/sample_character.json)
            P = []
            for j in ["chest", "head", "shoulder_L", "elbow_L", "shoulder_R", "elbow_R", "hip_L", "knee_L", "hip_R", "knee_R"]:
                part = asset["parts"].get(j)
                if not part:
                    continue
                v = [(x * s, y * s, z * s) for x, y, z in part["v"]]
                P.append((j, (v, [tuple(f) for f in part["f"]]), col, part.get("smooth", False),
                          {i: "marker" for i in part["marker_faces"]}))
            return J, P
        head = head_mesh(0.14 * s, S(0, 0.024, 0.145), seg=32, ring=30, rows=(13, 14))
        chest = chest_mesh(*S(0.31, 0.157, 0.509), S(0, 0, 0.2545), 0.219 * s, 0.347 * s, 0.429 * s)
        P = [("chest", chest[0], col, False, chest[1]), ("head", head[0], col, True, head[1])]
        for side in ("L", "R"):
            P += [("shoulder_" + side, box(*S(0.112, 0.112, 0.274), c=S(0, 0, -0.137)), col, False),
                  ("elbow_" + side, box(*S(0.10, 0.096, 0.297), c=S(0, 0, -0.1485)), col, False),
                  ("hip_" + side, box(*S(0.121, 0.121, 0.459), c=S(0, 0, -0.2295)), col, False),
                  ("knee_" + side, box(*S(0.121, 0.121, 0.483), c=S(0, 0, -0.2415)), col, False)]
        return J, P
    J = [  # name, parent, rest offset in parent space   (character faces +Y, its right side is +X)
        ("pelvis", "root", S(0, 0, 0.93)),
        ("spine", "pelvis", S(0, 0, 0.10)),
        ("chest", "spine", S(0, 0, 0.17)),
        ("neck", "chest", S(0, 0, 0.30)),
        ("head", "neck", S(0, 0, 0.08)),
    ]
    for side, sx in (("L", -1), ("R", 1)):
        J += [("shoulder_" + side, "chest", S(sx * 0.225, 0, 0.25)),
              ("elbow_" + side, "shoulder_" + side, S(0, 0, -0.28)),
              ("wrist_" + side, "elbow_" + side, S(0, 0, -0.25)),
              ("hip_" + side, "pelvis", S(sx * 0.095, 0, -0.04)),
              ("knee_" + side, "hip_" + side, S(0, 0, -0.42)),
              ("ankle_" + side, "knee_" + side, S(0, 0, -0.43))]
    HR = 0.13 * s
    head = head_mesh(HR, S(0, 0, 0.09))
    chest = chest_mesh(*S(0.38, 0.22, 0.31), S(0, 0, 0.13), 0.20 * s, 0.125 * s, 0.195 * s)
    P = [
        ("pelvis", box(*S(0.36, 0.21, 0.18), c=S(0, 0, 0.02)), col, False),
        ("spine", box(*S(0.36, 0.21, 0.19), c=S(0, 0, 0.08)), col, False),
        ("chest", chest[0], col, False, chest[1]),              # chest marker painted on the front face (body direction)
        ("neck", frustum(*S(0.05, 0.05, 0.10), c=S(0, 0, -0.03), seg=20), col, True),
        ("head", head[0], col, True, head[1]),                  # face marker painted on the sphere (face direction)
    ]
    for side in ("L", "R"):
        P += [("shoulder_" + side, box(*S(0.09, 0.09, 0.30), c=S(0, 0, -0.13)), col, False),
              ("elbow_" + side, box(*S(0.078, 0.078, 0.26), c=S(0, 0, -0.12)), col, False),
              ("wrist_" + side, box(*S(0.05, 0.09, 0.16), c=S(0, 0.0, -0.08)), col, False),
              ("hip_" + side, box(*S(0.13, 0.13, 0.45), c=S(0, 0, -0.20)), col, False),
              ("knee_" + side, box(*S(0.105, 0.105, 0.44), c=S(0, 0, -0.21)), col, False),
              ("ankle_" + side, box(*S(0.10, 0.25, 0.07), c=S(0, 0.06, -0.03)), col, False)]

    return J, P

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MASTER = []
def master_asset():
    """joint table of the official human_master.blend (extract_human_master.py)"""
    if not _MASTER:
        with open(os.path.join(ROOT_DIR, "assets", "human_master.json"), encoding="utf-8") as fh:
            _MASTER.append(json.load(fh))
    return _MASTER[0]

def add_master(kind, E, cc, mats, colour, height):
    """Append the official rig + its 10 skinned pieces from human_master.blend (unchanged skeleton), paint the body in
    the palette colour and the rig's own head / chest marker faces white, and drive each piece's deform bone from the
    fitted joint: bone world = joint world @ [R | 0] * model_scale (Copy Transforms, world space).  The bones' Rigify
    constraints are replaced by that one constraint and their B-Bone segments set to 1 so the rigid blocks never bend."""
    import bpy
    from mathutils import Matrix
    A = master_asset()
    blend = DATA.get("master_blend") or os.path.join(ROOT_DIR, "assets", "human_master", "human_master.blend")
    pieces = sorted({p["piece"] for p in A["parts"].values()})
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.objects = ["rig"] + pieces
    objs = [o for o in dst.objects if o is not None]
    rig = next(o for o in objs if o.type == "ARMATURE")
    for o in objs:
        cc.objects.link(o)
    ms = height / 1.75 * A["scale_to_1_75"]                 # one uniform model_scale (spec 2.4)
    rig.name = kind + "_human_master_rig"
    if rig.animation_data:
        rig.animation_data.action = None
    rig.parent = E["root"]; rig.location = (0, 0, 0); rig.rotation_euler = (0, 0, math.pi); rig.scale = (ms, ms, ms)
    rig.hide_render = True
    for o in objs:
        if o is rig:
            continue
        o.name = "%s_%s" % (kind, o.name)
        mm = o.data.materials
        mm[0] = mats[colour]
        if len(mm) > 1:
            mm[1] = mats["marker"]
    # a connected bone keeps its head on the parent's tail whatever its constraints say: free the driven bones
    # (hierarchy and rest positions unchanged)
    vl = bpy.context.view_layer
    for o in vl.objects:
        o.select_set(False)
    vl.objects.active = rig; rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for b in A["bones"].values():
        rig.data.edit_bones[b["bone"]].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.select_set(False)
    for j, b in A["bones"].items():
        bn = b["bone"]
        rig.data.bones[bn].bbone_segments = 1
        pb = rig.pose.bones[bn]
        for con in list(pb.constraints):
            pb.constraints.remove(con)
        t = bpy.data.objects.new("%s_%s_bone" % (kind, j), None)
        t.empty_display_size = 0.05; t.empty_display_type = "ARROWS"; cc.objects.link(t)
        t.parent = E[j]; t.location = (0, 0, 0)
        t.rotation_mode = "QUATERNION"; t.rotation_quaternion = Matrix(b["R"]).to_quaternion(); t.scale = (ms, ms, ms)
        con = pb.constraints.new("COPY_TRANSFORMS"); con.target = t
        con.target_space = "WORLD"; con.owner_space = "WORLD"
    return rig

def fk(J, rot, root_M, pelvis_z=None):
    W = {"root": root_M}
    for name, parent, off in J:
        if name == "pelvis" and pelvis_z is not None: off = (off[0], off[1], pelvis_z)
        r = rot.get(name, (0, 0, 0))
        W[name] = mmul(W[parent], mmul(T(off[0], off[1], off[2]), Rxyz(r[0], r[1], r[2])))
    return W

def char_pose(kind, f):
    """(root loc, root euler, {joint: euler}, pelvis height above the root) for frame index f.
    The root sits on the ground under the pelvis; the feet are placed on the ground by leg IK baked into the keys."""
    c = DATA["chars"][kind]
    r = c["root"][f]
    return (r[0], r[1], r[2]), (0.0, 0.0, r[3]), {n: c["rot"][n][f] for n in c["rot"]}, c["pelvis_z"][f]

# ------------------------------------------------------------------ set and props (geometry is baked into DATA)
def set_pieces():
    """list of (name, verts, faces, colour, layer, smooth)   layer: 0 backdrop, 1 terrain, 2 objects"""
    return [(m["name"], m["v"], m["f"], m["colour"], m["layer"], m.get("smooth", False)) for m in DATA["meshes"]]

def prop_pose(p, f):
    """per-frame transform of an animated prop: location, euler (deg), scale"""
    k = p["keys"][f]
    return k[0:3], k[3:6], k[6:9]

def lzhd_lighting(bpy, scn):
    """Eevee, Standard view, one sun (key light) + an even world (fill).  DATA["light"] (hints.json "light"):
    az = degrees the light comes from, around the camera's view (0 = from behind the camera, -90 = from screen left,
    90 = from screen right, 180 = back light), el = degrees above the horizon, sun = strength (W/m2), ambient = world
    strength, shadow = soft-shadow angle (deg)."""
    L = dict({"az": -35.0, "el": 50.0, "sun": 2.6, "ambient": 0.55, "shadow": 8.0}, **(DATA.get("light") or {}))
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scn.render.engine = eng; break
        except TypeError:
            continue
    ee = scn.eevee
    for a, v in (("taa_render_samples", 32), ("use_shadows", True), ("use_gtao", True), ("shadow_ray_count", 2), ("shadow_step_count", 8)):
        try: setattr(ee, a, v)
        except Exception: pass
    try:
        scn.view_settings.view_transform = "Standard"; scn.view_settings.look = "None"
    except Exception: pass
    scn.view_settings.exposure = 0.0
    w = scn.world
    w.use_nodes = True
    bg = next((n for n in w.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg:
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0); bg.inputs["Strength"].default_value = L["ambient"]
    from mathutils import Vector, Euler
    cam0 = DATA["cam"][0]
    vv = Euler([math.radians(x) for x in cam0[3:6]], "XYZ").to_matrix() @ Vector((0, 0, -1))     # camera view direction
    n = math.hypot(vv.x, vv.y) or 1.0
    view = (vv.x / n, vv.y / n)
    a = math.radians(L["az"])                      # direction the light travels, turned from the view direction
    dx = view[0] * math.cos(a) - view[1] * math.sin(a); dy = view[0] * math.sin(a) + view[1] * math.cos(a)
    e = math.radians(L["el"])
    d = (dx * math.cos(e), dy * math.cos(e), -math.sin(e))          # travel direction (down into the scene)
    sd = bpy.data.lights.new("Key_sun", "SUN"); sd.energy = L["sun"]; sd.angle = math.radians(L["shadow"])
    sun = bpy.data.objects.new("Key_sun", sd)
    col = bpy.data.collections.get("Lights") or bpy.data.collections.new("Lights")
    if col.name not in scn.collection.children: scn.collection.children.link(col)
    col.objects.link(sun)
    sun.rotation_euler = Vector(d).to_track_quat("-Z", "Y").to_euler()
    sun.location = (cam0[0], cam0[1], cam0[2] + 3.0)
    for i, q in enumerate(L.get("points") or []):          # practical lamps / light pools: [x, y, z, watts(, radius)]
        pd = bpy.data.lights.new("Lamp_%d" % i, "POINT"); pd.energy = float(q[3])
        pd.shadow_soft_size = float(q[4]) if len(q) > 4 else 0.3
        po = bpy.data.objects.new("Lamp_%d" % i, pd); col.objects.link(po); po.location = tuple(q[:3])

def build_blender():
    import bpy
    scn = bpy.context.scene
    # start clean: delete the previous render frames, done-marker and .blend, then recreate everything
    rdir = os.path.join(WORK_DIR, "render")
    if "render" in ARGS and os.path.isdir(rdir):
        for fn in os.listdir(rdir):
            if fn.lower().endswith((".png", ".jpg")):
                try: os.remove(os.path.join(rdir, fn))
                except OSError: pass
    for old in ([os.path.join(WORK_DIR, "render_done.txt")] if "render" in ARGS else []) + [BLEND_PATH, BLEND_PATH + "1"]:
        try: os.remove(old)
        except OSError: pass
    try: bpy.context.preferences.filepaths.save_version = 0      # no 1.blend1 backup inside the submission folder
    except Exception: pass
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    scn.render.resolution_x = int(round(DATA["res"][0] * RENDER_SCALE))          # 3840 wide (client sample)
    scn.render.resolution_y = int(round(DATA["res"][1] * RENDER_SCALE))
    scn.render.resolution_percentage = 100
    scn.render.fps = DATA["fps"]
    scn.frame_start, scn.frame_end = 1, DATA["frames"]
    scn.render.engine = "BLENDER_WORKBENCH"
    sh = scn.display.shading
    sh.light = "STUDIO"; sh.color_type = "MATERIAL"; sh.show_shadows = True; sh.shadow_intensity = 0.4
    sh.show_cavity = False
    try: sh.show_specular_highlight = False
    except Exception: pass
    if hasattr(sh, "background_type"): sh.background_type = "WORLD"
    try: scn.display.render_aa = "8"
    except Exception: pass
    try:
        scn.view_settings.view_transform = "Standard"
        scn.view_settings.look = "None"
    except Exception: pass
    scn.view_settings.exposure = EXPOSURE
    if scn.world is None:
        scn.world = bpy.data.worlds.new("World")
    scn.world.color = tuple(c / (2.0 ** EXPOSURE) for c in SKY[:3])

    LZHD = DATA.get("spec") == "lzhd1"           # LZHD V1.0 (2026-11-15): Eevee, palette = base colour, light per clip
    mats = {}
    for k, c in COLORS.items():
        m = bpy.data.materials.get("WM_" + k) or bpy.data.materials.new("WM_" + k)
        g = 1.0 if LZHD else GAIN.get(k, GAIN.get({"c": "_video", "g": "_ground"}[k[0]], 1.0) if (len(k) == 7 and k[0] in "cg") else 1.0)   # video colours
        m.diffuse_color = (min(1.0, c[0] * g), min(1.0, c[1] * g), min(1.0, c[2] * g), 1.0); m.roughness = 1.0
        if LZHD:
            m.use_nodes = True
            bs = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bs:
                bs.inputs["Base Color"].default_value = m.diffuse_color
                bs.inputs["Roughness"].default_value = 1.0
                for nm in ("Specular IOR Level", "Specular"):
                    if nm in bs.inputs:
                        bs.inputs[nm].default_value = 0.0
                if k == "lamp" and "Emission Color" in bs.inputs:          # light-source stand-ins glow a little
                    bs.inputs["Emission Color"].default_value = m.diffuse_color
                    bs.inputs["Emission Strength"].default_value = 1.5
        mats[k] = m
    if LZHD:
        lzhd_lighting(bpy, scn)

    def coll(name):
        c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
        if c.name not in scn.collection.children: scn.collection.children.link(c)
        return c

    def mesh_obj(name, v, f, colour, col, smooth=False, marks=None):
        me = bpy.data.meshes.new(name); me.from_pydata([tuple(p) for p in v], [], [tuple(x) for x in f]); me.update()
        me.materials.append(mats[colour])
        if marks:
            me.materials.append(mats["marker"])
            for idx in marks: me.polygons[idx].material_index = 1
        if smooth:
            for p in me.polygons: p.use_smooth = True
        ob = bpy.data.objects.new(name, me); col.objects.link(ob); return ob

    cs = coll("Set_Background")
    shot_range = {int(s["shot"]): s["frames"] for s in DATA.get("shots") or []}
    for name, v, f, colour, layer, smooth in set_pieces():
        ob = mesh_obj(name, v, f, colour, cs, smooth)
        m = re.search(r"_s(\d+)(?:_|$)", name)
        if len(shot_range) > 1 and m and int(m.group(1)) in shot_range:      # each shot's set only in its own frames:
            a, b = shot_range[int(m.group(1))]                                # a big ground / far hill must not show in another shot
            for fr, hide in ((1, a > 0), (a + 1, False), (b + 2, True)):
                if fr <= DATA["frames"] + 1:
                    ob.hide_render = hide; ob.hide_viewport = hide
                    ob.keyframe_insert("hide_render", frame=fr); ob.keyframe_insert("hide_viewport", frame=fr)

    for kind, cdat in DATA["chars"].items():
        label = cdat["label"]
        cc = coll(label)
        J, P = rig(kind, DATA["chars"][kind]["height"])
        E = {}
        root = bpy.data.objects.new(kind + "_root", None); root.empty_display_size = 0.3; cc.objects.link(root); E["root"] = root
        for name, parent, off in J:
            e = bpy.data.objects.new(kind + "_" + name, None); e.empty_display_size = 0.06; e.empty_display_type = "SPHERE"
            cc.objects.link(e); e.parent = E[parent]; e.location = off; E[name] = e
        if cdat.get("style") == "master":
            add_master(kind, E, cc, mats, cdat["colour"], cdat["height"])
        for i, part in enumerate(P):
            j, (v, f), colour, smooth = part[:4]
            ob = mesh_obj("%s_%s_geo%d" % (kind, j, i), v, f, colour, cc, smooth, part[4] if len(part) > 4 else None); ob.parent = E[j]
        for fr in range(DATA["frames"]):
            loc, rr, joints, pz = char_pose(kind, fr)
            root.location = loc; root.rotation_euler = [math.radians(a) for a in rr]
            root.keyframe_insert("location", frame=fr + 1); root.keyframe_insert("rotation_euler", frame=fr + 1)
            E["pelvis"].location = (0.0, 0.0, pz); E["pelvis"].keyframe_insert("location", frame=fr + 1)
            for n, r in joints.items():
                e = E[n]; e.rotation_euler = [math.radians(a) for a in r]
                e.keyframe_insert("rotation_euler", frame=fr + 1)
        vis = cdat.get("visible")
        if vis and not all(vis):                       # hidden before the person enters / after they leave
            geos = [o for o in cc.objects if o.type == "MESH" and o.name.startswith(kind + "_")]   # one person in several shots shares a collection
            prev = None
            for fr, v in enumerate(vis):
                if v != prev:
                    for o in geos:
                        o.hide_render = not v; o.hide_viewport = not v
                        o.keyframe_insert("hide_render", frame=fr + 1); o.keyframe_insert("hide_viewport", frame=fr + 1)
                    prev = v

    cp = coll("Props_Interactive")
    for p in DATA.get("props", []):
        ob = mesh_obj(p["name"], p["v"], p["f"], p["colour"], cp, p.get("smooth", False))
        for fr in range(DATA["frames"]):
            loc, rot, sc = prop_pose(p, fr)
            ob.location = loc; ob.rotation_euler = [math.radians(a) for a in rot]; ob.scale = sc
            ob.keyframe_insert("location", frame=fr + 1); ob.keyframe_insert("rotation_euler", frame=fr + 1); ob.keyframe_insert("scale", frame=fr + 1)

    cam_d = bpy.data.cameras.new("Camera"); cam_d.lens = DATA["lens"]; cam_d.sensor_width = DATA["sensor"]; cam_d.sensor_fit = "HORIZONTAL"
    cam_d.clip_start = 0.1; cam_d.clip_end = 20000
    cam = bpy.data.objects.new("Camera", cam_d); coll("Camera").objects.link(cam); scn.camera = cam
    zoom = DATA.get("zoom") or []                       # the lens zooms: focal length keyed on every frame
    zooms = bool(zoom) and max(abs(float(z) - 1.0) for z in zoom) > 1e-3
    for fr, c in enumerate(DATA["cam"]):
        cam.location = c[:3]; cam.rotation_euler = [math.radians(a) for a in c[3:]]
        cam.keyframe_insert("location", frame=fr + 1); cam.keyframe_insert("rotation_euler", frame=fr + 1)
        if zooms:
            cam_d.lens = DATA["lens"] * float(zoom[min(fr, len(zoom) - 1)])
            cam_d.keyframe_insert("lens", frame=fr + 1)
    if zooms:
        print("camera zoom: lens %.1f -> %.1f mm" % (DATA["lens"] * min(zoom), DATA["lens"] * max(zoom)))
    scn.frame_set(1)

    os.makedirs(os.path.join(WORK_DIR, "render"), exist_ok=True)
    os.makedirs(os.path.dirname(BLEND_PATH), exist_ok=True)
    try: scn.render.image_settings.media_type = "IMAGE"
    except Exception: pass
    scn.render.image_settings.file_format = "JPEG"      # ~1 MB per frame instead of ~6 MB PNG
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.quality = 97
    scn.render.filepath = os.path.join(WORK_DIR, "render", "wm_")
    # the saved project renders into its own folder (portable, no path of this machine): //render/wm_
    rel_out = scn.render.filepath
    scn.render.filepath = "//render/wm_"
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
    print("saved", BLEND_PATH)
    try:
        import scene_export
        sp = os.path.join(os.path.dirname(BLEND_PATH), "reconstruction", "scene.json")
        scene_export.export(bpy, sp, os.path.basename(BLEND_PATH), DATA)
        print("scene.json", sp)
    except Exception as e:                                  # the package step reports a missing scene.json
        import traceback; traceback.print_exc()
    scn.render.filepath = rel_out
    if "still" in ARGS:                                   # one preview frame for the AI edit page
        fr = int(ARGS[ARGS.index("still") + 1])
        scn.frame_set(fr)
        scn.render.resolution_percentage = 50
        out = os.path.join(WORK_DIR, "still.jpg")
        scn.render.filepath = out
        bpy.ops.render.render(write_still=True)
        scn.render.resolution_percentage = 100
        print("still", fr, out)
    if "render" in ARGS:
        bpy.ops.render.render(animation=True)
        n = len([f for f in os.listdir(os.path.join(WORK_DIR, "render")) if f.lower().endswith((".png", ".jpg"))])
        with open(os.path.join(WORK_DIR, "render_done.txt"), "w") as fh:
            fh.write("frames %d\nblender %s\n" % (n, bpy.app.version_string))
        print("rendered", n, "frames")

if __name__ == "__main__":
    try:
        import bpy  # noqa
    except ImportError:
        bpy = None
        print("not running inside Blender")
    if bpy is not None:
        build_blender()

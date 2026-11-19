# -*- coding: utf-8 -*-
"""
One-time: turn the client's sample character (11月2日试稿 blend, object L3.002 on Rigify rig.002) into
assets/sample_character.json, used by blender_build.py for character_style "sample".

The sample character is 10 rigid pieces, each bound to one deform bone (torso box, sphere head with the teal face
band, upper / lower arm and leg blocks).  Every piece is stored in its joint's local space for the pipeline's rest
pose (character faces +Y, its left on -X, arms and legs hanging straight down), scaled to 1.75 m:
  limbs  : proximal end centre at the origin, bone axis along -Z
  torso  : bottom centre at the origin (pipeline joint "chest" = pelvis height)
  head   : bottom of the sphere at the origin (pipeline joint "head", on top of the torso)

  blender -b <sample.blend> --factory-startup -P extract_sample_character.py -- <out.json> [mesh object]
"""
import bpy, json, sys, os
from mathutils import Vector, Matrix

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = ARGS[0]
MESH = ARGS[1] if len(ARGS) > 1 else "L3.002"
ob = bpy.data.objects[MESH]
rig = ob.find_armature() or ob.parent
rig.data.pose_position = "REST"
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
ev = ob.evaluated_get(dg); me = ev.to_mesh()
MW = ob.matrix_world; RW = rig.matrix_world
V = [MW @ v.co for v in me.vertices]
names = {g.index: g.name for g in ob.vertex_groups}

PART_OF = {"DEF-spine": "chest", "DEF-spine.006": "head",
           "DEF-upper_arm.L": "shoulder_L", "DEF-forearm.L": "elbow_L", "DEF-upper_arm.R": "shoulder_R", "DEF-forearm.R": "elbow_R",
           "DEF-thigh.L": "hip_L", "DEF-shin.L": "knee_L", "DEF-thigh.R": "hip_R", "DEF-shin.R": "knee_R"}
dom = []
for v in ob.data.vertices:
    gs = sorted(((g.weight, names[g.group]) for g in v.groups), reverse=True)
    dom.append(gs[0][1] if gs else None)

# which way does the character face?  the head's marker faces point to its front
mat_marker = []
for s in ob.material_slots:
    m = s.material; c = m.diffuse_color if m else (1, 1, 1, 1)
    if m and m.node_tree:
        for n in m.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED": c = n.inputs["Base Color"].default_value; break
    mat_marker.append(bool(c[2] > 0.3 and c[1] > 0.3 and c[0] < 0.6 * min(c[1], c[2])))
if not any(mat_marker) and len(mat_marker) > 1:             # markers not coloured yet (rig file): the second material is the marker
    mat_marker = [i == 1 for i in range(len(mat_marker))]
head_c = sum((V[i] for i, d in enumerate(dom) if d == "DEF-spine.006"), Vector()) / max(1, sum(d == "DEF-spine.006" for d in dom))
mk = [p for p in me.polygons if p.material_index < len(mat_marker) and mat_marker[p.material_index] and dom[p.vertices[0]] == "DEF-spine.006"]
front = (sum((MW @ p.center for p in mk), Vector()) / len(mk) - head_c) if mk else Vector((0, 1, 0))
front.z = 0; front.normalize()
yaw = Matrix.Rotation(-Vector((0, 1)).angle_signed(Vector((front.x, front.y))), 4, "Z")   # turn the front to +Y
# check: after the turn the character's left (Rigify .L) must be on -X
l = (yaw @ (RW @ rig.data.bones["DEF-upper_arm.L"].head_local)).x
r = (yaw @ (RW @ rig.data.bones["DEF-upper_arm.R"].head_local)).x
assert l < r, "unexpected handedness"
Vw = [yaw @ p for p in V]
zmin = min(p.z for p in Vw); zmax = max(p.z for p in Vw)
S = 1.75 / (zmax - zmin)

def bone_axis(b):
    bb = rig.data.bones[b]
    return ((yaw @ (RW @ bb.tail_local)) - (yaw @ (RW @ bb.head_local))).normalized()

parts = {}
for grp, joint in PART_OF.items():
    idx = sorted({i for i, d in enumerate(dom) if d == grp})
    if not idx:
        continue
    P = [Vw[i] for i in idx]
    if joint in ("chest", "head"):
        R = Matrix.Identity(3)
        cen = sum(P, Vector()) / len(P)
        origin = Vector((cen.x, cen.y, min(p.z for p in P)))
    else:
        ax = bone_axis(grp)
        R = ax.rotation_difference(Vector((0, 0, -1))).to_matrix()          # bone axis -> straight down
        Q = [R @ p for p in P]
        cen = sum(Q, Vector()) / len(Q)
        origin_r = Vector((cen.x, cen.y, max(q.z for q in Q)))                # top (proximal) end centre
        origin = R.inverted() @ origin_r
    loc = {i: [round(c, 5) for c in (R @ (Vw[i] - origin)) * S] for i in idx}
    faces, marks, vmap, verts = [], [], {}, []
    for p in me.polygons:
        if dom[p.vertices[0]] != grp:
            continue
        f = []
        for i in p.vertices:
            if i not in vmap:
                vmap[i] = len(verts); verts.append(loc[i])
            f.append(vmap[i])
        if p.material_index < len(mat_marker) and mat_marker[p.material_index]:
            marks.append(len(faces))
        faces.append(f)
    zs = [v[2] for v in verts]
    parts[joint] = {"v": verts, "f": faces, "marker_faces": marks, "smooth": joint == "head",
                    "length": round(max(zs) - min(zs), 4)}

ev.to_mesh_clear()
out = {"source": bpy.data.filepath, "mesh": MESH, "rig": rig.name, "scale_to_1_75": round(S, 5),
       "front_turn_deg": round(yaw.to_euler().z * 57.2958, 1), "parts": parts}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False)
print("ASSET", OUT, "scale", round(S, 4), "turn", out["front_turn_deg"])
for k, p in parts.items():
    print("PART %-11s verts %3d faces %3d marker %2d length %.3f" % (k, len(p["v"]), len(p["f"]), len(p["marker_faces"]), p["length"]))

# -*- coding: utf-8 -*-
r"""One-time (run inside Blender): dump the official human_master.blend (rig + its 10 skinned pieces) for
extract_human_master.py: each piece's rest geometry (armature space), material index per face, the deform bone it
follows, and the rest matrices / heads of the bones used.

  blender -b assets\human_master\human_master.blend --factory-startup -P pipeline\dump_human_master.py -- <out.json>
"""
import bpy, json, sys

argv = sys.argv[sys.argv.index("--") + 1:]
rig = bpy.data.objects["rig"]
res = {"blender": bpy.app.version_string, "pieces": {}, "bones": {}}
for o in bpy.data.objects:
    if o.type != "MESH" or o.parent != rig or o.name.startswith("WGT"):
        continue
    groups = {g.index: g.name for g in o.vertex_groups}
    bone = None
    for v in o.data.vertices:
        for g in v.groups:
            if g.weight > 0.5: bone = groups[g.group]
    M = o.matrix_world
    res["pieces"][o.name] = {"bone": bone, "v": [list(M @ v.co) for v in o.data.vertices],
                             "f": [list(p.vertices) for p in o.data.polygons],
                             "mat": [p.material_index for p in o.data.polygons],
                             "smooth": any(p.use_smooth for p in o.data.polygons),
                             "materials": [m.name if m else None for m in o.data.materials]}
for b in rig.data.bones:
    if b.name.startswith(("DEF-spine", "DEF-upper_arm", "DEF-forearm", "DEF-hand", "DEF-thigh", "DEF-shin", "DEF-foot")):
        res["bones"][b.name] = {"head": list(b.head_local), "tail": list(b.tail_local), "matrix": [list(r) for r in b.matrix_local],
                                "bbone_segments": b.bbone_segments, "parent": b.parent.name if b.parent else None}
with open(argv[0], "w", encoding="utf-8") as fh:
    json.dump(res, fh, ensure_ascii=False)
print("dumped", len(res["pieces"]), "pieces", len(res["bones"]), "bones")

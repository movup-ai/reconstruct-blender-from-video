# -*- coding: utf-8 -*-
"""One-time (run inside Blender): dump the v2.1 supplied base character (character-base.blend, the 物理_* pieces of
actor01) as world-space geometry, for extract_base_character.py.

  blender -b assets\\base_v21\\character-base.blend --factory-startup -P pipeline\\dump_base_character.py -- <out.json> [prefix]
"""
import bpy, json, sys

argv = sys.argv[sys.argv.index("--") + 1:]
out, prefix = argv[0], (argv[1] if len(argv) > 1 else "物理_")
res = {"blender": bpy.app.version_string, "pieces": {}}
for o in bpy.data.objects:
    if o.type != "MESH" or not o.name.startswith(prefix):
        continue
    dg = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = dg.to_mesh()
    M = o.matrix_world
    res["pieces"][o.name[len(prefix):]] = {
        "v": [list(M @ v.co) for v in me.vertices],
        "f": [list(p.vertices) for p in me.polygons],
        "smooth": any(p.use_smooth for p in me.polygons),
        "local_v": [list(v.co) for v in me.vertices],
        "matrix": [list(r) for r in M],
        "material": me.materials[0].name if me.materials and me.materials[0] else None}
    dg.to_mesh_clear()
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print("dumped", len(res["pieces"]), "pieces to", out)

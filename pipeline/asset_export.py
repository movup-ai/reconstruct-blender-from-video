# -*- coding: utf-8 -*-
"""Runs inside Blender (blender -b ASSET.blend -P asset_export.py -- OUT.json): every visible mesh of the asset file,
modifiers applied, instances included, in world space -> {"v": [[x, y, z]...], "f": [[i, j, k...]...], "blender": ver}.
assets_lib.py normalises it (stands on z = 0, centred, unit size) and caches it."""
import json
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
dg = bpy.context.evaluated_depsgraph_get()
V, F, names = [], [], []
for inst in dg.object_instances:
    ob = inst.object
    if ob.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT"}:
        continue
    src = ob.original
    if src.hide_render or (inst.is_instance is False and not src.visible_get()):
        continue
    me = ob.to_mesh()
    if me is None:
        continue
    M = inst.matrix_world.copy()
    base = len(V)
    V += [list(M @ v.co) for v in me.vertices]
    F += [[base + i for i in p.vertices] for p in me.polygons]
    names.append(src.name)
    ob.to_mesh_clear()
with open(out, "w", encoding="utf-8") as fh:
    json.dump({"v": [[round(x, 5) for x in p] for p in V], "f": F, "objects": names, "blender": bpy.app.version_string}, fh)

# -*- coding: utf-8 -*-
"""Rebuild the white-model shot from scene.json (schema "ba_whitemodel_scene" 1.0) in Blender.

  blender -b --factory-startup -P reconstruct.py -- scene.json rebuilt.blend

Creates the collections, materials, meshes, empties (character rigs), cameras and lights with their parents,
rest transforms, visibility and every keyframe from scene.json, sets the frame range, fps, resolution, render
settings and active camera, and saves to the given NEW file (it never overwrites the master project).
Needs only Blender (tested with the version in scene.json "exporter"); no add-ons, no network."""
import json, sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) < 2:
    raise SystemExit("usage: blender -b --factory-startup -P reconstruct.py -- scene.json out.blend")
src, dst = argv[0], argv[1]
S = json.load(open(src, encoding="utf-8"))
if S.get("schema") != "ba_whitemodel_scene":
    raise SystemExit("not a ba_whitemodel_scene file")

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
scn = bpy.context.scene
sc = S["scene"]
scn.frame_start, scn.frame_end = sc["frame_start"], sc["frame_end"]
scn.render.fps, scn.render.fps_base = sc["fps"], sc["fps_base"]
scn.render.resolution_x, scn.render.resolution_y, scn.render.resolution_percentage = sc["resolution"]
r = sc["render"]
scn.render.engine = r["engine"]
try:
    scn.view_settings.view_transform = r["view_transform"]; scn.view_settings.look = r["look"]
except Exception:
    pass
scn.view_settings.exposure = r["exposure"]
sh = scn.display.shading
sh.light = r["shading_light"]; sh.color_type = r["shading_color_type"]
sh.show_shadows = r["shadows"]; sh.shadow_intensity = r["shadow_intensity"]
if hasattr(sh, "background_type"):
    sh.background_type = "WORLD"
if r.get("world_color"):
    if scn.world is None:
        scn.world = bpy.data.worlds.new("World")
    scn.world.color = r["world_color"][:3]

mats = {}
for name, m in S["materials"].items():
    mm = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mm.diffuse_color = m["diffuse_color"]; mm.roughness = m.get("roughness", 1.0)
    mats[name] = mm

colls = {}
def coll(name):
    if name == "<scene>":
        return scn.collection
    if name not in colls:
        c = bpy.data.collections.new(name)
        par = S["collections"].get(name, {}).get("parent")
        (coll(par) if par else scn.collection).children.link(c)
        colls[name] = c
    return colls[name]

objs = {}
for rec in S["objects"]:
    t = rec["type"]
    if t == "MESH":
        m = rec["mesh"]
        me = bpy.data.meshes.new(m["name"])
        me.from_pydata(m["vertices"], [], m["faces"]); me.update()
        for mn in m["materials"]:
            me.materials.append(mats.get(mn))
        if m.get("face_material"):
            for p, i in zip(me.polygons, m["face_material"]):
                p.material_index = i
        if m.get("smooth"):
            for p in me.polygons:
                p.use_smooth = True
        ob = bpy.data.objects.new(rec["name"], me)
    elif t == "CAMERA":
        d = bpy.data.cameras.new(rec["name"]); c = rec["camera"]
        d.type = c["type"]; d.lens = c["lens"]; d.sensor_width = c["sensor_width"]; d.sensor_height = c["sensor_height"]
        d.sensor_fit = c["sensor_fit"]; d.clip_start = c["clip_start"]; d.clip_end = c["clip_end"]
        d.shift_x = c["shift_x"]; d.shift_y = c["shift_y"]
        ob = bpy.data.objects.new(rec["name"], d)
    elif t == "LIGHT":
        l = rec["light"]; d = bpy.data.lights.new(rec["name"], l["type"]); d.energy = l["energy"]; d.color = l["color"]
        ob = bpy.data.objects.new(rec["name"], d)
    else:
        ob = bpy.data.objects.new(rec["name"], None)
        e = rec.get("empty") or {}
        ob.empty_display_type = e.get("display_type", "PLAIN_AXES"); ob.empty_display_size = e.get("size", 0.1)
    for cn in rec["collections"] or ["<scene>"]:
        coll(cn).objects.link(ob)
    ob.rotation_mode = rec["rotation_mode"]
    ob.location = rec["location"]; ob.rotation_euler = rec["rotation_euler"]; ob.scale = rec["scale"]
    ob.hide_render = rec["hide_render"]; ob.hide_viewport = rec["hide_viewport"]
    objs[rec["name"]] = ob
for rec in S["objects"]:
    if rec["parent"]:
        ob = objs[rec["name"]]
        ob.parent = objs[rec["parent"]]
        from mathutils import Matrix
        ob.matrix_parent_inverse = Matrix(rec["matrix_parent_inverse"])

def fcurves(ob):
    act = ob.animation_data.action if ob.animation_data else None
    if act is None:
        return []
    try:
        return list(act.fcurves)
    except AttributeError:                       # Blender 4.4+: layered actions
        return [fc for lay in act.layers for st in lay.strips for cb in st.channelbags for fc in cb.fcurves]

for name, chans in S["animation"].items():
    ob = objs.get(name)
    if ob is None:
        continue
    for path, idx in chans.items():
        for i, keys in idx.items():
            i = int(i)
            prop = getattr(ob, path)
            vector = hasattr(prop, "__len__") and not isinstance(prop, str)
            for fr, val, interp in keys:
                if vector:
                    prop = getattr(ob, path); prop[i] = val
                else:
                    setattr(ob, path, bool(val) if isinstance(prop, bool) else val)
                ob.keyframe_insert(path, index=i if vector else -1, frame=fr)
            for fc in fcurves(ob):                 # interpolation as exported
                if fc.data_path == path and fc.array_index == i:
                    for kp, k in zip(fc.keyframe_points, keys):
                        kp.interpolation = k[2]

if sc.get("camera") and sc["camera"] in objs:
    scn.camera = objs[sc["camera"]]
scn.frame_set(scn.frame_start)
bpy.ops.wm.save_as_mainfile(filepath=dst)
print("rebuilt", len(objs), "objects ->", dst)

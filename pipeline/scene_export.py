# -*- coding: utf-8 -*-
"""Export the open Blender scene as model_3d/reconstruction/scene.json (v2.1 delivery rule: real scene data, not a
file list).  Runs inside Blender (called by blender_build.py right after the .blend is saved).

Schema "ba_whitemodel_scene" 1.0 — everything needed to rebuild the shot with reconstruct.py:
  scene      units (metre), coordinate system (Blender: right-handed, Z up), frame range, fps, resolution,
             active camera, render engine / colour management / world colour
  materials  name -> base colour (linear RGBA)
  collections name -> parent collection, objects
  objects    name, type (MESH / EMPTY / CAMERA / LIGHT), collection, parent, rest transform (location, XYZ euler in
             radians, scale), visibility, mesh geometry (vertices, faces, per-face material slot, smooth flag) or
             camera / empty data
  animation  every keyframed property: object -> data path -> channel index -> [[frame, value, interpolation], ...]
  characters per character: label, colour, height, root empty, joint empties (rig = empties hierarchy, keyed
             rotations), mesh blocks with the joint they follow, eye / chest band material
  camera     camera object, lens, sensor, per-shot frame ranges
The .blend stays the master project; this file is a faithful copy of its objects, geometry and keys."""
import json, os

SCHEMA = "ba_whitemodel_scene"
VERSION = "1.0"


def _r(x, n=5):
    return round(float(x), n)


def _vec(v, n=5):
    return [_r(a, n) for a in v]


def export(bpy, path, blend_name, data=None):
    scn = bpy.context.scene
    out = {
        "schema": SCHEMA, "schema_version": VERSION,
        "exporter": "pipeline/scene_export.py (Blender %s)" % bpy.app.version_string,
        "master_project": "model_3d/" + blend_name,
        "scene": {
            "name": scn.name, "units": "metre", "unit_scale": _r(scn.unit_settings.scale_length),
            "coordinate_system": "Blender world: right-handed, +Z up; rotations XYZ euler in radians; a camera looks along its local -Z",
            "frame_start": scn.frame_start, "frame_end": scn.frame_end, "fps": scn.render.fps, "fps_base": _r(scn.render.fps_base),
            "resolution": [scn.render.resolution_x, scn.render.resolution_y, scn.render.resolution_percentage],
            "camera": scn.camera.name if scn.camera else None,
            "render": {"engine": scn.render.engine, "view_transform": scn.view_settings.view_transform,
                       "look": scn.view_settings.look, "exposure": _r(scn.view_settings.exposure),
                       "shading_light": scn.display.shading.light, "shading_color_type": scn.display.shading.color_type,
                       "shadows": bool(scn.display.shading.show_shadows), "shadow_intensity": _r(scn.display.shading.shadow_intensity),
                       "world_color": _vec(scn.world.color) if scn.world else None,
                       "output": scn.render.filepath, "file_format": scn.render.image_settings.file_format}},
        "materials": {}, "collections": {}, "objects": [], "animation": {}, "characters": [], "camera": {}}
    for m in bpy.data.materials:
        if m.users:
            out["materials"][m.name] = {"diffuse_color": _vec(m.diffuse_color), "roughness": _r(m.roughness)}

    def walk(c, parent):
        out["collections"][c.name] = {"parent": parent, "objects": [o.name for o in c.objects]}
        for ch in c.children:
            walk(ch, c.name)
    for ch in scn.collection.children:
        walk(ch, None)
    if scn.collection.objects:
        out["collections"]["<scene>"] = {"parent": None, "objects": [o.name for o in scn.collection.objects]}

    for o in scn.objects:
        rec = {"name": o.name, "type": o.type, "collections": [c.name for c in o.users_collection],
               "parent": o.parent.name if o.parent else None, "rotation_mode": o.rotation_mode,
               "location": _vec(o.location), "rotation_euler": _vec(o.rotation_euler, 6), "scale": _vec(o.scale),
               "hide_render": bool(o.hide_render), "hide_viewport": bool(o.hide_viewport),
               "matrix_parent_inverse": [_vec(r, 6) for r in o.matrix_parent_inverse]}
        if o.type == "MESH":
            me = o.data
            rec["mesh"] = {"name": me.name, "vertices": [_vec(v.co) for v in me.vertices],
                           "faces": [list(p.vertices) for p in me.polygons],
                           "face_material": [p.material_index for p in me.polygons] if len(me.materials) > 1 else None,
                           "smooth": bool(me.polygons and me.polygons[0].use_smooth),
                           "materials": [m.name if m else None for m in me.materials]}
        elif o.type == "EMPTY":
            rec["empty"] = {"display_type": o.empty_display_type, "size": _r(o.empty_display_size)}
        elif o.type == "CAMERA":
            d = o.data
            rec["camera"] = {"lens": _r(d.lens), "sensor_width": _r(d.sensor_width), "sensor_height": _r(d.sensor_height),
                             "sensor_fit": d.sensor_fit, "clip_start": _r(d.clip_start), "clip_end": _r(d.clip_end),
                             "shift_x": _r(d.shift_x), "shift_y": _r(d.shift_y), "type": d.type}
        elif o.type == "LIGHT":
            d = o.data
            rec["light"] = {"type": d.type, "energy": _r(d.energy), "color": _vec(d.color)}
        out["objects"].append(rec)
        ad = o.animation_data
        act = ad.action if ad else None
        if act is not None:
            chans = {}
            fcs = []
            try:
                fcs = list(act.fcurves)
            except AttributeError:                           # Blender 4.4+: layered actions (layers / strips / channelbags)
                for lay in act.layers:
                    for st in lay.strips:
                        for cb in st.channelbags:
                            fcs += list(cb.fcurves)
            for fc in fcs:
                keys = [[_r(k.co[0], 3), _r(k.co[1], 6), k.interpolation] for k in fc.keyframe_points]
                chans.setdefault(fc.data_path, {})[str(fc.array_index)] = keys
            if chans:
                out["animation"][o.name] = chans

    if data:                                                 # what each object is: characters, camera, shots
        for kind, c in (data.get("chars") or {}).items():
            objs = [o.name for o in scn.objects if o.name.startswith(kind + "_")]
            out["characters"].append({
                "id": kind, "label": c.get("label"), "colour": c.get("colour"), "height_m": c.get("height"),
                "style": c.get("style"), "root": kind + "_root",
                "joints": [n for n in objs if not n.endswith(tuple("_geo%d" % i for i in range(40))) and n != kind + "_root"],
                "blocks": [n for n in objs if "_geo" in n],
                "rig": "empties hierarchy: <id>_root (location + yaw keys) -> <id>_pelvis (height key) -> joints (XYZ euler keys); "
                       "each mesh block is parented to its joint; the white eye / chest bands are faces with material WM_marker",
                "visible": c.get("visible")})
        out["camera"] = {"object": scn.camera.name if scn.camera else None, "lens_mm": data.get("lens"),
                         "sensor_width_mm": data.get("sensor"),
                         "shots": [{"shot": s.get("shot"), "frames": [s["frames"][0] + 1, s["frames"][1] + 1]}
                                   for s in (data.get("shots") or []) if "frames" in s],
                         "note": "camera keyed every frame (location + rotation); shot frames are 1-based Blender frames"}
        zoom = data.get("zoom") or []
        if zoom and max(abs(float(z) - 1.0) for z in zoom) > 1e-3:    # the lens zooms: the camera data's lens is keyed too
            out["camera"]["lens_mm_per_frame"] = [_r(float(data["lens"]) * float(z), 3) for z in zoom]
            out["camera"]["note"] = "camera keyed every frame (location + rotation, and the lens: it zooms); shot frames are 1-based Blender frames"
        out["source"] = {"clip": data.get("name"), "picture_size": data.get("res"), "fps": data.get("fps"), "frames": data.get("frames")}
        out['props'] = [{'id':p.get('object_id',p['name']), 'object':p['name'], 'label':p.get('label'),
                         'size_m':p.get('size_m'), 'events':p.get('events', []),
                         'event_frames':'0-based source frames'} for p in data.get('props',[]) if p.get('held')]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    return path

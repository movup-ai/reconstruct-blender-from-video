# -*- coding: utf-8 -*-
"""LZHD V1.0 trial check (6.2): render every character of a delivered .blend from the front, the side and the back
at a few frames (fixed lens, fixed distance) so torso width / height / depth, head-to-body and limb lengths can be
compared between views.  Run inside Blender; the .blend is not modified.

  blender -b <pkg>.blend -P pipeline\\multiview_check.py -- <out.jpg> [frames 1,55,110]"""
import bpy, math, os, sys, tempfile
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
frames = [int(x) for x in (argv[1] if len(argv) > 1 else "1").split(",")]
scn = bpy.context.scene
scn.render.resolution_x, scn.render.resolution_y, scn.render.resolution_percentage = 420, 560, 100
scn.render.image_settings.file_format = "JPEG"
tmp = tempfile.mkdtemp()
roots = sorted((o for o in bpy.data.objects if o.name.endswith("_root") and o.type == "EMPTY"), key=lambda o: o.name)
cd = bpy.data.cameras.new("mv_cam"); cd.lens = 50; cd.sensor_width = 36
cam = bpy.data.objects.new("mv_cam", cd); scn.collection.objects.link(cam); scn.camera = cam
fc = bpy.data.collections.new("mv_floor"); scn.collection.children.link(fc)      # plain floor instead of the set
fm = bpy.data.meshes.new("mv_floor"); fm.from_pydata([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)], [], [(0, 1, 2, 3)])
fo = bpy.data.objects.new("mv_floor", fm); fc.objects.link(fo)
if bpy.data.materials.get("WM_bg"):
    fm.materials.append(bpy.data.materials["WM_bg"])


def only(kind):
    """show this character, the lights and the plain floor; every other collection is excluded from the render"""
    for lc in bpy.context.view_layer.layer_collection.children:
        keep = lc.name in ("Lights", "mv_floor") or any(o.name == kind + "_root" for o in lc.collection.all_objects)
        lc.exclude = not keep
    cam_c = bpy.context.view_layer.layer_collection
    cam.hide_render = False


tiles = []
for fr in frames:
    for r in roots:
        kind = r.name[:-5]
        only(kind)
        scn.frame_set(fr)
        pel = bpy.data.objects.get(kind + "_pelvis")
        c = (pel.matrix_world.translation if pel else r.matrix_world.translation).copy()
        fwd = r.matrix_world.to_3x3() @ Vector((0, 1, 0)); fwd.z = 0; fwd.normalize()
        for view, ang in (("front", 0), ("side", 90), ("back", 180)):
            d = fwd.copy(); d.rotate(__import__("mathutils").Euler((0, 0, math.radians(ang))))
            target = Vector((c.x, c.y, 0.95))
            cam.location = target + d * 4.6 + Vector((0, 0, 0.1))
            cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
            p = os.path.join(tmp, "%s_%d_%s.jpg" % (kind, fr, view))
            scn.render.filepath = p; bpy.ops.render.render(write_still=True)
            tiles.append((kind, fr, view, p))
# grid: one row per character and frame, three views per row
import numpy as np
rows = {}
for kind, fr, view, p in tiles:
    img = bpy.data.images.load(p); w, h = img.size
    a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]
    rows.setdefault((kind, fr), []).append(a); bpy.data.images.remove(img)
grid = np.vstack([np.hstack(v) for k, v in sorted(rows.items())])
H, W = grid.shape[:2]
im = bpy.data.images.new("mv_grid", W, H)
im.pixels = np.dstack([grid[::-1], np.ones((H, W, 1), np.float32)]).ravel()
im.filepath_raw = out; im.file_format = "JPEG"; im.save()
print("multiview", out, "rows:", ", ".join("%s f%d" % k for k in sorted(rows)))

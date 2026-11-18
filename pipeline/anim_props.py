# -*- coding: utf-8 -*-
"""Moving set pieces (e.g. a chair a person pulls) for a clip: output/<clip>/anim_props.json -> scene_data["props"].

Run after build_env.py (fit / build_env rewrite scene_data) and before the render:
  python pipeline\\anim_props.py <clip>

anim_props.json: {"props": [{"name": "chair4", "parts": [{"name": "back", "colour": "white",
   "box": [cx, cy, z0, sx, sz, sy]}, ...],            # local box: centre x / y, bottom z, size along x / height / along y
   "keys": [[x, y, yaw_deg], ...]}]}                  # one per frame (world position of the local origin on the floor)
   or keys [[x, y, z, rx, ry, rz], ...] (degrees, Blender XYZ euler) for props that tip over / are lifted
A part may instead be a library shape: {"name": "car", "colour": "near", "prop": "car", "size": [length, height, width],
"yaw": 0} (props.build, e.g. the sedan; local origin = centre of its footprint on the floor, +x = front at yaw 0).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload, jsave, output_dir


def box_mesh(cx, cy, z0, sx, sz, sy):
    xs, ys = (cx - sx / 2, cx + sx / 2), (cy - sy / 2, cy + sy / 2)
    zs = (z0, z0 + sz)
    V = [[x, y, z] for z in zs for y in ys for x in xs]          # 0..3 bottom, 4..7 top
    F = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    return V, F


def main():
    name = sys.argv[1]
    c = cfg()
    src = os.path.join(output_dir(), name, "anim_props.json")
    dp = os.path.join(c["work_root"], name, "scene_data.json")
    data = jload(dp)
    data["props"] = [p for p in (data.get("props") or []) if not p.get("anim_prop")]
    if os.path.exists(src):
        for pr in jload(src)["props"]:
            keys = pr["keys"]
            meshes = []
            for part in pr["parts"]:
                if "prop" in part:
                    import props as _props
                    w, h, d = part["size"]
                    for pn, V, F, _sh in _props.place(_props.build(part["prop"], w, h, d), (0.0, 0.0), part.get("yaw", 0.0)):
                        meshes.append(("%s_%s" % (part["name"], pn), V, F, part["colour"]))
                else:
                    V, F = box_mesh(*part["box"])
                    meshes.append((part["name"], V, F, part["colour"]))
            for pname, V, F, colour in meshes:
                data["props"].append({"name": "%s_%s" % (pr["name"], pname), "v": V, "f": F, "colour": colour,
                                      "anim_prop": True,
                                      "keys": [([k[0], k[1], 0.0, 0.0, 0.0, k[2]] if len(k) == 3 else list(k[:6])) + [1.0, 1.0, 1.0]
                                               for k in keys]})
    jsave(dp, data)
    print("anim props:", [p["name"] for p in data["props"] if p.get("anim_prop")])


if __name__ == "__main__":
    main()

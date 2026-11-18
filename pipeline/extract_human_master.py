# -*- coding: utf-8 -*-
"""One-time: the official human_master.blend (dumped by dump_human_master.py) -> assets/human_master.json, used by
rig.py (fitting) and blender_build.py (style "master").

human_master.blend is a Rigify rig with 10 rigid pieces, each weighted 100 % to one deform bone:
  torso -> DEF-spine, head sphere -> DEF-spine.006, upper arm -> DEF-upper_arm.*, forearm -> DEF-forearm.*,
  thigh -> DEF-thigh.*, shin -> DEF-shin.*
The rig faces -Y (Blender convention), the pipeline's joints face +Y, so everything is turned 180 degrees about Z.
Joints sit on those bones' heads (shoulder = upper-arm head, elbow = forearm head, wrist = hand head, hip = thigh head,
knee = shin head, ankle = foot head, neck/head = DEF-spine.006 head, pelvis = DEF-spine head).  At the pipeline's rest
pose the limbs hang straight down, so each limb bone gets a swing from the rig's A-pose direction to straight down.
The whole figure is scaled to 1.75 m (sole to head top) for the joint table; the build scales it back uniformly
(model_scale = person height / 1.75 * scale_to_1_75, the only scale the spec allows).

For each joint the json stores the bone it drives and "R": the bone's rest rotation in the joint frame, so in Blender
bone world matrix = joint world matrix @ [R | 0] * model_scale (Copy Transforms on the bone).

  python pipeline\\extract_human_master.py assets\\human_master\\master_dump.json assets\\human_master.json"""
import json, sys
import numpy as np

G = np.diag([-1.0, -1.0, 1.0])                  # rig space (faces -Y) -> pipeline space (faces +Y)
PIECE = {"chest": "DEF-spine", "head": "DEF-spine.006"}
for _s in ("L", "R"):
    PIECE.update({"shoulder_" + _s: "DEF-upper_arm." + _s, "elbow_" + _s: "DEF-forearm." + _s,
                  "hip_" + _s: "DEF-thigh." + _s, "knee_" + _s: "DEF-shin." + _s})


def swing(a, b):
    """smallest rotation taking unit vector a to unit vector b"""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def main():
    src, out = sys.argv[1], sys.argv[2]
    d = json.load(open(src, encoding="utf-8"))
    B = {k: {"head": np.array(b["head"]), "M": np.array(b["matrix"])} for k, b in d["bones"].items()}
    P = d["pieces"]
    allv = np.vstack([np.array(p["v"]) for p in P.values()])
    sole_z, top_z = float(allv[:, 2].min()), float(allv[:, 2].max())
    k = 1.75 / (top_z - sole_z)
    h = lambda n: G @ B[n]["head"]
    L = lambda a, b: float(np.linalg.norm(B[b]["head"] - B[a]["head"]))
    pel = h("DEF-spine")
    upper = (L("DEF-upper_arm.L", "DEF-forearm.L") + L("DEF-upper_arm.R", "DEF-forearm.R")) / 2
    fore = (L("DEF-forearm.L", "DEF-hand.L") + L("DEF-forearm.R", "DEF-hand.R")) / 2
    thigh = (L("DEF-thigh.L", "DEF-shin.L") + L("DEF-thigh.R", "DEF-shin.R")) / 2
    shin = (L("DEF-shin.L", "DEF-foot.L") + L("DEF-shin.R", "DEF-foot.R")) / 2
    ankle_h = float(B["DEF-foot.L"]["head"][2] - sole_z)
    hip = (h("DEF-thigh.L") + G @ (B["DEF-thigh.R"]["head"] * [-1, 1, 1])) / 2      # left hip (pipeline -x)
    sh = (h("DEF-upper_arm.L") + G @ (B["DEF-upper_arm.R"]["head"] * [-1, 1, 1])) / 2
    neck = h("DEF-spine.006")
    pelvis_z = ankle_h + shin + thigh - (hip[2] - pel[2])           # pelvis height standing straight

    def r5(v):
        return [round(float(x), 5) for x in v]

    joints = [["pelvis", "root", [0, 0, pelvis_z]], ["spine", "pelvis", [0, 0, 0]], ["chest", "spine", [0, 0, 0]],
              ["neck", "chest", list(neck - pel)], ["head", "neck", [0, 0, 0]]]
    for side, sx in (("L", 1), ("R", -1)):                # sh / hip are the left ones (x < 0); right = mirrored
        joints += [["shoulder_" + side, "chest", [sx * sh[0], sh[1] - pel[1], sh[2] - pel[2]]],
                   ["elbow_" + side, "shoulder_" + side, [0, 0, -upper]], ["wrist_" + side, "elbow_" + side, [0, 0, -fore]],
                   ["hip_" + side, "pelvis", [sx * hip[0], hip[1] - pel[1], hip[2] - pel[2]]],
                   ["knee_" + side, "hip_" + side, [0, 0, -thigh]], ["ankle_" + side, "knee_" + side, [0, 0, -shin]]]
    joints = [[n, p, r5(np.array(o) * k)] for n, p, o in joints]

    bones, parts = {}, {}
    for j, bn in PIECE.items():
        Rb = B[bn]["M"][:3, :3]
        R = G @ Rb
        if j not in ("chest", "head"):                   # limbs: rig A-pose -> straight down at the pipeline's rest
            R = swing(R[:, 1], np.array([0, 0, -1.0])) @ R
        bones[j] = {"bone": bn, "R": [r5(r) for r in R]}
        piece = next(nm for nm, p in P.items() if p["bone"] == bn)
        p = P[piece]
        V = np.array(p["v"]) - B[bn]["head"]
        loc = V @ Rb                                      # rig rest -> bone-local (bone matrix is orthonormal)
        vj = (loc @ R.T) * k                              # bone-local -> joint frame, 1.75 m figure
        parts[j] = {"v": [r5(x) for x in vj], "f": p["f"], "smooth": bool(p["smooth"]), "piece": piece,
                    "marker_faces": [i for i, m in enumerate(p["mat"]) if m == 1]}
    hv = np.array(parts["head"]["v"]); hf = parts["head"]["f"]
    mk = np.vstack([hv[hf[i]].mean(0) for i in parts["head"]["marker_faces"]])
    head_c = (hv.max(0) + hv.min(0)) / 2; head_r = float((hv.max(0) - hv.min(0))[0] / 2)
    res = {"source": "human_master.blend (official, sha256 cbc13dbf...5e6c, blender %s)" % d.get("blender"),
           "scale_to_1_75": round(k, 5), "rig_height": round(top_z - sole_z, 4), "sole": round(ankle_h * k, 5),
           "head_centre": r5(head_c), "head_radius": round(head_r, 5), "eye_band": r5(mk.mean(0)),
           "eye_band_front": round(float(mk[:, 1].max()), 5), "joints": joints, "bones": bones, "parts": parts}
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("scale %.4f (rig %.3f m)  pelvis %.3f  neck %s  shoulder %s  upper %.3f fore %.3f  hip %s thigh %.3f shin %.3f  sole %.3f"
          % (k, top_z - sole_z, pelvis_z * k, r5((neck - pel) * k), r5(sh * k), upper * k, fore * k, r5(hip * k), thigh * k, shin * k, ankle_h * k))
    print("head centre %s r %.3f, eye band %s (front y %.3f); marker faces head %d chest %d"
          % (r5(head_c), head_r, r5(mk.mean(0)), mk[:, 1].max(), len(parts["head"]["marker_faces"]), len(parts["chest"]["marker_faces"])))


if __name__ == "__main__":
    main()

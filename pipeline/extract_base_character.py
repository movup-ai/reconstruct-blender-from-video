# -*- coding: utf-8 -*-
"""One-time: the v2.1 supplied base character (character-base.blend, dumped by dump_base_character.py) ->
assets/base_character_v21.json, used by rig.py and blender_build.py for character_style "base21".

The base character is 10 rigid blocks (beveled box torso / upper arms / forearms / thighs / shins, sphere head) plus
an eye band on the head and a chest band on the torso.  In the .blend every block's own frame has the bone along
local +Y from the joint, the front on local +Z and the character's left on local +X.  Here every block is put in
the pipeline's joint frame (character faces +Y, its right on +X, limbs hanging along -Z), the whole figure is scaled
to 1.75 m (proportions kept), and the two bands are joined into the head / torso meshes as "marker" faces
(drawn pure white).  The chest band, a flat quad lying on the torso's front face in the .blend, is lifted 4 mm off
the face so it never z-fights.

Joints (1.75 m figure): pelvis = hip height standing (thigh + shin), chest at the pelvis, neck at the torso top,
shoulders / hips where the arm / leg blocks start, elbow / knee = upper-arm / thigh length, wrist = forearm length,
ankle 6.3 cm (scaled) above the bottom of the shin block (the sole), like the earlier sample character.

  python pipeline\\extract_base_character.py assets\\base_v21\\base_dump.json assets\\base_character_v21.json"""
import json, sys
import numpy as np

A_BODY = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)     # local (x left, y up, z front) -> (x right, y front, z up)
A_LIMB = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)     # local (y along the limb, z front) -> limb along -Z


def main():
    src, out = sys.argv[1], sys.argv[2]
    d = json.load(open(src, encoding="utf-8"))
    P = d["pieces"]
    M = {k: np.array(p["matrix"], float) for k, p in P.items()}
    L = {k: np.array(p["local_v"], float) for k, p in P.items()}

    def rel(a, b):                               # origin of b in a's local frame
        return (np.linalg.inv(M[a]) @ M[b][:, 3])[:3]

    def ext(k, ax):
        return float(L[k][:, ax].max())

    torso_len = ext("torso", 1)
    head_c = rel("torso", "head")[1]                       # sphere centre above the torso origin
    sh = rel("torso", "left_upper_arm")                    # (+x = left)
    hip = rel("torso", "left_thigh")
    upper = float(np.mean([rel("left_upper_arm", "left_forearm")[1], rel("right_upper_arm", "right_forearm")[1]]))
    fore = (ext("left_forearm", 1) + ext("right_forearm", 1)) / 2
    thigh = float(np.mean([rel("left_thigh", "left_shin")[1], rel("right_thigh", "right_shin")[1]]))
    shin = (ext("left_shin", 1) + ext("right_shin", 1)) / 2
    head_r = ext("head", 1)
    pelvis = thigh + shin
    k = 1.75 / (pelvis + head_c + head_r)                  # whole figure -> 1.75 m
    sole = 0.063
    ankle = shin - sole / k

    def r5(v):
        return [round(float(x), 5) for x in v]

    joints = [["pelvis", "root", [0, 0, pelvis]], ["spine", "pelvis", [0, 0, 0]], ["chest", "spine", [0, 0, 0]],
              ["neck", "chest", [0, 0, torso_len]], ["head", "neck", [0, 0, 0]]]
    for side, sx in (("L", -1), ("R", 1)):
        joints += [["shoulder_" + side, "chest", [sx * abs(sh[0]), 0, sh[1]]], ["elbow_" + side, "shoulder_" + side, [0, 0, -upper]],
                   ["wrist_" + side, "elbow_" + side, [0, 0, -fore]], ["hip_" + side, "pelvis", [sx * abs(hip[0]), 0, 0]],
                   ["knee_" + side, "hip_" + side, [0, 0, -thigh]], ["ankle_" + side, "knee_" + side, [0, 0, -ankle]]]
    joints = [[n, p, r5(np.array(o) * k)] for n, p, o in joints]

    def part(piece, A, off=(0, 0, 0), bands=()):
        v = (L[piece] @ A.T + np.array(off)) * k
        f = [list(x) for x in P[piece]["f"]]
        marks = []
        for b, lift in bands:                              # join a band: its local frame -> its parent block's frame
            Mb = np.linalg.inv(M[piece]) @ M[b]
            bl = (np.c_[L[b], np.ones(len(L[b]))] @ Mb.T)[:, :3]
            if lift:                                       # push off the block's front face (local +z)
                bl = bl + np.array([0, 0, lift])
            bv = (bl @ A.T + np.array(off)) * k
            base = len(v)
            v = np.vstack([v, bv])
            for x in P[b]["f"]:
                marks.append(len(f)); f.append([base + i for i in x])
        return {"v": [r5(x) for x in v], "f": f, "smooth": bool(P[piece]["smooth"]), "marker_faces": marks}

    # bands sized like the approved client sample (11月2日试稿, L3): eye visor ~71% of the head's width and ~22% of its
    # height, wrapped on the sphere at eye level; chest band ~71% of the torso's width, ~16% of its height, on the
    # upper chest (66%..82% of the torso height).  The base model's own thin strips are replaced by these.
    parts = {"chest": part("torso", A_BODY), "head": part("head", A_BODY, off=(0, 0, head_c - torso_len))}
    hr = head_r * k; hcz = (head_c - torso_len) * k
    w_eye, h_eye = 0.71 * 2 * hr, 0.22 * 2 * hr
    az = np.arcsin(min(0.95, w_eye / 2 / hr)); el = np.arcsin(h_eye / 2 / hr)
    R = hr * 1.012
    nu, nv = 14, 3
    ev = [[R * np.cos(e) * np.sin(a), R * np.cos(e) * np.cos(a), hcz + R * np.sin(e)]
          for e in np.linspace(-el, el, nv + 1) for a in np.linspace(-az, az, nu + 1)]
    ef = [[j * (nu + 1) + i, j * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i] for j in range(nv) for i in range(nu)]
    tw = (ext("torso", 0) - L["torso"][:, 0].min()) * k; tl = torso_len * k; td = ext("torso", 2) * k
    cw, z0, z1 = 0.71 * tw, 0.66 * tl, 0.82 * tl
    yf = td + 0.004
    cv = [[-cw / 2, yf, z0], [cw / 2, yf, z0], [cw / 2, yf, z1], [-cw / 2, yf, z1]]
    cf = [[0, 1, 2, 3]]
    for nm, (bv, bf) in (("head", (ev, ef)), ("chest", (cv, cf))):
        p = parts[nm]; base = len(p["v"])
        p["v"] += [r5(x) for x in bv]
        for f in bf:
            p["marker_faces"].append(len(p["f"])); p["f"].append([base + i for i in f])
    for side, nm in (("L", "left"), ("R", "right")):
        parts["shoulder_" + side] = part(nm + "_upper_arm", A_LIMB)
        parts["elbow_" + side] = part(nm + "_forearm", A_LIMB)
        parts["hip_" + side] = part(nm + "_thigh", A_LIMB)
        parts["knee_" + side] = part(nm + "_shin", A_LIMB)
    hc = (head_c - torso_len) * k
    res = {"source": "v2.1 supplied base model (character-base.blend, blender %s), pieces 物理_*" % d.get("blender"),
           "scale_to_1_75": round(k, 5), "sole": sole, "head_centre": round(hc, 5), "head_radius": round(head_r * k, 5),
           "eye_level": round(hc, 5), "joints": joints, "parts": parts}
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("scale %.4f  pelvis %.3f  neck %.3f  shoulder x %.3f z %.3f  upper %.3f fore %.3f  hip x %.3f thigh %.3f ankle %.3f  head c %.3f r %.3f"
          % (k, pelvis * k, torso_len * k, abs(sh[0]) * k, sh[1] * k, upper * k, fore * k, abs(hip[0]) * k, thigh * k, ankle * k, hc, head_r * k))
    print("marker faces: head %d, chest %d" % (len(parts["head"]["marker_faces"]), len(parts["chest"]["marker_faces"])))


if __name__ == "__main__":
    main()

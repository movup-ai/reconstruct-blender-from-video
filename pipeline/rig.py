# -*- coding: utf-8 -*-
"""numpy copy of the character rig in blender_build.py (same joints, same Blender XYZ euler order).
character faces +Y, its right side is +X, z up."""
import numpy as np

JOINTS = ["pelvis", "spine", "chest", "neck", "head",
          "shoulder_L", "elbow_L", "wrist_L", "shoulder_R", "elbow_R", "wrist_R",
          "hip_L", "knee_L", "ankle_L", "hip_R", "knee_R", "ankle_R"]

def joint_defs(height, style="sample"):
    """style "sample": the client's sample character (11_2 blend; the default): one torso block, head on the torso, no neck / hands / feet.
    style "classic": the earlier mannequin (neck, hands, feet; the fallback when the sample mesh asset is missing)."""
    s = height / 1.75
    if style == "sample":
        J = [("pelvis", "root", (0, 0, 0.94)), ("spine", "pelvis", (0, 0, 0)), ("chest", "spine", (0, 0, 0)),
             ("neck", "chest", (0, 0, 0.52)), ("head", "neck", (0, 0, 0))]
        for side, sx in (("L", -1), ("R", 1)):
            J += [("shoulder_" + side, "chest", (sx * 0.222, 0, 0.482)), ("elbow_" + side, "shoulder_" + side, (0, 0, -0.274)),
                  ("wrist_" + side, "elbow_" + side, (0, 0, -0.335)), ("hip_" + side, "pelvis", (sx * 0.081, 0, 0)),
                  ("knee_" + side, "hip_" + side, (0, 0, -0.459)), ("ankle_" + side, "knee_" + side, (0, 0, -0.42))]
        return [(n, p, np.array(o, float) * s) for n, p, o in J], s
    J = [("pelvis", "root", (0, 0, 0.93)), ("spine", "pelvis", (0, 0, 0.10)), ("chest", "spine", (0, 0, 0.17)),
         ("neck", "chest", (0, 0, 0.30)), ("head", "neck", (0, 0, 0.08))]
    for side, sx in (("L", -1), ("R", 1)):
        J += [("shoulder_" + side, "chest", (sx * 0.225, 0, 0.25)), ("elbow_" + side, "shoulder_" + side, (0, 0, -0.28)),
              ("wrist_" + side, "elbow_" + side, (0, 0, -0.25)), ("hip_" + side, "pelvis", (sx * 0.095, 0, -0.04)),
              ("knee_" + side, "hip_" + side, (0, 0, -0.42)), ("ankle_" + side, "knee_" + side, (0, 0, -0.43))]
    return [(n, p, np.array(o, float) * s) for n, p, o in J], s

def Rxyz(r):
    a, b, c = np.radians(r)
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

# COCO-17 keypoints as points on the rig: (joint, local offset / s)
COCO_ON_RIG = {
  "classic": [("head", (0, 0.120, 0.07)), ("head", (-0.035, 0.110, 0.11)), ("head", (0.035, 0.110, 0.11)),
              ("head", (-0.125, 0.0, 0.09)), ("head", (0.125, 0.0, 0.09))],
  "sample":  [("head", (0, 0.140, 0.125)), ("head", (-0.040, 0.125, 0.165)), ("head", (0.040, 0.125, 0.165)),
              ("head", (-0.135, 0.02, 0.145)), ("head", (0.135, 0.02, 0.145))]}
BODY_ON_RIG = [("shoulder_L", (0, 0, 0)), ("shoulder_R", (0, 0, 0)), ("elbow_L", (0, 0, 0)), ("elbow_R", (0, 0, 0)),
               ("wrist_L", (0, 0, 0)), ("wrist_R", (0, 0, 0)), ("hip_L", (-0.03, 0, 0)), ("hip_R", (0.03, 0, 0)),
               ("knee_L", (0, 0, 0)), ("knee_R", (0, 0, 0)), ("ankle_L", (0, 0, 0)), ("ankle_R", (0, 0, 0))]

class Rig:
    def __init__(self, height, style="sample"):
        if style not in COCO_ON_RIG:
            raise ValueError("unknown character style %r (Stage 1 uses 'sample')" % style)
        self.style = style
        self.J, self.s = joint_defs(height, style)
        face = COCO_ON_RIG[style]
        self.coco = [(j, np.array(o) * self.s) for j, o in face + BODY_ON_RIG]
        self.sole = (0.063 if style == "sample" else 0.065) * self.s    # ankle joint height when standing

    def fk(self, loc, yaw, rot, pelvis_z):
        """world (R, t) of every joint.  rot: {joint: (rx,ry,rz) deg}"""
        W = {"root": (Rxyz((0, 0, yaw)), np.asarray(loc, float))}
        for name, parent, off in self.J:
            Rp, tp = W[parent]
            if name == "pelvis":
                off = np.array([0, 0, pelvis_z])
            W[name] = (Rp @ Rxyz(rot.get(name, (0, 0, 0))), tp + Rp @ off)
        return W

    def coco_points(self, W):
        return np.array([W[j][1] + W[j][0] @ o for j, o in self.coco])

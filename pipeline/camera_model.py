# -*- coding: utf-8 -*-
"""The Blender camera of scene_data.json as a pinhole camera: one definition of the camera key -> rotation, position,
used by the quality numbers, the colour card and the comparison tool (compare.py).

  cam key  [x, y, z, ex, ey, ez]   position (m) and XYZ Euler angles (degrees), Blender convention (looks along -z)
"""
import math
import numpy as np


def cam_mats(c):
    x, y, z, ex, ey, ez = c
    a, b, g = [math.radians(v) for v in (ex, ey, ez)]
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx, np.array([x, y, z], float)


class Cam:
    def __init__(self, data):
        self.d = data; self.W, self.H = data["res"]; self.f = data["lens"] / data["sensor"] * self.W
        self.zoom = data.get("zoom") or None                 # focal scale per frame when the lens zooms
    def f_at(self, fr):
        """focal length (px) at frame fr"""
        return self.f * float(self.zoom[min(int(fr), len(self.zoom) - 1)]) if self.zoom else self.f
    def project(self, P, fr):
        R, C = cam_mats(self.d["cam"][fr])
        Xc = (np.atleast_2d(P) - C) @ R
        z = -Xc[:, 2]
        zz = np.maximum(z, 1e-6); f = self.f_at(fr)
        return np.c_[self.W / 2 + f * Xc[:, 0] / zz, self.H / 2 - f * Xc[:, 1] / zz], z
    def ray(self, u, v, fr):
        R, C = cam_mats(self.d["cam"][fr]); f = self.f_at(fr)
        d = R @ np.array([(u - self.W / 2) / f, -(v - self.H / 2) / f, -1.0])
        return C, d / np.linalg.norm(d)

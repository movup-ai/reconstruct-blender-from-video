# -*- coding: utf-8 -*-
"""ground plane + camera height from the people (their feet stand on the ground, their heads are ~height above).
builds the World: z up, ground z=0, camera at (0,0,h) looking along +Y at frame 0."""
import numpy as np
from scipy.spatial.transform import Rotation
from common import *

class World:
    def __init__(self, cam, Rlist, up_c, hc, zoom=None, yaw=0.):
        self.cam = cam
        self.Ri = [np.asarray(r) for r in Rlist]          # frame-0 cam -> frame-i cam
        self.zoom = None if zoom is None or np.allclose(zoom, 1.0) else np.asarray(zoom, float)   # focal f_i = zoom[i] * cam.f
        Z = up_c / np.linalg.norm(up_c)
        Y = np.array([0, 0, 1.0]) - Z * Z[2]
        if np.linalg.norm(Y) < 1e-6:
            Y = np.array([0., 1., 0.]) - Z * Z[1]
        Y /= np.linalg.norm(Y)
        X = np.cross(Y, Z)
        self.M = np.c_[X, Y, Z]                            # world -> cam0 (columns = world axes in cam0 coords)
        self.M = self.M @ Rotation.from_euler("z", yaw).as_matrix()
        self.C = np.array([0, 0, hc])
        self.hc, self.up_c = hc, Z
        self.T = None                                      # per-frame camera travel (world metres), set by set_path()
    def C_at(self, i):
        return self.C if self.T is None else self.C + self.T[min(i, len(self.T) - 1)]
    def set_path(self, keys, N):
        """camera move set by hand: keys [[frame, yaw, pitch, roll, right, forward, up], ...] relative to frame 0
        (degrees: yaw + = turn left, pitch + = tilt up, roll + = clockwise; metres along the frame-0 axes).
        Replaces the tracked rotation and adds the travel a rotation-only track cannot see."""
        keys = sorted(keys)
        fr = [k[0] for k in keys]
        V = np.array([[np.interp(f, fr, [k[j] for k in keys]) for j in range(1, 7)] for f in range(N)])
        Ri, T = [], []
        for yaw, pitch, roll, dx, dy, dz in V:
            Yw = Rotation.from_euler("z", -yaw, degrees=True).as_matrix()          # the world turned the other way
            P = Rotation.from_euler("xz", [-pitch, roll], degrees=True).as_matrix()  # then tilt / roll in camera axes
            Ri.append(P @ self.M @ Yw @ self.M.T)
            T.append([dx, dy, dz])
        self.Ri = Ri; self.T = np.array(T, float)
    def R_cw(self, i):                                     # world -> cam i
        return self.Ri[min(i, len(self.Ri) - 1)] @ self.M
    def f_at(self, i):                                     # focal length (px) at frame i: the lens zooms
        return self.cam.f if self.zoom is None else self.cam.f * float(self.zoom[min(i, len(self.zoom) - 1)])
    def project(self, X, i):
        Xc = (np.atleast_2d(X) - self.C_at(i)) @ self.R_cw(i).T
        z = np.maximum(Xc[:, 2], 1e-6); f = self.f_at(i)
        return np.c_[Xc[:, 0] / z * f + self.cam.W / 2, Xc[:, 1] / z * f + self.cam.H / 2], Xc[:, 2]
    def norm(self, i, x, y):                               # pixel -> normalised camera coords at frame i
        f = self.f_at(i)
        return np.array([(x - self.cam.W / 2) / f, (y - self.cam.H / 2) / f])
    def ray(self, i, x, y):
        d = self.R_cw(i).T @ np.r_[self.norm(i, x, y), 1.0]
        return d / np.linalg.norm(d)
    def on_ground(self, i, x, y, z=0.0):
        d = self.ray(i, x, y)
        if d[2] >= -1e-4:
            return None
        C = self.C_at(i)
        return C + d * (z - C[2]) / d[2]
    def blender_cam(self, i):
        Rwb = self.R_cw(i).T @ np.diag([1, -1, -1])         # blender camera looks down its -Z, +Y up
        e = Rotation.from_matrix(Rwb).as_euler("xyz", degrees=True)
        C = self.C_at(i)
        return [float(C[0]), float(C[1]), float(C[2])] + [float(a) for a in e]

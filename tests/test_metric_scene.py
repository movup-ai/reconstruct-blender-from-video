"""Geometric regressions: metric observation maths, the camera model and the body raster."""
import copy
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline")]
import metric_scene as MS
from sam3d_body import rasterize
from camera_model import Cam
from common import Camera
from ground_solve import World
from rig import Rig, JOINTS


def cube(low=(-1, -1, -1), high=(1, 1, 1)):
    x0, y0, z0 = low; x1, y1, z1 = high
    return {"name": "cube", "v": [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
            "f": [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]], "colour": "set"}


def scene():
    world = World(Camera(80, 60, 18), [np.eye(3)], np.array([0., -1., 0.]), 1.6)
    data = {"res": [80,60], "lens":18., "sensor":36., "cam":[world.blender_cam(0)],
            "frames":1, "chars":{}, "meshes":[], "zoom":[1.]}
    return data, world


def evidence(world):
    K = world.cam.K()
    return {"frames":np.array([0]), "depth":np.full((1,60,80), 5., np.float32), "K":K[None],
            "R":world.R_cw(0)[None], "C":world.C_at(0)[None], "person":np.zeros((1,60,80),bool),
            "valid":np.ones((1,60,80),bool), "labels":np.zeros((1,60,80),np.int16), "names":np.array(["wall"])}


class ObservationTests(unittest.TestCase):
    def test_scale_rejects_outliers_and_unreliable_samples(self):
        self.assertAlmostEqual(MS.robust_scale([2, 2.01, 1.99, 2, 40, np.nan]), 2., places=2)
        self.assertIsNone(MS.robust_scale([2, np.inf]))

    def test_backprojection_roundtrip_with_noncentral_principal_point(self):
        K = np.array([[90.,0,12.],[0,80.,9.],[0,0,1.]])
        p = MS.points(np.full((24,32),3.), K)
        uv = p @ K.T; uv = uv[...,:2]/uv[...,2:]
        y,x = np.indices((24,32))
        np.testing.assert_allclose(uv, np.stack((x,y),-1), atol=1e-12)

    def test_environment_reuses_shared_depth_at_a_different_resolution(self):
        K = np.array([[40.,0,40.],[0,40.,30.],[0,0,1.]])
        p = MS.resample_depth(np.full((60,80),5.),K,(120,160))
        np.testing.assert_allclose(p[60,80],[0.,0.,5.])
        np.testing.assert_allclose(p[60,120],[2.5,0.,5.])

    def test_box_mask_uses_picture_crop_and_clips_boundaries(self):
        pose = {"people":[{"boxes":{"0":[-20,30,50,100]}}]}
        mask = MS.person_mask(pose,0,(40,100),20,(200,80))
        self.assertTrue(mask[10,10]); self.assertFalse(mask[10,80])
        self.assertFalse(MS.person_mask(pose,1,(40,100),20,(200,80)).any())

    def test_gravity_uses_floor_ceiling_and_walls_with_outliers(self):
        up = np.array([0.,-1.,0.]); prior = Rotation.from_euler('z',5,degrees=True).apply(up)
        normals = np.vstack([np.tile(up,(100,1)),np.tile(-up,(100,1)),np.tile([1,0,0],(100,1)),np.tile([0,0,1],(10,1))])
        labels = np.r_[np.zeros(100),np.full(100,2),np.ones(100),np.zeros(10)]
        got = MS.solve_up(normals,labels,prior)
        self.assertGreater(got @ up, .999)
        np.testing.assert_allclose(MS.solve_up([],[],prior),prior)

    def test_world_yaw_is_a_projection_preserving_change_of_basis(self):
        cam = Camera(80,60,18); up = np.array([0.,-1.,0.])
        a = World(cam,[np.eye(3)],up,1.6)
        b = World(cam,[np.eye(3)],up,1.6,yaw=.4)
        point = np.array([[1.,5.,.8]])
        reexpressed = (point-a.C) @ a.M.T @ b.M+b.C
        np.testing.assert_allclose(a.project(point,0)[0],b.project(reexpressed,0)[0])

    def test_observation_has_no_capture_or_set_proposals(self):
        """Stage 1 keeps metric_observation*.npz (the saved depth); the old evidence/set-proposal writers are gone"""
        self.assertFalse(hasattr(MS, 'capture')); self.assertFalse(hasattr(MS, 'plane_proposals'))


class RasterTests(unittest.TestCase):
    def test_raster_depth_and_near_plane_clipping(self):
        K = np.array([[20.,0,20],[0,20,20],[0,0,1]])
        verts = [[-2,-2,4],[2,-2,4],[2,2,4],[-2,2,4]]
        z = rasterize(verts,[[0,1,2,3]],K,np.eye(3),np.zeros(3),(40,40))
        self.assertAlmostEqual(z[20,20],4.)
        crossing = [[-1,-1,-1],[2,-1,3],[0,2,3]]
        z = rasterize(crossing,[[0,1,2]],K,np.eye(3),np.zeros(3),(40,40))
        self.assertGreater(np.isfinite(z).sum(),0)
        self.assertTrue((z[np.isfinite(z)]>=.01).all())


class CameraModelTests(unittest.TestCase):
    def test_blender_camera_projects_like_the_fitted_world(self):
        """camera_model.Cam (scene_data.json camera keys) and ground_solve.World agree on every projection"""
        cam = Camera(80, 60, 18); up = Rotation.from_euler('x', 8, degrees=True).apply([0., -1., 0.])
        world = World(cam, [np.eye(3), Rotation.from_euler('y', 3, degrees=True).as_matrix()], up, 1.6, zoom=[1., 1.2])
        data = {"res": [80, 60], "lens": 18., "sensor": 36., "frames": 2, "cam": [world.blender_cam(i) for i in range(2)],
                "zoom": [world.f_at(i) / cam.f for i in range(2)]}
        points = np.array([[.5, 5., .8], [-1., 7., 0.], [0., 4., 1.7]])
        for f in range(2):
            np.testing.assert_allclose(Cam(data).project(points, f)[0], world.project(points, f)[0], atol=1e-6)

    def test_rig_points_project_in_front_of_the_camera(self):
        data, _ = scene(); cam = Cam(data)
        ch = {'height':1.75,'style':'sample','root':[[.5,5.,0.,20.]],'pelvis_z':[.94],'rot':{j:[[0.,0.,0.]] for j in JOINTS}}
        rig = Rig(ch['height'], ch['style']); root = ch['root'][0]
        P = rig.coco_points(rig.fk(root[:3], root[3], {j: ch['rot'][j][0] for j in JOINTS}, ch['pelvis_z'][0]))
        uv, z = cam.project(P, 0)
        self.assertTrue((z > 0).all()); self.assertEqual(uv.shape, (17, 2))

if __name__ == '__main__': unittest.main()

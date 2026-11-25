"""Geometric regressions: independent projection, scale, raster and repair fixtures."""
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
import scene_verify as SV
from build_env import Cam
from camera_solve import Camera
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

    def test_observation_capture_preserves_shot_offset(self):
        data, world = scene(); bundle = evidence(world)
        obs = dict(bundle, scale=2., groups=np.zeros_like(bundle['labels']))
        with tempfile.TemporaryDirectory() as tmp:
            path = MS.capture(obs,world,tmp,'_s2',start=100,dx=200)
            with np.load(path) as saved:
                self.assertEqual(saved['frames'][0],100)
                self.assertEqual(saved['C'][0,0],200)
                self.assertEqual(saved['depth'][0,0,0],10)


class VerificationTests(unittest.TestCase):
    def test_depth_detects_foreground_conflict_and_missing_surface(self):
        obs = np.full((10,10),5.); pred = np.full_like(obs,np.inf)
        pred[:5] = 3
        result,_ = SV.compare_depth(obs,pred,np.zeros_like(obs,bool))
        self.assertEqual(result['conflict_fraction'],1.)
        self.assertEqual(result['coverage_fraction'],0.)
        result,_ = SV.compare_depth(obs,obs,np.ones_like(obs,bool))
        self.assertEqual(result['observed_pixels'],0)

    def test_raster_depth_and_near_plane_clipping(self):
        K = np.array([[20.,0,20],[0,20,20],[0,0,1]])
        verts = [[-2,-2,4],[2,-2,4],[2,2,4],[-2,2,4]]
        z = SV.rasterize(verts,[[0,1,2,3]],K,np.eye(3),np.zeros(3),(40,40))
        self.assertAlmostEqual(z[20,20],4.)
        crossing = [[-1,-1,-1],[2,-1,3],[0,2,3]]
        z = SV.rasterize(crossing,[[0,1,2]],K,np.eye(3),np.zeros(3),(40,40))
        self.assertGreater(np.isfinite(z).sum(),0)
        self.assertTrue((z[np.isfinite(z)]>=.01).all())

    def test_penetration_requires_closed_volume(self):
        box = cube()
        self.assertTrue(SV.inside_mesh(np.zeros(3),box['v'],box['f']))
        self.assertFalse(SV.inside_mesh(np.array([2.,0,0]),box['v'],box['f']))
        self.assertFalse(SV.inside_mesh(np.zeros(3),box['v'],box['f'][:-1]))

    def test_support_uses_real_polygon_and_excludes_vertical_wall(self):
        box = cube()
        self.assertAlmostEqual(SV.support_height([0,0],box['v'],box['f'],2),1.)
        self.assertIsNone(SV.support_height([2,0],box['v'],box['f'],2))
        self.assertIsNone(SV.support_height([0,0],box['v'],[[0,1,5,4]],2))

    def test_stale_evidence_is_rejected_after_camera_edit(self):
        data, world = scene()
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(os.path.join(tmp,'metric_evidence.npz'),**evidence(world))
            bundles,skipped = SV.load_evidence(data,tmp)
            self.assertEqual(len(bundles),1)
            data['cam'][0][0] += 1
            bundles,skipped = SV.load_evidence(data,tmp)
            self.assertEqual(len(bundles),0)
            self.assertEqual(len(skipped),1)

    def test_changed_metric_evidence_invalidates_environment_cache(self):
        from run_pipeline import same_set_inputs
        a = {'frames':1,'cam':[[0,0,1,90,0,0]],'lens':35.,'metric_evidence':'before'}
        b = dict(a,metric_evidence='after')
        self.assertFalse(same_set_inputs(a,b))

    def test_missing_evidence_does_not_claim_depth_checks_passed(self):
        data,_ = scene()
        with tempfile.TemporaryDirectory() as tmp:
            result = SV.process(data,tmp,{},repair=True)
        self.assertFalse(result['depth_checked'])
        self.assertEqual(result['repairs'],[])

    def test_evidence_query_returns_world_depth_and_rejects_outside_pixel(self):
        from scene_evidence import query
        data,world = scene()
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(os.path.join(tmp,'metric_evidence.npz'),**evidence(world))
            result = query(data,tmp,frame=4,pixel=[40,30])
            self.assertEqual(result['frame'],0)
            self.assertEqual(result['requested_frame'],4)
            np.testing.assert_allclose(result['pixel']['world'],[0.,5.,1.6])
            with self.assertRaises(ValueError): query(data,tmp,pixel=[1000,30])

    def test_repair_adds_missing_measured_surface_and_improves_coverage(self):
        data, world = scene()
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(os.path.join(tmp,'metric_evidence.npz'),**evidence(world))
            before = SV.verify(data,tmp)
            after = SV.process(data,tmp,{'validation_mode':'audit','scene_repair_limit':2},repair=True)
            self.assertTrue(after['repairs'])
            self.assertGreater(after['evidence'][0]['coverage_fraction'],before['evidence'][0]['coverage_fraction'])
            self.assertEqual(after['evidence'][0]['conflict_fraction'],0.)
            repaired = [m for m in data['meshes'] if '_evidence' in m]
            self.assertTrue(repaired)
            self.assertTrue(all(m['layer'] in (0, 1, 2) for m in repaired),
                            'repaired geometry must satisfy the Blender scene contract')

    def test_occlusion_repair_moves_a_whole_primitive_toward_measured_depth(self):
        data,world = scene()
        first = cube((-2,4.,-.5),(0,4.1,3.)); first['_primitive']='test_wall'
        second = cube((0,4.,-.5),(2,4.1,3.)); second['name']='cube_right'; second['_primitive']='test_wall'
        data['meshes']=[first,second]
        before_v = np.asarray([m['v'] for m in data['meshes']])
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(os.path.join(tmp,'metric_evidence.npz'),**evidence(world))
            result = SV.process(data,tmp,{'validation_mode':'audit','scene_repair_limit':1},repair=True)
        moved = [r for r in result['repairs'] if r['action']=='depth-aligned primitive translation']
        self.assertEqual(len(moved),1)
        after_v = np.asarray([m['v'] for m in data['meshes'][:2]])
        np.testing.assert_allclose(after_v-before_v,np.broadcast_to([0,1,0],before_v.shape),atol=1e-5)

    def test_global_structure_flags_isolated_geometry(self):
        data,_ = scene()
        floor = cube((-10,0,-.1),(10,20,0)); floor['name']='ground'
        floating = cube((2,3,2),(3,4,3)); floating['name']='floating'
        data['meshes']=[floor,floating]
        with tempfile.TemporaryDirectory() as tmp:
            result = SV.verify(data,tmp)
        self.assertTrue(any(i['kind']=='isolated_geometry' and i['object']=='floating' for i in result['issues']))

    def test_justifications_preserve_issues_without_marking_them_resolved(self):
        data,world = scene()
        with tempfile.TemporaryDirectory() as tmp:
            initial = SV.verify(data,tmp,bundles=[evidence(world)])
            key = initial['issues'][0]['id']
            result = SV.verify(data,tmp,{key:'Reflective surface excluded after review'},bundles=[evidence(world)])
            self.assertEqual(result['issues'][0]['status'],'justified')
            self.assertNotIn(key,result['unresolved'])


class HumanScaleTests(unittest.TestCase):
    def character(self):
        return {'height':1.75,'style':'sample','root':[[.5,5.,0.,20.]],'visible':[True],
                'pelvis_z':[.94],'rot':{j:[[0.,0.,0.]] for j in JOINTS},'track_id':1}

    def test_track_scale_preserves_all_joint_projections(self):
        data,_ = scene(); ch = self.character(); cam = Cam(data)
        rig = Rig(ch['height'],ch['style']); root = ch['root'][0]
        before = rig.coco_points(rig.fk(root[:3],root[3],{j:ch['rot'][j][0] for j in JOINTS},ch['pelvis_z'][0]))
        SV.scale_track(ch,data['cam'],1.2)
        rig = Rig(ch['height'],ch['style']); root = ch['root'][0]
        after = rig.coco_points(rig.fk(root[:3],root[3],{j:ch['rot'][j][0] for j in JOINTS},ch['pelvis_z'][0]))
        np.testing.assert_allclose(cam.project(before,0)[0],cam.project(after,0)[0],atol=1e-10)

    def test_explicit_scale_is_idempotent_and_invalid_height_rejected(self):
        data,_ = scene(); data['chars'] = {'main':self.character()}
        hints = {'people':{'1':{'metric_scale':1.1}}}
        SV.place_people(data,hints); once = copy.deepcopy(data)
        self.assertEqual(SV.place_people(data,hints),[])
        self.assertEqual(data,once)
        with self.assertRaises(ValueError): SV.place_people(data,{'people':{'1':{'metric_scale':10}}})

    def test_support_repair_uses_one_scale_for_the_whole_track(self):
        data,_ = scene(); ch = self.character()
        ch['root'][0][2] = .12
        ch['root'] = [list(ch['root'][0]) for _ in range(3)]
        ch['visible'] *= 3; ch['pelvis_z'] *= 3
        ch['rot'] = {j: values*3 for j,values in ch['rot'].items()}
        data['cam'] *= 3; data['frames'] = 3; data['chars'] = {'main':ch}
        data['meshes'] = [cube((-10,0,-.1),(10,20,0))]
        with tempfile.TemporaryDirectory() as tmp:
            before = SV.verify(data,tmp)
            after = SV.process(data,tmp,{'validation_mode':'audit'},repair=True)
        self.assertTrue(any(i['kind']=='foot_support' for i in before['issues']))
        self.assertTrue(any(r['action']=='constant support scale' for r in after['repairs']))
        self.assertFalse(any(i['kind']=='foot_support' for i in after['issues']))

    def test_scaling_keeps_carried_props_attached(self):
        data,_ = scene(); ch = self.character(); data['chars'] = {'main':ch}
        data['props'] = [{'track_id':1,'keys':[[.5,5,1.,0,0,0,1,1,1]]}]
        SV.place_people(data,{'people':{'1':{'metric_scale':1.1}}})
        key = data['props'][0]['keys'][0]
        np.testing.assert_allclose(key[:3],[.55,5.5,.94])
        np.testing.assert_allclose(key[6:9],[1.1]*3)


if __name__ == '__main__': unittest.main()

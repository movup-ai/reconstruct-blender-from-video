"""Geometry contracts for metric body observations, mesh scale and placement."""
import os
import sys
import tempfile
import unittest
import types
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT,os.path.join(ROOT,'pipeline')]
import sam3d_body as BODY
import scene_verify
from camera_solve import Camera
from ground_solve import World
from fit_people import p3_target,_P3A,_P3B


class MeshSurfaceTests(unittest.TestCase):
    def test_vectorized_surface_matches_reference_perspective_zbuffer(self):
        K=np.array([[30.,0,20],[0,30.,20],[0,0,1]])
        vertices=np.array([[-1,-1,3],[1,-1,5],[1,1,5],[-1,1,3]])
        faces=[[0,1,2],[0,2,3]]
        a=BODY.surface_depth(vertices,faces,K,(40,40))
        b=scene_verify.rasterize(vertices,faces,K,np.eye(3),np.zeros(3),(40,40))
        np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b))
        np.testing.assert_allclose(a[np.isfinite(a)],b[np.isfinite(b)],atol=1e-12)

    def test_visible_surface_selects_near_mesh_and_ignores_offscreen_faces(self):
        K=np.array([[20.,0,20],[0,20.,20],[0,0,1]])
        vertices=np.array([[-2,-2,4],[2,-2,4],[0,2,4],[-2,-2,6],[2,-2,6],[0,2,6],
                           [100,100,2],[101,100,2],[100,101,2]])
        z=BODY.surface_depth(vertices,[[3,4,5],[0,1,2],[6,7,8]],K,(40,40))
        self.assertAlmostEqual(z[20,20],4.)

    def test_surface_scale_uses_only_visible_mask_and_finite_depth(self):
        body=np.full((20,20),4.); measured=np.full_like(body,2.)
        mask=np.zeros_like(body,bool); mask[:10]=True
        measured[10:]=.01; measured[0,0]=np.inf
        self.assertAlmostEqual(BODY.surface_ratio(body,measured,mask,np.ones_like(mask)),2.)
        self.assertIsNone(BODY.surface_ratio(body,measured,np.zeros_like(mask),np.ones_like(mask)))


class PlacementTests(unittest.TestCase):
    def test_neck_placement_changes_translation_but_body_scale_is_constant(self):
        neck=np.array([[0.,0.,4.],[0.,0.,5.]])
        relative=np.array([[[-.2,0,0],[.2,0,0]], [[-.2,0,0],[.2,0,0]]])
        factors=np.array([.8,1.2]); joints=relative+neck[:,None]
        placed=BODY.place(joints,neck,factors)
        np.testing.assert_allclose(placed.mean(1),neck*factors[:,None])
        np.testing.assert_allclose(np.linalg.norm(placed[:,1]-placed[:,0],axis=1),[.4,.4])

    def test_invalid_placement_factor_is_rejected(self):
        for factors in ([0.],[np.nan],[-1.]):
            with self.assertRaises(ValueError): BODY.place(np.zeros((1,17,3)),np.zeros((1,3)),factors)

    def test_alignment_uses_global_instance_scale_and_interpolates_neck_factors(self):
        joints=np.zeros((3,17,3)); joints[:,:,2]=4.
        body={'surface_ratios':np.array([2.,2.,2.,40.]),'p1_f':np.array([0,1,2]),'p1_j':joints,
              'p1_n':np.tile([0,0,4.],(3,1)),'p1_neck_depth':np.array([1.6,np.nan,2.4])}
        report=BODY.align_depth(body)
        self.assertTrue(report['calibrated']); self.assertAlmostEqual(report['meters_per_unit'],2.)
        np.testing.assert_allclose(body['p1_factors'],[.8,1.,1.2])

    def test_unavailable_depth_retains_metric_body_camera_coordinates(self):
        joints=np.arange(51).reshape(1,17,3)/100+[0,0,4.]
        body={'surface_ratios':np.array([]),'p1_f':np.array([0]),'p1_j':joints,'p1_n':np.array([[0,0,4.]]),
              'p1_neck_depth':np.array([np.nan])}
        report=BODY.align_depth(body)
        self.assertFalse(report['calibrated'])
        np.testing.assert_allclose(body['p1_placed'],joints)

    def test_explicit_metric_scale_has_priority(self):
        report=BODY.align_depth({'surface_ratios':np.array([2.,2.,2.])},explicit_scale=3.)
        self.assertEqual(report['meters_per_unit'],3.)
        with self.assertRaises(ValueError): BODY.align_depth({},explicit_scale=-1.)

    def test_h36m_mapping_preserves_left_right_hips_and_wrists(self):
        coco=np.tile(np.arange(17)[None,:,None],(1,1,3))
        h=BODY.h36m(coco)
        np.testing.assert_allclose(h[0,0],11.5)
        np.testing.assert_allclose(h[0,13],9.)
        np.testing.assert_allclose(h[0,16],10.)
        np.testing.assert_allclose(h[0,1],12.)

    def test_sampler_respects_visibility_and_includes_depth_reference_frames(self):
        person={'boxes':{str(f):[0,0,1,1] for f in (0,1,2,8,9,10)}}
        self.assertEqual(BODY.sample_frames(person,3,evidence=[1,4,9]),[0,1,8,9,10])

    def test_camera_space_prior_is_not_rotated_onto_image_ray_twice(self):
        world=World(Camera(320,180,35),[np.eye(3)],np.array([0.,-1.,0.]),1.6)
        points=np.arange(51,dtype=float).reshape(1,17,3)/10
        confidence=np.ones((1,17)); k=np.tile([290,90,.9],(17,1))
        target,_=p3_target((np.array([0]),points,confidence,'camera'),0,k,world)
        delta=points[0][_P3B]-points[0][_P3A]
        delta/=np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-9)
        np.testing.assert_allclose(target,delta)
        # Off-axis image joints must not rotate native camera-space body directions.
        centered,_=p3_target((np.array([0]),points,confidence),0,k,world)
        np.testing.assert_allclose(centered,target)

    def test_missing_body_assets_stop_before_model_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(BODY.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'required file missing'):
                BODY.recover([],{'people':[]},Camera(320,180,35),0,{}, {},{'sam3d_checkpoint':os.path.join(tmp,'missing.ckpt')})
            run.assert_not_called()

    def test_review_uses_inferred_body_height_and_keeps_short_sam_track(self):
        from agent import review
        pose={'tracker_backend':'sam3','people':[{'id':1,'frames':{str(f):[] for f in range(8)}}]}
        scene={'chars':{'P1':{'track_id':1,'height':1.55,'human_backend':'sam3d_body'}}}
        def read(path,default=None):
            return pose if path=='pose' else scene if path.endswith('scene_data.json') else {}
        folders={'pose':'pose','ai':'ai','work':'work'}
        c={'person_height':1.72,'lens_mm':35.}
        with patch.object(review,'folders',return_value=folders),patch.object(review,'read_json',side_effect=read):
            result=review.suggested(c,types.SimpleNamespace(name='clip',video='video'))
        self.assertAlmostEqual(result['people']['1']['height'],1.55)
        self.assertEqual(result['people']['1']['role'],'main')


if __name__=='__main__': unittest.main()

"""Production-standard regressions for geometry, timing, color and delivery checks."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'pipeline'), str(ROOT/'tests')]
from rig import Rig, style_asset
import source_exposure
import quality
import colour_card
from common import normalize_backends, DEFAULT_CFG
from test_body_constraints import fixture


class L3StandardTests(unittest.TestCase):
    def test_runtime_descriptor_preserves_ten_parts_and_seventeen_mark_faces(self):
        asset = style_asset('l3')
        self.assertEqual(len(asset['parts']), 10)
        self.assertEqual(sum(len(p['v']) for p in asset['parts'].values()), 578)
        self.assertEqual(sum(len(p['f']) for p in asset['parts'].values()), 590)
        self.assertEqual(sum(len(p['marker_faces']) for p in asset['parts'].values()), 17)
        for bone in asset['bones'].values():
            rotation = np.array(bone['R'])
            np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=2e-5)
            self.assertGreater(np.linalg.det(rotation), .999)

    def test_scale_is_uniform_in_numerical_rig(self):
        a,b = Rig(1.75,'l3'),Rig(1.4,'l3')
        for (_,_,pa),(_,_,pb) in zip(a.J,b.J):
            np.testing.assert_allclose(pb,.8*pa)
        np.testing.assert_allclose([o for j,o in b.coco],.8*np.array([o for j,o in a.coco]))

    def test_default_migration_preserves_explicit_character_choice(self):
        self.assertEqual(normalize_backends({'character_style':'master'})['character_style'],'master')
        self.assertEqual(normalize_backends({'character_style':'master','previs_standard':'legacy'})['character_style'],'master')

    def test_old_asset_fit_is_not_adopted_under_new_l3_geometry(self):
        import tempfile
        import dependencies
        from common import jsave
        from body_constraints import VERSION
        with tempfile.TemporaryDirectory() as tmp:
            jsave(str(Path(tmp)/'scene_data.json'),{'human_backend':'sam3d_body',
                'chars':{'P1':{'style':'master','body_constraints':{'version':VERSION}}}})
            jsave(str(Path(tmp)/'pose.json'),{'tracker_backend':'sam3'})
            dependencies.adopt_existing('video',DEFAULT_CFG,tmp,tmp,tmp)
            self.assertFalse((Path(tmp)/'build_state.json').exists())

    def test_background_source_color_and_semantic_palette_are_idempotent(self):
        data={'frames':0,'previs_standard':'l3','background_policy':'source','chars':{},
              'palette':{'wall_sample':[21,52,89]},
              'meshes':[{'name':'wall','colour':'wall_sample','layer':0,'v':[[0,0,0]],'f':[]}]}
        colour_card.apply(data, log=None)
        first=copy.deepcopy(data)
        colour_card.apply(data, log=None)
        self.assertEqual(data,first)
        self.assertEqual(data['palette'][data['meshes'][0]['colour']],[21,52,89])
        self.assertEqual(data['palette']['red'],[194,70,58])

    def test_explicit_gray_background_mode_remains_available(self):
        data={'frames':0,'previs_standard':'l3','background_policy':'grey','chars':{},
              'palette':{},'meshes':[{'name':'wall','colour':'bg','layer':0,'v':[[0,0,0]],'f':[]}]}
        colour_card.apply(data,log=None)
        self.assertEqual(data['meshes'][0]['colour'],'bg')


class ExposureTests(unittest.TestCase):
    def test_lights_out_and_recovery_stay_on_exact_source_frames(self):
        samples=np.full((30,48),.4);samples[10:20]*=.0625
        result=source_exposure.from_tiles(samples,[(0,30)])
        np.testing.assert_allclose(result['stops'][:10],0)
        np.testing.assert_allclose(result['stops'][10:20],-4)
        np.testing.assert_allclose(result['stops'][20:],0)

    def test_cuts_do_not_become_exposure_changes(self):
        samples=np.full((30,48),.4);samples[15:]*=.0625
        result=source_exposure.from_tiles(samples,[(0,15),(15,30)])
        np.testing.assert_allclose(result['stops'],0)

    def test_local_object_brightness_change_does_not_darken_scene(self):
        samples=np.full((30,48),.4);samples[10:20,:8]*=.01
        np.testing.assert_allclose(source_exposure.from_tiles(samples,[(0,30)])['stops'],0)

    def test_no_background_evidence_leaves_exposure_unchanged(self):
        samples=np.full((30,48),np.nan)
        np.testing.assert_allclose(source_exposure.from_tiles(samples,[(0,30)])['stops'],0)


class ShotCoverageTests(unittest.TestCase):
    def test_actor_is_not_missing_outside_its_own_shot(self):
        _,_,_,data,pose=fixture(8)
        other=copy.deepcopy(data['chars']['P1'])
        data['chars']['P1']['source_range']=[0,3]
        data['chars']['P1']['visible']=[True]*4+[False]*4
        other['source_range']=[4,7];other['visible']=[False]*4+[True]*4
        data['chars']['s2_P1']=other
        result=quality.alignment_audit(data,pose)
        self.assertTrue(result['passed'])
        self.assertEqual([r['checked_frames'] for r in result['people']],[4,4])

    def test_empty_reconstruction_cannot_pass_with_visible_required_actor(self):
        _,_,_,data,pose=fixture(8);data['chars']={}
        result=quality.alignment_audit(data,pose)
        self.assertFalse(result['passed'])
        with self.assertRaises(RuntimeError):quality.require_alignment(result)

    def test_missing_selected_actor_is_reported_even_when_other_actor_exists(self):
        _,_,_,data,pose=fixture(8);data['required_track_ids']=[1,2]
        result=quality.alignment_audit(data,pose)
        self.assertFalse(result['passed'])

    def test_actor_height_is_fixed_across_shots(self):
        # Reuse the fitting fixture's mocked numeric solver to exercise real height selection.
        from unittest.mock import patch
        import fit_people as fitting
        from common import DEFAULT_CFG
        world,keys,xyz,_,_=fixture()
        settings=dict(DEFAULT_CFG,character_style='sample',_sam3d_body={'p1_f':np.arange(8),'p1_placed':xyz})
        q=np.zeros(fitting.NP);q[1]=4;q[3]=1.5
        with patch.object(fitting,'cached_fit_track',return_value=(Rig(1.72,'sample'),{0:q,7:q},{0:0,7:0})):
            first,_=fitting.fit_all(world,0,{1:(list(range(8)),keys)},8,settings)
            settings['_sam3d_body']['p1_placed']=xyz*1.3
            second,_=fitting.fit_all(world,0,{1:(list(range(8)),keys)},8,settings)
        self.assertEqual(first['P1']['height'],second['P1']['height'])


class TemporalRefinementTests(unittest.TestCase):
    def observations(self, n=9):
        import fit_people as fitting
        world,_,_,_,_=fixture(n)
        rig=Rig(1.72,'l3')
        q=np.zeros((n,fitting.NP));q[:,1]=4.;q[:,2]=180.;q[:,3]=rig.J[0][2][2]
        elbow=next(fitting.NB+i for i,a in enumerate(fitting.ANG) if a[:2]==('elbow_L',0))
        q[:,elbow]=np.linspace(20.,70.,n)
        keys=[]
        for f,row in enumerate(q):
            loc,yaw,rot,pz=fitting.unpack(row,rig)
            uv,_=world.project(rig.coco_points(rig.fk(loc,yaw,rot,pz)),f)
            keys.append(np.c_[uv,np.full(17,.95)])
        return world,rig,q,np.asarray(keys),elbow

    def test_observed_gesture_refined_without_changing_root_or_bone_lengths(self):
        from temporal_refine import arms
        world,rig,q,keys,elbow=self.observations()
        bad=q.copy();bad[:,elbow]+=15.
        result,report=arms(rig,bad,list(range(len(q))),keys,world,0)
        self.assertTrue(report['applied'])
        self.assertLess(report['after_px'],report['before_px']*.8)
        np.testing.assert_array_equal(result[:,:4],bad[:,:4])
        self.assertGreater(np.ptp(result[:,elbow]),35.)

    def test_unobserved_arms_keep_existing_depth_prior(self):
        from temporal_refine import arms
        world,rig,q,keys,_=self.observations();keys[:,7:11,2]=0
        result,report=arms(rig,q,list(range(len(q))),keys,world,0)
        self.assertFalse(report['applied']);np.testing.assert_array_equal(result,q)

    def test_long_shot_windows_preserve_unobserved_prior_without_seams(self):
        from temporal_refine import arms
        world,rig,q,keys,_=self.observations(400);keys[:,7:11,2]=0
        result,report=arms(rig,q,list(range(len(q))),keys,world,0)
        self.assertFalse(report['applied'])
        self.assertGreater(len(report['windows']),1)
        np.testing.assert_allclose(result,q,atol=1e-12)


class FrameReviewTests(unittest.TestCase):
    def test_portable_review_has_every_frame_and_keeps_letterbox_aspect(self):
        import tempfile
        from unittest.mock import patch
        import cv2
        import frame_review
        class Capture:
            def read(self):return True,np.full((24,32,3),80,np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);renders=root/'render';renders.mkdir()
            for f in (1,2):cv2.imwrite(str(renders/('wm_%04d.png'%f)),np.full((16,32,3),240,np.uint8))
            info={'frames':2,'W':32,'Hfull':24,'Hpic':16,'y0':4,'fps':29.97}
            with patch.object(frame_review,'open_video',return_value=(Capture(),None)),patch.object(frame_review,'close_video'):
                result=frame_review.build('source',renders,root/'review',info)
            self.assertEqual(result['frames'],2)
            self.assertEqual(result['fps'],29.97)
            self.assertEqual(len(list((root/'review/model').glob('*.jpg'))),2)
            image=cv2.imread(str(root/'review/model/000000.jpg'))
            self.assertEqual(image.shape,(24,32,3))
            self.assertLess(image[0].mean(),5)
            text=(root/'review/index.html').read_text()
            self.assertNotIn('__COUNT__',text);self.assertNotIn('__FPS__',text)

    def test_missing_render_cannot_silently_create_partial_review(self):
        import tempfile
        from unittest.mock import patch
        import frame_review
        class Capture:
            def read(self):return True,np.zeros((24,32,3),np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(frame_review,'open_video',return_value=(Capture(),None)),patch.object(frame_review,'close_video'):
                with self.assertRaisesRegex(ValueError,'Render review frame'):
                    frame_review.build('source',Path(tmp),Path(tmp)/'review',{'frames':1,'W':32,'Hfull':24,'Hpic':16,'y0':4,'fps':30})


if __name__=='__main__':
    unittest.main()

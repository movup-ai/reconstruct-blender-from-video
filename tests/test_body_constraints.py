"""Regressions for partial-body placement, uncertain arms and delivery alignment."""
import copy
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT)]
import body_constraints as BC
import fit_people as FP
import quality
import dependencies
from common import DEFAULT_CFG,jsave
from camera_solve import Camera
from ground_solve import World
from rig import Rig,JOINTS


def fixture(n=8):
    world=World(Camera(640,360,35),[np.eye(3)]*n,np.array([0.,-1.,0.]),1.5)
    rig=Rig(1.72,'sample'); root=[0.,4.,0.,0.]; rot={j:[0.,0.,0.] for j in JOINTS}
    xyz=rig.coco_points(rig.fk(root[:3],0,rot,1.5))
    uv,z=world.project(xyz,0)
    keys=np.tile(np.c_[uv,np.full(17,.9)][None],(n,1,1))
    camera_xyz=(xyz-world.C_at(0))@world.R_cw(0).T
    ch={'height':1.72,'style':'sample','track_id':1,'root':[root.copy() for _ in range(n)],
        'pelvis_z':[1.5]*n,'rot':{j:[rot[j].copy() for _ in range(n)] for j in JOINTS},'visible':[True]*n}
    data={'res':[640,360],'lens':35.,'sensor':36.,'frames':n,'fps':30,
          'cam':[world.blender_cam(f) for f in range(n)],'chars':{'P1':ch},'meshes':[]}
    pose={'people':[{'id':1,'frames':{str(f):k.tolist() for f,k in enumerate(keys)}}]}
    return world,keys,np.tile(camera_xyz[None],(n,1,1)),data,pose


class ObservationTests(unittest.TestCase):
    def test_masked_ankle_guesses_cannot_establish_full_body_contact(self):
        _,keys,_,_,_=fixture()
        self.assertEqual(BC.full_body_support(keys),1.)
        keys[:,15:17,2]=.1
        self.assertEqual(BC.full_body_support(keys),0.)

    def test_unobserved_and_inconsistent_joints_get_less_influence(self):
        world,keys,xyz,_,_=fixture()
        keys[:,9,2]=.02; xyz[:,10,0]+=2
        _,_,confidence,info=BC.prepare(world,0,np.arange(8),keys,np.arange(8),xyz)
        self.assertLess(confidence[:,9].max(),.1)
        self.assertLess(confidence[:,10].max(),.1)
        self.assertGreater(confidence[:,5].min(),.7)

    def test_missing_frames_do_not_receive_interpolated_observation_confidence(self):
        world,keys,xyz,_,_=fixture()
        _,_,confidence,_=BC.prepare(world,0,[0,7],keys[[0,7]],[0,7],xyz[[0,7]])
        self.assertLess(confidence[3].max(),.1)

    def test_low_confidence_model_spike_is_damped(self):
        world,keys,xyz,_,_=fixture(21)
        keys[:,9,2]=.01; xyz[10,9,0]+=2
        _,smooth,_,_=BC.prepare(world,0,np.arange(21),keys,np.arange(21),xyz)
        self.assertLess(abs(smooth[10,9,0]-xyz[9,9,0]),.5)

    def test_uncertain_arm_flip_damped_while_observed_gesture_preserved(self):
        q=np.zeros((21,FP.NP)); i=next(i for i,a in enumerate(FP.ANG) if a[:2]==('elbow_L',0))
        q[10,4+i]=100
        low=BC.stabilize_arms(q,np.zeros((21,17)),FP.ANG)
        high=BC.stabilize_arms(q,np.ones((21,17)),FP.ANG)
        self.assertLess(low[10,4+i],30)
        np.testing.assert_array_equal(high,q)

    def test_zero_confidence_has_no_twenty_percent_prior_floor(self):
        world,keys,xyz,_,_=fixture()
        import sam3d_body
        _,weight=FP.p3_target((np.arange(8),sam3d_body.h36m(xyz),np.zeros((8,17)),'camera'),0,keys[0],world)
        np.testing.assert_array_equal(weight,0)

    def test_cropped_wrist_endpoint_is_not_a_visible_hand_observation(self):
        keys=np.array([[100.,718.,.9],[100.,650.,.9]])
        result=BC.image_keys(keys,1280,720)
        self.assertLess(result[0,2],.1);self.assertEqual(result[1,2],.9)
        self.assertEqual(keys[0,2],.9)

    def test_hidden_wrist_does_not_freeze_a_visible_upper_arm(self):
        q=np.zeros((21,FP.NP));j=next(i for i,a in enumerate(FP.ANG) if a[:2]==('shoulder_L',0))
        q[:,4+j]=np.arange(21)*4
        confidence=np.ones((21,17));confidence[:,9]=0
        result=BC.stabilize_arms(q,confidence,FP.ANG)
        np.testing.assert_array_equal(result[:,4+j],q[:,4+j])


class PlacementTests(unittest.TestCase):
    def test_partial_body_can_align_above_the_guessed_floor_pelvis_limit(self):
        world,keys,xyz,data,pose=fixture(4); keys[:,11:,2]=.01
        hints={'legs':'free','_body_unanchored':True,'_body_start':[[0,0.,4.,1.5],[3,0.,4.,1.5]],
               '_body_hip':[[0,0.,4.,1.5],[3,0.,4.,1.5]]}
        rig,q,error=FP.fit_track(world,0,list(range(4)),keys,1.72,step=3,style='sample',hint=hints)
        self.assertLess(np.median(list(error.values())),5)
        self.assertGreater(np.median([v[3] for v in q.values()]),1.3)

    def capture_fit(self, keys, xyz, hints=None, reviewed_height=None):
        n=len(keys); world,_,_,_,_=fixture(n)
        q=np.zeros(FP.NP);q[1]=4.;q[3]=1.5
        config=dict(DEFAULT_CFG,_sam3d_body={'p1_f':np.arange(n),'p1_placed':xyz},
                    _hints={'people':{'1':hints or {}}},_review={'people':{'1':{'height':reviewed_height}}})
        with patch.object(FP,'cached_fit_track',return_value=(Rig(1.72,'sample'),{0:q,n-1:q},{0:0.,n-1:0.})) as fit:
            chars,_=FP.fit_all(world,0,{1:(list(range(n)),keys)},n,config)
        return chars['P1'],fit.call_args.args[7]

    def test_full_body_keeps_contacts_and_partial_body_releases_unknown_ground(self):
        _,keys,xyz,_,_=fixture()
        ch,hint=self.capture_fit(keys,xyz)
        self.assertFalse(hint.get('_body_unanchored',False))
        keys[:,11:,2]=.01
        ch,hint=self.capture_fit(keys,xyz)
        self.assertTrue(hint['_body_unanchored']);self.assertNotIn('distance',hint)

    def test_explicit_seated_and_floor_choices_keep_priority(self):
        _,keys,xyz,_,_=fixture();keys[:,11:,2]=.01
        ch,hint=self.capture_fit(keys,xyz,{'legs':'seated','seat_height':.5})
        self.assertFalse(hint.get('_body_unanchored',False));self.assertEqual(hint['legs'],'seated')
        ch,hint=self.capture_fit(keys,xyz,{'floor_z':1.})
        self.assertFalse(hint.get('_body_unanchored',False));self.assertNotIn('_body_hip',hint)

    def test_bad_partial_body_height_uses_configured_prior_but_review_wins(self):
        _,keys,xyz,_,_=fixture();keys[:,11:,2]=.01;xyz[:,:,0]+=3
        ch,_=self.capture_fit(keys,xyz)
        self.assertEqual(ch['height'],DEFAULT_CFG['person_height'])
        self.assertEqual(ch['body_constraints']['height_source'],'configured prior')
        ch,_=self.capture_fit(keys,xyz,reviewed_height=1.25)
        self.assertEqual(ch['height'],1.25)

    def test_old_floor_constrained_scene_is_not_adopted_as_a_new_fit(self):
        _,_,_,data,pose=fixture(); data['human_backend']='sam3d_body';pose['tracker_backend']='sam3'
        with tempfile.TemporaryDirectory() as tmp:
            jsave(os.path.join(tmp,'scene_data.json'),data);jsave(os.path.join(tmp,'pose.json'),pose)
            dependencies.adopt_existing('video',{},tmp,tmp,tmp)
            self.assertFalse(pathlib.Path(tmp,'build_state.json').exists())


class AlignmentGateTests(unittest.TestCase):
    def test_correct_projection_passes_and_large_displacement_is_rejected(self):
        _,_,_,data,pose=fixture()
        result=quality.alignment_audit(data,pose)
        self.assertTrue(result['passed']);quality.require_alignment(result)
        for r in data['chars']['P1']['root']: r[2]-=1.
        result=quality.alignment_audit(data,pose)
        self.assertFalse(result['passed'])
        with self.assertRaisesRegex(RuntimeError,'Human alignment failed'): quality.require_alignment(result)

    def test_tracked_but_hidden_character_cannot_be_delivered(self):
        _,_,_,data,pose=fixture();data['chars']['P1']['visible']=[False]*8
        result=quality.alignment_audit(data,pose)
        self.assertFalse(result['passed']);self.assertEqual(result['people'][0]['missing_frames'],8)

    def test_render_path_rejects_bad_cached_scene_before_blender(self):
        import run_pipeline as RP
        _,_,_,data,pose=fixture()
        for r in data['chars']['P1']['root']: r[2]-=1.
        with tempfile.TemporaryDirectory() as tmp:
            jsave(os.path.join(tmp,'scene_data.json'),data);jsave(os.path.join(tmp,'pose.json'),pose)
            c=dict(DEFAULT_CFG,submit_root=tmp,scene_verify=False)
            with patch.object(RP,'marks_dir',return_value=tmp),patch.object(RP.pkg,'package_name',return_value='office-5'), \
                 patch.object(RP,'blender_render') as render:
                with self.assertRaisesRegex(RuntimeError,'Human alignment failed'):
                    RP.render_only('clip','video',tmp,c,types.SimpleNamespace(),0)
                render.assert_not_called()

    def test_refit_does_not_restore_old_ground_height_for_unanchored_body(self):
        import run_pipeline as RP
        _,_,_,data,_=fixture()
        previous={'frames':8,'chars':{}}
        previous['chars']=copy.deepcopy(data['chars'])
        for r in previous['chars']['P1']['root']: r[2]=1.
        data['chars']['P1']['body_constraints']={'unanchored':True}
        with tempfile.TemporaryDirectory() as tmp:
            jsave(os.path.join(tmp,'scene_data.json'),previous)
            RP.carry_over(tmp,'clip',{},data['chars'],8,None)
        self.assertTrue(all(r[2]==0 for r in data['chars']['P1']['root']))


if __name__=='__main__': unittest.main()

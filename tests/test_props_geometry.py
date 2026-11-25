"""Regression coverage for prop edits, lifecycle events, prototypes and source-visible people."""
import copy
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'pipeline'), str(ROOT/'tests')]
import dependencies as DEP
import prop_timeline as PT
import geometry_protection as GP
import prototypes
import held_objects as HO
import fit_people as FP
import run_pipeline as RP
import sam3_scene as SAM
from common import DEFAULT_CFG, jload, jsave
from rig import JOINTS, Rig
from test_metric_scene import scene, evidence, cube
from agent import review


def fixture(n=12):
    data, world = scene()
    data.update(frames=n, fps=30, cam=data['cam']*n, zoom=[1.]*n, props=[], human_backend='sam3d_body')
    def char(track, x):
        return {'height':1.72,'style':'sample','track_id':track,
                'root':[[x+f*.03,4.,0.,0.] for f in range(n)],'visible':[True]*n,
                'pelvis_z':[.94]*n, 'rot':{j:[[0.,0.,0.] for f in range(n)] for j in JOINTS}}
    data['chars']={'main':char(1,0), 'second':char(2,1)}
    return data, world


def prop(pid='bottle', events=None):
    return {'id':pid,'name':pid,'shape':'cyl','size_m':[.08,.2,.08],
            'events':events or [{'frame':0,'state':'held','person':1,'joint':'right'}], 'keyframes':[]}


class TimelineTests(unittest.TestCase):
    def test_release_pickup_transfer_and_hidden_are_separate_intervals(self):
        data,_=fixture(); before=copy.deepcopy(data['chars'])
        item=prop(events=[{'frame':0,'state':'held','person':1,'joint':'right'},
                          {'frame':3,'state':'free'}, {'frame':6,'state':'held','person':2,'joint':'left'},
                          {'frame':9,'state':'hidden'}])
        item['keyframes']=[{'frame':3,'box':[30,20,35,30]},{'frame':5,'box':[50,20,55,30]}]
        PT.apply(data,[item]); keys=np.asarray(data['props'][0]['keys'])
        self.assertTrue(np.all(keys[:9,6:]==1)); self.assertTrue(np.all(keys[9:,6:]==0))
        self.assertGreater(np.linalg.norm(keys[5,:3]-keys[3,:3]),.1)
        rig=Rig(1.72,'sample'); ch=data['chars']['second']; root=ch['root'][6]
        hand=rig.fk(root[:3],root[3],{j:ch['rot'][j][6] for j in JOINTS},ch['pelvis_z'][6])['wrist_L'][1]
        np.testing.assert_allclose(keys[6,:3],hand)
        self.assertEqual(data['chars'],before)

    def test_two_objects_same_hand_keep_identity_and_dimensions(self):
        data,_=fixture(); a,b=prop('phone'),prop('bottle')
        a['events'].append({'frame':4,'state':'hidden'})
        b['events'][0]['frame']=6
        b['size_m']=[.1,.4,.1]
        PT.apply(data,[a,b]); PT.apply(data,[a,b])
        self.assertEqual(len(data['props']),2)
        self.assertEqual({p['object_id'] for p in data['props']},{'phone','bottle'})
        self.assertEqual(data['props'][1]['size_m'],[.1,.4,.1])
        self.assertTrue(all(k[6]==0 for k in data['props'][1]['keys'][:6]))

    def test_release_without_spatial_keys_stays_at_release_and_warns(self):
        data,_=fixture(); item=prop(); item['events'].append({'frame':3,'state':'free'})
        PT.apply(data,[item]); keys=np.array(data['props'][0]['keys'])
        np.testing.assert_allclose(keys[3:,:3],np.repeat(keys[2,None,:3],9,axis=0))
        self.assertTrue(data['prop_timeline_warnings'])

    def test_free_object_does_not_cross_camera_cut_without_evidence(self):
        data,_=fixture(); data['shots']=[{'frames':[0,5]},{'frames':[6,11]}]
        item=prop(); item['events'].append({'frame':3,'state':'free'})
        item['keyframes']=[{'frame':3,'box':[20,20,30,30]}]
        PT.apply(data,[item]); self.assertTrue(all(k[6]==0 for k in data['props'][0]['keys'][6:]))

    def test_box_depth_can_place_first_free_object(self):
        data,_=fixture(); item=prop(events=[{'frame':0,'state':'free'}])
        item['keyframes']=[{'frame':0,'box':[30,20,40,30],'depth_m':4.}]
        PT.apply(data,[item]); self.assertTrue(all(k[6]==1 for k in data['props'][0]['keys']))

    def test_hand_offsets_do_not_interpolate_between_owners(self):
        data,_=fixture(); item=prop()
        item['events'].append({'frame':6,'state':'held','person':2,'joint':'left'})
        item['keyframes']=[{'frame':0,'box':[30,20,35,30]}]
        PT.apply(data,[item]); ch=data['chars']['second']; rig=Rig(1.72,'sample'); r=ch['root'][6]
        hand=rig.fk(r[:3],r[3],{j:ch['rot'][j][6] for j in JOINTS},.94)['wrist_L'][1]
        np.testing.assert_allclose(data['props'][0]['keys'][6][:3],hand)

    def test_visibility_of_person_is_respected(self):
        data,_=fixture(); data['chars']['main']['visible'][2]=False
        PT.apply(data,[prop()]); self.assertEqual(data['props'][0]['keys'][2][6:], [0,0,0])

    def test_old_static_prop_is_removed_and_restored_without_rebuilding_background(self):
        data,_=fixture(); mesh=cube(); mesh.update(label='bottle',cls='object',_primitive='bottle',
                         _source={'frame':0,'px':[30,20,40,40]})
        wall=copy.deepcopy(mesh); wall.update(name='wall',cls='wall',_primitive='wall')
        data['meshes']=[mesh,wall]; item=prop(); item['keyframes']=[{'frame':0,'box':[30,20,40,40]}]
        PT.apply(data,[item]); self.assertEqual([m['name'] for m in data['meshes']],['wall'])
        PT.apply(data,[]); self.assertEqual({m['name'] for m in data['meshes']},{'cube','wall'})

    def test_reject_invalid_events_dimensions_and_owner(self):
        for value in [float('nan'),-1,1.3,100]:
            p=prop(); p['events'][0]['frame']=value
            with self.assertRaises(ValueError): PT.validate([p],12)
        for size in [[0,1,1],[1,float('inf'),1],[1,2],[-1,1,1]]:
            p=prop(); p['size_m']=size
            with self.assertRaises(ValueError): PT.validate([p])
        p=prop(); p['events'][0]['person']=None
        with self.assertRaises(ValueError): PT.validate([p])
        with self.assertRaises(ValueError): PT.validate([prop(),prop()])
        with self.assertRaises(ValueError): review.validate({'people':{},'prop_timeline':[prop()]})

    def test_review_roundtrip_preserves_depth_events_and_mode(self):
        p=prop(); p['keyframes']=[{'frame':0,'box':[0,0,10,10],'depth_m':4}]
        rv=review.validate({'people':{'1':{}},'prop_timeline':[p]})
        self.assertEqual(rv['prop_mode'],'manual')
        self.assertEqual(rv['prop_timeline'][0]['keyframes'][0]['depth_m'],4)

    def test_detection_episodes_and_labels_never_merge_across_long_gap(self):
        dets=[{'track':1,'joint':'wrist_R','kind':'held','label':label,'score':.9,'f':f,'box':[0,0,4,4]}
              for label,fs in [('phone',[0,2,4,40,42,44]),('bottle',[60,62,64])] for f in fs]
        result=HO.groups(dets,{1:10})
        self.assertEqual(len(result),3)
        self.assertEqual(sorted(o['span'] for o in result),[[0,4],[40,44],[60,64]])


class ProtectionTests(unittest.TestCase):
    def test_source_visible_person_protected_but_real_occlusion_kept(self):
        data,world=fixture(1); b=evidence(world)
        b['person'][0,20:45,25:55]=True
        # Camera-space slab, 2m nearer than the source person.
        verts=np.array([[-2,-2,3],[2,-2,3],[2,2,3],[-2,2,3]]) @ world.R_cw(0)+world.C_at(0)
        mesh={'name':'oversized','v':verts.tolist(),'f':[[0,1,2,3]],'colour':'set'}
        data['meshes']=[mesh]
        actions=GP.protect(data,[b]); self.assertTrue(actions)
        result=GP.protect(data,[b]); self.assertFalse(result)
        data['meshes']=[copy.deepcopy(mesh)]; b['person'][:]=False
        self.assertEqual(GP.protect(data,[b]),[])
        self.assertEqual(data['meshes'][0]['v'],mesh['v'])

    def test_huge_conflict_suppressed_with_recoverable_backup(self):
        data,world=fixture(1); b=evidence(world); b['depth'][:]=20; b['person'][0,10:50,10:70]=True
        verts=np.array([[-3,-3,2],[3,-3,2],[3,3,2],[-3,3,2]])@world.R_cw(0)+world.C_at(0)
        data['meshes']=[{'name':'bad','v':verts.tolist(),'f':[[0,1,2,3]]}]
        GP.protect(data,[b]); self.assertEqual(data['meshes'],[])
        self.assertEqual(data['suppressed_geometry'][0]['meshes'][0]['name'],'bad')

    def test_observed_depth_tolerance_and_invalid_pixels(self):
        depth=np.full((10,10),5.); person=np.ones((10,10),bool)
        self.assertEqual(GP.person_conflict(depth-.1,depth,person)[0],0)
        self.assertEqual(GP.person_conflict(depth-1,depth,person,np.zeros_like(person))[0],0)
        self.assertEqual(GP.person_conflict(depth-1,depth,person)[0],100)

    def test_prototypes_sized_from_measurements_and_named_explicitly(self):
        from build_env import build, Cam
        from describe import parse
        rng=np.random.default_rng(1); points=rng.random((1000,3))*[.8,.6,1.7]+[0,5,0]
        self.assertEqual(prototypes.vegetation_kind('tree'),'pine')
        self.assertEqual(prototypes.vegetation_kind('bushes'),'bush')
        self.assertIsNone(prototypes.vegetation_kind('office first floor'))
        self.assertEqual(parse('table next to a tree')['kind'],'table')
        self.assertEqual(parse('potted plant')['kind'],'plant_pot')
        data,_=fixture(1)
        for kind in ('pine','bush','palm','plant_pot'):
            ob=prototypes.measured_object(kind,points,'plant',{})
            self.assertLess(ob['size'][1],1.8)
            meshes=build({'ground':{'size':0},'objects':[ob]},Cam(data))
            self.assertTrue(meshes); self.assertTrue(all(m['prototype']==kind for m in meshes))
            self.assertLess(np.ptp(np.concatenate([m['v'] for m in meshes]),axis=0)[2],2.)

    def test_automatic_footprint_is_bounded(self):
        data,world=fixture(1)
        m=cube((-3,3,0),(3,4,3)); source={'frame':0,'px':[25,15,55,45]}
        result=GP.match_source_box([m],source,data)
        self.assertLess(np.ptp(result[0]['v'],axis=0)[0],np.ptp(m['v'],axis=0)[0])

    def test_environment_preview_camera_can_build_prototypes(self):
        from env_auto import AutoCam
        from build_env import build
        cam=AutoCam(80,60,40,None)
        ob={'type':'pine','name':'tree','size':[.8,1.5,.6],'pos':[0,4,0],
            '_src':{'px':[10,10,20,30],'frame':0}}
        self.assertTrue(build({'ground':{'size':0},'objects':[ob]},cam))

    def test_explicit_static_dimensions_survive_box_placement(self):
        from build_env import build, Cam
        data,_=fixture(1)
        ob={'type':'box','name':'small','bbox':[20,15,60,45],'frame':0,'dist':4,
            'size':[.3,.5,.2],'size_locked':True,'_src':{'px':[20,15,60,45],'frame':0}}
        meshes=build({'ground':{'size':0},'objects':[ob]},Cam(data))
        extent=np.ptp(np.concatenate([m['v'] for m in meshes]),axis=0)
        np.testing.assert_allclose(extent,[.3,.2,.5],atol=1e-6)

    def test_rendered_character_depth_protects_against_scale_disagreement(self):
        obs=np.full((10,10),2.); background=np.full((10,10),3.)
        person=np.ones((10,10),bool); rendered=np.full((10,10),4.)
        self.assertEqual(GP.person_conflict(background,obs,person)[0],0)
        self.assertEqual(GP.person_conflict(background,obs,person,rendered=rendered)[0],100)


class DependencyTests(unittest.TestCase):
    def setup_case(self,tmp):
        root=pathlib.Path(tmp); video=root/'clip.mp4'; video.write_bytes(b'video')
        work,marks,out=[root/x for x in ('work','marks','out')]
        for p in (work,marks,out): p.mkdir()
        jsave(str(marks/'pose.json'),{'tracker_backend':'sam3'})
        rv={'people':{'1':{'height':1.72}},'camera':{},'prop_timeline':[]}
        jsave(str(marks/'review.json'),rv)
        data,_=fixture(); jsave(str(work/'scene_data.json'),data)
        jsave(str(work/'clip_info.json'),{'y0':0})
        return str(video),str(work),str(marks),str(out),rv

    def test_props_and_background_never_invalidate_human_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            video,work,marks,out,rv=self.setup_case(tmp)
            a=DEP.inputs(video,{},work,marks,out)
            rv['prop_timeline']=[prop()]; rv['prop_mode']='manual'; jsave(os.path.join(marks,'review.json'),rv)
            b=DEP.inputs(video,{},work,marks,out)
            self.assertEqual(a['fit'],b['fit']); self.assertEqual(a['environment'],b['environment']); self.assertNotEqual(a['props'],b['props'])
            jsave(os.path.join(marks,'env_ai.json'),{'shots':{'_000':{'objects':[{'type':'pine'}]}}})
            d=DEP.inputs(video,{},work,marks,out)
            self.assertEqual(b['fit'],d['fit']); self.assertNotEqual(b['environment'],d['environment'])
            rv['people']['1']['height']=1.8; jsave(os.path.join(marks,'review.json'),rv)
            self.assertNotEqual(d['fit'],DEP.inputs(video,{},work,marks,out)['fit'])

    def test_real_prop_only_run_skips_all_expensive_stages_and_preserves_people(self):
        with tempfile.TemporaryDirectory() as tmp:
            video,work,marks,out,rv=self.setup_case(tmp)
            c=dict(DEFAULT_CFG, work_root=tmp,scene_verify=False)
            # run_clip puts work under <root>/<name>
            name='work'; DEP.record(video,c,work,marks,out)
            before=jload(os.path.join(work,'scene_data.json'))
            rv.update(prop_timeline=[prop()],prop_mode='manual'); jsave(os.path.join(marks,'review.json'),rv)
            args=types.SimpleNamespace(name=name,stage='fit',require_confirm=False,force=True)
            with patch.object(RP,'marks_dir',return_value=marks),patch.object(RP,'output_dir',return_value=tmp), \
                 patch.object(RP,'load_review',return_value=rv),patch.object(DEP,'inputs',wraps=DEP.inputs) as calc, \
                 patch.object(RP,'find_marks',side_effect=AssertionError('people detection rerun')), \
                 patch.object(RP,'make_set',side_effect=AssertionError('background rerun')), \
                 patch.object(RP,'read_frames',side_effect=AssertionError('video decode rerun')):
                # Both out paths are empty; neither contains a hand set or hints.
                result=RP.run_clip(video,c,args)
            after=jload(os.path.join(work,'scene_data.json'))
            self.assertTrue(result['reused_people']); self.assertTrue(result['rebuilt_props'])
            self.assertEqual(after['chars'],before['chars']); self.assertEqual(after['cam'],before['cam'])
            self.assertEqual(len(after['props']),1)

    def test_legacy_migration_records_old_review_before_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            video,work,marks,out,rv=self.setup_case(tmp)
            DEP.adopt_existing(video,{},work,marks,out)
            state=jload(os.path.join(work,'build_state.json'))
            self.assertIsNone(state['environment'])
            self.assertEqual(state['fit'],DEP.inputs(video,{},work,marks,out)['fit'])

    def test_review_lens_and_hints_have_same_cache_key_in_server_and_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            video,work,marks,out,rv=self.setup_case(tmp)
            rv['camera']['lens']=42.; jsave(os.path.join(marks,'review.json'),rv)
            jsave(os.path.join(marks,'hints.json'),{'fit_step':3})
            a=DEP.inputs(video,{'lens_mm':34.9},work,marks,out)
            b=DEP.inputs(video,{'lens_mm':42.,'fit_step':3},work,marks,out)
            self.assertEqual(a,b)

    def test_numeric_person_cache_invalidates_height_and_keypoint_edits(self):
        data,world=fixture(); fs=[0,1,2,3]; kp=np.zeros((4,17,3)); q={0:np.zeros(FP.NP)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(FP,'fit_track',return_value=(Rig(1.72,'sample'),q,{0:1.})) as fit:
            args=(world,0,fs,kp,1.72,{'_work':tmp},'sample',{},None,{})
            FP.cached_fit_track(*args); FP.cached_fit_track(*args)
            self.assertEqual(fit.call_count,1)
            FP.cached_fit_track(world,0,fs,kp,1.8,{'_work':tmp},'sample',{},None,{})
            kp[0,0,0]=1
            FP.cached_fit_track(*args)
            self.assertEqual(fit.call_count,3)

    def test_build_state_uses_processed_snapshot_if_user_edits_during_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            video,work,marks,out,rv=self.setup_case(tmp)
            c=dict(DEFAULT_CFG,scene_verify=False); rv.update(prop_timeline=[prop()],prop_mode='manual')
            jsave(os.path.join(marks,'review.json'),rv)
            before=DEP.inputs(video,c,work,marks,out)
            def edit_during_job(*args,**kwargs):
                changed=copy.deepcopy(rv); changed['prop_timeline'][0]['name']='New name during job'
                jsave(os.path.join(marks,'review.json'),changed)
                return []
            with patch.object(RP,'held_place',side_effect=edit_during_job),patch.object(RP,'marks_dir',return_value=marks),patch.object(RP,'output_dir',return_value=tmp):
                RP.rebuild_edits('work',video,work,c,types.SimpleNamespace(stage='fit'),0,
                    dict(before,props='old'),before)
            state=jload(os.path.join(work,'build_state.json'))
            self.assertEqual(state,before)
            self.assertNotEqual(state['props'],DEP.inputs(video,c,work,marks,out)['props'])

    def test_scene_batch_reuses_cached_results_without_starting_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt=pathlib.Path(tmp)/'sam.pt'; ckpt.write_bytes(b'weights')
            c={'sam3_checkpoint':str(ckpt),'sam3_python':sys.executable}
            rgb=np.zeros((12,16,3),np.uint8); pending=[]
            SAM.segment(rgb,c,cache_dir=tmp,_queue=pending)
            np.savez_compressed(pending[0][3],masks=np.zeros((0,12,16),bool),box_masks=np.zeros((0,12,16),bool),peak_vram_mb=1.)
            with patch.object(SAM.subprocess,'run',side_effect=AssertionError('Worker should not run')):
                self.assertEqual(SAM.prefetch([{'rgb':rgb}],c,tmp),0)


if __name__=='__main__': unittest.main()

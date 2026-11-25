"""Fast processing, opt-in audit coverage, and content-based validation reuse."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'pipeline'),str(ROOT/'tests')]
import validation as V
import scene_verify as S
import quality
import dependencies as D
from test_metric_scene import scene,evidence,cube


class ValidationModeTests(unittest.TestCase):
    def test_view_budget_is_global_across_many_shots(self):
        bundles=[{'frames':np.arange(i*4,i*4+4),'depth':np.ones((4,2,2)),'names':np.array(['wall'])} for i in range(5)]
        selected=V.sampled_bundles(bundles)
        self.assertEqual(sum(len(b['frames']) for b in selected),3)
        self.assertEqual([int(b['frames'][0]) for b in selected],[0,4,8])

    def fixture(self,tmp,frames=9):
        data,world=scene();data['frames']=frames;data['cam']*=frames;data['zoom']*=frames
        bundle=evidence(world)
        bundle={k:np.repeat(v,frames,axis=0) if k not in ('names',) else v for k,v in bundle.items()}
        bundle['frames']=np.arange(frames)
        np.savez_compressed(Path(tmp)/'metric_evidence.npz',**bundle)
        return data

    def test_normal_skips_exhaustive_work_and_audit_uses_all_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=self.fixture(tmp)
            with patch.object(S,'support_scale',side_effect=AssertionError('normal contact search')), \
                 patch.object(S,'depth_correction',side_effect=AssertionError('normal exhaustive repair')), \
                 patch.object(S,'overview',side_effect=AssertionError('normal diagnostic export')), \
                 patch('metric_scene.plane_proposals',side_effect=AssertionError('normal surface proposals')):
                result=S.process(data,tmp,{},repair=True)
            self.assertEqual(len(result['checked_depth_frames']),3)
            self.assertFalse(result['checks']['detailed_contacts']);self.assertFalse(result['exhaustive'])
            with patch.object(S,'overview') as preview:
                audit=S.process(data,tmp,{'validation_mode':'audit'})
            self.assertEqual(len(audit['checked_depth_frames']),9);self.assertTrue(audit['exhaustive'])
            preview.assert_called_once()

    def test_unchanged_prepare_and_finish_reuse_post_repair_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=self.fixture(tmp)
            def repair(scene,*args,**kwargs):
                scene['meshes'].append(cube());return [{'action':'bounded test correction'}]
            with patch('geometry_protection.protect',side_effect=repair) as protect:
                first=S.process(data,tmp,{},repair=True)
            self.assertEqual(protect.call_args.kwargs,dict(max_groups=2,refine=False,resolution=96,only_objects=set()))
            self.assertFalse(first['cache_hit'])
            with patch.object(S,'verify',side_effect=AssertionError('unchanged checks reran')), \
                 patch('geometry_protection.protect',side_effect=AssertionError('unchanged repair reran')):
                second=S.process(data,tmp,{'_freeze_people':True})
                third=S.process(data,tmp,{},repair=True)
            self.assertTrue(second['cache_hit']);self.assertTrue(third['cache_hit'])
            self.assertEqual(len(data['meshes']),1)

    def test_changed_geometry_camera_evidence_and_policy_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=self.fixture(tmp);data['meshes']=[cube()]
            S.process(data,tmp,{},repair=False)
            data['meshes'][0]['v'][0][0]+=.1
            self.assertFalse(S.process(data,tmp,{})['cache_hit'])
            data['cam'][0][0]+=.1
            self.assertFalse(S.process(data,tmp,{})['cache_hit'])
            evidence_path=Path(tmp)/'metric_evidence.npz'
            evidence_path.unlink()
            self.assertFalse(S.process(data,tmp,{})['cache_hit'])
            self.assertFalse(S.process(data,tmp,{'validation_mode':'audit'})['cache_hit'])
            with patch.object(S,'verify',side_effect=AssertionError('normal cache was lost after audit')):
                self.assertTrue(S.process(data,tmp,{})['cache_hit'])

    def test_corrupt_cache_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=self.fixture(tmp);S.process(data,tmp,{})
            path=Path(tmp)/'scene_validation_cache_normal.json';bad=json.loads(path.read_text())
            bad['result']['unresolved']=[];bad['result']['issues']=[];path.write_text(json.dumps(bad))
            self.assertFalse(S.process(data,tmp,{})['cache_hit'])
            path.write_text('{broken');self.assertFalse(S.process(data,tmp,{})['cache_hit'])

    def test_cached_profile_is_fresh_inside_a_prepare_profile(self):
        from profiling import recording
        with tempfile.TemporaryDirectory() as tmp:
            data=self.fixture(tmp);S.process(data,tmp,{})
            with recording() as parent:S.process(data,tmp,{})
            profile=json.loads((Path(tmp)/'verification_profile.json').read_text())
            self.assertEqual(profile['counters']['validation_cache_hits'],1)
            self.assertNotIn('scene_checks',profile['stages'])
            self.assertEqual(parent['counters']['validation_cache_hits'],1)

    def test_modes_do_not_invalidate_inference_or_background_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'clip.mp4';video.write_bytes(b'video')
            for name in ('work','marks','out'):(root/name).mkdir()
            normal=D.inputs(str(video),{},root/'work',root/'marks',root/'out')
            audit=D.inputs(str(video),{'validation_mode':'audit'},root/'work',root/'marks',root/'out')
            self.assertEqual(normal,audit)
            self.assertFalse(V.review_enabled({}));self.assertTrue(V.review_enabled({'validation_mode':'audit'}))
            self.assertFalse(V.review_enabled({'validation_mode':'audit','frame_review':False}))

    def test_audit_switch_reuses_pipeline_fit_and_preserves_failed_gate(self):
        import types
        import run_pipeline as R
        from common import DEFAULT_CFG,jsave
        from test_props_geometry import DependencyTests
        with tempfile.TemporaryDirectory() as tmp:
            video,work,marks,out,review=DependencyTests().setup_case(tmp)
            config=dict(DEFAULT_CFG,work_root=tmp,_review=review)
            D.record(video,config,work,marks,out)
            jsave(str(Path(work)/'report.json'),{'human_alignment':{'passed':False,'people':[{'char':'test','passed':False}]}})
            args=types.SimpleNamespace(name='work',stage='fit',require_confirm=False,force=True)
            with patch.object(R,'marks_dir',return_value=marks),patch.object(R,'output_dir',return_value=tmp), \
                 patch.object(R,'load_review',return_value=review), \
                 patch.object(R,'find_marks',side_effect=AssertionError('tracking reran')), \
                 patch.object(R,'fit_all',side_effect=AssertionError('human fitting reran')), \
                 patch.object(R,'make_set',side_effect=AssertionError('background discovery reran')):
                result=R.run_clip(video,dict(config,validation_mode='audit'),args)
            self.assertTrue(result['reused_people']);self.assertEqual(result['status'],'needs_review')

    def test_human_checks_reuse_failure_but_invalidate_changed_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,_=scene();pose={'people':[]};failure={'passed':False,'people':[{'char':'missing','passed':False}]}
            with patch.object(quality,'alignment_audit',return_value=failure) as check:
                quality.cached_alignment(data,pose,0,tmp)
                result=quality.cached_alignment(data,pose,0,tmp)
                self.assertFalse(result['passed']);self.assertTrue(result['cache_hit']);self.assertEqual(check.call_count,1)
                pose['changed']=True;quality.cached_alignment(data,pose,0,tmp);self.assertEqual(check.call_count,2)
            with self.assertRaises(RuntimeError):quality.require_alignment(result)

    def test_essential_checks_reject_invalid_camera_even_in_normal_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,_=scene();data['cam'][0][0]=float('nan')
            with self.assertRaisesRegex(ValueError,'camera'):S.process(data,tmp,{})

    def test_rig_sampling_and_cache_scope(self):
        import l3_blender as L
        from types import SimpleNamespace
        data={'frames':100,'chars':{},'shots':[{'frames':[0,49]},{'frames':[50,99]}]}
        self.assertLess(len(L.validation_frames(data)),10);self.assertEqual(len(L.validation_frames(data,'audit')),100)
        with tempfile.TemporaryDirectory() as tmp,patch.object(L,'validate_actors',return_value={'passed':True}) as check:
            bpy=SimpleNamespace(app=SimpleNamespace(version_string='test'))
            L.cached_validation(bpy,data,tmp);L.cached_validation(bpy,data,tmp)
            self.assertEqual(check.call_count,1)
            data['validation_mode']='audit';L.cached_validation(bpy,data,tmp);self.assertEqual(check.call_count,2)
            data['validation_mode']='normal';L.cached_validation(bpy,data,tmp);self.assertEqual(check.call_count,2)

    def test_settings_accept_modes_and_reject_unknown_policy(self):
        from agent import config
        with tempfile.TemporaryDirectory() as tmp,patch.object(config,'CFG_PATH',str(Path(tmp)/'config.json')):
            self.assertEqual(config.save({'validation_mode':'audit'})['validation_mode'],'audit')
            with self.assertRaises(ValueError):config.save({'validation_mode':'disabled'})
            self.assertEqual(config.save({'validation_mode':'normal'})['validation_mode'],'normal')

    def test_review_export_reuses_unchanged_frames_and_repairs_missing_artifacts(self):
        import cv2
        import frame_review as F
        class Capture:
            def read(self):return True,np.zeros((24,32,3),np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);renders=root/'render';renders.mkdir();output=root/'review'
            cv2.imwrite(str(renders/'wm_0001.png'),np.full((16,32,3),100,np.uint8))
            info=dict(frames=1,W=32,Hfull=24,Hpic=16,y0=4,fps=30)
            with patch.object(F,'open_video',return_value=(Capture(),None)) as decode,patch.object(F,'close_video'):
                self.assertFalse(F.build('source',renders,output,info)['cache_hit'])
                self.assertTrue(F.build('source',renders,output,info)['cache_hit']);self.assertEqual(decode.call_count,1)
                (output/'model/000000.jpg').unlink()
                self.assertFalse(F.build('source',renders,output,info)['cache_hit']);self.assertEqual(decode.call_count,2)
                cv2.imwrite(str(renders/'wm_0001.png'),np.full((16,32,3),150,np.uint8))
                self.assertFalse(F.build('source',renders,output,info)['cache_hit']);self.assertEqual(decode.call_count,3)

    def test_normal_delivery_removes_stale_optional_review_only(self):
        import package
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);review=root/'review';review.mkdir();(review/'index.html').write_text('old audit')
            scene=root/'scene.json';scene.write_text('{}')
            package.clear_optional_review(root,{'validation_mode':'audit'});self.assertTrue(review.exists())
            package.clear_optional_review(root,{});self.assertFalse(review.exists());self.assertTrue(scene.exists())


if __name__=='__main__':unittest.main()

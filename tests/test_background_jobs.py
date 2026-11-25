"""Independent background rebuilds and concurrent preparation/finishing."""
import copy
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT)]
from common import DEFAULT_CFG,jsave,jload
import dependencies as DEP
import env_auto as EA
import gpu_lease
import run_pipeline as RP
from agent import jobs
from test_props_geometry import fixture


class QueueLanesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=jobs.JobStore(os.path.join(self.tmp.name,'queue.db'))

    def tearDown(self):self.store.db.close();self.tmp.cleanup()

    def take(self):return self.store.take({'u'},prepare_limit=1,finish_limit=1)

    def test_prepare_and_finish_overlap_but_second_job_in_each_lane_waits(self):
        for clip,stage in [('p1','fit'),('p2','background'),('f1','all'),('f2','all')]:
            self.store.enqueue(clip,stage,user='u')
        finish=self.take();prepare=self.take()
        self.assertEqual(finish['stage'],'all');self.assertEqual(prepare['stage'],'background')
        self.assertIsNone(self.take())
        self.store.finish(finish['id'],True)
        self.assertEqual(self.take()['clip'],'f2')
        self.store.finish(prepare['id'],True)
        self.assertEqual(self.take()['clip'],'p1')

    def test_same_clip_remains_serialized_across_lanes(self):
        self.store.enqueue('clip','fit',user='u');first=self.take()
        self.store.enqueue('clip','all',user='u',force=True)
        self.assertIsNone(self.take());self.store.finish(first['id'],True)
        self.assertEqual(self.take()['stage'],'all')

    def test_repeated_background_requests_do_not_duplicate_a_waiting_job(self):
        self.store.enqueue('clip','fit',user='u');self.take()
        first=self.store.enqueue('clip','background',user='u',force=True)
        self.assertIsNotNone(first)
        self.assertIsNone(self.store.enqueue('clip','background',user='u',force=True))
        self.assertEqual(self.store.counts()['queued'],1)

    def test_worker_pool_starts_both_lanes_with_parallel_one(self):
        app=types.SimpleNamespace(store=self.store,c={'parallel':1,'finish_parallel':1,'autopilot':'off'})
        runner=jobs.Runner(app);started={k:threading.Event() for k in ('fit','all')};release=threading.Event()
        def run(job):
            started[job['stage']].set();release.wait(5);self.store.finish(job['id'],True)
        runner._run=run
        for clip,stage in [('p','fit'),('f','all')]:self.store.enqueue(clip,stage,user='u')
        runner.active.add('u')
        threads=[threading.Thread(target=runner._worker) for _ in range(2)]
        for t in threads:t.start()
        try:
            self.assertTrue(started['fit'].wait(3));self.assertTrue(started['all'].wait(3))
            self.assertEqual(len(self.store.running()),2)
        finally:
            release.set();runner.stop.set();runner.wake()
            for t in threads:t.join(3)

    def test_background_time_is_separate_from_prepare(self):
        jid=self.store.enqueue('clip','background',user='u');self.take();self.store.finish(jid,True)
        self.assertIn('background',self.store.timing()['clip']['steps'])
        self.assertNotIn('prepare',self.store.timing()['clip']['steps'])


class BackgroundPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)
        self.video=str(self.root/'clip.mp4');pathlib.Path(self.video).write_bytes(b'video')
        self.work=self.root/'clip';self.work.mkdir();self.marks=self.root/'marks';self.marks.mkdir()
        self.data,_=fixture(12)
        for fn,value in [('scene_data.json',self.data),('camera.json',{}),('clip_info.json',{'video':self.video})]:jsave(str(self.work/fn),value)
        self.c=dict(DEFAULT_CFG,work_root=str(self.root),_review={},_hints={})
        self.previous={'fit':'existing-human-fit','environment':'old','props':'old'}
        jsave(str(self.work/'build_state.json'),self.previous)
        self.args=types.SimpleNamespace(stage='background',name='clip',require_confirm=False,force=True)

    def tearDown(self):self.tmp.cleanup()

    def invoke(self, mutate=False, fail=False):
        def command(cmd,**kw):
            self.assertEqual(os.environ.get('BA_FREEZE_PEOPLE'),'1')
            if fail:raise subprocess.CalledProcessError(1,cmd)
            if cmd[1].endswith('build_env.py'):
                data=jload(str(self.work/'scene_data.json'))
                if mutate:data['chars']['main']['root'][0][2]+=1
                data['meshes']=[{'name':'new-background','v':[],'f':[]}]
                jsave(str(self.work/'scene_data.json'),data)
        with patch.object(RP,'marks_dir',return_value=str(self.marks)),patch.object(RP,'output_dir',return_value=str(self.root)), \
             patch.object(RP.subprocess,'run',side_effect=command) as commands,patch.object(RP,'held_place',return_value=[]), \
             patch.object(RP,'find_marks',side_effect=AssertionError('tracking rerun')), \
             patch.object(RP,'fit_all',side_effect=AssertionError('human refit')), \
             patch.object(RP,'estimate_shot',side_effect=AssertionError('camera refit')), \
             patch.object(RP,'read_frames',side_effect=AssertionError('full video decoded')), \
             patch.object(RP,'held_detect',side_effect=AssertionError('automatic props rerun')):
            result=RP.run_clip(self.video,self.c,self.args)
        return result,commands

    def test_background_route_preserves_humans_camera_and_pending_fit_state(self):
        result,commands=self.invoke()
        self.assertTrue(result['reused_people']);self.assertTrue(result['reused_camera'])
        after=jload(str(self.work/'scene_data.json'))
        for key in ('chars','cam','zoom'):self.assertEqual(after.get(key),self.data.get(key))
        self.assertEqual(after['meshes'][0]['name'],'new-background')
        self.assertEqual(jload(str(self.work/'build_state.json'))['fit'],'existing-human-fit')
        self.assertTrue((self.work/'background_revision.json').exists())
        self.assertIn('--annotations-only',commands.call_args_list[0].args[0]);self.assertEqual(commands.call_count,2)

    def test_accidental_human_change_rolls_back(self):
        with self.assertRaisesRegex(RuntimeError,'original scene restored'):self.invoke(mutate=True)
        self.assertEqual(jload(str(self.work/'scene_data.json')),self.data)
        self.assertFalse((self.work/'background_revision.json').exists())

    def test_finish_reuses_scene_after_background_update(self):
        DEP.record(self.video,self.c,str(self.work),str(self.marks),str(self.root/'clip'))
        self.invoke()
        args=types.SimpleNamespace(stage='all',name='clip',require_confirm=False,force=True)
        with patch.object(RP,'marks_dir',return_value=str(self.marks)),patch.object(RP,'output_dir',return_value=str(self.root)), \
             patch.object(RP,'render_only',return_value={'status':'ok'}) as render, \
             patch.object(RP,'find_marks',side_effect=AssertionError('tracking rerun')), \
             patch.object(RP,'make_set',side_effect=AssertionError('background rerun')):
            result=RP.run_clip(self.video,self.c,args)
        self.assertTrue(result['reused_people']);self.assertFalse(result['rebuilt_environment']);render.assert_called_once()

    def test_failed_object_worker_preserves_scene_and_dependency_state(self):
        with self.assertRaises(subprocess.CalledProcessError):self.invoke(fail=True)
        self.assertEqual(jload(str(self.work/'scene_data.json')),self.data)
        self.assertEqual(jload(str(self.work/'build_state.json')),self.previous)

    def test_background_requires_an_existing_fit(self):
        (self.work/'camera.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'Prepare the clip first'):self.invoke()


class InferenceLeaseTests(unittest.TestCase):
    def test_copied_scene_cannot_render_into_a_different_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=os.path.join(tmp,'scene.json')
            original={'name':'clip','work_dir':os.path.join(tmp,'original'),'submit_dir':os.path.join(tmp,'submit','clip')}
            jsave(path,original)
            with patch.object(RP,'blender_exe',return_value='blender'),patch.object(RP.subprocess,'run') as launch:
                with self.assertRaisesRegex(RuntimeError,'render workspace'):
                    RP.blender_render({'submit_root':os.path.join(tmp,'submit')},tmp,path,1)
                launch.assert_not_called()
            self.assertEqual(jload(path),original)

    def test_another_process_waits_until_lease_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=os.path.join(tmp,'gpu.lock');marker=os.path.join(tmp,'acquired')
            lease=gpu_lease.Lease(path);lease.acquire()
            code="import sys,pathlib;sys.path.insert(0,sys.argv[1]);from gpu_lease import Lease;g=Lease(sys.argv[2]);g.acquire();pathlib.Path(sys.argv[3]).write_text('yes');g.release()"
            proc=subprocess.Popen([sys.executable,'-c',code,str(ROOT/'pipeline'),path,marker],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                self.assertIn('waiting for model inference slot',proc.stdout.readline())
                self.assertFalse(os.path.exists(marker))
                lease.release();out,err=proc.communicate(timeout=5)
                self.assertEqual(proc.returncode,0,err);self.assertTrue(os.path.exists(marker))
            finally:
                lease.release()
                if proc.poll() is None:proc.kill();proc.communicate()

    def test_job_exception_releases_model_slot(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'BA_GPU_LOCK':os.path.join(tmp,'gpu.lock')}):
            @gpu_lease.job
            def failed():gpu_lease.acquire();raise ValueError('failure')
            with self.assertRaises(ValueError):failed()
            lease=gpu_lease.Lease(os.environ['BA_GPU_LOCK']);lease.acquire();lease.release()

    def test_blender_releases_inference_slot_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'BA_GPU_LOCK':os.path.join(tmp,'gpu.lock')}):
            @gpu_lease.job
            def render():
                gpu_lease.acquire();lease=gpu_lease._active.get()
                with patch.object(RP,'blender_exe',return_value=None):
                    with self.assertRaisesRegex(RuntimeError,'Blender not found'):RP.blender_render({},tmp,'scene.json',1)
                self.assertIsNone(lease.file)
            render()


if __name__=='__main__':unittest.main()

"""Reset must stop every writer before deleting outputs and timing history."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent import config,clips,jobs
from agent.server import App,HttpError


class ResetLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.patch=patch.object(config,'DATA',str(self.root/'data'));self.patch.start()
        self.addCleanup(self.patch.stop)
        import package  # pipeline, on sys.path through config; queueing names a clip, so keep the names file in the temp folder
        names=patch.object(package,'NAMES_P',str(self.root/'package_names.json'));names.start()
        self.addCleanup(names.stop)
        source=self.root/'source';source.mkdir()
        for name in ('one','two'):(source/(name+'.mp4')).write_bytes(b'synthetic source')
        c={**config.DEFAULT_CFG,**config.APP_DEFAULTS,'source_root':str(source),'work_root':str(self.root/'work'),
           'submit_root':str(self.root/'submit'),'marks_root':str(self.root/'marks'),'max_attempts':3}
        self.app=App(c);self.addCleanup(self.app.store.db.close)
        self.ws=self.app.ws
        self.runner=self.app.runner;self.runner.after=Mock();self.threads=[]

    def tearDown(self):
        self.runner.shutdown()
        for thread in self.threads:thread.join(5)

    def wait_for(self,predicate):
        deadline=time.monotonic()+5
        while not predicate():
            if time.monotonic()>deadline:self.fail('Timed out waiting for test worker')
            threading.Event().wait(.01)

    def start_worker(self):
        thread=threading.Thread(target=self.runner._worker,daemon=True)
        self.threads.append(thread);thread.start();self.runner.resume()

    def fill(self,cl):
        f=clips.folders(self.ws.c,cl)
        for key in ('marks','work','submit'):
            Path(f[key]).mkdir(parents=True,exist_ok=True)
            (Path(f[key])/'old.json').write_text('{}')
        Path(f['zip']).write_bytes(b'generated package')
        return f

    def test_live_process_tree_stops_before_work_and_history_are_removed(self):
        cl=self.ws.lib.clips[0];f=self.fill(cl)
        heartbeat=self.root/'child-heartbeat'
        child="from pathlib import Path;import sys,time;p=Path(sys.argv[1]);\nwhile True:p.write_text(str(time.time()));time.sleep(.02)"
        parent="from pathlib import Path;import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);p=Path(sys.argv[3]);\nwhile True:p.mkdir(parents=True,exist_ok=True);(p/'active').write_text('writing');time.sleep(.02)"
        self.runner._cmd=lambda *_:[sys.executable,'-c',parent,child,str(heartbeat),f['work']]
        jid=self.ws.enqueue(cl,'movement');self.start_worker()
        self.wait_for(lambda:heartbeat.exists() and jid in self.runner.procs)
        process=self.runner.procs[jid]
        self.app.store.enqueue(cl.id,'background',force=True)
        result=self.ws.reset([],all_clips=True)
        self.assertTrue(result['ok']);self.assertGreaterEqual(result['stopped'],2)
        self.assertIsNotNone(process.poll())
        self.assertFalse(Path(f['work']).exists())
        last=heartbeat.read_text();threading.Event().wait(.15)
        self.assertEqual(heartbeat.read_text(),last,'Child process must also stop')
        self.assertFalse(Path(f['work']).exists(),'A late writer must not recreate the folder')
        self.assertEqual(self.app.store.counts(),{})
        self.assertNotIn(cl.id,self.app.store.timing())
        self.assertTrue(self.runner.is_paused())
        self.runner.after.assert_not_called()

    def test_reset_covers_skipped_clips(self):
        selected=list(self.ws.lib.clips)
        for cl in selected:self.fill(cl)
        clips.set_skipped(selected[0].id,True,self.ws.c)
        clips.set_skipped(selected[1].id,True,self.ws.c)
        jid=self.app.store.enqueue(selected[1].id,'movement')
        self.app.store.take();self.app.store.finish(jid,True)
        Path(self.ws.c['work_root'],'CLAUDE_HANDOFF.json').write_text('{}')
        result=self.ws.reset([],all_clips=True)
        self.assertTrue(result['ok']);self.assertEqual(len(result['clips']),2)
        self.assertEqual(len(self.ws.lib.clips),2)
        self.assertFalse(clips.skipped_set(self.ws.c))
        for cl in selected:
            self.assertEqual(clips.prepared_paths(self.ws.c,cl),[])
            self.assertEqual(clips.status(self.ws.c,cl)['label'],'new')
            self.assertTrue(Path(cl.video).exists())
        self.assertFalse(Path(self.ws.c['work_root'],'CLAUDE_HANDOFF.json').exists())
        self.assertIsNone(self.app.store.latest_for(selected[1].id),'its job history is gone too')
        self.assertEqual(self.app.store.timing(),{})

    def test_timing_only_reset_and_dry_run_do_not_need_work_files(self):
        cl=self.ws.lib.clips[0]
        jid=self.ws.enqueue(cl,'movement');self.app.store.take();self.app.store.finish(jid,True)
        preview=self.ws.reset([],dry=True,all_clips=True)
        self.assertEqual(len(preview['clips']),2)
        self.assertIn(cl.id,self.app.store.timing())
        result=self.ws.reset([],all_clips=True)
        self.assertTrue(result['ok']);self.assertEqual(self.app.store.timing(),{})

    def test_reset_during_startup_prevents_launch_and_new_enqueues(self):
        cl=self.ws.lib.clips[0];entered=threading.Event();release=threading.Event()
        def command(*_):
            entered.set();release.wait(3)
            return [sys.executable,'-c',"raise AssertionError('Must not launch')"]
        self.runner._cmd=command
        self.ws.enqueue(cl,'movement');self.start_worker();self.assertTrue(entered.wait(3))
        results=[]
        reset_thread=threading.Thread(target=lambda:results.append(self.ws.reset([],all_clips=True)))
        reset_thread.start()
        try:
            self.wait_for(lambda:self.runner.resetting)
            with self.assertRaises(HttpError):self.ws.enqueue(cl,'movement')
            self.runner.resume()
            self.assertTrue(self.runner.is_paused())
            with patch.object(jobs.subprocess,'Popen',side_effect=AssertionError('Cancelled worker launched')):
                release.set();reset_thread.join(3)
            self.assertFalse(reset_thread.is_alive());self.assertTrue(results[0]['ok'])
            self.assertFalse(Path(clips.folders(self.ws.c,cl)['work']).exists())
            self.assertEqual(self.app.store.counts(),{})
        finally:
            release.set();reset_thread.join(3)

    def test_timeout_keeps_files_and_leaves_reset_all_paused(self):
        cl=self.ws.lib.clips[0];f=self.fill(cl)
        self.runner.inflight[99]={'id':99,'clip':cl.id}
        try:
            with self.assertRaises(TimeoutError):
                with self.runner.reset_jobs([cl.id],all_clips=True,timeout=.01):
                    self.fail('Must not delete files while a worker can write')
            self.assertTrue(Path(f['work']).exists())
            self.assertTrue(self.runner.is_paused())
        finally:self.runner.inflight.clear()

    def test_pause_stops_active_work_then_resume_reuses_completed_cache(self):
        import numpy as np
        cl=self.ws.lib.clips[0];f=self.fill(cl)
        checkpoint=Path(f['work'])/'checkpoint';done=self.root/'resumed-done'
        valid=Path(f['work'])/'complete.npz';np.savez(valid,value=[1,2,3]);before=valid.read_bytes()
        broken=Path(f['work'])/'incomplete.npz';broken.write_bytes(b'partial interrupted numpy output')
        code="from pathlib import Path;import sys,time;p=Path(sys.argv[1]);\nif p.exists():Path(sys.argv[2]).write_text('reused checkpoint');sys.exit(0)\np.write_text('completed expensive stage');\nwhile True:time.sleep(.02)"
        self.runner._cmd=lambda *_:[sys.executable,'-c',code,str(checkpoint),str(done)]
        jid=self.ws.enqueue(cl,'movement');self.start_worker()
        self.wait_for(lambda:checkpoint.exists() and jid in self.runner.procs)
        self.app.store.db.execute('UPDATE jobs SET started=? WHERE id=?',(time.time()-20,jid))
        process=self.runner.procs[jid]
        result=self.runner.interrupt(pause=True)
        self.assertTrue(result['resumable']);self.assertTrue(result['paused'])
        self.assertIsNotNone(process.poll());self.runner.after.assert_not_called()
        self.assertEqual(valid.read_bytes(),before);self.assertFalse(broken.exists())
        self.assertTrue(checkpoint.exists());self.assertFalse(done.exists())
        queued=self.app.store.latest_for(cl.id)
        self.assertEqual(queued['state'],'queued');self.assertNotEqual(queued['id'],jid)
        self.assertGreaterEqual(self.app.store.timing()[cl.id]['machine'],20)
        self.runner.resume()
        self.wait_for(lambda:done.exists() and self.app.store.latest_for(cl.id)['state']=='done')
        self.assertEqual(done.read_text(),'reused checkpoint')
        self.assertEqual(self.app.store.latest_for(cl.id)['attempts'],1,'Pause does not consume a retry')
        self.runner.after.assert_called_once()

    def test_cancel_stops_process_without_deleting_completed_work_or_requeueing(self):
        cl=self.ws.lib.clips[0];f=self.fill(cl);ready=self.root/'cancel-ready'
        code="from pathlib import Path;import sys,time;Path(sys.argv[1]).write_text('ready');\nwhile True:time.sleep(.02)"
        self.runner._cmd=lambda *_:[sys.executable,'-c',code,str(ready)]
        jid=self.ws.enqueue(cl,'movement');self.start_worker()
        self.wait_for(lambda:ready.exists() and jid in self.runner.procs)
        process=self.runner.procs[jid]
        result=self.runner.interrupt()
        self.assertTrue(result['paused']);self.assertFalse(result['resumable'])
        self.assertIsNotNone(process.poll())
        self.assertEqual(self.app.store.counts().get('queued',0),0)
        self.assertEqual(self.app.store.latest_for(cl.id)['state'],'stopped')
        self.assertTrue(Path(f['work'],'old.json').exists());self.assertTrue(Path(f['zip']).exists())
        self.assertEqual(clips.status(self.ws.c,cl,self.app.store.latest_for(cl.id))['label'],'done',
                         'the zip made earlier still counts after a later run is cancelled')
        Path(f['zip']).unlink()
        self.assertEqual(clips.status(self.ws.c,cl,self.app.store.latest_for(cl.id))['label'],'cancelled')
        self.assertEqual(self.app.store.recover(),0,'Cancelled work must stay cancelled after restart')
        self.runner.after.assert_not_called()

    def test_pause_restarts_interrupted_stage_before_later_finish_on_same_clip(self):
        cl=self.ws.lib.clips[0]
        fit=self.ws.enqueue(cl,'movement');self.app.store.take()
        finish=self.app.store.enqueue(cl.id,'background',force=True)
        self.runner.interrupt(pause=True)
        resumed=self.app.store.take()
        self.assertEqual(resumed['stage'],'movement')
        self.assertNotEqual(resumed['id'],fit)
        self.assertIsNone(self.app.store.take())
        self.app.store.finish(resumed['id'],True)
        self.assertEqual(self.app.store.take()['id'],finish)

    def test_pause_only_interrupts_active_job_and_keeps_ready_clip_and_waiting_row(self):
        active,ready_clip=self.ws.lib.clips
        paths=self.fill(ready_clip)
        Path(paths['scene']).write_text('{}')
        Path(paths['preview']).write_bytes(b'prepared preview')
        clips.record_finish(self.ws.c,ready_clip,clips.fingerprint(self.ws.c,ready_clip))
        before_status=clips.status(self.ws.c,ready_clip)
        self.assertEqual(before_status['label'],'done')
        ready=self.root/'pause-only-running'
        self.runner._cmd=lambda *_:[sys.executable,'-c',
            "from pathlib import Path;import sys,time;Path(sys.argv[1]).touch();time.sleep(60)",str(ready)]
        running_id=self.ws.enqueue(active,'movement');self.start_worker()
        self.wait_for(lambda:ready.exists() and running_id in self.runner.procs)
        waiting_id=self.ws.enqueue(active,'background',force=True)
        def waiting_row():
            return dict(self.app.store.db.execute('SELECT * FROM jobs WHERE id=?',(waiting_id,)).fetchone())
        before_waiting=waiting_row()
        result=self.runner.interrupt(pause=True)
        self.assertEqual(result['interrupted'],1)
        self.assertEqual(waiting_row(),before_waiting,'Pause must not rewrite waiting jobs')
        self.assertEqual(clips.status(self.ws.c,ready_clip),before_status)
        paused=self.app.store.latest_for(active.id)
        self.assertEqual(clips.status(self.ws.c,active,paused)['label'],'paused')

    def test_pause_with_only_waiting_work_does_not_rewrite_it(self):
        cl=self.ws.lib.clips[0]
        self.ws.enqueue(cl,'movement')
        before=self.app.store.latest_for(cl.id)
        self.runner.resume()
        result=self.runner.interrupt(pause=True)
        self.assertEqual(result['interrupted'],0)
        self.assertEqual(self.app.store.latest_for(cl.id),before)

    def test_cancel_one_clip_does_not_cancel_other_waiting_work(self):
        first,second=self.ws.lib.clips
        self.ws.enqueue(first,'movement');self.ws.enqueue(second,'movement')
        self.runner.resume()
        result=self.runner.interrupt(clip=first.id)
        self.assertFalse(result['paused'],'Other clips may continue')
        self.assertEqual(self.app.store.latest_for(first.id)['state'],'stopped')
        self.assertEqual(self.app.store.latest_for(second.id)['state'],'queued')
        with patch.object(self.runner.stop,'wait',side_effect=[False,True]):self.runner._autopilot()
        self.assertEqual(self.app.store.latest_for(first.id)['state'],'stopped','Autopilot must not restart a cancelled clip')

    def test_api_pauses_resumes_and_cancels_a_live_job(self):
        import urllib.request
        from agent.server import Server,make_handler
        server=Server(('127.0.0.1',0),make_handler(self.app))
        service=threading.Thread(target=server.serve_forever,daemon=True);service.start()
        base='http://127.0.0.1:%d'%server.server_address[1]
        def post(path):
            request=urllib.request.Request(base+path,data=b'{}',method='POST')     # no login, no cookie
            with urllib.request.urlopen(request,timeout=5) as response:return json.load(response)
        cl=self.ws.lib.clips[0];f=self.fill(cl);ready=self.root/'api-ready'
        code="from pathlib import Path;import sys,time;Path(sys.argv[1]).write_text('ready');\nwhile True:time.sleep(.02)"
        self.runner._cmd=lambda *_:[sys.executable,'-c',code,str(ready)]
        jid=self.ws.enqueue(cl,'movement');self.start_worker()
        try:
            self.wait_for(lambda:ready.exists() and jid in self.runner.procs)
            process=self.runner.procs[jid]
            self.assertIsNone(process.poll())
            self.assertTrue(post('/api/pause')['resumable'])
            self.assertIsNotNone(process.poll());self.assertTrue(Path(f['work'],'old.json').exists())
            next_id=self.app.store.latest_for(cl.id)['id']
            post('/api/resume')
            self.wait_for(lambda:next_id in self.runner.procs)
            resumed=self.runner.procs[next_id]
            self.assertGreater(post('/api/cancel_session')['cancelled'],0)
            self.assertIsNotNone(resumed.poll())
            self.assertEqual(self.app.store.latest_for(cl.id)['state'],'stopped')
        finally:
            server.shutdown();server.server_close();service.join(3)

    def test_pause_timeout_still_queues_the_interrupted_job_and_keeps_its_clip_waiting(self):
        cl=self.ws.lib.clips[0]
        jid=self.ws.enqueue(cl,'movement');job=self.app.store.take()
        self.runner.inflight[jid]=job                        # a worker that does not let go in time
        try:
            with self.assertRaises(TimeoutError):
                with self.runner.reset_jobs([],all_clips=True,timeout=.05,mode='pause'):
                    self.fail('Must not continue while the worker still runs')
            waiting=self.app.store.latest_for(cl.id)
            self.assertNotEqual(waiting['id'],jid)
            self.assertEqual((waiting['state'],waiting['error']),('queued',clips.PAUSED),'the paused job is not lost')
            self.assertEqual(clips.status(self.ws.c,cl,waiting)['label'],'paused')
            self.assertIn(jid,self.runner.cancelled,'its old process still counts as cancelled')
            with self.assertRaises(TimeoutError):            # a second Pause does not queue it twice
                with self.runner.reset_jobs([],all_clips=True,timeout=.05,mode='pause'):pass
            self.assertEqual(self.app.store.counts().get('queued'),1)
            self.runner.resume()
            with self.runner.cond:
                self.assertIn(cl.id,self.runner._busy_clips())
                self.assertIsNone(self.app.store.take(exclude=self.runner._busy_clips()),
                                  'no new job for the clip while the old process may still run')
        finally:
            self.runner.inflight.pop(jid,None);self.runner.cancelled.discard(jid)
        resumed=self.app.store.take(exclude=self.runner._busy_clips())
        self.assertEqual((resumed['clip'],resumed['error']),(cl.id,None))

    def test_cancel_timeout_still_marks_the_job_stopped(self):
        cl=self.ws.lib.clips[0]
        jid=self.ws.enqueue(cl,'movement');job=self.app.store.take()
        self.runner.inflight[jid]=job
        try:
            with self.assertRaises(TimeoutError):
                with self.runner.reset_jobs([cl.id],timeout=.05,mode='cancel'):pass
        finally:
            self.runner.inflight.pop(jid,None);self.runner.cancelled.discard(jid)
        self.assertEqual(self.app.store.latest_for(cl.id)['state'],'stopped','autopilot must not pick a cancelled clip up again')

    def test_paused_label_clears_on_resume(self):
        cl=self.ws.lib.clips[0];ready=self.root/'label-ready'
        self.runner._cmd=lambda *_:[sys.executable,'-c',"from pathlib import Path;import sys,time;Path(sys.argv[1]).touch();time.sleep(60)",str(ready)]
        jid=self.ws.enqueue(cl,'movement');self.start_worker()
        self.wait_for(lambda:ready.exists() and jid in self.runner.procs)
        self.runner.interrupt(pause=True)
        self.assertEqual(clips.status(self.ws.c,cl,self.app.store.latest_for(cl.id))['label'],'paused')
        ready.unlink()
        self.runner.resume()
        self.assertNotEqual(clips.status(self.ws.c,cl,self.app.store.latest_for(cl.id))['label'],'paused',
                            'resumed: the job is waiting or running, not paused')
        self.wait_for(lambda:ready.exists())
        self.assertEqual(clips.status(self.ws.c,cl,self.app.store.latest_for(cl.id))['label'],'processing')
        self.runner.interrupt()


if __name__=='__main__':unittest.main()

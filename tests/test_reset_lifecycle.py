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
        source=self.root/'source';source.mkdir()
        for name in ('one','two'):(source/(name+'.mp4')).write_bytes(b'synthetic source')
        c={**config.DEFAULT_CFG,**config.APP_DEFAULTS,'source_root':'','work_root':str(self.root/'work'),
           'submit_root':str(self.root/'submit'),'marks_root':str(self.root/'marks'),'autopilot':'off',
           'ai_enabled':False,'max_attempts':3}
        self.app=App(c);self.addCleanup(self.app.store.db.close)
        user,_=self.app.users.login('reset-tester')
        self.app.users.set_source(user,str(source));self.ws=self.app.ws(user)
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
        self.threads.append(thread);thread.start();self.runner.resume(self.ws.user)

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
        jid=self.ws.enqueue(cl,'fit');self.start_worker()
        self.wait_for(lambda:heartbeat.exists() and jid in self.runner.procs)
        process=self.runner.procs[jid]
        self.app.store.enqueue(cl.id,'all',force=True,user=self.ws.user)
        result=self.ws.reset([],all_clips=True)
        self.assertTrue(result['ok']);self.assertGreaterEqual(result['stopped'],2)
        self.assertIsNotNone(process.poll())
        self.assertFalse(Path(f['work']).exists())
        last=heartbeat.read_text();threading.Event().wait(.15)
        self.assertEqual(heartbeat.read_text(),last,'Child process must also stop')
        self.assertFalse(Path(f['work']).exists(),'A late writer must not recreate the folder')
        self.assertEqual(self.app.store.counts(self.ws.user),{})
        self.assertNotIn(cl.id,self.app.store.timing())
        self.assertTrue(self.runner.is_paused(self.ws.user))
        self.runner.after.assert_not_called()

    def test_reset_covers_removed_and_taken_out_clips_but_preserves_other_members(self):
        selected=list(self.ws.lib.clips)
        for cl in selected:self.fill(cl)
        clips.set_held(selected[0].id,True,self.ws.c)
        self.ws.lib.remove(selected[1])
        jid=self.app.store.enqueue(selected[1].id,'fit',user=self.ws.user)
        self.app.store.take({self.ws.user});self.app.store.finish(jid,True)
        other_source=self.root/'other-source';other_source.mkdir();(other_source/'one.mp4').write_bytes(b'other')
        other,_=self.app.users.login('another-member');self.app.users.set_source(other,str(other_source))
        other_ws=self.app.ws(other);other_clip=other_ws.lib.clips[0]
        other_work=Path(clips.folders(other_ws.c,other_clip)['work']);other_work.mkdir(parents=True)
        (other_work/'keep').write_text('keep')
        other_jid=other_ws.enqueue(other_clip,'fit');self.runner.resume(other)
        Path(self.ws.c['work_root'],'CLAUDE_HANDOFF.json').write_text('{}')
        result=self.ws.reset([],all_clips=True)
        self.assertTrue(result['ok']);self.assertEqual(len(result['clips']),2)
        self.assertEqual(len(self.ws.lib.clips),2);self.assertFalse(self.ws.lib.removed)
        self.assertFalse(clips.held_set(self.ws.c))
        for cl in selected:
            self.assertEqual(clips.prepared_paths(self.ws.c,cl),[])
            self.assertEqual(clips.status(self.ws.c,cl)['label'],'not prepared')
            self.assertTrue(Path(cl.video).exists())
        self.assertFalse(Path(self.ws.c['work_root'],'CLAUDE_HANDOFF.json').exists())
        self.assertTrue((other_work/'keep').exists())
        self.assertEqual(self.app.store.latest_for(other_clip.id)['id'],other_jid)
        self.assertFalse(self.runner.is_paused(other))

    def test_timing_only_reset_and_dry_run_do_not_need_work_files(self):
        cl=self.ws.lib.clips[0]
        jid=self.ws.enqueue(cl,'fit');self.app.store.take({self.ws.user});self.app.store.finish(jid,True)
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
        self.ws.enqueue(cl,'fit');self.start_worker();self.assertTrue(entered.wait(3))
        results=[]
        reset_thread=threading.Thread(target=lambda:results.append(self.ws.reset([],all_clips=True)))
        reset_thread.start()
        try:
            self.wait_for(lambda:self.ws.user in self.runner.resetting)
            with self.assertRaises(HttpError):self.ws.enqueue(cl,'all')
            self.runner.resume(self.ws.user)
            self.assertTrue(self.runner.is_paused(self.ws.user))
            with patch.object(jobs.subprocess,'Popen',side_effect=AssertionError('Cancelled worker launched')):
                release.set();reset_thread.join(3)
            self.assertFalse(reset_thread.is_alive());self.assertTrue(results[0]['ok'])
            self.assertFalse(Path(clips.folders(self.ws.c,cl)['work']).exists())
            self.assertEqual(self.app.store.counts(self.ws.user),{})
        finally:
            release.set();reset_thread.join(3)

    def test_timeout_keeps_files_and_leaves_reset_all_paused(self):
        cl=self.ws.lib.clips[0];f=self.fill(cl)
        self.runner.inflight[99]={'id':99,'user':self.ws.user,'clip':cl.id}
        try:
            with self.assertRaises(TimeoutError):
                with self.runner.reset_jobs(self.ws.user,[cl.id],all_user=True,timeout=.01):
                    self.fail('Must not delete files while a worker can write')
            self.assertTrue(Path(f['work']).exists())
            self.assertTrue(self.runner.is_paused(self.ws.user))
        finally:self.runner.inflight.clear()


if __name__=='__main__':unittest.main()

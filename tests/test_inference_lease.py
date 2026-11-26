"""The shared model-inference slot (pipeline/gpu_lease.py) and the render workspace guard.
(Kept from the older test_background_jobs.py; the queue lanes and background rebuilds it also tested are gone.)"""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT)]
from common import jsave,jload
import gpu_lease
import run_pipeline as RP


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

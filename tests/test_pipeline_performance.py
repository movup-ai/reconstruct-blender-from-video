"""Scheduling and geometric invariants for CPU verification optimizations."""
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path[:0]=[str(pathlib.Path(__file__).resolve().parents[1]/'pipeline')]
import geometry_protection as GP
import gpu_lease as GPU
import run_pipeline as RP
import scene_verify as SV
from profiling import recording


class VerificationPerformanceTests(unittest.TestCase):
    def camera(self):
        return np.array([[80.,0,40.],[0,80.,30.],[0,0,1.]]),np.eye(3),np.zeros(3)

    def test_clipping_protects_visible_people_and_preserves_supported_surfaces(self):
        K,R,C=self.camera();person=np.zeros((60,80),bool);person[20:40,32:48]=True
        observation=(0,np.full((60,80),10.),K,R,C,person,np.ones((60,80),bool),np.full((60,80),np.inf))
        mesh={'name':'wall','v':[[-2.,-1.4,3.],[2.,-1.4,3.],[2.,1.4,3.],[-2.,1.4,3.]],'f':[[0,1,2,3]]}
        clipped=GP.clip_unsupported_faces(mesh,[observation]*3)
        rendered=SV.rasterize(clipped['v'],clipped['f'],K,R,C,(60,80))
        self.assertFalse(np.isfinite(rendered[25:35,35:45]).any())
        self.assertTrue(np.isfinite(rendered[5:15,5:15]).all())
        from object_fit import bits
        mesh['_observations']=[{'mask':bits(np.ones((60,80),bool)),'shape':[60,80],
                                'K':K.tolist(),'R':R.tolist(),'C':C.tolist()}]
        supported=GP.clip_unsupported_faces(mesh,[observation])
        original=SV.rasterize(mesh['v'],mesh['f'],K,R,C,(60,80))
        np.testing.assert_allclose(SV.rasterize(supported['v'],supported['f'],K,R,C,(60,80)),original)

    def test_refinement_budget_retains_unprocessed_geometry(self):
        K,R,C=self.camera();person=np.ones((60,80),bool)
        observation=(0,np.full((60,80),10.),K,R,C,person,np.ones_like(person),np.full((60,80),np.inf))
        mesh={'name':'wall','v':[[-2.,-1.4,3.],[2.,-1.4,3.],[2.,1.4,3.],[-2.,1.4,3.]],'f':[[0,1,2,3]]}
        result=GP.clip_unsupported_faces(mesh,[observation],max_triangles=0)
        self.assertTrue(result['_occlusion_clip_limited']);self.assertTrue(result['f'])
        np.testing.assert_allclose(SV.rasterize(result['v'],result['f'],K,R,C,(60,80)),
                                   SV.rasterize(mesh['v'],mesh['f'],K,R,C,(60,80)))

    def test_limited_refinement_is_reported_as_unresolved(self):
        from tests.test_metric_scene import scene,cube
        data,_=scene();mesh=cube();mesh['_occlusion_clip_limited']=True;data['meshes']=[mesh]
        report=SV.verify(data,'unused',bundles=[])
        issue=next(i for i in report['issues'] if i['kind']=='occlusion_refinement_limit')
        self.assertEqual(issue['status'],'unresolved');self.assertIn(issue['id'],report['unresolved'])

    def test_raster_culling_keeps_near_plane_and_camera_edge_intersections(self):
        K,R,C=self.camera()
        vertices=[[-.1,-.1,-.2],[.6,-.2,2.],[.1,.6,2.],[100.,0,2.],[101.,0,2.],[100.,1,2.]]
        with patch.object(SV,'_clip_near',wraps=SV._clip_near) as clip:
            depth=SV.rasterize(vertices,[[0,1,2],[3,4,5]],K,R,C,(60,80))
        self.assertTrue(np.isfinite(depth).any());self.assertEqual(clip.call_count,1)
        self.assertTrue((depth[np.isfinite(depth)]>=.01).all())
        self.assertFalse(np.isfinite(SV.rasterize(vertices,[[3,4,5]],K,R,C,(60,80))).any())

    def test_background_cpu_build_releases_inference_slot(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{'BA_GPU_LOCK':str(pathlib.Path(tmp)/'gpu.lock')}):
            @GPU.job
            def build():
                GPU.acquire();self.assertIsNotNone(GPU._active.get().file)
                def cpu_stage(*args,**kwargs):
                    self.assertIsNone(GPU._active.get().file)
                    return types.SimpleNamespace(returncode=0)
                with patch.object(RP.subprocess,'run',side_effect=cpu_stage):
                    return RP.make_set('clip','video',tmp,{},'fit',True)
            with recording() as profile:result=build()
            self.assertIn('hand-built',result)
            self.assertIn('inference_slot_wait',profile['stages'])
            self.assertGreaterEqual(profile['stages']['inference_slot_wait']['seconds'],0.)


if __name__=='__main__':unittest.main()

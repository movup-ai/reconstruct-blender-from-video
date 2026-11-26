"""Character opt-in, renderer independence, exact protection and optional exposure."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'pipeline'),str(ROOT/'tests')]
from common import DEFAULT_CFG,normalize_backends,jsave,jload
from test_props_geometry import fixture
import dependencies as D
import geometry_protection as G
import render_settings as R
import source_exposure as E


class SkillRegressionTests(unittest.TestCase):
    def test_l3_requires_explicit_character_choice(self):
        self.assertEqual(DEFAULT_CFG['character_style'],'sample')
        self.assertFalse(DEFAULT_CFG['source_exposure']);self.assertEqual(DEFAULT_CFG['render_engine'],'workbench')
        for style in ('sample','master','l3'):
            result=normalize_backends({'character_style':style,'previs_standard':'l3'})
            self.assertEqual(result['character_style'],style)
            self.assertEqual(result['previs_standard'],'l3' if style=='l3' else 'legacy')

    def test_renderer_choice_is_independent_of_character(self):
        for style in ('sample','l3','master'):
            self.assertEqual(R.engine({'character_style':style}),'workbench')
            self.assertEqual(R.engine({'character_style':style,'render_engine':'eevee'}),'eevee')
        self.assertEqual(R.engine({'render_engine':'workbench'},{'spec':'lzhd1'}),'eevee')
        with self.assertRaises(ValueError):R.engine({'render_engine':'unknown'})

    def test_exact_l3_parts_include_the_full_torso_and_head(self):
        from l3_blender import descriptor
        parts=G.character_parts('l3')
        self.assertEqual(set(parts),set(descriptor()['parts']))
        self.assertEqual(len(parts),10)
        self.assertGreater(np.ptp(np.asarray(parts['chest']['v'])[:,0]),.1)
        data,world=fixture(1)
        for character in data['chars'].values():character['style']='l3'
        with patch('props.bar',side_effect=AssertionError('L3 fell back to a limb proxy')):
            depth=G.rendered_people_depth(data,0,world.cam.K(),world.R_cw(0),world.C_at(0),(60,80))
        self.assertTrue(np.isfinite(depth).any())

    def test_render_and_exposure_settings_do_not_invalidate_model_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'clip.mp4';video.write_bytes(b'video')
            for name in ('work','marks','out'):(root/name).mkdir()
            def state(config):return D.inputs(str(video),config,root/'work',root/'marks',root/'out')
            normal=state({});self.assertEqual(normal,state({'render_engine':'eevee','source_exposure':True}))
            self.assertNotEqual(normal['fit'],state({'character_style':'l3'})['fit'])

    def test_disabled_exposure_removes_saved_curve_without_decoding(self):
        data={'source_exposure':{'stops':[-1.]}}
        with patch('common.read_frames',side_effect=AssertionError('Disabled exposure decoded video')):
            E.attach(data,'unused','unused',{})
        self.assertNotIn('source_exposure',data)

    def test_brief_ambiguous_exposure_does_not_flash_to_neutral(self):
        samples=np.full((30,48),.4);samples[10:]*=.25
        samples[15,:24]=.4
        result=E.from_tiles(samples,[(0,30)])
        self.assertTrue(result['shots'][0]['applied'])
        np.testing.assert_allclose(result['stops'][14:17],[-2.,-2.,-2.])

    def test_long_ambiguous_exposure_rejects_the_shot(self):
        samples=np.full((30,48),.4);samples[5:]*=.25;samples[10:18,:24]=.4
        result=E.from_tiles(samples,[(0,30)])
        self.assertFalse(result['shots'][0]['applied']);self.assertEqual(result['shots'][0]['reason'],'ambiguous brightness interval')
        np.testing.assert_allclose(result['stops'],0.)

    def test_optional_exposure_reuses_unchanged_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'clip.mp4';video.write_bytes(b'video')
            data={'frames':4,'shots':[{'frames':[0,3]}]};pose={'people':[]};info={'y0':0,'Hpic':16,'fps':30}
            result={'stops':[-1.]*4,'shots':[]}
            with patch.object(E,'measure',return_value=result) as measure:
                E.attach(data,str(video),tmp,{'source_exposure':True},pose,info,[None]*4)
                E.attach(data,str(video),tmp,{'source_exposure':True},pose,info,[None]*4)
                self.assertEqual(measure.call_count,1);self.assertTrue(data['source_exposure']['cache_hit'])
                pose['people']=[{'frames':{'0':[[0,0,1.]]*17}}]
                E.attach(data,str(video),tmp,{'source_exposure':True},pose,info,[None]*4);self.assertEqual(measure.call_count,2)

    def test_blender_handoff_honors_engine_and_removes_stale_exposure(self):
        import run_pipeline as P
        for style in ('sample','l3'):
            for engine in ('workbench','eevee'):
                with tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);work=root/'work';work.mkdir();sub=root/'sub';data,_=fixture(1)
                    data.update(name='clip',work_dir=str(work),submit_dir=str(sub/'clip'),source_exposure={'stops':[-1.]})
                    for ch in data['chars'].values():ch['style']=style
                    dp=work/'scene_data.json';jsave(str(dp),data)
                    def builder(*args,**kwargs):
                        saved=jload(str(dp));self.assertEqual(saved['render_engine'],engine);self.assertNotIn('source_exposure',saved)
                        (work/'render_done.txt').write_text('frames 1');return types.SimpleNamespace(returncode=0)
                    with patch.object(P,'blender_exe',return_value='fake'),patch.object(P.subprocess,'run',side_effect=builder),patch('colour_card.apply'):
                        P.blender_render({'submit_root':str(sub),'render_engine':engine},str(work),str(dp),1)

    def test_settings_validate_new_controls(self):
        from agent import config
        with tempfile.TemporaryDirectory() as tmp,patch.object(config,'CFG_PATH',str(Path(tmp)/'config.json')):
            result=config.save({'character_style':'l3','render_engine':'workbench','source_exposure':False})
            self.assertEqual(result['character_style'],'l3');self.assertEqual(result['render_engine'],'workbench')
            for setting in ('character_style','render_engine'):
                with self.assertRaises(ValueError):config.save({setting:'unknown'})

    def test_source_background_colors_survive_character_selection(self):
        import colour_card
        from test_metric_scene import cube
        for style in ('sample','l3'):
            data,_=fixture(1);mesh=cube();mesh.update(colour='plant_green',kind='wall',cls='wall')
            data.update(meshes=[mesh],palette={'plant_green':[30,90,45]},background_policy='video',previs_standard='legacy')
            for ch in data['chars'].values():ch['style']=style
            colour_card.apply(data,log=None)
            self.assertEqual(data['palette'][mesh['colour']],[30,90,45])


if __name__=='__main__':unittest.main()

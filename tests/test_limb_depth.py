"""Depth correction regressions, including an explicit inference-free job path."""
import copy
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
import shutil
import subprocess
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
sys.path.insert(0,str(ROOT))
import limb_depth as LD
import dependencies as DEP
import run_pipeline as RP
from rig import Rig
from fit_people import assemble, NP
from common import jsave, jload
from agent import review


def fixture(n=13):
    rig=Rig(1.75,'sample');q=np.zeros(NP);q[3]=.94
    keys,_=assemble(rig,{0:q,n-1:q},n,0)
    a=dict(copy.deepcopy(keys),height=1.75,style='sample',track_id=1,colour='red')
    b=dict(copy.deepcopy(keys),height=1.75,style='sample',track_id=2,colour='blue')
    for root in a['root']:root[1]=.15
    scene=dict(frames=n,fps=30,res=[640,480],lens=35,sensor=36,cam=[[0,-5,1.5,90,0,0]]*n,chars={'A':a,'B':b})
    row=dict(person='1',other='2',part='left_forearm',other_part='left_forearm',order='front',start=4,end=8,margin_m=.08)
    return scene,row


def wrist_gap(data,frame=6):
    out=[]
    for c in data['chars'].values():
        root=c['root'][frame];r=Rig(c['height'],c['style'])
        w=r.fk(root[:3],root[3],{j:v[frame] for j,v in c['rot'].items()},c['pelvis_z'][frame])
        out.append(w['wrist_L'][1][1])
    return out[1]-out[0]


class LimbDepthTests(unittest.TestCase):
    def test_front_order_reverses_wrong_depth_with_fixed_geometry(self):
        data,row=fixture();before=copy.deepcopy(data)
        self.assertLess(wrist_gap(data),0)
        report=LD.apply(data,[row])
        self.assertGreater(wrist_gap(data),.04)
        self.assertEqual(report['unresolved'],[])
        self.assertEqual(data['cam'],before['cam'])
        for k,c in data['chars'].items():
            for field in ('root','pelvis_z','height','visible'): self.assertEqual(c[field],before['chars'][k][field])
            self.assertEqual(c['rot']['hip_L'],before['chars'][k]['rot']['hip_L'])

    def test_repeat_is_idempotent_and_removal_restores_baseline(self):
        data,row=fixture();before=copy.deepcopy(data)
        LD.apply(data,[row]);first=copy.deepcopy(data['chars'])
        LD.apply(data,[row]);self.assertEqual(first,data['chars'])
        LD.apply(data,[])
        for k,c in data['chars'].items():self.assertEqual(c['rot'],before['chars'][k]['rot'])

    def test_behind_order_and_unaffected_frames(self):
        data,row=fixture(25);row.update(order='behind',person='2',other='1',start=10,end=14)
        before=copy.deepcopy(data)
        LD.apply(data,[row]);self.assertGreater(wrist_gap(data,12),.04)
        for k,c in data['chars'].items():
            self.assertEqual(c['rot']['shoulder_L'][0],before['chars'][k]['rot']['shoulder_L'][0])
            self.assertEqual(c['rot']['shoulder_L'][-1],before['chars'][k]['rot']['shoulder_L'][-1])

    def test_missing_or_invisible_tracks_are_skipped(self):
        data,row=fixture();data['chars']['B']['visible']=[False]*13
        report=LD.apply(data,[row]);self.assertEqual(report['solved_frames'],0);self.assertGreater(report['skipped_frames'],0)

    def test_invalid_ranges_and_conflicts(self):
        _,row=fixture()
        for change in ({'end':100},{'start':-1},{'start':4.5},{'start':True},{'margin_m':float('nan')},{'margin_m':0},{'part':'torso'},{'other':'1'},{'other':'7'},{'order':'unknown'}):
            with self.subTest(change=change),self.assertRaises(ValueError):LD.validate([dict(row,**change)],{'1','2'},13)
        with self.assertRaises(ValueError):LD.validate([row,dict(row,order='behind')])

    def test_review_roundtrip(self):
        _,row=fixture();rv={'people':{'1':{},'2':{}},'limb_depth':[row]}
        self.assertEqual(review.validate(rv)['limb_depth'],[row])
        self.assertEqual(review.validate({'people':{}})['limb_depth'],[])

    def test_dependency_key_does_not_invalidate_fit_environment_or_props(self):
        _,row=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            a=DEP.inputs('missing.mp4',{},tmp,tmp,tmp,review={})
            b=DEP.inputs('missing.mp4',{},tmp,tmp,tmp,review={'limb_depth':[row]})
        for key in ('fit','environment','props'):self.assertEqual(a[key],b[key])
        self.assertNotEqual(a['limb_depth'],b['limb_depth'])

    def test_depth_job_never_falls_back_to_inference_or_overwrites_pending_edits(self):
        data,row=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            jsave(str(Path(tmp)/'scene_data.json'),data)
            original={'fit':'stale-fit','environment':'pending-env','props':'pending-props'}
            jsave(str(Path(tmp)/'build_state.json'),original)
            with patch.object(RP,'held_place',return_value=[]) as held,patch.object(LD,'preview') as preview, \
                 patch.object(RP,'fit_all',side_effect=AssertionError('Body fit reran')), \
                 patch.object(RP,'held_detect',side_effect=AssertionError('Prop detection reran')), \
                 patch.object(RP,'make_set',side_effect=AssertionError('Set rebuilt')), \
                 patch.object(RP,'read_frames',side_effect=AssertionError('Full video was reloaded')), \
                 patch.object(RP.gpu_lease,'acquire',side_effect=AssertionError('Inference lock acquired')):
                RP.depth_only('clip','video',tmp,{'_review':{'limb_depth':[row]}},types.SimpleNamespace(stage='depth'),time.time())
            held.assert_called_once();preview.assert_called_once()
            state=jload(str(Path(tmp)/'build_state.json'))
            for k,v in original.items():self.assertEqual(state[k],v)
            self.assertEqual(state['limb_depth'],DEP.digest([row]))
            self.assertGreater(wrist_gap(jload(str(Path(tmp)/'scene_data.json'))),.04)

    def test_depth_job_missing_scene_fails_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(RuntimeError,'Prepare'):
            RP.depth_only('clip','video',tmp,{},None,time.time())

    def test_finish_edit_path_reuses_scene_for_pending_depth_ranges(self):
        data,row=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            jsave(str(Path(tmp)/'scene_data.json'),data)
            current={'fit':'same','environment':'same','props':'same','limb_depth':DEP.digest([row])}
            previous=dict(current,limb_depth=DEP.digest([]))
            with patch.object(LD,'preview'),patch.object(RP,'held_place',return_value=[]), \
                 patch.object(RP,'held_detect',side_effect=AssertionError('Prop inference reran')), \
                 patch.object(RP,'make_set',side_effect=AssertionError('Set rebuilt')), \
                 patch.object(RP,'fit_all',side_effect=AssertionError('Body fit reran')):
                RP.rebuild_edits('clip','video',tmp,{'scene_verify':False,'_review':{'limb_depth':[row]}},types.SimpleNamespace(stage='fit'),time.time(),previous,current)
            self.assertGreater(wrist_gap(jload(str(Path(tmp)/'scene_data.json'))),.04)

    @unittest.skipUnless(shutil.which('node'),'Node required for UI controls')
    def test_timeline_controls_execute_and_preserve_ranges(self):
        script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
global.window=global;global.esc=s=>s;global.toast=s=>{throw Error(s)};
class Host {
 constructor(index=null){this.nodes={};this.dataset={row:index};}
 set innerHTML(html){this.html=html;this.nodes={};this.rows=[...html.matchAll(/data-row="(\d+)"/g)].map(m=>new Host(m[1]));}
 querySelector(s){return this.nodes[s] ||= {value:'',dataset:{}};}
 querySelectorAll(s){if(s==='[data-row]')return this.rows;
  return ['person','part','order','other','other_part','start','end','margin_m'].map(k=>{
   const n=this.querySelector('[data-key="'+k+'"]');n.dataset.key=k;return n;});}
}
vm.runInThisContext(fs.readFileSync('web/limb_depth.js','utf8'));
let changed=0,applied=0;const host=new Host(),video={currentTime:1,addEventListener(){},pause(){this.paused=true}};
const editor=new LimbDepthEditor(host,video,()=>changed++,()=>applied++);
const info={frames:100,fps:25,people:[{id:1},{id:2}]};
editor.load({},info);host.querySelector('[data-add]').onclick();
assert.equal(editor.value().limb_depth[0].start,25);
let row=host.rows[0];const order=row.querySelector('[data-key="order"]');order.value='behind';order.onchange();
const gap=row.querySelector('[data-key="margin_m"]');gap.value='12';gap.onchange();
video.currentTime=2;row.querySelector('[data-end]').onclick();
assert.equal(editor.value().limb_depth[0].end,50);assert.equal(editor.value().limb_depth[0].margin_m,.12);
host.rows[0].querySelector('[data-go]').onclick();assert(video.paused);assert(video.currentTime>1&&video.currentTime<1.01);
const saved=editor.value();editor.load(saved,info);assert.equal(editor.items[0].order,'behind');
host.querySelector('[data-apply]').onclick();assert.equal(applied,1);
editor.report({...info,limb_depth_report:{solved_frames:12,unresolved:[{}],skipped_frames:0}});
assert(host.querySelector('[data-report]').textContent.includes('1 unresolved'));
host.rows[0].querySelector('[data-remove]').onclick();assert.equal(editor.items.length,0);assert(changed>=4);
'''
        subprocess.run(['node','-e',script],cwd=ROOT,check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main()

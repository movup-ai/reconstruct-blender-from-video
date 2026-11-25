"""Run real card/table actions and detail controls without a browser or network."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ProcessingControlsUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'),'Node is needed for JavaScript control checks')
    def test_running_clip_has_cancel_and_all_stages_have_active_pause_controls(self):
        script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert'),root=process.argv[1],elements={};
const calls=[];const context=vm.createContext({localStorage:{removeItem(){}},
 document:{getElementById(id){return elements[id] ||= {}; }},calls});
vm.runInContext(fs.readFileSync(root+'/web/common.js','utf8'),context);
const index=fs.readFileSync(root+'/web/index.html','utf8');
const actions=index.slice(index.indexOf('function clipActions('),index.indexOf('function renderTable('));
vm.runInContext(actions,context);
context.clip={id:'abc',name:'film_clip',status:{job:{stage:'fit',state:'running'}}};
assert(vm.runInContext('clipActions(clip).queue',context).includes('data-cancel-job="abc"'));
const detail=fs.readFileSync(root+'/web/clip.html','utf8');
const begin=detail.indexOf("  const finishQueue = $('finishqueue');");
const end=detail.indexOf("  $('geometry-actions')",begin);
assert(begin>=0&&end>begin);
vm.runInContext("const CID='abc';function refresh(){};processingControl=async(url)=>calls.push(url);function wire(){"+detail.slice(begin,end)+'}',context);
(async()=>{
 for(const stage of ['fit','background','ai','all']){
  context.INFO={job:{stage,state:'running'},paused:false};vm.runInContext('wire()',context);
  assert.equal(elements.finishqueue.hidden,false);assert.equal(elements['cancel-processing'].hidden,false);
  await elements.finishqueue.onclick();assert.equal(calls.pop(),'/api/pause');
  await elements['cancel-processing'].onclick();assert.equal(calls.pop(),'/api/clip/abc/cancel');
 }
 context.INFO={job:{stage:'all',state:'queued'},paused:true};vm.runInContext('wire()',context);
 assert.equal(elements.finishqueue.textContent,'Resume processing');await elements.finishqueue.onclick();assert.equal(calls.pop(),'/api/resume');
 context.INFO={job:{stage:'fit',state:'running'},paused:true};vm.runInContext('wire()',context);
 await elements.finishqueue.onclick();assert.equal(calls.pop(),'/api/pause');
 context.INFO={job:{stage:'fit',state:'stopped'},paused:true};vm.runInContext('wire()',context);
 assert.equal(elements.finishqueue.hidden,true);assert.equal(elements['cancel-processing'].hidden,true);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result=subprocess.run(['node','-e',script,str(ROOT)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()

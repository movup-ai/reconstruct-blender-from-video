"""Run real card/table actions and detail controls without a browser or network."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ProcessingControlsUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'),'Node is needed for JavaScript control checks')
    def test_pausing_keeps_done_and_waiting_statuses_and_only_marks_interrupted_row(self):
        script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('web/index.html','utf8');
const context=vm.createContext({});
const start=html.indexOf('function render() {'),end=html.indexOf('  // ---- filter tabs',start);
assert(start>=0&&end>start);
vm.runInContext(html.slice(start,end)+'return cs;}',context);
context.DATA={queue:{paused:true},clips:[
 {status:{label:'done',group:'done'}},
 {status:{label:'waiting',group:'busy'}},
 {status:{label:'paused',group:'busy'}}]};
const before=JSON.stringify(context.DATA.clips);vm.runInContext('render()',context);
assert.equal(JSON.stringify(context.DATA.clips),before);
const step=html.slice(html.indexOf('const STEP_OF ='),html.indexOf('function clipActions('));
vm.runInContext('function fmtDur(s){return s+" s";}function progressBar(){return "";}'+step,context);
context.waiting={status:{job:{stage:'movement',state:'queued'}}};
assert(vm.runInContext("stepCell(waiting,'movement')",context).includes('>waiting<'));
context.waiting.status.job.error='Paused by user';
assert(vm.runInContext("stepCell(waiting,'movement')",context).includes('>paused<'));
context.cancelledButDone={status:{done:true,job:{stage:'movement',state:'stopped'}},took:{steps:{movement:{state:'done',ran:60}}}};
assert(vm.runInContext("stepCell(cancelledButDone,'movement')",context).includes('&#10003; 60 s'),'a valid zip wins over a later cancel');
context.failed={status:{job:{stage:'movement',state:'failed'}},took:{steps:{movement:{state:'failed',ran:5}}}};
assert(vm.runInContext("stepCell(failed,'movement')",context).includes('failed after 5 s'));
context.fresh={status:{}};
assert(vm.runInContext("stepCell(fresh,'movement')",context).includes('–'));
'''
        result=subprocess.run(['node','-e',script],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(shutil.which('node'),'Node is needed for JavaScript control checks')
    def test_running_clip_has_cancel_and_all_stages_have_active_pause_controls(self):
        script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert'),root=process.argv[1],elements={};
const calls=[];const context=vm.createContext({localStorage:{removeItem(){}},
 document:{getElementById(id){return elements[id] ||= {}; }},calls});
vm.runInContext(fs.readFileSync(root+'/web/common.js','utf8'),context);
const index=fs.readFileSync(root+'/web/index.html','utf8');
const actions=index.slice(index.indexOf('const downloadUrl ='),index.indexOf('\n',index.indexOf('const downloadUrl =')))+'\n'+
  index.slice(index.indexOf('function clipActions('),index.indexOf('function renderTable('));
vm.runInContext(actions,context);
context.clip={id:'abc',name:'film_clip',status:{job:{stage:'movement',state:'running'}}};
assert(vm.runInContext('clipActions(clip).queue',context).includes('data-cancel-job="abc"'));
context.clip={id:'abc',name:'film_clip',package:'film-1',status:{done:true,label:'done',job:{stage:'movement',state:'done'}}};
assert(vm.runInContext('clipActions(clip).queue',context).includes('href="/download/film_clip"'),'a done clip offers its zip');
context.clip={id:'abc',name:'film_clip',status:{label:'new'}};
assert(vm.runInContext('clipActions(clip).remove',context).includes('data-skip="abc"'),'one button skips a video');
assert.equal(vm.runInContext('clipActions(clip).queue',context),'','no second "take out" button');
context.clip={id:'abc',name:'film_clip',status:{label:'skipped',skipped:true}};
assert(vm.runInContext('clipActions(clip).queue',context).includes('data-include="abc"'),'a skipped video offers Include');
assert.equal(vm.runInContext('clipActions(clip).remove',context),'','a skipped video has no skip button');
context.clip={id:'abc',name:'film_clip',status:{job:{stage:'movement',state:'running'}}};
assert(vm.runInContext('clipActions(clip).remove',context).includes('disabled'),'a running video cannot be skipped');
const detail=fs.readFileSync(root+'/web/clip.html','utf8');
const begin=detail.indexOf("  const finishQueue = $('finishqueue');");
const end=detail.indexOf("  // ---- end of the processing controls",begin);
assert(begin>=0&&end>begin);
vm.runInContext("const CID='abc';function refresh(){};processingControl=async(url)=>calls.push(url);function wire(){"+detail.slice(begin,end)+'}',context);
(async()=>{
 for(const stage of ['movement','background']){
  context.INFO={job:{stage,state:'running'},paused:false};vm.runInContext('wire()',context);
  assert.equal(elements.finishqueue.hidden,false);assert.equal(elements['cancel-processing'].hidden,false);
  await elements.finishqueue.onclick();assert.equal(calls.pop(),'/api/pause');
  await elements['cancel-processing'].onclick();assert.equal(calls.pop(),'/api/clip/abc/cancel');
 }
 context.INFO={job:{stage:'movement',state:'queued'},paused:true};vm.runInContext('wire()',context);
 assert.equal(elements.finishqueue.textContent,'Resume processing');await elements.finishqueue.onclick();assert.equal(calls.pop(),'/api/resume');
 context.INFO={job:{stage:'movement',state:'running'},paused:true};vm.runInContext('wire()',context);
 await elements.finishqueue.onclick();assert.equal(calls.pop(),'/api/pause');
 context.INFO={job:{stage:'movement',state:'stopped'},paused:true};vm.runInContext('wire()',context);
 assert.equal(elements.finishqueue.hidden,true);assert.equal(elements['cancel-processing'].hidden,true);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result=subprocess.run(['node','-e',script,str(ROOT)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(shutil.which('node'),'Node is needed for JavaScript control checks')
    def test_processing_control_failure_is_caught_and_buttons_come_back(self):
        script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert'),root=process.argv[1];
const buttons=[{disabled:false}],toasts=[];
const context=vm.createContext({localStorage:{removeItem(){}},setTimeout(){},clearTimeout(){},
 document:{getElementById(){return {}},querySelectorAll(){return buttons;}},
 fetch:async()=>({status:409,ok:false,statusText:'Conflict',json:async()=>({error:'A worker did not stop'})})});
vm.runInContext(fs.readFileSync(root+'/web/common.js','utf8'),context);
(async()=>{
 const result=await vm.runInContext("processingControl('/api/pause','Pausing…')",context);
 assert.equal(result,null);assert.equal(buttons[0].disabled,false);
 assert.equal(vm.runInContext('processingBusy',context),false);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result=subprocess.run(['node','-e',script,str(ROOT)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()

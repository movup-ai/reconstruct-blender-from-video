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
const step=html.slice(html.indexOf('function statusCell('),html.indexOf('function clipActions('));
vm.runInContext('function esc(s){return String(s);}function fmtDur(s){return s+" s";}function progressBar(){return "<bar>";}'+step,context);
context.waiting={status:{label:'waiting'}};
assert(vm.runInContext("statusCell(waiting)",context).includes('>waiting<'));
context.waiting.status.label='paused';
assert(vm.runInContext("statusCell(waiting)",context).includes('>paused<'));
context.cancelledButDone={status:{label:'done',done:true},took:{done:{machine:60,end:0},last:{state:'stopped',ran:5}}};
assert(vm.runInContext("statusCell(cancelledButDone)",context).includes('&#10003; 60 s'),'a valid zip wins over a later cancel');
context.failed={status:{label:'failed'},took:{last:{state:'failed',ran:5}}};
assert(vm.runInContext("statusCell(failed)",context).includes('failed after 5 s'));
context.running={status:{label:'processing'},took:{last:{state:'running',ran:9}}};
context.prog={progress:40,step:'finding objects and surfaces',since:9};
assert(vm.runInContext("statusCell(running,prog)",context).includes('running 9 s</div><bar>'));
context.fresh={status:{label:'new'}};
assert(vm.runInContext("statusCell(fresh)",context).includes('–'));
context.mv={status:{label:'done',group:'done'}};
assert(vm.runInContext("statusChip(mv)",context).includes('>Done<'),'one chip: the clip\'s status');
context.mv.status={label:'skipped',group:'skip'};
assert(vm.runInContext("statusChip(mv)",context).includes('>Skipped<'));
'''
        result=subprocess.run(['node','-e',script],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(shutil.which('node'),'Node is needed for JavaScript control checks')
    def test_running_clip_has_cancel_and_the_clip_page_has_pause_and_cancel(self):
        script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert'),root=process.argv[1],elements={};
const calls=[];const context=vm.createContext({localStorage:{removeItem(){}},
 document:{getElementById(id){return elements[id] ||= {}; }},calls});
vm.runInContext(fs.readFileSync(root+'/web/common.js','utf8'),context);
const index=fs.readFileSync(root+'/web/index.html','utf8');
const actions=index.slice(index.indexOf('function clipActions('),index.indexOf('function renderTable('));
vm.runInContext(actions,context);
const one=()=>{const a=vm.runInContext('clipActions(clip).action',context);assert.equal((a.match(/<button/g)||[]).length,1,'exactly one button');return a;};
// a job running or waiting: Cancel (never a disabled skip next to it)
for(const state of ['running','queued']){
 context.clip={id:'abc',name:'film_clip',status:{label:'done',job:{stage:'run',state}}};
 const a=one();assert(a.includes('data-cancel-job="abc"')&&a.includes('>Cancel<'),state);
 assert(!a.includes('disabled'));
}
// not done: Skip, labelled, and no download links (the zips are in SUBMIT and on the clip page)
for(const status of [{label:'new'},{label:'failed',job:{stage:'run',state:'failed'}},{label:'outdated'}]){
 context.clip={id:'abc',name:'film_clip',package:'film-1',status};
 const a=one();assert(a.includes('data-skip="abc"')&&a.includes('>Skip<'));
 assert(!a.includes('/download/'),'no download buttons on the list');
}
// done: nothing left to skip, no button at all
context.clip={id:'abc',name:'film_clip',package:'film-1',status:{label:'done',done:true,job:{stage:'run',state:'done'}}};
assert.equal(vm.runInContext('clipActions(clip).action',context),'','a finished video has no button');
// skipped: Include
context.clip={id:'abc',name:'film_clip',status:{label:'skipped',skipped:true}};
assert(one().includes('data-include="abc"'),'a skipped video offers Include');
const detail=fs.readFileSync(root+'/web/clip.html','utf8');
const begin=detail.indexOf("  {                                                           // the run's Pause / Cancel");
const end=detail.indexOf("  // ---- end of the processing controls",begin);
assert(begin>=0&&end>begin);
vm.runInContext("const CID='abc';function refresh(){};processingControl=async(url)=>calls.push(url);function wire(){"+detail.slice(begin,end)+'}',context);
(async()=>{
 context.INFO={job:{stage:'run',state:'running'},paused:false};vm.runInContext('wire()',context);
 assert.equal(elements['run-pause'].hidden,false);assert.equal(elements['run-cancel'].hidden,false);
 await elements['run-pause'].onclick();assert.equal(calls.pop(),'/api/pause');
 await elements['run-cancel'].onclick();assert.equal(calls.pop(),'/api/clip/abc/cancel');
 context.INFO={job:{stage:'run',state:'queued'},paused:true};vm.runInContext('wire()',context);
 assert.equal(elements['run-pause'].textContent,'Resume');await elements['run-pause'].onclick();assert.equal(calls.pop(),'/api/resume');
 context.INFO={job:{stage:'run',state:'running'},paused:true};vm.runInContext('wire()',context);
 await elements['run-pause'].onclick();assert.equal(calls.pop(),'/api/pause');
 context.INFO={job:{stage:'run',state:'stopped'},paused:true};vm.runInContext('wire()',context);
 assert.equal(elements['run-pause'].hidden,true);assert.equal(elements['run-cancel'].hidden,true);
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

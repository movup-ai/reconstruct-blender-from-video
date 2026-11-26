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
const step=html.slice(html.indexOf('const BG_GROUP ='),html.indexOf('function clipActions('));
vm.runInContext('function esc(s){return String(s);}function fmtDur(s){return s+" s";}function progressBar(){return "<bar>";}'+step,context);
// Movement: its own status (the label comes from its own latest job)
context.waiting={status:{label:'waiting'}};
assert(vm.runInContext("stepCell(waiting,'movement')",context).includes('>waiting<'));
context.waiting.status.label='paused';
assert(vm.runInContext("stepCell(waiting,'movement')",context).includes('>paused<'));
context.cancelledButDone={status:{label:'done',done:true},took:{steps:{movement:{state:'done',ran:60}}}};
assert(vm.runInContext("stepCell(cancelledButDone,'movement')",context).includes('&#10003; 60 s'),'a valid zip wins over a later cancel');
context.failed={status:{label:'failed'},took:{steps:{movement:{state:'failed',ran:5}}}};
assert(vm.runInContext("stepCell(failed,'movement')",context).includes('failed after 5 s'));
context.fresh={status:{label:'new'}};
assert(vm.runInContext("stepCell(fresh,'movement')",context).includes('–'));
// Background: a column of its own; a Background run never shows in the Movement column
context.bgrun={status:{label:'done',done:true,background:{state:'processing'}},took:{steps:{movement:{state:'done',ran:60},background:{state:'running',ran:9}}}};
context.prog={stage:'background',progress:40,step:'finding objects',since:9};
assert(vm.runInContext("stepCell(bgrun,'background',prog)",context).includes('running 9 s</div><bar>'));
assert(vm.runInContext("stepCell(bgrun,'movement',prog)",context).includes('&#10003; 60 s'),'Movement stays done while Background runs');
context.bgdone={status:{label:'done',done:true,background:{state:'done'}},took:{steps:{background:{state:'done',ran:120}}}};
assert(vm.runInContext("stepCell(bgdone,'background')",context).includes('&#10003; 120 s'));
context.ready={status:{label:'done',done:true,background:{state:'none'}}};
assert(vm.runInContext("stepCell(ready,'background')",context).includes('ready'));
context.notyet={status:{label:'new',background:{state:'none'}}};
assert(vm.runInContext("stepCell(notyet,'background')",context).includes('–'));
assert.equal(vm.runInContext("bgChip(notyet)",context),'','no Background chip before there is one');
assert(vm.runInContext("bgChip(bgdone)",context).includes('Background done'));
assert(vm.runInContext("bgChip(bgrun)",context).includes('Background running'));
context.mv={status:{label:'done',group:'done'}};
assert(vm.runInContext("mvChip(mv)",context).includes('Movement done'),'each chip names its stage');
context.mv.status={label:'skipped',group:'skip'};
assert(vm.runInContext("mvChip(mv)",context).includes('>Skipped<'));
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
const line=key=>index.slice(index.indexOf(key),index.indexOf('\n',index.indexOf(key)))+'\n';
const actions=line('const bgState =')+
  index.slice(index.indexOf('function clipActions('),index.indexOf('function renderTable('));
vm.runInContext(actions,context);
const one=()=>{const a=vm.runInContext('clipActions(clip).action',context);assert.equal((a.match(/<button/g)||[]).length,1,'exactly one button');return a;};
// a job running or waiting, in either stage: Cancel (never a disabled skip next to it)
for(const stage of ['movement','background'])for(const state of ['running','queued']){
 context.clip={id:'abc',name:'film_clip',status:{label:'done',job:{stage,state}}};
 const a=one();assert(a.includes('data-cancel-job="abc"')&&a.includes('>Cancel<'),stage+' '+state);
 assert(!a.includes('disabled'));
}
// idle (new or done): Skip, labelled, and no download links (the zips are in SUBMIT and on the clip page)
for(const status of [{label:'new'},{label:'done',done:true,job:{stage:'movement',state:'done'},background:{state:'none'}}]){
 context.clip={id:'abc',name:'film_clip',package:'film-1',status};
 const a=one();assert(a.includes('data-skip="abc"')&&a.includes('>Skip<'));
 assert(!a.includes('/download/'),'no download buttons on the list');
}
// both stages done: nothing left to skip, no button at all
context.clip={id:'abc',name:'film_clip',package:'film-1',status:{label:'done',done:true,job:{stage:'background',state:'done'},background:{state:'done'}}};
assert.equal(vm.runInContext('clipActions(clip).action',context),'','a finished video has no button');
// skipped: Include
context.clip={id:'abc',name:'film_clip',status:{label:'skipped',skipped:true}};
assert(one().includes('data-include="abc"'),'a skipped video offers Include');
const detail=fs.readFileSync(root+'/web/clip.html','utf8');
const begin=detail.indexOf("  for (const stage of ['movement', 'background']) {           // each stage's row has its own Pause / Cancel");
const end=detail.indexOf("  // ---- end of the processing controls",begin);
assert(begin>=0&&end>begin);
vm.runInContext("const CID='abc';function refresh(){};processingControl=async(url)=>calls.push(url);function wire(){"+detail.slice(begin,end)+'}',context);
const other={movement:'background',background:'movement'};
(async()=>{
 for(const stage of ['movement','background']){
  context.INFO={job:{stage,state:'running'},paused:false};vm.runInContext('wire()',context);
  assert.equal(elements[stage+'-pause'].hidden,false);assert.equal(elements[stage+'-cancel'].hidden,false);
  assert.equal(elements[other[stage]+'-pause'].hidden,true,'the other stage\'s row has no Pause');
  assert.equal(elements[other[stage]+'-cancel'].hidden,true,'nor Cancel');
  await elements[stage+'-pause'].onclick();assert.equal(calls.pop(),'/api/pause?stage='+stage,'pauses only this job\'s stage');
  await elements[stage+'-cancel'].onclick();assert.equal(calls.pop(),'/api/clip/abc/cancel');
 }
 context.INFO={job:{stage:'movement',state:'queued'},paused:true};vm.runInContext('wire()',context);
 assert.equal(elements['movement-pause'].textContent,'Resume');await elements['movement-pause'].onclick();assert.equal(calls.pop(),'/api/resume?stage=movement');
 context.INFO={job:{stage:'movement',state:'running'},paused:true};vm.runInContext('wire()',context);
 await elements['movement-pause'].onclick();assert.equal(calls.pop(),'/api/pause?stage=movement');
 context.INFO={job:{stage:'movement',state:'stopped'},paused:true};vm.runInContext('wire()',context);
 assert.equal(elements['movement-pause'].hidden,true);assert.equal(elements['movement-cancel'].hidden,true);
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

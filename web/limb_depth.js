/* Ordinal limb depth ranges use source frame numbers and persistent track IDs. */
class LimbDepthEditor {
  constructor(host, video, changed, apply) {
    this.host=host; this.video=video; this.changed=changed; this.apply=apply;
    video.addEventListener('timeupdate',()=>this.sync());
  }
  load(review, info) { this.items=structuredClone(review?.limb_depth||[]); this.info=info; this.render(); }
  value() { return {limb_depth:structuredClone(this.items||[])}; }
  sync() { const stamp=this.host.querySelector('[data-current]'); if(stamp) stamp.textContent=`Current frame: ${this.frame()}`; }
  frame() { return Math.max(0,Math.min((this.info.frames||1)-1,Math.floor(this.video.currentTime*(this.info.fps||30)+.01))); }
  report(info) {
    this.info=info;
    const report=info.limb_depth_report, node=this.host.querySelector('[data-report]');
    if(node) node.textContent=report?`Last correction: ${report.solved_frames} frames fitted; ${report.unresolved.length} unresolved overlap constraints; ${report.skipped_frames} frames skipped because a character was not visible. Inspect the preview before Finish.`:'';
  }
  render() {
    const people=(this.info.people||[]).map(p=>`<option value="${esc(String(p.id))}">Person ${esc(String(p.id))}</option>`).join('');
    const parts=(torso=false)=>[['left_upper_arm','Left upper arm'],['left_forearm','Left forearm'],['right_upper_arm','Right upper arm'],['right_forearm','Right forearm'],...(torso?[['torso','Torso']]:[])].map(([v,n])=>`<option value="${v}">${n}</option>`).join('');
    this.host.innerHTML=`<p class="hint">Mark which arm is in front during an overlap. Frame ranges are inclusive. Apply reuses saved tracking, bodies, camera and set; only arm fitting and the preview update. Roots and bone lengths stay fixed. Ambiguous or unreachable corrections are reported.</p>
      <p data-current></p><div class="row"><button data-add>+ Add depth range at current frame</button><button data-apply>Apply depth corrections</button></div>
      ${this.items.map((r,i)=>`<div class="step" data-row="${i}"><div class="row">
      <label>Character <select data-key="person">${people}</select></label><label>Arm segment <select data-key="part">${parts()}</select></label>
      <label>Order <select data-key="order"><option value="front">In front of</option><option value="behind">Behind</option></select></label>
      <label>Character <select data-key="other">${people}</select></label><label>Body part <select data-key="other_part">${parts(true)}</select></label></div>
      <div class="row"><label>Start frame <input data-key="start" type="number" min="0" max="${(this.info.frames||1)-1}" style="width:90px"></label><button data-start>Set start here</button>
      <label>End frame <input data-key="end" type="number" min="0" max="${(this.info.frames||1)-1}" style="width:90px"></label><button data-end>Set end here</button>
      <label>Clearance (cm) <input data-key="margin_m" type="number" min="1" max="50" step="1" style="width:70px"></label>
      <button data-go>Go to start</button><button data-remove>Remove range</button></div></div>`).join('')}
      <p data-report class="small muted"></p>`;
    this.host.querySelector('[data-add]').onclick=()=>{
      const p=this.info.people||[];
      if(p.length<2) { toast('Prepare a clip with at least two characters first.'); return; }
      this.items.push({person:String(p[0].id),part:'right_forearm',other:String(p[1].id),other_part:'left_forearm',order:'front',start:this.frame(),end:this.frame(),margin_m:.08});
      this.changed(); this.render();
    };
    this.host.querySelector('[data-apply]').onclick=()=>this.apply();
    this.host.querySelectorAll('[data-row]').forEach(row=>{
      const r=this.items[+row.dataset.row];
      row.querySelectorAll('[data-key]').forEach(input=>{
        const k=input.dataset.key; input.value=k==='margin_m'?Math.round(r[k]*100):r[k];
        input.onchange=()=>{r[k]=['start','end','margin_m'].includes(k)?Number(input.value)/(k==='margin_m'?100:1):input.value; this.changed();};
      });
      for(const k of ['start','end']) row.querySelector(`[data-${k}]`).onclick=()=>{r[k]=this.frame();this.changed();this.render();};
      row.querySelector('[data-go]').onclick=()=>{this.video.pause();this.video.currentTime=(r.start+.05)/(this.info.fps||30);};
      row.querySelector('[data-remove]').onclick=()=>{this.items.splice(+row.dataset.row,1);this.changed();this.render();};
    });
    this.sync();
    this.report(this.info);
  }
}
window.LimbDepthEditor=LimbDepthEditor;

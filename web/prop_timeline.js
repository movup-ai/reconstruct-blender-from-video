/* Explicit object identities and ownership events on the source-video timeline. */
class PropTimelineEditor {
  constructor(host, video, changed) {
    this.host = host; this.video = video; this.changed = changed;
    this.items = []; this.selected = null; this.frame = 0; this.info = {};
    video.addEventListener('seeked', () => this.sync());
    video.addEventListener('timeupdate', () => this.sync());
  }
  load(review, info) {
    this.info = info; this.items = JSON.parse(JSON.stringify(review?.prop_timeline || []));
    this.mode = review?.prop_mode || (this.items.length ? 'manual' : 'auto');
    this.selected = this.items[0]?.id; this.render();
  }
  value() { return {prop_mode: this.mode, prop_timeline: JSON.parse(JSON.stringify(this.items))}; }
  current() { return this.items.find(o => o.id === this.selected); }
  seek(f) {
    this.frame = Math.max(0, Math.min((this.info.frames || 1)-1, Math.round(f)));
    this.video.pause(); this.video.currentTime = (this.frame+.05)/(this.info.fps || 30);
    this.sync(false);
  }
  sync(fromVideo = true) {
    if (fromVideo) this.frame = Math.max(0, Math.min((this.info.frames || 1)-1, Math.floor(this.video.currentTime*(this.info.fps || 30)+.01)));
    const f = this.host.querySelector('[data-frame]'), slider = this.host.querySelector('[data-slider]');
    if (f) f.value = this.frame; if (slider) slider.value = this.frame;
    const stamp = this.host.querySelector('[data-time]');
    if (stamp) stamp.textContent = `${(this.frame/(this.info.fps || 30)).toFixed(2)} s`;
    this.draw();
  }
  update() { this.changed(); this.render(); }
  event(state) {
    const o = this.current(); if (!o) return;
    const person = +this.host.querySelector('[data-owner]').value;
    if (['held','worn'].includes(state) && !person) { toast('Choose a character first.'); return; }
    o.events = o.events.filter(e => e.frame !== this.frame);
    o.events.push({frame:this.frame, state, person:['held','worn'].includes(state) ? person : null,
      joint:this.host.querySelector('[data-joint]').value});
    o.events.sort((a,b) => a.frame-b.frame); this.update();
  }
  draw() {
    const cv = this.host.querySelector('canvas'); if (!cv) return;
    const g = cv.getContext('2d'); g.fillStyle = '#202327'; g.fillRect(0,0,cv.width,cv.height);
    if (this.video.readyState >= 2) g.drawImage(this.video,0,0,cv.width,cv.height);
    const o = this.current(), keys = o?.keyframes.filter(k => k.box) || [];
    const active = o?.events.filter(e => e.frame <= this.frame).at(-1);
    const box = keys.find(k => k.frame === this.frame)?.box;
    if (box && active?.state !== 'hidden') {
      g.strokeStyle = '#ffca45'; g.lineWidth = Math.max(2,cv.width/400);
      g.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);
    }
    if (this.drag) {
      const [a,b] = this.drag; g.strokeStyle = '#ffca45'; g.lineWidth = 3;
      g.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);
    }
  }
  render() {
    const o = this.current(), current = o?.events.filter(e => e.frame <= this.frame).at(-1);
    const owner = current?.person || this.lastOwner || this.info.people?.[0]?.id;
    const joint = current?.joint || this.lastJoint || 'right';
    const people = this.info.people || [];
    this.host.innerHTML = `<p class="hint">Add a named object, choose its character and hand, then set events at the current frame. Use <b>Release / throw</b>, <b>Rest</b>, <b>Hide</b>, or <b>Hold / pick up</b> as the action changes. A different object gets its own row. Save keeps edits; Re-check updates props using the fitted people and cameras.</p>
      <div class="row"><label>Props <select data-mode><option value="auto">Automatic + timeline</option><option value="manual">Timeline only</option><option value="off">Off</option></select></label>
      <button data-add>+ Add object</button></div>
      <div class="row" data-objects>${this.items.map(p => `<button data-select="${esc(p.id)}" class="${p.id===o?.id?'primary':''}">${esc(p.name)}</button>`).join('')}</div>
      ${o ? `<div class="row"><label>Name <input data-name maxlength="120" value="${esc(o.name)}"></label>
      <label>Shape <select data-shape>${[['box','Box'],['flat','Flat / phone'],['cyl','Cylinder / bottle'],['ball','Ball'],['bag','Bag'],['rod','Rod']].map(([v,n])=>`<option value="${v}" ${o.shape===v?'selected':''}>${n}</option>`).join('')}</select></label><button data-remove>Remove object</button></div>
      <div class="row"><label>Size in metres (optional: width, height, depth) <input data-size placeholder="e.g. 0.10, 0.20, 0.06" value="${o.size_m?.join(', ') || ''}"></label></div>
      <div class="row"><button data-prev aria-label="Previous prop frame">−1 frame</button><label>Frame <input data-frame type="number" min="0" max="${(this.info.frames || 1)-1}" value="${this.frame}" style="width:90px"></label><button data-next>+1 frame</button><span data-time></span></div>
      <input data-slider aria-label="Prop timeline frame" type="range" min="0" max="${(this.info.frames || 1)-1}" value="${this.frame}" style="width:100%">
      <canvas aria-label="Draw the object's box on this frame" style="width:100%;max-width:820px;touch-action:none;cursor:crosshair"></canvas>
      <p class="small muted">Pause and drag a box around the object to set its size and location. For a thrown object, add boxes along its path; positions interpolate within the event and shot. Depth stays at release unless you enter a depth for a box. No motion is inferred across a cut.</p>
      <div class="row"><label>Character <select data-owner>${people.map(p=>`<option value="${p.id}" ${+owner===+p.id?'selected':''}>Person ${p.id}</option>`).join('')}</select></label>
      <label>Attachment <select data-joint>${[['right','Right hand'],['left','Left hand'],['both','Both hands'],['back','Back']].map(([v,n])=>`<option value="${v}" ${joint===v?'selected':''}>${n}</option>`).join('')}</select></label></div>
      <div class="row">${[['held','Hold / pick up'],['worn','Wear'],['free','Release / throw'],['resting','Rest'],['hidden','Hide']].map(([v,n])=>`<button data-event="${v}">${n}</button>`).join('')}</div>
      <div class="row"><label>Depth at this box (metres, optional) <input data-depth type="number" min="0.06" max="99" step="0.1" value="${o.keyframes.find(k=>k.frame===this.frame)?.depth_m || ''}"></label><button data-depth-save>Set depth</button></div>
      <div class="timeline-events"><b>Events</b>${o.events.length?'':'<p class="muted">Hidden until the first event. Choose Hold / pick up to begin.</p>'}${o.events.map(e=>`<div class="row"><button data-go="${e.frame}">Frame ${e.frame} · ${(e.frame/(this.info.fps||30)).toFixed(2)} s</button><span>${esc(e.state)}${e.person ? ` · Person ${e.person} · ${esc(e.joint)}`:''}</span><button data-delete-event="${e.frame}" aria-label="Delete event at frame ${e.frame}">Remove event</button></div>`).join('')}</div>
      <div><b>Box / position keys</b>${o.keyframes.map(k=>`<div class="row"><button data-go="${k.frame}">Frame ${k.frame}</button><span>${k.box?'Box':'World position'}${k.depth_m?` · ${k.depth_m} m`:''}</span><button data-delete-key="${k.frame}">Remove key</button></div>`).join('')}</div>` : '<p class="muted">No timeline objects yet. Automatic detections remain available.</p>'}
      <div class="small muted">${(this.info.prop_warnings || []).map(esc).join('<br>')}</div>`;
    const $h = q => this.host.querySelector(q);
    $h('[data-mode]').value = this.mode;
    $h('[data-mode]').onchange = e => { this.mode=e.target.value; this.changed(); };
    $h('[data-add]').onclick = () => {
      const id = 'prop_'+Date.now().toString(36)+'_'+Math.random().toString(36).slice(2,7);
      this.items.push({id,name:'Object '+(this.items.length+1),shape:'box',size_m:null,events:[],keyframes:[]});
      this.selected=id; this.mode='manual'; this.update();
    };
    this.host.querySelectorAll('[data-select]').forEach(b=>b.onclick=()=>{ this.selected=b.dataset.select; this.render(); });
    if (!o) return;
    $h('[data-name]').oninput=e=>{ o.name=e.target.value; this.changed(); };
    $h('[data-name]').onchange=()=>this.render();
    $h('[data-shape]').onchange=e=>{ o.shape=e.target.value; this.changed(); };
    $h('[data-size]').onchange=e=>{
      const text=e.target.value.trim(), vals=text.split(/[,\s]+/).map(Number);
      if (text && (vals.length!==3 || vals.some(x=>!Number.isFinite(x)||x<=0||x>20))) { toast('Enter three positive dimensions in metres.'); e.target.value=o.size_m?.join(', ')||''; return; }
      o.size_m=text?vals:null; this.changed();
    };
    $h('[data-remove]').onclick=()=>{ this.items=this.items.filter(p=>p!==o); this.selected=this.items[0]?.id; this.update(); };
    $h('[data-frame]').oninput=e=>{ if (e.target.value !== '') this.seek(+e.target.value); };
    $h('[data-frame]').onchange=()=>this.render();
    $h('[data-slider]').oninput=e=>this.seek(+e.target.value);
    $h('[data-slider]').onchange=()=>this.render();
    $h('[data-prev]').onclick=()=>{ this.seek(this.frame-1); this.render(); };
    $h('[data-next]').onclick=()=>{ this.seek(this.frame+1); this.render(); };
    $h('[data-owner]').onchange=e=>this.lastOwner=+e.target.value;
    $h('[data-joint]').onchange=e=>this.lastJoint=e.target.value;
    this.host.querySelectorAll('[data-event]').forEach(b=>b.onclick=()=>this.event(b.dataset.event));
    this.host.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>{ this.seek(+b.dataset.go); this.render(); });
    this.host.querySelectorAll('[data-delete-event]').forEach(b=>b.onclick=()=>{ o.events=o.events.filter(e=>e.frame!==+b.dataset.deleteEvent); this.update(); });
    this.host.querySelectorAll('[data-delete-key]').forEach(b=>b.onclick=()=>{ o.keyframes=o.keyframes.filter(k=>k.frame!==+b.dataset.deleteKey); this.update(); });
    $h('[data-depth-save]').onclick=()=>{
      const k=o.keyframes.find(k=>k.frame===this.frame), text=$h('[data-depth]').value, depth=+text;
      if (!k) { toast('Draw a box on this frame first.'); return; }
      if (!text) delete k.depth_m;
      else if (depth>.05 && depth<100) k.depth_m=depth;
      else { toast('Enter a depth between .05 and 100 metres.'); return; }
      this.update();
    };
    const cv=$h('canvas'); cv.width=this.info.size?.[0]||1280; cv.height=this.info.size?.[1]||720;
    const point=e=>{ const r=cv.getBoundingClientRect(); return [Math.max(0,Math.min(cv.width,(e.clientX-r.left)*cv.width/r.width)),Math.max(0,Math.min(cv.height,(e.clientY-r.top)*cv.height/r.height))]; };
    cv.onpointerdown=e=>{ this.video.pause(); cv.setPointerCapture(e.pointerId); const p=point(e); this.drag=[p,p]; this.draw(); };
    cv.onpointermove=e=>{ if (this.drag) { this.drag[1]=point(e); this.draw(); } };
    cv.onpointercancel=()=>{ this.drag=null; this.draw(); };
    cv.onpointerup=e=>{
      if (!this.drag) return; const [a,b]=this.drag; this.drag=null;
      const box=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];
      if (box[2]-box[0]<3 || box[3]-box[1]<3) { this.draw(); return; }
      const prior=o.keyframes.find(k=>k.frame===this.frame)||{};
      o.keyframes=o.keyframes.filter(k=>k.frame!==this.frame); o.keyframes.push({...prior,frame:this.frame,box}); o.keyframes.sort((a,b)=>a.frame-b.frame); this.update();
    };
    this.sync(false);
  }
}
window.PropTimelineEditor = PropTimelineEditor;

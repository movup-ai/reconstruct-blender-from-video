// Blender Agent - shared helpers for the pages
const $ = id => document.getElementById(id);
const esc = s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
const STAGE = { fit: 'local models', ai: 'local AI', all: 'render' };
function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== '' && v != null) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}
async function api(url) {
  const r = await fetch(url);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}
async function post(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? '' : JSON.stringify(body) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) { toast(d.error || r.statusText, true); throw new Error(d.error || r.statusText); }
  return d;
}
let toastT = null;
function toast(msg, bad) {
  const t = $('toast'); if (!t) return;
  t.textContent = msg; t.className = 'toast show' + (bad ? ' bad' : '');
  clearTimeout(toastT); toastT = setTimeout(() => t.className = 'toast', bad ? 6000 : 3000);
}
function chip(st) { return `<span class="chip g-${st.group}">${esc(st.label)}</span>`; }
function fmtTime(s) { s = Math.max(0, Math.round(s)); return s < 60 ? s + 's' : Math.floor(s / 60) + 'm ' + (s % 60) + 's'; }
function segInit(box, onPick) { box.querySelectorAll('button').forEach(b => b.onclick = () => { segSet(box, b.dataset.v); onPick && onPick(b.dataset.v); }); }
function segSet(box, v) { box.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === v)); }
function segGet(box) { const b = box.querySelector('button.on'); return b ? b.dataset.v : null; }

// ---- shared top bar: Clips · Submissions · Settings · Help, with a live queue indicator
function nav(active) {
  if (!document.querySelector('link[rel=icon]')) document.head.insertAdjacentHTML('beforeend', '<link rel="icon" href="/favicon.svg" type="image/svg+xml">');
  const links = [['/', 'Clips', 'clips'], ['/submissions', 'Submissions', 'submissions'], ['/settings', 'Settings', 'settings'], ['/help', 'Help', 'help']];
  const bar = el('nav', { class: 'topnav' });            // full-width bar; its contents line up with the page column
  bar.innerHTML = `<div class="navin"><a class="brand" href="/"><img src="/favicon.svg" alt=""> Blender Agent</a>` +
    links.map(([h, l, k]) => `<a class="nl${k === active ? ' on' : ''}" href="${h}"${k === active ? ' aria-current="page"' : ''}>${l}</a>`).join('') +
    `<span class="qind" id="qind" title="Jobs"></span>` +
    `<button class="quitbtn" id="quit" title="Stop Blender Agent. Unfinished jobs wait in the queue for the next Start.">Quit</button></div>`;
  document.body.prepend(bar);
  $('quit').onclick = async () => { if (!(await ask('Unfinished jobs wait in the queue for the next Start.', 'Quit', { title: 'Quit Blender Agent?' }))) return;
    await post('/api/quit');
    document.body.innerHTML = '<div class="wrap"><h1>Blender Agent stopped</h1><p class="sub">Start it again with “Start_Blender_Agent.bat”.</p></div>'; };
  const tick = async () => { try { const d = await api('/api/clips'); const q = d.queue; window.__clips = d;
      $('qind').innerHTML = q.running ? `<span class="pulse"></span>${q.running} working${q.queued ? ` · ${q.queued} waiting` : ''}` :
        q.queued ? `<span class="idle"></span>${q.paused ? 'Paused · ' : ''}${q.queued} waiting` : '<span class="idle"></span>Idle';
      const base = document.title.replace(/^\(\d+\)\s*/, ''); document.title = (q.running || q.queued ? `(${q.running + q.queued}) ` : '') + base;
    } catch (e) { $('qind').innerHTML = '<span class="bad">offline</span>'; } };
  tick(); setInterval(tick, 7000);
}
function progressBar(pct, step) {
  if (pct == null) return '';
  return `<div class="pbar" title="${esc(step || '')}"><i style="width:${Math.max(2, pct)}%"></i></div><div class="small muted">${esc(step || '')} · ${pct}%</div>`;
}
async function openFolder(path) { try { await post('/api/open', { path }); } catch (e) {} }

// ---- app-styled confirmation dialog: await ask('Short explanation.', 'Do it', { title, name, points, danger })  -> true / false
//      title: the question as a heading · name: the file or item it acts on · points: short facts as a list · danger: red button
function ask(msg, okLabel, o = {}) {
  return new Promise(resolve => {
    const bg = el('div', { class: 'modal-bg', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'asktitle' });
    bg.innerHTML = `<div class="modal">${o.title ? `<h3 class="m-title" id="asktitle">${esc(o.title)}</h3>` : ''}` +
      (o.name ? `<div class="m-name" title="${esc(o.name)}">${esc(o.name)}</div>` : '') +
      (msg ? `<p${o.title ? '' : ' id="asktitle"'}>${esc(msg)}</p>` : '') +
      (o.points ? `<ul class="m-points">${o.points.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : '') +
      `<div class="actions" style="justify-content:flex-end">
      <button class="m-no">Cancel</button><button class="${o.danger ? 'danger' : 'primary'} m-yes">${esc(okLabel || 'Continue')}</button></div></div>`;
    const done = v => { bg.remove(); document.removeEventListener('keydown', key); resolve(v); };
    const key = e => { if (e.key === 'Escape') done(false); if (e.key === 'Enter') done(true); };
    bg.querySelector('.m-no').onclick = () => done(false); bg.querySelector('.m-yes').onclick = () => done(true);
    bg.onclick = e => { if (e.target === bg) done(false); };
    document.addEventListener('keydown', key); document.body.appendChild(bg); bg.querySelector('.m-yes').focus();
  });
}

// one light theme for the whole app (an older dark-theme choice saved in this browser is cleared)
try { localStorage.removeItem('ba-theme'); } catch (e) {}

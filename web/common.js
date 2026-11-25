// Blender Agent - shared helpers for the pages
const $ = id => document.getElementById(id);
const esc = s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
const STAGE = { fit: 'Prepare', background: 'Render background only', ai: 'local AI', all: 'Finish' };
function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== '' && v != null) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}
// not logged in (or the login ran out): to the login page, back here afterwards
function toLogin() { location.href = '/login?next=' + encodeURIComponent(location.pathname + location.search); }
async function api(url) {
  const r = await fetch(url);
  if (r.status === 401) { toLogin(); throw new Error('log in first'); }
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}
async function post(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? '' : JSON.stringify(body) });
  if (r.status === 401) { toLogin(); throw new Error('log in first'); }
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
// a run's length: "45 s", "12 min", "1 h 05 min"
function fmtDur(s) { s = Math.max(0, Math.round(s)); if (s < 60) return s + ' s'; const m = Math.round(s / 60);
  return m < 60 ? m + ' min' : Math.floor(m / 60) + ' h ' + String(m % 60).padStart(2, '0') + ' min'; }
// how long the work on a video took (computer time, not waiting): took = {done, open, last} from /api/clips, st = its status.
// running: this job so far · prepared: how long preparing took · finished: all its jobs, from Start to the zip
function tookText(t, st) {
  if (!t) return null;
  const L = t.last, what = L && L.stage === 'all' ? 'finishing' : 'preparing';
  if (L && L.state === 'running') return { text: `${what} ${fmtDur(L.ran)}`, tip: `This job has been ${what} for ${fmtDur(L.ran)}.` };
  if (t.done && (!st || st.done)) return { text: 'done in ' + fmtDur(t.done.machine), fin: true,
    tip: `The computer worked ${fmtDur(t.done.machine)} to make this package. From Start to the zip took ${fmtDur(t.done.wall)}, with the queue and your review.` };
  if (L && L.stage === 'fit' && L.state === 'done' && (!st || st.fit)) return { text: 'prepared in ' + fmtDur(L.ran), tip: `Preparing took ${fmtDur(L.ran)}.` };
  return null;
}
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
    `<span class="qind" id="qind" title="Your jobs"></span>` +
    `<span class="who" id="who"></span><button class="quitbtn" id="logout" title="Log out of Blender Agent on this browser. Your jobs keep running.">Log out</button>` +
    `<button class="quitbtn" id="quit" style="display:none" title="Stop Blender Agent for every member. Unfinished jobs wait in the queue for the next Start.">Quit</button></div>`;
  document.body.prepend(bar);
  $('logout').onclick = async () => { await post('/api/logout'); location.href = '/login'; };
  $('quit').onclick = async () => { if (!(await ask('It stops for every member. Unfinished jobs wait in the queue for the next Start.', 'Quit', { title: 'Quit Blender Agent?' }))) return;
    await post('/api/quit');
    document.body.innerHTML = '<div class="wrap"><h1>Blender Agent stopped</h1><p class="sub">Start it again with “Start_Blender_Agent.bat”.</p></div>'; };
  const tick = async () => { try { const d = await api('/api/clips'); const q = d.queue; window.__clips = d;
      const o = q.others || {}, busy = (o.running || 0) + (o.queued || 0);
      $('qind').innerHTML = (q.running ? `<span class="pulse"></span>${q.running} working${q.queued ? ` · ${q.queued} waiting` : ''}` :
        q.queued ? `<span class="idle"></span>${q.paused ? 'Paused · ' : ''}${q.queued} waiting` : '<span class="idle"></span>Idle') +
        (busy ? `<span class="muted" title="Jobs of other members on the same NVIDIA card: yours wait for a free place">· others: ${o.running || 0} working${o.queued ? `, ${o.queued} waiting` : ''}</span>` : '');
      $('who').textContent = d.me ? d.me.name : ''; $('quit').style.display = d.me && d.me.local ? '' : 'none';
      const base = document.title.replace(/^\(\d+\)\s*/, ''); document.title = (q.running || q.queued ? `(${q.running + q.queued}) ` : '') + base;
    } catch (e) { $('qind').innerHTML = '<span class="bad">offline</span>'; } };
  tick(); setInterval(tick, 7000);
}
function progressBar(pct, step) {
  if (pct == null) return '';
  return `<div class="pbar" title="${esc(step || '')}"><i style="width:${Math.max(2, pct)}%"></i></div><div class="small muted">${esc(step || '')} · ${pct}%</div>`;
}
async function openFolder(path) { try { await post('/api/open', { path }); } catch (e) {} }

// ---- Start over: delete what preparing left for one clip (cid) or every clip (no cid), after a confirmation
async function startOver(cid, name) {
  const base = cid ? `/api/clip/${cid}/reset` : '/api/reset_all';
  const r = await post(base + '?dry=1');
  if (cid && r.busy.length) { toast('This clip is being processed right now. Start over when that job is done.', true); return false; }
  if (!r.clips.length) { toast(r.busy.length ? 'Every clip with files is being processed right now. Try again when those jobs are done.' : 'Nothing to delete: no clip has prepared files.'); return false; }
  const what = [...new Set(r.clips.flatMap(c => c.paths.map(p => p.what)))].join('; ');
  const pts = [(cid ? 'Deleted: ' : `Deleted for ${r.clips.length} clip(s): `) + what + '.',
    'Your roles, joint fixes and objects are deleted too. This cannot be undone.',
    cid ? 'Kept: the video, its delivery name and its thumbnail.' : 'Kept: the videos, their delivery names and their thumbnails.',
    cid ? 'Press Prepare this clip (or Start) to detect the people again.' : 'Press Start to prepare them again from the videos.'];
  if (r.busy.length) pts.push(`Skipped while they are being processed: ${r.busy.join(', ')}.`);
  if (!(await ask('', cid ? 'Delete and start over' : 'Reset all', { title: cid ? 'Start this clip over?' : `Reset all ${r.clips.length} clip(s)?`, name: cid ? name : '', danger: true, points: pts }))) return false;
  const d = await post(base);
  if (d.failed.length) toast(`${d.failed.length} item(s) could not be deleted (${d.failed[0].reason}): close any program using them and try again.`, true);
  else toast(cid ? 'Started over: prepare the clip again.' : `Reset ${d.clips.length} clip(s). Press Start to prepare them.`);
  return true;
}

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

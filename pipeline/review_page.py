# -*- coding: utf-8 -*-
"""One static page to review every clip: Work_auto\\REVIEW.html (+ thumbnails in Work_auto\\_review\\).

  python pipeline\\review_page.py [--film TEXT] [--frames 4] [--width 480] [--force] [--threads 3]

Per film, per clip: 4 frames film | white model side by side (the source video and render\\wm_NNNN.jpg), status
(rendered or not, zip), people and roles (report.json + the marks folder's review.json), warnings and notes (camera /
people hints from output\\<clip>\\hints.json, missing set, stale render ...) and links to the comparison video and the
check pictures.  Reads the clip folders only; thumbnails are made again only when the clip's render or video changed."""
import argparse, html, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from common import cfg, read_frames_at, imwrite_u, output_dir  # noqa: E402
from agent.clips import Library, folders, read_json, status  # noqa: E402


def uri(p):
    return Path(p).resolve().as_uri()


def esc(s):
    return html.escape(str(s), quote=True)


def mtime(p):
    try:
        return round(os.path.getmtime(p), 1)
    except OSError:
        return 0


def n_frames(video, ci):
    if ci and ci.get("frames"):
        return int(ci["frames"])
    cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    return max(n, 1)


def make_thumbs(job):
    """[(frame, file name)] film | white model pair per frame, each half `width` px wide"""
    video, work, ci, frs, width, out_dir, cid = job
    got = read_frames_at(video, frs)
    names = []
    for f in frs:
        im = got.get(f)
        if im is None:
            continue
        if ci and "y0" in ci and "Hpic" in ci:
            im = im[ci["y0"]:ci["y0"] + ci["Hpic"]]
        h = max(1, int(round(im.shape[0] * width / float(im.shape[1]))))
        film = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
        rp = os.path.join(work, "render", "wm_%04d.jpg" % (f + 1))
        if os.path.exists(rp):
            r = cv2.imdecode(np.fromfile(rp, np.uint8), cv2.IMREAD_COLOR)
            wm = cv2.resize(r, (width, h), interpolation=cv2.INTER_AREA) if r is not None else None
        else:
            wm = None
        if wm is None:
            wm = np.full_like(film, 70)
            cv2.putText(wm, "not rendered", (width // 2 - 80, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        for im_, t in ((film, "film f%d" % f), (wm, "white model")):
            cv2.putText(im_, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im_, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        pair = np.hstack([film, np.full((h, 4, 3), 255, np.uint8), wm])
        nm = "%s_f%04d.jpg" % (cid, f)
        imwrite_u(os.path.join(out_dir, nm), pair, 80)
        names.append((f, nm, pair.shape[1], pair.shape[0]))
    return names


def clip_info(c, cl):
    """everything shown for one clip (files only, no video decoding)"""
    fo = folders(c, cl); work = fo["work"]; name = cl.name
    stem = os.path.splitext(os.path.basename(cl.video))[0]
    if not os.path.isdir(work) and os.path.isdir(os.path.join(c["work_root"], stem)):     # older clips: folders named by file stem
        name = stem; work = os.path.join(c["work_root"], stem); sub = os.path.join(c["submit_root"], stem)
        fo = dict(fo, work=work, submit=sub, zip=sub + ".zip", preview=os.path.join(work, "preview.jpg"),
                  scene=os.path.join(work, "scene_data.json"))
    ci = read_json(os.path.join(work, "clip_info.json"))
    rep = read_json(os.path.join(work, "report.json")) or {}
    rv = read_json(fo["review"]) or {}
    hints = read_json(os.path.join(output_dir(), name, "hints.json"))
    try:
        st = status(c, cl)
    except Exception as e:                                   # a half-written file must not stop the page
        st = {"label": "unknown (%s)" % e, "group": "fail"}
    if name != cl.name and os.path.exists(fo["zip"]):
        st = {"label": "done", "group": "done"}
    rdir = os.path.join(work, "render")
    try:
        n_wm = len([x for x in os.listdir(rdir) if x.startswith("wm_") and x.endswith(".jpg")])
    except OSError:
        n_wm = 0
    done_txt = os.path.join(work, "render_done.txt")
    rendered = os.path.exists(done_txt) or n_wm > 0
    comp = os.path.join(fo["submit"], "comparison", name + ".mp4")
    N = n_frames(cl.video, ci)
    warn, notes = [], []
    for w in rep.get("warnings") or []:
        warn.append(str(w))
    if rep.get("status") not in (None, "ok"):
        warn.append("fit status: %s" % rep.get("status"))
    if not ci:
        warn.append("not fitted yet (no clip_info.json)")
    if n_wm and n_wm < N:
        warn.append("render incomplete: %d of %d frames" % (n_wm, N))
    if os.path.exists(fo["scene"]) and os.path.exists(done_txt) and mtime(fo["scene"]) > mtime(done_txt) + 5:
        warn.append("scene_data.json changed after the render (render may be stale)")
    if rendered and not os.path.exists(fo["zip"]):
        warn.append("rendered but no zip")
    if os.path.exists(fo["zip"]) and not os.path.exists(comp):
        warn.append("zip but no comparison video")
    has_set = os.path.exists(os.path.join(output_dir(), name, "env_claude.json")) or \
        os.path.exists(os.path.join(fo["marks"], "env_claude.json"))
    if ci and not has_set:
        notes.append("no hand-built set (env_claude.json): automatic set")
    if hints:
        cam = hints.get("camera")
        if cam:
            notes.append("camera hint used: " + cam_text(cam))
        for tid, ph in (hints.get("people") or {}).items():
            if isinstance(ph, dict):
                notes.append("person hint (track %s): %s" % (tid, ", ".join(sorted(ph.keys()))))
        for k in hints:
            if k not in ("camera", "people"):
                notes.append("hint: %s" % k)
    rvc = rv.get("camera") or {}
    if rvc.get("note"):
        notes.append("camera note: %s" % rvc["note"])
    if rv.get("notes"):
        notes.append("review notes: %s" % rv["notes"])
    # people: report rows (fitted) joined with the review's roles by track id
    roles = rv.get("people") or {}
    people = []
    seen = set()
    for p in rep.get("people") or []:
        t = str(p.get("track")); seen.add(t); r = roles.get(t, {})
        people.append({"name": p.get("person", "?"), "colour": p.get("colour") or r.get("colour", ""), "role": r.get("role", ""),
                       "height": r.get("height"), "label": r.get("name", ""), "frames": p.get("frames"),
                       "err": p.get("median_error_px"), "worst": p.get("worst_error_px")})
        if (p.get("median_error_px") or 0) > 40 or (p.get("worst_error_px") or 0) > 250:   # a loose fit (the chip shows all errors)
            warn.append("%s fit error: median %.0f px, worst %.0f px" % (p.get("person"), p.get("median_error_px") or 0, p.get("worst_error_px") or 0))
    for t, r in roles.items():
        if t not in seen and r.get("role") not in (None, "", "ignore", "skip"):
            people.append({"name": "track %s" % t, "colour": r.get("colour", ""), "role": r.get("role", ""), "height": r.get("height"),
                           "label": r.get("name", ""), "frames": None, "err": None, "worst": None})
    frs = sorted(set(int(x) for x in np.linspace(0, N - 1, 4).round()))
    if name != cl.name:
        notes.append("folders named %s" % name)
    return {"cl": cl, "fo": fo, "ci": ci, "rep": rep, "rv": rv, "st": st, "rendered": rendered, "n_wm": n_wm, "N": N,
            "comp": comp, "warn": warn, "notes": notes, "people": people, "frs": frs,
            "sig": [mtime(cl.video), mtime(done_txt), n_wm, mtime(os.path.join(work, "clip_info.json"))]}


def cam_text(cam):
    """camera hint in one line (also per shot)"""
    if not isinstance(cam, dict):
        return str(cam)
    bits = ["%s=%s" % (k, v) for k, v in cam.items() if k not in ("path", "shots") and not isinstance(v, (dict, list))]
    if cam.get("path"):
        bits.append("moving path, %d keys" % len(cam["path"]))
    for s, v in sorted((cam.get("shots") or {}).items()):
        bits.append("shot %s: %s" % (s, cam_text(v)))
    return ", ".join(bits) or "(set)"


CSS = """
:root{--bg:#f4f4f2;--card:#fff;--ink:#1d1d1f;--mute:#666;--line:#ddd;--ok:#1d7a3a;--warn:#a15c00;--bad:#b3261e;--acc:#2456a6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:5;background:#222;color:#eee;padding:8px 16px;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center}
header b{font-size:16px}header a{color:#9cf}header .f{display:flex;gap:6px;flex-wrap:wrap}
button{font:inherit;border:1px solid #666;background:#333;color:#eee;border-radius:4px;padding:2px 9px;cursor:pointer}button.on{background:#9cf;color:#000}
main{padding:12px 16px;max-width:2000px;margin:auto}h2{margin:22px 0 8px;font-size:20px;border-bottom:2px solid #333}
.clip{background:var(--card);border:1px solid var(--line);border-radius:6px;margin:0 0 14px;padding:10px 12px}
.hd{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline}.hd .nm{font-size:17px;font-weight:700}.hd .no{color:var(--mute)}
.badge{display:inline-block;border-radius:10px;padding:0 9px;font-size:12px;font-weight:600;color:#fff;background:#888}
.g-done{background:var(--ok)}.g-conf{background:var(--acc)}.g-ready{background:#6a4fb0}.g-busy{background:var(--warn)}.g-fail{background:var(--bad)}.g-new{background:#888}
.meta{color:var(--mute);font-size:13px;margin:3px 0;word-break:break-all}.meta a{color:var(--acc)}
.ppl span{display:inline-block;margin:2px 8px 2px 0;padding:0 6px;border-radius:4px;background:#eee}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
ul{margin:4px 0;padding-left:20px}li.w{color:var(--bad)}li.n{color:#444}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:6px;margin-top:6px}
.strip img{width:100%;height:auto;display:block;background:#ccc}
.hide{display:none}
"""

JS = """
function flt(k,b){document.querySelectorAll('header button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
document.querySelectorAll('.clip').forEach(c=>{const d=c.dataset;let s=true;
if(k=='rend')s=d.r=='1';if(k=='norend')s=d.r=='0';if(k=='warn')s=d.w=='1';c.classList.toggle('hide',!s)});
document.querySelectorAll('h2').forEach(h=>{let n=h.nextElementSibling,any=false;while(n&&n.tagName!='H2'){if(n.classList.contains('clip')&&!n.classList.contains('hide'))any=true;n=n.nextElementSibling}h.classList.toggle('hide',!any)})}
"""
DOT = {"red": "#eb3e3c", "blue": "#3278e6", "green": "#46be5a", "yellow": "#ebd23c", "purple": "#965abe"}


def card(k, info, thumbs, out_dir):
    cl, fo, st = info["cl"], info["fo"], info["st"]
    h = ['<section class="clip" id="c%s" data-r="%d" data-w="%d">' % (cl.id, info["rendered"], bool(info["warn"]))]
    h.append('<div class="hd"><span class="no">#%d</span><span class="nm">%s</span><span class="badge g-%s">%s</span>'
             % (k, esc(cl.name), esc(st.get("group", "new")), esc(st.get("label", "?"))))
    h.append('<span>%s</span>' % ("rendered %d/%d frames" % (info["n_wm"], info["N"]) if info["rendered"] else "not rendered"))
    if info["rep"].get("shots"):
        h.append('<span>%d shot(s)</span>' % len(info["rep"]["shots"]))
    h.append('</div>')
    links = []
    if os.path.exists(info["comp"]):
        links.append('<a href="%s">comparison video</a>' % esc(uri(info["comp"])))
    for nm, lab in (("env_check.jpg", "set check"), ("compare.jpg", "compare"), ("compare_zoom.jpg", "zoom"), ("preview.jpg", "fit preview")):
        p = os.path.join(fo["work"], nm)
        if os.path.exists(p):
            links.append('<a href="%s">%s</a>' % (esc(uri(p)), lab))
    links.append('<a href="%s">work folder</a>' % esc(uri(fo["work"])) if os.path.isdir(fo["work"]) else "")
    zp = fo["zip"] if os.path.exists(fo["zip"]) else None
    h.append('<div class="meta">zip: %s</div>' % (esc(zp) if zp else "none"))
    h.append('<div class="meta">%s</div>' % " · ".join(x for x in links if x))
    h.append('<div class="meta">source: %s</div>' % esc(cl.video))
    if info["people"]:
        ps = []
        for p in info["people"]:
            t = '<span><i class="dot" style="background:%s"></i>%s' % (DOT.get(p["colour"], "#aaa"), esc(p["name"]))
            if p["role"]:
                t += " · " + esc(p["role"])
            if p["label"]:
                t += " · " + esc(p["label"])
            if p["height"]:
                t += " · %.2f m" % p["height"]
            if p["frames"] is not None:
                t += " · %s fr" % p["frames"]
            if p["err"] is not None:
                t += " · err %.1f/%.0f px" % (p["err"], p["worst"] or 0)
            ps.append(t + "</span>")
        h.append('<div class="ppl"><b>%d people</b> %s</div>' % (len(info["people"]), "".join(ps)))
    else:
        h.append('<div class="ppl"><b>0 people</b></div>')
    if info["warn"] or info["notes"]:
        h.append("<ul>" + "".join('<li class="w">%s</li>' % esc(w) for w in info["warn"])
                 + "".join('<li class="n">%s</li>' % esc(n) for n in info["notes"]) + "</ul>")
    if thumbs:
        h.append('<div class="strip">' + "".join(
            '<img loading="lazy" src="_review/%s" width="%d" height="%d" alt="%s f%d" title="%s  frame %d">'
            % (esc(nm), w, hh, esc(cl.name), f, esc(cl.name), f) for f, nm, w, hh in thumbs) + "</div>")
    else:
        h.append('<div class="meta">(no frames: video could not be read)</div>')
    h.append("</section>")
    return "\n".join(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default=""); ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    c = cfg(); work_root = c["work_root"]
    out_dir = os.path.join(work_root, "_review"); os.makedirs(out_dir, exist_ok=True)
    man_p = os.path.join(out_dir, "manifest.json")
    man = read_json(man_p, {}) or {}
    lib = Library(c)
    clips = [cl for cl in lib.clips if a.film.lower() in cl.film.lower()]
    infos = []
    for cl in clips:
        try:
            infos.append(clip_info(c, cl))
        except Exception as e:
            print("skip %s: %s" % (cl.name, e))
    jobs, todo = [], []
    for info in infos:
        cl = info["cl"]; sig = info["sig"] + [a.width, info["frs"]]
        old = man.get(cl.id)
        if not a.force and old and old.get("sig") == sig and all(os.path.exists(os.path.join(out_dir, t[1])) for t in old["thumbs"]):
            info["thumbs"] = [tuple(t) for t in old["thumbs"]]
            continue
        todo.append(info)
        jobs.append((cl.video, info["fo"]["work"], info["ci"], info["frs"], a.width, out_dir, cl.id))
    if jobs:
        print("thumbnails for %d of %d clips ..." % (len(jobs), len(infos)), flush=True)
        with ThreadPoolExecutor(max(1, a.threads)) as ex:
            res = list(ex.map(lambda j: _safe(make_thumbs, j), jobs))
        for info, th in zip(todo, res):
            info["thumbs"] = th
            man[info["cl"].id] = {"sig": info["sig"] + [a.width, info["frs"]], "thumbs": th, "name": info["cl"].name}
    with open(man_p, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False)
    films = {}
    for info in infos:
        films.setdefault(info["cl"].film or "(no film folder)", []).append(info)
    n_r = sum(i["rendered"] for i in infos); n_w = sum(bool(i["warn"]) for i in infos)
    body = []
    k = 0
    for film in sorted(films):
        lst = sorted(films[film], key=lambda i: i["cl"].name)
        body.append('<h2 id="film%d">%s <small>(%d clips, %d rendered)</small></h2>'
                    % (sorted(films).index(film), esc(film), len(lst), sum(i["rendered"] for i in lst)))
        for info in lst:
            k += 1
            body.append(card(k, info, info.get("thumbs") or [], out_dir))
    nav = " ".join('<a href="#film%d">%s</a>' % (n, esc(f.split("「")[0].split("！")[0][:14])) for n, f in enumerate(sorted(films)))
    page = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>White model review</title><style>%s</style><script>%s</script></head><body>
<header><b>White model review</b><span>%d clips · %d rendered · %d with warnings · made %s</span>
<span class="f"><button class="on" onclick="flt('all',this)">all</button><button onclick="flt('rend',this)">rendered</button>
<button onclick="flt('norend',this)">not rendered</button><button onclick="flt('warn',this)">warnings</button></span>
<span class="f">%s</span></header>
<main><p class="meta">Each row: film frame (left) | white model render (right), 4 frames per clip. When reporting a problem,
screenshot the clip's header (name) together with the frames.</p>
%s</main></body></html>""" % (CSS, JS, len(infos), n_r, n_w, time.strftime("%Y-%m-%d %H:%M"), nav, "\n".join(body))
    out = os.path.join(work_root, "REVIEW.html")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(page)
    os.replace(tmp, out)
    print("%s\n%d clips (%d rendered, %d with warnings), %d thumbnail sets made, %.1f s"
          % (out, len(infos), n_r, n_w, len(jobs), time.time() - t0))


def _safe(fn, job):
    try:
        return fn(job)
    except Exception as e:
        print("thumbnails failed for %s: %s" % (job[0], e), flush=True)
        return []


if __name__ == "__main__":
    main()

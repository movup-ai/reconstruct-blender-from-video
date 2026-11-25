"""Portable, exact-frame review of source and render without video seek ambiguity."""
import json
from pathlib import Path

import cv2
import numpy as np

from common import imwrite_u, open_video, close_video


def build(video, render_dir, output, info, width=640):
    import validation as V
    from profiling import counter
    output=Path(output);count=int(info['frames'])
    renders=[next((Path(render_dir)/('wm_%04d.%s'%(f+1,ext)) for ext in ('jpg','png')
                   if (Path(render_dir)/('wm_%04d.%s'%(f+1,ext))).is_file()),None) for f in range(count)]
    key=V.digest({'source':V.file_digest(video),'video':str(video),'info':info,'width':width,
        'renders':[V.file_digest(p) if p else None for p in renders],
        'implementation':V.implementation(('frame_review.py',))})
    cache=output/'build_cache.json';cached=V.load_cache(cache,key)
    artifacts=[output/'index.html',output/'review.json']+[output/folder/('%06d.jpg'%f)
        for folder in ('source','model') for f in range(count)]
    if cached and all(p.is_file() for p in artifacts):
        counter('review_export_cache_hits');return dict(cached['result'],cache_hit=True)
    result=_build(video,render_dir,output,info,width);V.save_cache(cache,key,result)
    return dict(result,cache_hit=False)


def _build(video, render_dir, output, info, width=640):
    output=Path(output);source_dir=output/'source';model_dir=output/'model'
    for directory in (source_dir,model_dir):
        directory.mkdir(parents=True,exist_ok=True)
        for old in directory.glob('*.jpg'):
            if len(old.stem)==6 and old.stem.isdigit():old.unlink()
    capture,temporary=open_video(str(video))
    count=int(info['frames']); full_height=int(info['Hfull']);picture_height=int(info['Hpic']);y0=int(info['y0'])
    target_width=min(int(width),int(info['W']));target_height=max(1,round(full_height*target_width/info['W']))
    try:
        for frame in range(count):
            ok,source=capture.read()
            if not ok:raise ValueError('Source review frame %d is missing'%frame)
            path=next((Path(render_dir)/('wm_%04d.%s'%(frame+1,ext)) for ext in ('jpg','png') if (Path(render_dir)/('wm_%04d.%s'%(frame+1,ext))).exists()),None)
            if path is None:raise ValueError('Render review frame %d is missing'%frame)
            model=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
            if model is None:raise ValueError('Render frame cannot be decoded: '+str(path))
            # Recreate source letterboxing rather than stretch the picture area.
            full=np.zeros_like(source)
            full[y0:y0+picture_height]=cv2.resize(model,(source.shape[1],picture_height))
            for directory,image in ((source_dir,source),(model_dir,full)):
                imwrite_u(str(directory/('%06d.jpg'%frame)),cv2.resize(image,(target_width,target_height)),90)
    finally:
        close_video(capture,temporary)
    metadata={'frames':count,'fps':float(info['fps']),'source_frame_numbering':'zero-based',
              'review_status':'unreviewed','scope':'Synchronized source/render frames; controls are not burned into video.'}
    (output/'review.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    page=HTML.replace('__COUNT__',str(count)).replace('__FPS__',str(float(info['fps'])))
    (output/'index.html').write_text(page,encoding='utf-8')
    return {'path':'model_3d/reconstruction/review/index.html',**metadata}


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Source and 3D frame review</title>
<style>body{margin:20px;background:#172027;color:#eee;font:15px system-ui}nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:15px}button,select,input{font:inherit}button,select{padding:6px 10px}#stage{position:relative;max-width:960px}img{display:block;width:100%}#stage.overlay #model{position:absolute;inset:0;opacity:.5}#stage.source #model,#stage.model #source{display:none}#scrub{width:min(800px,95vw)}p{color:#b7c5cb}</style>
<h1>Source and 3D frame review</h1><nav><button id="previous">Previous</button><button id="play">Play</button><button id="next">Next</button>
<label>Mode <select id="mode"><option value="stack">Source above / 3D below</option><option value="source">Source</option><option value="model">3D</option><option value="overlay">Overlay</option></select></label>
<label>Speed <select id="speed"><option value="1">1x</option><option value="0.5">0.5x</option><option value="0.25">0.25x</option></select></label><output id="position"></output></nav>
<input id="scrub" type="range" min="0" value="0" step="1"><div id="stage"><img id="source" alt="Source frame"><img id="model" alt="3D frame"></div>
<p>Arrow keys step one source frame; Space plays or pauses. Inspect full motion and occlusions. Automated checks do not establish visual acceptance.</p>
<script>
const count=__COUNT__,fps=__FPS__;let frame=0,timer=null;const byId=id=>document.getElementById(id);
byId('scrub').max=count-1;
function show(value){frame=Math.max(0,Math.min(count-1,Math.round(value)));const name=String(frame).padStart(6,'0')+'.jpg';byId('source').src='source/'+name;byId('model').src='model/'+name;byId('scrub').value=frame;byId('position').textContent='Source frame '+frame+' / '+(count-1)+' · '+(frame/fps).toFixed(3)+' s';}
function pause(){if(timer!==null)clearInterval(timer);timer=null;byId('play').textContent='Play';}
function play(){pause();if(frame===count-1)show(0);byId('play').textContent='Pause';timer=setInterval(()=>{if(frame===count-1)pause();else show(frame+1);},1000/(fps*Number(byId('speed').value)));}
byId('previous').onclick=()=>{pause();show(frame-1);};byId('next').onclick=()=>{pause();show(frame+1);};byId('play').onclick=()=>timer===null?play():pause();
byId('scrub').oninput=event=>{pause();show(Number(event.target.value));};byId('mode').onchange=event=>byId('stage').className=event.target.value;byId('speed').onchange=()=>{if(timer!==null)play();};
document.onkeydown=event=>{if(['INPUT','SELECT'].includes(event.target.tagName))return;if(event.key==='ArrowLeft'){event.preventDefault();pause();show(frame-1);}if(event.key==='ArrowRight'){event.preventDefault();pause();show(frame+1);}if(event.code==='Space'){event.preventDefault();timer===null?play():pause();}};show(0);
</script></html>'''

"""Validation policy and content-based caches, independent of inference dependencies."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

VERSION = 1


def mode(config=None):
    value=(config or {}).get('validation_mode','normal')
    if value not in ('normal','audit'):raise ValueError('validation_mode must be normal or audit')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        default=lambda x:x.tolist()).encode()).hexdigest()


def file_digest(path):
    path=Path(path)
    if not path.is_file():return None
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def implementation(files=()):
    root=Path(__file__).resolve().parent
    return {p:file_digest(root/p) for p in ('validation.py',*files)}


def rig_assets(data):
    from rig import ASSET_FILES
    root=Path(__file__).resolve().parents[1]/'assets'
    styles={c.get('style','sample') for c in data.get('chars',{}).values()}
    paths={ASSET_FILES[s] for s in styles if s in ASSET_FILES}
    if 'sample' in styles:paths.add('sample_character.json')
    if 'l3' in styles:paths.update(('l3/L3_standard.blend','l3/model-profile.json'))
    return {p:file_digest(root/p) for p in sorted(paths)}


def geometry(data, people_only=False):
    keys=('frames','res','sensor','lens','cam','zoom','required_track_ids')
    value={k:data.get(k) for k in keys}
    value['chars']={n:{k:c.get(k) for k in ('height','style','track_id','root','rot','pelvis_z','visible',
        'source_range','root_z_fit','metric_scale','body_constraints')} for n,c in data.get('chars',{}).items()}
    if not people_only:
        presentation={'colour','shade','layer','source_rgb','semantic_role','label'}
        value['meshes']=[{k:v for k,v in m.items() if k not in presentation} for m in data.get('meshes',[])]
        value['props']=[{k:v for k,v in p.items() if k not in presentation} for p in data.get('props',[])]
        value['shots']=[s.get('frames') for s in data.get('shots',[])]
    return value


def sampled_bundles(bundles, limit=3):
    """Keep shot coverage, then endpoints/middle, with a fixed total view budget."""
    import numpy as np
    entries=[(i,j) for i,b in enumerate(bundles) for j in range(len(b['frames']))]
    if len(entries)<=limit:return bundles
    selected=[]
    for i,b in enumerate(bundles):
        if len(selected)>=limit:break
        if len(b['frames']):selected.append((i,0))
    for ix in [len(entries)-1,len(entries)//2,*np.linspace(0,len(entries)-1,limit,dtype=int)]:
        if len(selected)>=limit:break
        candidate=entries[int(ix)]
        if candidate not in selected:selected.append(candidate)
        if len(selected)>=limit:break
    result=[]
    frame_keys={'frames','depth','K','R','C','person','valid','labels','groups'}
    for i,b in enumerate(bundles):
        ids=[j for bi,j in selected if bi==i]
        if ids:result.append({k:v[ids] if k in frame_keys and hasattr(v,'shape') and len(v)==len(b['frames']) else v for k,v in b.items()})
    return result


def ensure_scene(data):
    """Reject unusable animation arrays before any repair or rendering."""
    import numpy as np
    count=int(data['frames'])
    def finite_rows(value,shape,label):
        rows=np.asarray(value,float)
        if rows.shape!=shape or not np.isfinite(rows).all():raise ValueError('Invalid '+label+' animation; prepare the clip again')
    if count<1:raise ValueError('Scene has no frames')
    finite_rows(data['cam'],(count,6),'camera')
    for name,ch in data.get('chars',{}).items():
        finite_rows(ch['root'],(count,4),name+' root')
        finite_rows(ch['pelvis_z'],(count,),name+' pelvis')
        for joint,values in ch['rot'].items():finite_rows(values,(count,3),name+' '+joint)
        if not np.isfinite(ch['height']) or ch['height']<=0:raise ValueError('Invalid actor height: '+name)
        if 'visible' in ch and len(ch['visible'])!=count:raise ValueError('Visibility length does not match scene: '+name)


def load_cache(path, key):
    try:
        value=json.loads(Path(path).read_text(encoding='utf-8'))
        valid=value.get('key')==key and value.get('version')==VERSION
        return value if valid and value.get('result_digest')==digest(value.get('result')) else None
    except (OSError,ValueError,AttributeError):return None


def save_cache(path, key, result, **extra):
    write_json(path,dict(version=VERSION,key=key,result=result,result_digest=digest(result),**extra))


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=path.parent,suffix='.json')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:json.dump(value,stream)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.remove(temp)


def cached_report(path, key, compute):
    """Reports keep their public schema; the sidecar stores cache identity."""
    from profiling import counter
    path=Path(path);cache_path=path.parent/'validation_cache'/(path.stem+'-'+key+'.json')
    cached=load_cache(cache_path,key)
    if cached and isinstance(cached.get('result'),dict):
        result=dict(cached['result'],cache_hit=True);counter('validation_cache_hits')
    else:
        result=dict(compute(),cache_hit=False);save_cache(cache_path,key,result)
    write_json(path,result)
    return result


def review_enabled(config):
    return mode(config)=='audit' and config.get('frame_review',True) is not False

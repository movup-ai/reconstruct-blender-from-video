"""Content-based caches for the checks (quality.cached_alignment), independent of inference dependencies."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

VERSION = 1


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


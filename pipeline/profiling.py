"""Per-clip wall timings, including subprocess stages and failures."""
import contextvars
import functools
import json
import os
import tempfile
import time
from contextlib import contextmanager

_active = contextvars.ContextVar("clip_profile", default=None)


def add_seconds(name, seconds, calls=1):
    profile=_active.get()
    if profile is None:return
    row=profile['stages'].setdefault(name,{'calls':0,'seconds':0.})
    row['seconds']+=float(seconds);row['calls']+=calls


def counter(name, count=1):
    profile=_active.get()
    if profile is not None:profile.setdefault('counters',{})[name]=profile.get('counters',{}).get(name,0)+count


@contextmanager
def recording(path=None):
    profile={'stages':{},'scope':'worker; nested timings are not additive'}
    token=_active.set(profile);started=time.perf_counter()
    try:yield profile
    finally:
        profile['total_seconds']=time.perf_counter()-started;_active.reset(token)
        if path:
            os.makedirs(os.path.dirname(path),exist_ok=True)
            from common import jsave
            jsave(path,profile)


def verification_profile(function):
    @functools.wraps(function)
    def measured(data,work,*args,**kwargs):
        parent=_active.get()
        with recording(os.path.join(work,'verification_profile.json')) as child:
            try:return function(data,work,*args,**kwargs)
            finally:
                if parent is not None:
                    for name,row in child['stages'].items():
                        target=parent['stages'].setdefault(name,{'calls':0,'seconds':0.})
                        target['calls']+=row['calls'];target['seconds']+=row['seconds']
                    for name,count in child.get('counters',{}).items():
                        counters=parent.setdefault('counters',{});counters[name]=counters.get(name,0)+count
    return measured


def stage(name):
    def decorate(function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                profile = _active.get()
                if profile is not None:
                    row = profile["stages"].setdefault(name, {"calls": 0, "seconds": 0.0})
                    row["calls"] += 1
                    row["seconds"] += time.perf_counter() - started
        return measured
    return decorate


def profile_clip(function):
    @functools.wraps(function)
    def measured(video, config, args):
        name = getattr(args, "name", None) or function.__globals__["clip_name"](video, config)
        work = os.path.join(config["work_root"], name)
        profile = {"clip": name, "stage": args.stage, "stages": {}, "status": "failed"}
        profile["models"] = {"pose": "rtmpose", "camera": config.get("camera_backend", "flow"),
                             "tracker": "sam3", "human": "sam3d_body"}
        if args.stage == 'background':
            profile['models'] = {'scene_segmenter':'sam3 box prompts or user outline',
                                 'geometry':'saved metric depth or MoGe-2', 'objects':'annotated prototypes',
                                 'people':'reused', 'camera':'reused'}
        token = _active.set(profile)
        started = time.perf_counter()
        detail_paths={name:os.path.join(work,name) for name in ('environment_profile.json','verification_profile.json')}
        before={name:os.stat(path).st_mtime_ns if os.path.isfile(path) else None for name,path in detail_paths.items()}
        try:
            result = function(video, config, args)
            profile["status"] = result.get("status", "ok")
            return result
        finally:
            profile["total_seconds"] = time.perf_counter() - started
            for name,path in detail_paths.items():
                if os.path.isfile(path) and os.stat(path).st_mtime_ns!=before[name]:
                    try:
                        with open(path,encoding='utf-8') as stream:profile.setdefault('details',{})[name]=json.load(stream)
                    except (OSError,ValueError):pass
            _active.reset(token)
            os.makedirs(work, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=work, suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(profile, stream, indent=2)
                os.replace(temp, os.path.join(work, "profile_%s.json" % args.stage))
            finally:
                if os.path.exists(temp):
                    os.remove(temp)
    return measured

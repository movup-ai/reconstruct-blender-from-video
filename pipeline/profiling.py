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
        profile["models"] = {"pose": "rtmpose", "camera": "flow", "depth": "moge2 + mask2former",
                             "tracker": "sam3", "human": "sam3d_body"}
        token = _active.set(profile)
        started = time.perf_counter()
        try:
            result = function(video, config, args)
            profile["status"] = result.get("status", "ok")
            return result
        finally:
            profile["total_seconds"] = time.perf_counter() - started
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

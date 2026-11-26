# -*- coding: utf-8 -*-
"""Settings and folders.  One config.json next to the launcher holds both the pipeline settings and the app settings."""
import json
import math
import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(ROOT, "pipeline")
WEB = os.path.join(ROOT, "web")
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

from common import DEFAULT_CFG, CFG_PATH, member_cfg, normalize_backends  # noqa: E402  (pipeline module)

APP_DEFAULTS = {
    "port": 8765,
    "host": "127.0.0.1",              # 0.0.0.0: the other PCs of the network can open the app (members log in by name)
    "session_days": 30,               # a login lasts this long on one browser
    "parallel": 1,                    # Prepare/background slots; model inference shares one GPU lease
    "finish_parallel": 1,             # separate rendering/packaging slots
    "autopilot": "off",               # off | prepare | full   (full = prepare, AI fill, auto-confirm, render)
    "timeout_fit_min": 40,
    "timeout_ai_min": 20,
    "timeout_render_min": 180,
    "max_attempts": 2,                # a failed job is tried once more automatically
    "ai_enabled": True,               # AI suggestions through the DeepSeek API (needs a key)
    "deepseek_model": "deepseek-flash",    # DeepSeek V4.1 Flash: the model that accepts images
    "deepseek_url": "https://api.deepseek.com",
    "ai_send_images": True,           # send frames + person crops (falls back to measurements if the model refuses images)
    "open_browser": True,
    "render_on_confirm": False,       # False: confirming writes the hand-off for Claude, who builds the set and renders
    "set_detail": "clean",            # clean = readable L2 set (whole facades, simple masses); detailed = every depth block
    "set_builder": "auto",            # auto = the local models build the set; claude = Claude does (pipeline/claude_set.py)
    "set_model": "claude-sonnet-5-5",
    "set_rounds": 2,                  # Claude: write the set, then check it over the film and correct it once
}
SECRETS = ("deepseek_key", "anthropic_key")   # saved in config.json, never sent back to the browser
SET_BUILDERS = ("auto", "claude")       # who builds the background set: the local models, or Claude (needs a key)
SET_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5")
EDITABLE = {"parallel": int, "finish_parallel": int, "autopilot": str, "ai_enabled": bool, "deepseek_model": str, "deepseek_key": str,
            "ai_send_images": bool, "max_attempts": int, "set_detail": str,
            "set_builder": str, "set_model": str, "set_rounds": int, "anthropic_key": str,
            "source_root": str, "submit_root": str, "work_root": str, "blender_exe": str, "person_height": float, "lens_mm": float}
AUTOPILOT_MODES = ("off", "prepare", "full")
EDITABLE.update({"camera_backend": str, "da3_model": str,
                 "character_style": str, "render_engine": str, "source_exposure": bool,
                 "da3_python": str, "da3_meters_per_unit": float,
                 "foot_contact": bool, "set_adaptive_frames": bool, "set_max_frames": int,
                 "metric_scene": bool, "scene_verify": bool, "scene_repair": bool, "validation_mode": str,
                 "sam3_checkpoint": str, "sam3_python": str,
                 "sam3d_checkpoint": str, "sam3d_mhr": str, "sam3d_python": str, "sam3d_stride": int})

_lock = threading.Lock()


def load():
    """pipeline defaults + app defaults + config.json; inside a job the app runs for a member (BA_USER), that
    member's own folders (common.member_cfg)"""
    c = dict(DEFAULT_CFG)
    c.update(APP_DEFAULTS)
    if os.path.exists(CFG_PATH):
        try:
            with open(CFG_PATH, encoding="utf-8") as fh:
                c.update(json.load(fh))
        except (OSError, ValueError):
            pass
    c = normalize_backends(c)
    if os.environ.get("BA_USER"):
        c = member_cfg(c, os.environ["BA_USER"], os.environ.get("BA_SOURCE", ""))
    return c


def save(updates):
    """validate and write settings changed in the app; returns the new config"""
    clean = {}
    for k, v in updates.items():
        if k not in EDITABLE:
            raise ValueError("setting %r cannot be changed here" % k)
        if k == "da3_meters_per_unit" and v in (None, ""):
            clean[k] = None
            continue
        try:
            v = EDITABLE[k](v) if EDITABLE[k] is not bool else (v in (True, "true", "1", 1, "on"))
        except (TypeError, ValueError):
            raise ValueError("bad value for %s: %r" % (k, v))
        if k in ("parallel", "finish_parallel") and not 1 <= v <= 8:
            raise ValueError("%s must be 1-8" % k)
        if k == "camera_backend" and v not in ("flow", "da3"):
            raise ValueError("camera_backend must be flow or da3")
        if k == 'validation_mode' and v not in ('normal','audit'):
            raise ValueError('validation_mode must be normal or audit')
        if k == 'character_style' and v not in ('sample','l3','classic','base21','master'):
            raise ValueError('Unknown character style')
        if k == 'render_engine' and v not in ('workbench','eevee'):
            raise ValueError('render_engine must be workbench or eevee')
        if k == 'sam3d_stride' and not 1<=v<=16:
            raise ValueError('sam3d_stride must be 1-16')
        if k == "set_max_frames" and not 2 <= v <= 12:
            raise ValueError("set_max_frames must be 2-12")
        if k == "da3_meters_per_unit" and (not math.isfinite(v) or v <= 0):
            raise ValueError("DA3 translation scale must be positive and finite")
        if k == "set_detail" and v not in ("clean", "detailed"):
            raise ValueError("set_detail must be clean or detailed")
        if k == "autopilot" and v not in AUTOPILOT_MODES:
            raise ValueError("autopilot must be one of %s" % ", ".join(AUTOPILOT_MODES))
        if k in ("deepseek_key", "anthropic_key"):
            v = v.strip()
            if v and (len(v) < 20 or len(v) > 300 or any(ch.isspace() for ch in v)):
                raise ValueError("that does not look like an API key")
        if k == "set_builder" and v not in SET_BUILDERS:
            raise ValueError("set_builder must be one of %s" % ", ".join(SET_BUILDERS))
        if k == "set_model" and v not in SET_MODELS:
            raise ValueError("set_model must be one of %s" % ", ".join(SET_MODELS))
        if k == "set_rounds" and not 1 <= v <= 3:
            raise ValueError("set_rounds must be 1-3")
        if k in ("source_root", "submit_root", "work_root") and not os.path.isabs(v):
            raise ValueError("%s must be a full path" % k)
        clean[k] = v
    with _lock:
        cur = {}
        if os.path.exists(CFG_PATH):
            with open(CFG_PATH, encoding="utf-8") as fh:
                cur = json.load(fh)
        cur.update(clean)
        cur = normalize_backends(cur)
        tmp = CFG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cur, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, CFG_PATH)
    return load()


def user_data(c):
    """where a member's own lists live (removed / taken-out clips): data\\users\\<user>; data\\ without a member"""
    return os.path.join(DATA, "users", c["_user"]) if c.get("_user") else DATA


def ensure_dirs(c):
    for d in (DATA, LOGS, c["work_root"], c["submit_root"]):
        os.makedirs(d, exist_ok=True)

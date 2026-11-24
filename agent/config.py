# -*- coding: utf-8 -*-
"""Settings and folders.  One config.json next to the launcher holds both the pipeline settings and the app settings."""
import json
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

from common import DEFAULT_CFG, CFG_PATH, member_cfg  # noqa: E402  (pipeline module)

APP_DEFAULTS = {
    "port": 8765,
    "host": "127.0.0.1",              # 0.0.0.0: the other PCs of the network can open the app (members log in by name)
    "session_days": 30,               # a login lasts this long on one browser
    "parallel": 3,                    # clips processed at the same time
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
}
SECRETS = ("deepseek_key",)            # saved in config.json, never sent back to the browser
EDITABLE = {"parallel": int, "autopilot": str, "ai_enabled": bool, "deepseek_model": str, "deepseek_key": str,
            "ai_send_images": bool, "max_attempts": int, "set_detail": str,
            "source_root": str, "submit_root": str, "work_root": str, "blender_exe": str, "person_height": float, "lens_mm": float}
AUTOPILOT_MODES = ("off", "prepare", "full")

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
    if os.environ.get("BA_USER"):
        c = member_cfg(c, os.environ["BA_USER"], os.environ.get("BA_SOURCE", ""))
    return c


def save(updates):
    """validate and write settings changed in the app; returns the new config"""
    clean = {}
    for k, v in updates.items():
        if k not in EDITABLE:
            raise ValueError("setting %r cannot be changed here" % k)
        try:
            v = EDITABLE[k](v) if EDITABLE[k] is not bool else (v in (True, "true", "1", 1, "on"))
        except (TypeError, ValueError):
            raise ValueError("bad value for %s: %r" % (k, v))
        if k == "parallel" and not 1 <= v <= 8:
            raise ValueError("parallel must be 1-8")
        if k == "set_detail" and v not in ("clean", "detailed"):
            raise ValueError("set_detail must be clean or detailed")
        if k == "autopilot" and v not in AUTOPILOT_MODES:
            raise ValueError("autopilot must be one of %s" % ", ".join(AUTOPILOT_MODES))
        if k == "deepseek_key":
            v = v.strip()
            if v and (len(v) < 20 or len(v) > 200 or any(ch.isspace() for ch in v)):
                raise ValueError("that does not look like an API key")
        if k in ("source_root", "submit_root", "work_root") and not os.path.isabs(v):
            raise ValueError("%s must be a full path" % k)
        clean[k] = v
    with _lock:
        cur = {}
        if os.path.exists(CFG_PATH):
            with open(CFG_PATH, encoding="utf-8") as fh:
                cur = json.load(fh)
        cur.update(clean)
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

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

from common import DEFAULT_CFG, CFG_PATH, member_cfg, normalize_backends  # noqa: E402  (pipeline module)

APP_DEFAULTS = {
    "port": 8765,
    "host": "127.0.0.1",              # 0.0.0.0: the other PCs of the network can open the app (members log in by name)
    "session_days": 30,               # a login lasts this long on one browser
    "parallel": 1,                    # clips processed at once; model inference shares one GPU lease
    "timeout_movement_min": 220,      # Stage 1 Movement: people, camera, floor, Blender render and the v2.1 zip
    "max_attempts": 2,                # a failed job is tried once more automatically
    "open_browser": True,
}
# Older timeouts: Prepare ("fit") and Finish ("render") were separate jobs; Movement does both in one run.
OLD_TIMEOUTS = {"timeout_fit_min": 40, "timeout_render_min": 180}
SECRETS = ()                          # nothing secret is stored any more (no AI keys)
_EDITABLE = {"parallel": int, "max_attempts": int,
             "source_root": str, "submit_root": str, "work_root": str, "blender_exe": str,
             "person_height": float, "lens_mm": float,
             "foot_contact": bool, "metric_scene": bool,
             "sam3_checkpoint": str, "sam3_python": str,
             "sam3d_checkpoint": str, "sam3d_mhr": str, "sam3d_python": str, "sam3d_stride": int}
# only settings the pipeline still has (a setting it dropped is not offered here)
EDITABLE = {k: t for k, t in _EDITABLE.items() if k in DEFAULT_CFG or k in APP_DEFAULTS}

_lock = threading.Lock()


def migrate(saved):
    """settings saved by an older version -> today's names (unknown or retired keys are simply ignored)"""
    saved = dict(saved)
    if "timeout_movement_min" not in saved and any(k in saved for k in OLD_TIMEOUTS):
        total = 0
        for k, default in OLD_TIMEOUTS.items():
            try:
                total += float(saved.get(k, default))
            except (TypeError, ValueError):
                total += default
        saved["timeout_movement_min"] = total
    return saved


def load():
    """pipeline defaults + app defaults + config.json; inside a job the app runs for a member (BA_USER), that
    member's own folders (common.member_cfg)"""
    c = dict(DEFAULT_CFG)
    c.update(APP_DEFAULTS)
    if os.path.exists(CFG_PATH):
        try:
            with open(CFG_PATH, encoding="utf-8") as fh:
                saved = json.load(fh)
            if isinstance(saved, dict):
                c.update(migrate(saved))
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
        try:
            v = EDITABLE[k](v) if EDITABLE[k] is not bool else (v in (True, "true", "1", 1, "on"))
        except (TypeError, ValueError):
            raise ValueError("bad value for %s: %r" % (k, v))
        if k == "parallel" and not 1 <= v <= 8:
            raise ValueError("parallel must be 1-8")
        if k == "max_attempts" and not 1 <= v <= 5:
            raise ValueError("max_attempts must be 1-5")
        if k == "sam3d_stride" and not 1 <= v <= 16:
            raise ValueError("sam3d_stride must be 1-16")
        if k == "person_height" and not 0.5 <= v <= 2.5:
            raise ValueError("person_height must be 0.5-2.5 m")
        if k == "lens_mm" and not 5 <= v <= 400:
            raise ValueError("lens_mm must be 5-400 mm")
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

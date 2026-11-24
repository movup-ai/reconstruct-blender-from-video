# -*- coding: utf-8 -*-
"""Members and their logins.  A login is a name, not a password: it tells members apart (each has their own source
folder and results), it does not keep anyone out.

data\\users.json      {key: {"name": shown name, "source_root": folder or "", "created": time, "last_login": time}}
data\\sessions.json   {token: {"user": key, "created": time}}  - a browser stays logged in for config "session_days"
The key is the name in lower case; it is also the member's folder name (output\\_users\\<key>, ...)."""
import json
import os
import re
import secrets
import threading
import time

from . import config

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")
RESERVED = {"con", "prn", "aux", "nul"} | {"com%d" % i for i in range(1, 10)} | {"lpt%d" % i for i in range(1, 10)}
COOKIE = "ba_session"


def check_name(name):
    """a member name -> its key; ValueError with a plain message when it cannot be one"""
    name = str(name or "").strip()
    if not NAME_RE.match(name) or name.endswith("."):
        raise ValueError("use 1-32 letters, numbers, - _ or . (starting with a letter or number), e.g. anna or team-2")
    key = name.lower()
    if key.split(".")[0] in RESERVED:
        raise ValueError("%s is a name Windows keeps for itself: choose another" % name)
    return key


class Users:
    def __init__(self, session_days=30):
        self.lock = threading.Lock()
        self.session_days = session_days
        self.users = self._read(self._users_p())
        self.sessions = self._read(self._sessions_p())

    # ---- files (read config.DATA at call time: the tests point it elsewhere)
    @staticmethod
    def _users_p():
        return os.path.join(config.DATA, "users.json")

    @staticmethod
    def _sessions_p():
        return os.path.join(config.DATA, "sessions.json")

    @staticmethod
    def _read(p):
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write(p, d):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, p)

    # ---- members
    def names(self):
        with self.lock:
            return sorted((u["name"] for u in self.users.values()), key=str.lower)

    def get(self, key):
        with self.lock:
            u = self.users.get(key)
            return dict(u) if u else None

    def set_source(self, key, path):
        with self.lock:
            self.users[key]["source_root"] = path
            self._write(self._users_p(), self.users)

    # ---- sessions
    def login(self, name):
        """-> (key, token).  A new name becomes a new member."""
        key = check_name(name)
        now = time.time()
        token = secrets.token_urlsafe(24)
        with self.lock:
            u = self.users.setdefault(key, {"name": str(name).strip(), "source_root": "", "created": now})
            u["last_login"] = now
            self.sessions = {t: s for t, s in self.sessions.items() if not self._expired(s, now)}
            self.sessions[token] = {"user": key, "created": now}
            self._write(self._users_p(), self.users)
            self._write(self._sessions_p(), self.sessions)
        return key, token

    def logout(self, token):
        with self.lock:
            if self.sessions.pop(token, None) is not None:
                self._write(self._sessions_p(), self.sessions)

    def user_for(self, token):
        """the member a session token belongs to, or None (unknown, expired, or the member is gone)"""
        if not token:
            return None
        with self.lock:
            s = self.sessions.get(token)
            if not s or self._expired(s, time.time()) or s.get("user") not in self.users:
                return None
            return s["user"]

    def _expired(self, s, now):
        return now - float(s.get("created") or 0) > self.session_days * 86400


def token_from_cookie(header):
    """the session token in a Cookie header, or None"""
    for part in (header or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == COOKIE and v:
            return v
    return None

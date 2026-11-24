# -*- coding: utf-8 -*-
"""python -m agent [--port N] [--no-browser]

Starts the app.  If it is already running, just opens the browser on it (exit code 3).
Exit code 0 = closed from the app (Quit), anything else = crashed (the launcher restarts it)."""
import argparse
import json
import logging
import logging.handlers
import os
import socket
import sys
import threading
import urllib.request
import webbrowser

from . import APP_NAME, __version__, config


def setup_logging():
    os.makedirs(config.LOGS, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(os.path.join(config.LOGS, "agent.log"), maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)


def running_instance(port):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/api/health" % port, timeout=2) as r:
            return json.loads(r.read().decode("utf-8")).get("app") == APP_NAME
    except Exception:
        return False


def port_free(port, host="127.0.0.1"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(prog="agent", description=APP_NAME)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    setup_logging()
    log = logging.getLogger("agent")
    c = config.load()
    want = args.port or int(os.environ.get("PORT") or c.get("port", 8765))
    if running_instance(want):
        log.info("%s is already running on port %d - opening it", APP_NAME, want)
        if not args.no_browser:
            webbrowser.open("http://127.0.0.1:%d" % want)
        return 3
    host = str(c.get("host") or "127.0.0.1")
    port = next((p for p in range(want, want + 20) if port_free(p, host)), None)
    if port is None:
        log.error("no free port between %d and %d", want, want + 19)
        return 1
    if port != want:
        log.info("port %d is used by another program - using %d", want, port)
    from .server import App, serve
    app = App(c)
    url = "http://127.0.0.1:%d" % port
    lan = ""
    if host not in ("127.0.0.1", "localhost"):                # the addresses the other PCs of the network can open
        try:                                                   # (every adapter: a VPN or virtual switch has its own)
            ips = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
                          if not a[4][0].startswith(("127.", "169.254."))})
            lan = "".join("\n  Other PCs: http://%s:%d" % (ip, port) for ip in ips)
        except OSError:
            pass
    print("\n  %s %s\n  Open %s   (keep this window open; close it to stop)%s\n" % (APP_NAME, __version__, url, lan), flush=True)
    if c.get("open_browser", True) and not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve(app, port, host)
    return 0


if __name__ == "__main__":
    sys.exit(main())

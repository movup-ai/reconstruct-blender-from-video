# -*- coding: utf-8 -*-
"""Per clip: every detected track (pose.json) with its review role / colour, frames, mean box, and which fitted
character it became.   python pipeline\\tracks_info.py <package or clip> [...]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from agent import config
from agent.clips import Library, folders

c = config.load(); lib = Library(c)
with open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8") as fh:
    names = json.load(fh)
inv = {v: k for k, v in names.items()}
sys.stdout.reconfigure(encoding="utf-8")
for arg in sys.argv[1:]:
    clip = inv.get(arg, arg)
    cl = next(x for x in lib.clips if x.name == clip)
    f = folders(c, cl)
    with open(f["pose"], encoding="utf-8") as fh:
        pose = json.load(fh)
    with open(f["review"], encoding="utf-8") as fh:
        rv = json.load(fh)
    with open(f["scene"], encoding="utf-8") as fh:
        sd = json.load(fh)
    print("=== %s  %s  frames %s  shots %s" % (arg, clip, sd["frames"], [s["frames"] for s in sd.get("shots", [])]))
    fit = {str(ch["track_id"]): k for k, ch in sd["chars"].items()}
    people = pose.get("people") or pose.get("tracks") or {}
    if isinstance(people, list):
        people = {str(p.get("id", i)): p for i, p in enumerate(people)}
    for pid, p in sorted(people.items(), key=lambda t: int(t[0]) if str(t[0]).isdigit() else 0):
        pid = p.get("id", pid)
        fr = sorted(int(k) for k in (p.get("frames") or p.get("boxes") or {}).keys())
        bx = p.get("boxes") or p.get("box")
        mb = ""
        try:
            B = np.array(list(bx.values()) if isinstance(bx, dict) else bx, float)
            mb = "x %.0f-%.0f y %.0f-%.0f" % tuple(np.median(B, 0)[[0, 2, 1, 3]])
        except Exception:
            pass
        r = rv["people"].get(str(pid), {})
        print("  track %-3s %-7s %-7s frames %4s-%-4s n=%-4d %s  -> %s" % (pid, r.get("role", "-"), r.get("colour", "-"),
              fr[0] if len(fr) else "-", fr[-1] if len(fr) else "-", len(fr), mb, fit.get(str(pid), "")))

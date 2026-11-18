# -*- coding: utf-8 -*-
"""python -m agent.selftest - checks the installation and prints what is missing."""
import sys

from . import config


def main():
    c = config.load()
    config.ensure_dirs(c)
    from .server import App
    h = App(c).health()
    print("\nChecks:")
    for x in h["checks"]:
        mark = "OK " if x["ok"] else ("-- " if x.get("optional") else "!! ")
        print("  %s %-20s %s" % (mark, x["name"], x["detail"]))
    print("\nResult:", "ready" if h["ok"] else "something is missing (see !! above)")
    return 0 if h["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

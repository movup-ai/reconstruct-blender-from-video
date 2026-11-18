# -*- coding: utf-8 -*-
"""Round 2 of the v2.1 review (32 frames per clip + every-frame check): roles and pose hints.
Merged into output\\<clip>\\hints.json (people entries updated key by key) and the marks review.json."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline")); sys.path.insert(0, ROOT)
from agent import config
from agent.clips import Library, folders

N0 = None
SEIZA = {"hip": 105, "knee": -135}
OVERHEAD = {"shoulder_L": [160, -10, 0], "shoulder_R": [160, 10, 0], "elbow_L": [25], "elbow_R": [25]}
UPRIGHT = {"spine": [0, 0, 0], "chest": [0, 0, 0]}

ROLES = {   # package: {track: role}
    "the_hidden_blade-14": {"1": "second"},                      # the brown-kimono man: blue in both shots
    "the_hidden_blade-23": {"5": "third", "6": "fourth"},        # wall companion green, staff walker yellow (order seen)
}
R2C = {"main": "red", "second": "blue", "third": "green", "fourth": "yellow", "extra": "purple", "ignore": "purple"}

FIX = {
    # clip 1: the lord walks AWAY in shot 2 (back to camera, heading ~350); a little taller (wide kataginu);
    # the dark-robed man kneels in seiza bowing in the corridor
    "the_hidden_blade-1": {"1": {"yaw": 350, "yaw_sigma": 8},
                           "6": {"legs": [[210, 448, "kneeling"]], "leg_pose": [[210, 448, SEIZA]], "pelvis_z": 0.38,
                                 "keys": [[289, {"spine": [-30, 0, 0], "chest": [-10, 0, 0]}], [448, {"spine": [-30, 0, 0], "chest": [-10, 0, 0]}]],
                                 "key_strength": 3}},
    # clip 3: head up between the raised arms at the end
    "the_hidden_blade-3": {"1": {"keys_more_add": [[139, {"head": [15, 0, None]}], [149, {"head": [15, 0, None]}]]}},
    # clip 4: very wide, low horse stance at the start and the end
    "the_hidden_blade-4": {"1": {"keys_more": [[[0, {"hip_L": [60, 35, 0], "hip_R": [60, -35, 0], "knee_L": [-95], "knee_R": [-95]}],
                                                [10, {"hip_L": [60, 35, 0], "hip_R": [60, -35, 0], "knee_L": [-95], "knee_R": [-95]}]],
                                               [[132, {"hip_L": [80, 35, 0], "hip_R": [80, -35, 0], "knee_L": [-125], "knee_R": [-125]}],
                                                [149, {"hip_L": [80, 35, 0], "hip_R": [80, -35, 0], "knee_L": [-125], "knee_R": [-125]}]]]}},
    # clip 5: both arms stretched to the left holding the staff (f115-134)
    "the_hidden_blade-5": {"1": {"keys_more_add": [[115, {"shoulder_L": [70, 40, 0], "shoulder_R": [70, -20, 0], "elbow_L": [10], "elbow_R": [30]}],
                                                   [134, {"shoulder_L": [70, 40, 0], "shoulder_R": [70, -20, 0], "elbow_L": [10], "elbow_R": [30]}]]}},
    # clip 11: the young man leans ~30 deg forward at the waist, hand on the sword (f0-111)
    "the_hidden_blade-11": {"1": {"keys": [[0, {"spine": [-28, 0, 0], "chest": [-5, 0, 0]}], [111, {"spine": [-28, 0, 0], "chest": [-5, 0, 0]}]],
                                  "key_strength": 3}},
    # clip 12: red bends DEEP at the waist carrying the box (f57-110), then kneels in seiza (f125+);
    # blue rests her head on her raised left hand (f76+)
    "the_hidden_blade-12": {"1": {"keys": [[57, {"spine": [-60, 0, 0], "chest": [-15, 0, 0], "hip_L": [45, 0, 0], "hip_R": [45, 0, 0], "knee_L": [-55], "knee_R": [-55],
                                                 "shoulder_L": [85, 0, 0], "shoulder_R": [85, 0, 0], "elbow_L": [20], "elbow_R": [20]}],
                                           [110, {"spine": [-60, 0, 0], "chest": [-15, 0, 0], "hip_L": [45, 0, 0], "hip_R": [45, 0, 0], "knee_L": [-55], "knee_R": [-55],
                                                  "shoulder_L": [85, 0, 0], "shoulder_R": [85, 0, 0], "elbow_L": [20], "elbow_R": [20]}],
                                           [125, {"spine": [-25, 0, 0], "shoulder_L": [60, 0, 0], "shoulder_R": [60, 0, 0], "elbow_L": [40], "elbow_R": [40]}],
                                           [149, {"spine": [-25, 0, 0], "shoulder_L": [60, 0, 0], "shoulder_R": [60, 0, 0], "elbow_L": [40], "elbow_R": [40]}]],
                                  "legs": [[120, 149, "kneeling"]], "leg_pose": [[120, 149, SEIZA]],
                                  "pelvis_z": [[110, 0.85], [124, 0.4], [149, 0.38]], "key_strength": 2},
                            "2": {"keys": [[76, {"shoulder_L": [45, 25, 0], "elbow_L": [140]}], [149, {"shoulder_L": [45, 25, 0], "elbow_L": [140]}]]}},
    # clip 21: tiny far runner, unreliable marks: crouched in the grass (f15-33), then an upright run, arms low
    "the_hidden_blade-21": {"1": {"pose": {"spine": [None, 0, 0], "chest": [None, 0, 0]},
                                  "legs": [[15, 33, "kneeling"]],
                                  "keys": [[15, {"spine": [-30, 0, 0]}], [33, {"spine": [-30, 0, 0]}],
                                           [40, {"spine": [-12, 0, 0], "head": [0, 0, None], "shoulder_L": [25, 5, 0], "shoulder_R": [25, -5, 0], "elbow_L": [70], "elbow_R": [70]}],
                                           [149, {"spine": [-12, 0, 0], "head": [0, 0, None], "shoulder_L": [25, 5, 0], "shoulder_R": [25, -5, 0], "elbow_L": [70], "elbow_R": [70]}]],
                                  "key_strength": 3}},
    # clip 22: forward hunch only, never a sideways lean
    "the_hidden_blade-22": {"1": {"pose": {"spine": [None, 0, 0], "chest": [None, 0, 0], "shoulder_L": [10, 8, None], "shoulder_R": [10, -8, None],
                                           "elbow_L": [20], "elbow_R": [20]},
                                  "keys": [[0, {"spine": [-15, 0, 0]}], [149, {"spine": [-15, 0, 0]}]]}},
    # clip 26: seated blue stays upright (no recline / side lean / twist) f20-75; 3/4 toward the camera at the start
    "the_hidden_blade-26": {"2": {"keys": [[20, UPRIGHT], [75, UPRIGHT]], "key_strength": 2,
                                  "yaw": [[0, 150], [19, 145], [30, 135]], "yaw_sigma": 10}},
}


def main():
    c = config.load(); lib = Library(c)
    names = json.load(open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8"))
    inv = {v: k for k, v in names.items()}
    for pkg, roles in ROLES.items():
        cl = next(x for x in lib.clips if x.name == inv[pkg]); p = folders(c, cl)["review"]
        r = json.load(open(p, encoding="utf-8"))
        for t, role in roles.items():
            r["people"].setdefault(t, {"height": 1.72, "name": ""}).update(role=role, colour=R2C[role])
        json.dump(r, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(pkg, "roles", roles)
    for pkg, people in FIX.items():
        p = os.path.join(ROOT, "output", inv[pkg], "hints.json")
        h = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        for tid, d in people.items():
            e = h.setdefault("people", {}).setdefault(tid, {})
            d = dict(d)
            add = d.pop("keys_more_add", None)
            e.update(d)
            if add:
                e.setdefault("keys_more", []).append(add)
        json.dump(h, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(pkg, "hints", sorted(people))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Pose hints from the v2.1 review of the 26 隐剑鬼爪 clips, merged into output\\<clip>\\hints.json (backup
hints.before_v21fix.json).  Angles: XYZ euler deg on the rig (forward bend = spine/chest x < 0; hip / shoulder x > 0 =
thigh / arm forward; knee x < 0 = bent; elbow x > 0 = bent; shoulder y: + moves the hand toward -x (L outward, R inward);
hip y: + = toward -x (L abduct, R adduct))."""
import json, os, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARMS_DOWN = {"shoulder_L": [8, 0, None], "shoulder_R": [8, 0, None], "elbow_L": [15], "elbow_R": [15]}
HANDS_FRONT_WAIST = {"shoulder_L": [20, -12, 0], "shoulder_R": [20, 12, 0], "elbow_L": [70], "elbow_R": [70]}
OVERHEAD = {"shoulder_L": [160, -10, 0], "shoulder_R": [160, 10, 0], "elbow_L": [25], "elbow_R": [25]}


def K(*pairs):
    return [[f, p] for f, p in pairs]


FIX = {
    # clip 1: old lord (tracks 2 front / 1 back): hands together at the waist / arms hang in the sleeves;
    # dark-robed man (5 in shot 1 upright, 6 in shot 2 kneeling by the wall)
    "the_hidden_blade-1": {
        "2": {"pose": HANDS_FRONT_WAIST},
        "1": {"pose": ARMS_DOWN},
        "5": {"pose": {"spine": [0, None, None], "chest": [0, None, None]}},
        "6": {"legs": [[210, 448, "kneeling"]]}},
    # clip 3: f120+ upright, wide low stance, sword overhead
    "the_hidden_blade-3": {"1": {"keys": K((125, dict(OVERHEAD, spine=[0, None, None], chest=[0, None, None],
                                                       hip_L=[None, 20, None], hip_R=[None, -20, None])),
                                          (149, dict(OVERHEAD, spine=[0, None, None], chest=[0, None, None],
                                                     hip_L=[None, 20, None], hip_R=[None, -20, None])))}},
    # clip 4: f90-105 both hands overhead, head up; f135+ deep squat knees wide
    "the_hidden_blade-4": {"1": {"keys": K((88, dict(OVERHEAD, head=[0, None, None])), (106, dict(OVERHEAD, head=[0, None, None]))),
                                 "keys_more": [K((135, {"hip_L": [75, 25, 0], "hip_R": [75, -25, 0], "knee_L": [-115], "knee_R": [-115]}),
                                                 (149, {"hip_L": [75, 25, 0], "hip_R": [75, -25, 0], "knee_L": [-115], "knee_R": [-115]}))]}},
    # clip 5: f7 hands low at the right hip; f45-60 sword overhead, torso bent forward-left
    "the_hidden_blade-5": {"1": {"keys": K((0, {"shoulder_L": [25, -30, 0], "elbow_L": [40], "shoulder_R": [15, 0, 0], "elbow_R": [30]}),
                                          (15, {"shoulder_L": [25, -30, 0], "elbow_L": [40], "shoulder_R": [15, 0, 0], "elbow_R": [30]})),
                                 "keys_more": [K((45, dict(OVERHEAD, spine=[-20, 0, 15])), (60, dict(OVERHEAD, spine=[-20, 0, 15])))]}},
    # clip 10: red swings the staff with both arms high around f52
    "the_hidden_blade-10": {"1": {"keys": K((46, OVERHEAD), (58, OVERHEAD))}},
    # clip 12: red crouches with a box (f52), bends deep at the waist lifting it (f97), kneels in seiza (f130+)
    "the_hidden_blade-12": {"1": {"keys": K((45, {"spine": [-30, 0, 0], "hip_L": [90, 0, 0], "hip_R": [90, 0, 0], "knee_L": [-120], "knee_R": [-120],
                                                  "shoulder_L": [60, 0, 0], "shoulder_R": [60, 0, 0], "elbow_L": [60], "elbow_R": [60]}),
                                           (60, {"spine": [-30, 0, 0], "hip_L": [90, 0, 0], "hip_R": [90, 0, 0], "knee_L": [-120], "knee_R": [-120],
                                                 "shoulder_L": [60, 0, 0], "shoulder_R": [60, 0, 0], "elbow_L": [60], "elbow_R": [60]}),
                                           (92, {"spine": [-55, 0, 0], "chest": [-15, 0, 0], "hip_L": [35, 0, 0], "hip_R": [35, 0, 0], "knee_L": [-45], "knee_R": [-45],
                                                 "shoulder_L": [80, 0, 0], "shoulder_R": [80, 0, 0], "elbow_L": [20], "elbow_R": [20]}),
                                           (104, {"spine": [-55, 0, 0], "chest": [-15, 0, 0], "hip_L": [35, 0, 0], "hip_R": [35, 0, 0], "knee_L": [-45], "knee_R": [-45],
                                                  "shoulder_L": [80, 0, 0], "shoulder_R": [80, 0, 0], "elbow_L": [20], "elbow_R": [20]})),
                                  "legs": [[130, 149, "kneeling"]], "leg_pose": [[130, 149, {"hip": 105, "knee": -135}]]}},
    # clip 18: blue bows low / kneels beside the old man from f40 on
    "the_hidden_blade-18": {"2": {"legs": [[40, 149, "kneeling"]],
                                  "keys": K((40, {"spine": [-55, None, None], "chest": [-15, None, None]}),
                                            (149, {"spine": [-55, None, None], "chest": [-15, None, None]}))}},
    # clip 21: running toward the camera, hunched, arms low
    "the_hidden_blade-21": {"1": {"pose": dict(ARMS_DOWN, spine=[-20, None, None], chest=[0, None, None])}},
    # clip 22: arms down at the sides (no raised / crossed arms)
    "the_hidden_blade-22": {"1": {"pose": {"shoulder_L": [10, 8, None], "shoulder_R": [10, -8, None], "elbow_L": [20], "elbow_R": [20]}}},
    # clip 24: blue holds both hands together in front of the chest (f130+)
    "the_hidden_blade-24": {"2": {"keys": K((128, {"shoulder_L": [40, -15, 0], "shoulder_R": [40, 15, 0], "elbow_L": [90], "elbow_R": [90]}),
                                            (149, {"shoulder_L": [40, -15, 0], "shoulder_R": [40, 15, 0], "elbow_L": [90], "elbow_R": [90]}))}},
    # clip 25: blue in a deep squat, both knees up, torso forward, hands in the dirt
    "the_hidden_blade-25": {"2": {"pose": {"hip_L": [110, 12, 0], "hip_R": [110, -12, 0], "knee_L": [-140], "knee_R": [-140],
                                           "spine": [-30, None, None], "shoulder_L": [70, 0, None], "shoulder_R": [70, 0, None]}}},
    # clip 26: seated blue stays nearly upright around f40-65, left hand on the knee
    "the_hidden_blade-26": {"2": {"keys": K((38, {"spine": [0, None, None], "chest": [0, None, None], "shoulder_L": [30, 0, 0], "elbow_L": [40]}),
                                            (66, {"spine": [0, None, None], "chest": [0, None, None], "shoulder_L": [30, 0, 0], "elbow_L": [40]}))}},
}


def main():
    names = json.load(open(os.path.join(ROOT, "data", "package_names.json"), encoding="utf-8"))
    inv = {v: k for k, v in names.items()}
    for pkg, people in FIX.items():
        p = os.path.join(ROOT, "output", inv[pkg], "hints.json")
        h = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        bk = p.replace("hints.json", "hints.before_v21fix.json")
        if os.path.exists(p) and not os.path.exists(bk):
            shutil.copyfile(p, bk)
        for tid, d in people.items():
            h.setdefault("people", {}).setdefault(tid, {}).update(d)
        json.dump(h, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(pkg, "->", sorted(people))


if __name__ == "__main__":
    main()

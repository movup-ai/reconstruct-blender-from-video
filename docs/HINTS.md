# Fixing a character: pose hints, roles and tracks

Everything a person does comes from the detected joints (marks) + `output\<clip>\hints.json` (by hand) + the roles
in the marks `review.json`. Fix by editing those, then refit and re-render the clip. Never edit pipeline code for a
single clip.

## Commands (always via package name; PowerShell mangles the Chinese clip names)

```
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe pipeline\redo_clip.py the_hidden_blade-12 --stages none --zoom 24        # close-up sheets only
.venv\Scripts\python.exe pipeline\redo_clip.py the_hidden_blade-12 --angles 40,90 --chars P1       # fitted angles + yaw
.venv\Scripts\python.exe pipeline\redo_clip.py the_hidden_blade-12 --zoom 24                       # refit + set + render + sheets
.venv\Scripts\python.exe pipeline\redo_clip.py the_hidden_blade-12 --frames 60,85,106              # sheets at chosen frames
.venv\Scripts\python.exe pipeline\tracks_info.py the_hidden_blade-12                                # tracks, roles, colours
.venv\Scripts\python.exe pipeline\track_edit.py the_hidden_blade-12 show 1 2                        # box x / height per 5 frames
.venv\Scripts\python.exe pipeline\jerk_check.py <clip name>                                         # sudden moves (clip name from data\package_names.json)
```
Close-up sheets: `Work_auto\review_v21\<package>_zoom_K.jpg`, one row per frame, per character three tiles:
film | film + detected joints (yellow dots) + fitted skeleton (colour lines) | render. Label `fNN Pk`.
`--angles` prints `kind colour fNN yaw=… (faces camera at A, back to camera at B)` and every joint angle.
Frames are 0-based source frames (the labels on the sheets).

## Rig and angle conventions (XYZ euler, degrees, Blender order R = Rz·Ry·Rx)
Rest pose: the character faces +Y, its right side is +X, Z up, arms and legs hanging straight down.
| joint | x | y | z |
| --- | --- | --- | --- |
| spine, chest | < 0 bends forward, > 0 leans back | side tilt (+ = upper body toward its right, +X) | twist |
| head | < 0 looks down, > 0 looks up | tilt | + turns toward its left (screen-right when it faces camera) |
| shoulder_L / _R | + raises the arm forward: 90 = horizontal in front, 180 = straight up | + moves the hand toward -X: L outward (90 = arm out to the side), R inward (R arm out to the side = -90) | twist of the upper arm |
| elbow_L / _R `[x]` | + bends the forearm forward/up (0 straight, 90 right angle, 140 hand at shoulder) | | |
| hip_L / _R | + thigh forward (90 = thigh horizontal, seated) | + toward -X: L spreads out, R crosses in (R out = negative) | twist |
| knee_L / _R `[x]` | < 0 bends (−90 shin back square, −135 seiza) | | |
`pelvis` exists but is normally 0. Body heading = root `yaw`; `--angles` tells you the yaw that faces the camera.

## hints.json  (`{"people": {"<track id>": {...}}}`; track id = the track number from tracks_info)
**Frame numbers in people hints (and the camera path) count from the start of THE SHOT the person is in**, not
from the start of the clip (run_pipeline.slice_pose renumbers per shot). In a one-shot clip that is the same thing;
in a multi-shot clip subtract the shot's first frame (shots: `tracks_info.py` header). Yaw keys are interpolated the
short way round (12 → 357 turns 15°).
- `keys`: `[[frame, {"joint": [x, y, z], "elbow_L": [x]}], ...]` — poses at frames, interpolated between keys,
  only inside the first..last key frame. `null` = axis left to the picture. Two keys with the same pose hold it over
  the range. `keys_more`: a list of further key lists (separate ranges / joints).
- `key_strength`: degrees the fit may leave a key (default 6; 2 = strict, 1.5 = very strict).
- `pose`: `{"joint": [x, y, z]}` held over the WHOLE track (strong, σ 3°). Use for "arms always down" etc.
- `legs`: `"seated"` | `"hidden"` (legs out of picture: standing legs assumed) | `"kneeling"`, or per range
  `[[f0, f1, "kneeling"], [f2, f3, "standing"]]`. `leg_pose`: `{"hip": 105, "knee": -135}` (seiza) or ranges
  `[[f0, f1, {...}]]`. `seat_height` (m). `pelvis_z`: number or `[[frame, m], ...]` — only used while legs are
  seated / kneeling / hidden.
- `yaw`: number or `[[frame, deg], ...]`, `yaw_sigma` (deg, smaller = stricter, 8 is firm), `yaw_range` [f0, f1].
- `distance`: metres from the camera along the floor, number or `[[frame, m], ...]` (fixes size / depth).
- `use_kps`: list of trusted COCO joints (others ignored): 0 nose, 1/2 eyes L/R, 3/4 ears, 5/6 shoulders L/R,
  7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles. Use when detections sit on a wrong object.
- `smooth`: gaussian sigma in frames (default 1.5; 3 removes jitter, softens fast moves).
- `head_px`: `[[frame, u, v], ...]` head centre in the picture (px of the full frame) when the head is not detected.
Left / right are the CHARACTER's: a person facing the camera has their left hand on screen-right.

## Feet on the ground (clip-level `"plant_feet": true`, used for the_hidden_blade-1, -13, -21)
At render time `pipeline\ground_feet.py` moves each body so the lowest foot touches the floor and re-solves the legs
of standing / walking frames (pelvis above 70% of standing height) into planted alternating steps. That re-solve
OVERRIDES leg keys. Per person (hints people entry):
- `"keep_legs": [[f0, f1], ...]` - keep your hand-set / fitted legs in these clip frames (only the vertical fix).
- `"stand": [[f0, f1], ...]` - stands still here: position held, legs straight and together.
- `"step_len": 0.6` - max foot slide before a new step (m, default 0.36; larger = longer strides).
- `"plant": false` - leave this person exactly as fitted.
The vertical fix and IK are applied in the RENDER stage, so after a hint change run fit + build_env + render, and look at
the rendered sheets (redo_clip.py --zoom / all_frames.py), not the fitted angles alone.
`pipeline\all_frames.py <pkg> [--crop x0,y0,x1,y1] [--from F --to T]`: every frame film | render, 30-42 per sheet.

## Roles / colours (marks review.json, `people` → `{"role", "colour"}`)
main=red, second=blue, third=green, fourth=yellow, extra=purple, ignore = not built. One person keeps one colour
in every shot (several track ids may share a role/colour when they are the same person in different shots).
The path is printed by tracks_info (`folders(c, cl)["review"]`); edit with Python (json, ensure_ascii=False).

## Tracks (marks pose.json) — `pipeline\track_edit.py`
`swap A B F0 [F1]` (tracker switched people), `move A B F0 [F1]`, `drop A F0 [F1]` (garbage detections),
`hold A F0 F1` (person keeps still after detection stops), `fill A F0 F1` (bridge a short gap). A backup
`pose.before_trackedit.json` is kept.

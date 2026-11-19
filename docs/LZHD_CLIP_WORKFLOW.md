# One clip to the LZHD V1.0 spec (11月17日 batch)

Spec: `SOURCE\LZHD白模视频复刻需求及验收标准.pdf` + palette `SOURCE\LZHD 白模配色语义色板.pdf`.
The approved trial is **`Dragon_11月17日(22)`**: copy its approach. Its files are the examples:
`output\Dragon_11月17日(22)\make_env.py` (set from camera rays), `output\Dragon_11月17日(22)\make_stool.py` (moving prop),
`output\11月17日原始素材191秒\Dragon\11月17日(22)\hints.json` (camera, light, people), `...\review.json` (heights),
trial package `DELIVERY\lzhd_trial\dragon-22.zip`.

Run everything from the project folder with `.venv\Scripts\python.exe` and `PYTHONIOENCODING=utf-8`.
Video of a clip: `SOURCE\11月17日原始素材191秒\<Dragon|Swan>\<stem>.mp4`; clip name `<Dragon|Swan>_<stem>`, e.g.
`Swan_11月17日(5)`. Marks folder (pose.json, review.json, **hints.json**): `output\11月17日原始素材191秒\<film>\<stem>\`.
Set files (env_claude.json, make_env.py, anim_props.json): `output\<clip name>\`. Work folder: `Work_auto\<clip name>\`.
Package name: `dragon-<n>` / `swan-<n>` with n = the number in the file name; Dragon's `11月17日.mp4` -> `dragon-0`.

## What the client checks (the spec, short)
- People: the official human_master figure (done by the pipeline: style "master"), position / size / facing /
  every limb matching the film frame by frame (yellow dots = detected joints = the truth; fit poses to them).
  No floating feet, no limbs through bodies or furniture, front/back order as in the film (a hand in front stays in front).
- Set: perspective, depth, key reference objects at the right place and size, contact (sitting, holding), occlusion.
  Not an empty background. Static objects never move. Model what affects space, occlusion, interaction; skip tiny things.
- Colours by meaning (env_claude `"colour"`): `bg` walls / floor / ground / stairs / ceiling / buildings / terrain,
  trees and plants too; `fg` objects between the camera and the people (a table in front of them);
  `near` objects a person touches or handles in the shot (cup, stool, car door they open, the car they lean on);
  `far_i` furniture / vehicles / doors nobody touches; `lamp` visible light fixtures; `special` anything else.
  All parts of an object take its colour.
- Light: main light direction / strength and the bright-dark relation of the film (hints `"light"`).
- Camera: the pipeline's tracked camera; fix height / tilt with a hint when the set or the people disagree.

## Steps
1. **Look.** `pipeline\env_frames.py <clip>` -> `Work_auto\<clip>\env_frames.jpg` (frames with a px grid);
   `Work_auto\<clip>\env_auto.jpg` + `output\<clip>\env_auto.json` (MoGe/SAM proposal, camera comparison in
   `logs\env_auto_1117.log`); `pipeline\tracks_info.py <clip>` (people, roles). Check the roles in review.json:
   a detection that is not a real person -> role "ignore"; one person = one colour; main red, second blue, third
   green, fourth yellow (biggest / most important first).
2. **Camera.** If the fitted camera (height, tilt) and MoGe disagree, decide with known sizes: door 2.0-2.1 m,
   desk / table 0.72-0.78, chair seat 0.45, stair rise 0.15-0.26, car 1.45 tall, and the people's feet on the floor.
   Write hints.json `{"camera": {"height": h, "pitch_down_deg": p}}` (keep other keys), then
   `pipeline\run_pipeline.py <video> --stage fit --name <clip>` (always pass --name).
3. **Heights.** The master figure has short legs: adults need ~1.80-1.90 m (women ~1.70-1.80, children by brief
   only). Set `"height"` per person in review.json (start at 1.86), refit, then `pipeline\foot_gaps.py <clip>`:
   aim for gaps within +-0.05 m where the feet are in the picture. Feet out of the picture for a stretch:
   hints people `"legs": [[f0, f1, "standing"], [f2, f3, "hidden"]]` + `"pelvis_z"`; sitting: `"legs": "seated"`
   (+ `"seat_height"`). See docs\HINTS.md for all person hints (keys, pose, yaw, distance, use_kps).
4. **Set.** Write `output\<clip>\make_env.py` that computes `env_claude.json` from the fitted camera
   (`build_env.Cam(d).ray(u, v, frame)`; floor points, planes) like the trial's, with the colours above.
   Then `pipeline\build_env.py <clip>` and read `Work_auto\<clip>\env_check.jpg` (film | overlay | set);
   2-3 rounds. Format: docs\ENV_AUTHORING.md. Re-run make_env.py + build_env.py after every refit
   (a fit rewrites scene_data). Do not add a ceiling unless it is seen (it blocks the sun).
5. **Moving props** (only when a person moves a sizeable object): `output\<clip>\anim_props.json` per-frame keys
   `[x, y, z, rx, ry, rz]` (see make_stool.py), then `pipeline\anim_props.py <clip>` after build_env.
6. **Light.** hints `"light": {"az": deg (0 = from behind the camera, -90 from screen left, 90 right, 180 back
   light), "el": deg, "sun": 1-3, "ambient": 0.3-0.6, "points": [[x, y, z, watts, radius]]}` (indoor pools: ~100-150 W
   at ~2.9 m). Outdoors: sun 2.5-3, ambient 0.6-0.8, no points.
7. **Render + review.** `pipeline\render_lzhd.py <video> <clip> DELIVERY\lzhd_batch --pkg <pkg> --no-package`
   (~30-60 s), then `pipeline\lzhd_review.py <video> <clip> --n 8 --out <jpg>` (more frames / `--frames` around
   fast action, `--crop` to zoom) and READ the sheet: every person on their dots, feet on the floor, props in place,
   nothing through anything, occlusion order right. Fix (hints keys / heights / set) and repeat. Usually 2-4 rounds.
8. **Package.** `pipeline\render_lzhd.py <video> <clip> DELIVERY\lzhd_batch --pkg <pkg>`: all
   `output_checks` must be true. Then the multi-view check:
   `"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe" -b "DELIVERY\lzhd_batch\<clip>\model_3d\<pkg>.blend" -P pipeline\multiview_check.py -- DELIVERY\lzhd_batch\multiview_<pkg>.jpg 1,<mid>,<last>`
   and look at it (same figure proportions from every view).
9. Write the final review sheet to `DELIVERY\lzhd_batch\review_<pkg>.jpg` (lzhd_review --n 8).

## Rules
- Touch only your own clips' files (marks hints/review, output\<clip>\, Work_auto\<clip>\, your packages).
  Never edit pipeline\*.py, config.json, assets\ or another clip. If the pipeline itself is wrong, stop that clip and
  report it.
- Never skip a fit with `--name`; never run batch scripts.
- A clip you cannot make match (unreadable, very wrong detections): package nothing, report why.

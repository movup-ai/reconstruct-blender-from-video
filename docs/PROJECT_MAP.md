# Project map

What every folder and tracked file is for, and what the 2026-11-18 review decided about it. Regenerate the file
table after big changes; the decisions below are written by hand.

## Folders

| Folder | Purpose | In git |
|---|---|---|
| `agent\` | The app: local server, job queue, review logic, package validation, optional DeepSeek calls | yes |
| `web\` | The app's pages | yes |
| `pipeline\` | Processing (camera, fit, set, Blender, packaging) and the command-line tools | yes |
| `tests\` | Unit tests (`.venv\Scripts\python -m unittest discover -s tests`) | yes |
| `tools\v21\` | Stand-alone checker for v2.1 delivery packages | yes |
| `docs\` | Working notes; `docs\v21\` holds the client's delivery spec | yes |
| `assets\` | Character definitions used by the fit and Blender, and their source `.blend` files | yes |
| `samples\` | The client's original trial `.blend` files (reference only, nothing reads them) | yes |
| `inputs\test\` | Three short test clips | yes |
| `data\` | Clip-name tables; `jobs.db` (the app's job queue) is not in git | partly |
| `.venv\` | Python environment, made by `Setup.bat` | no |
| `models\` | Model weights: Depth Anything, MoGe-2, MotionBERT, SAM 2.1 | no |
| `third_party\` | MoGe and MotionBERT source code, downloaded by `Setup.bat` | no |
| `SOURCE\` | Input clips | no |
| `output\` | Per clip: detected joints, review, hints (what was detected and decided) | no |
| `Work_auto\` | Per clip: computed files, previews, renders, quality report; rebuilt by a refit | no |
| `SUBMIT\` | Finished `.blend` and videos | no |
| `logs\` | App log | no |

## Review decisions (2026-11-18)

Removed (all still in git history): `_archive\` (22 files: earlier apps and dev backups),
`pipeline\blender_build.before_master.py`, `data\v21_pose_fixes.py`, `data\v21_pose_fixes2.py`,
`assets\sample_character_11_2.json`, `assets\sample_character_check.jpg`, `models\trt_cache\` (292 MB, not in git).

Fixed: unused imports in ten modules; every file handle closed (about 80 statements); machine-specific
`D:\` paths in code and docs; the `bush` prop below the floor (failing test); `tracks_info.py` crashing on a clip
without a review.

Kept on purpose:

| What | Why |
|---|---|
| The tools in "Command-line tools" and "Client packaging" below that no code calls | They are run by hand; the docs or other tools name them. Every one imports or runs. Owner confirmed keeping them (2026-11-19). |
| `samples\*.blend`, `data\package_tasks.json` | Client originals and task records |
| `models\mask2former_ade\` (826 MB) | Needed by `semseg.py` (facebook/mask2former-swin-large-ade-semantic, downloaded by Setup.bat) |
| `models\grounding_dino\` (890 MB) | Needed by `held_objects.py` (IDEA-Research/grounding-dino-base, downloaded by Setup.bat) |
| `models\sam2\` (857 MB) | Needed by `env_auto.py`; owner confirmed keeping it (2026-11-19) |
| `Work_auto\report_*.csv` | One small file per run: run history |

## Speed (measured 2026-11-19 on the earlier PC: i5-13400, no NVIDIA card)

Since 2026-11-23 the project is GPU only (owner's decision): the processor paths and `requirements-cpu.txt` are
removed, and every model runs on the NVIDIA card (RTX 4070 here). The numbers below are from the processor and are
kept as history; they have not been measured again on the card.

One clip from raw video to `.blend` and videos: 214–320 s for 137–149 frames.

| Candidate | Measured | Decision |
|---|---|---|
| Detection on the processor used rtmlib's one-crop-at-a-time path | 125 → 98 s, 260 → 85–164 s | Done: batched path |
| Two depth models per clip | one load saved | Done: shared |
| Smaller pose models (`"pose_mode": "balanced"`) | detection 87 → 26 s; loses part of small far people | Opt-in setting; owner chose to keep the exact models as default (2026-11-19) |
| CPU thread count | default fastest of 6 settings | Unchanged |
| Lower lens-measurement resolution | 1.5–3× faster, lens 5–10% off | Rejected |
| Caching depth runs | 1 repeat in 17 runs, < 1 s | Rejected |
| Remaining refit time | depth 15 s, inpainting 7 s, optical flow 5 s, fit 9 s | No waste found; only a different method would be faster |
| An NVIDIA card | detection is ~36% of the time on the processor | The largest gain left |

## Every tracked file

### Run by every clip (app → pipeline)

| File | Role | Referred to by | Check |
|---|---|---|---|
| `automark.py` | automark.py - automatic body keypoint marking for the white-model clips. | `pipeline/common.py`, `pipeline/run_pipeline.py` | imports |
| `pipeline/blender_build.py` | White-model L3 rebuild - generic Blender builder of the automatic pipeline. | `pipeline/build_env.py`, `pipeline/extract_base_character.py`, `pipeline/extract_human_master.py`, `pipeline/extract_sample_character.py` +5 | Blender script |
| `pipeline/build_env.py` | Hand-built set for one clip (the client-sample style: clean grey room / landscape made of simple recognisable | `README.md`, `docs/ENV_AUTHORING.md`, `docs/HINTS.md` +13 | imports |
| `pipeline/camera_estimate.py` | automatic camera for one shot: rotation (+ zoom) tracking, steadying, and camera height / pitch / roll. | `pipeline/camera_solve.py`, `pipeline/ground_solve.py`, `pipeline/run_pipeline.py` | imports |
| `pipeline/camera_solve.py` | per-frame camera rotation from tracked background features (pan / tilt / roll shots). | `pipeline/camera_estimate.py`, `pipeline/run_pipeline.py` | imports |
| `pipeline/common.py` | shared settings and helpers for the automatic white-model pipeline | `Setup.bat`, `agent/config.py`, `agent/server.py`, `agent/validate.py` +43 | imports |
| `pipeline/compare.py` | Film frame next to the rendered frame for a few frames of a clip - to see what does not match. | `README.md`, `docs/ENV_AUTHORING.md`, `pipeline/audit.py`, `pipeline/build_env.py` +7 | imports |
| `pipeline/dedupe_roles.py` | Find people the detector saw twice (two tracks on the same body) in the automatic roles, and mark the smaller / | `pipeline/run_pipeline.py` | imports |
| `pipeline/fit_people.py` | 3D pose of every person: the L3 rig is fitted to the 2D marks of each frame (COCO-17) | `pipeline/build_env.py`, `pipeline/pose3d.py`, `pipeline/run_pipeline.py` | imports |
| `pipeline/get_model.py` | download the Depth Anything V2 Small depth model (Apache-2.0, ~99 MB) into models\ | `Setup.bat` | imports |
| `pipeline/ground_feet.py` | Feet on the ground and a real stepping gait (client feedback: "äººç‰©çš„æ‚¬æµ®æ„Ÿ/æ¼‚æµ®æ„Ÿå¤ªå¼º", "æ­¥æ€å’ŒåŽŸç‰‡ä¸ç¬¦"). | `docs/HINTS.md`, `pipeline/render_lzhd.py`, `pipeline/run_pipeline.py` | imports |
| `pipeline/ground_solve.py` | ground plane + camera height from the people (their feet stand on the ground, their heads are ~height above). | `pipeline/run_pipeline.py` | imports |
| `pipeline/lens_auto.py` | The clip's lens from the picture (MoGe-2), instead of one fixed lens for every clip. | `README.md`, `pipeline/run_pipeline.py`, `tests/test_quality.py` | imports |
| `pipeline/motion_prior.py` | Real-world movement knowledge as a SOFT recommendation for the fitted characters. | `docs/MOTION_RESEARCH.md`, `pipeline/fit_people.py` | imports |
| `pipeline/package.py` | Rendered frames -> a submission package.  package() follows the v2.1 rules (2026-11-12, see its docstring); | `agent/server.py`, `agent/validate.py`, `docs/HINTS.md` +24 | imports |
| `pipeline/pose3d.py` | 3D pose lifting: COCO-17 2D tracks -> H36M-17 3D joints per frame (MotionBERT, GPU). | `pipeline/common.py`, `pipeline/fit_people.py` | imports |
| `pipeline/props.py` | Library of simple, recognisable white-model objects (like the client samples: a door has a frame, a panel and a | `README.md`, `agent/ai.py`, `agent/brief.py`, `agent/review.py` +12 | imports |
| `pipeline/quality.py` | Quality of a fitted clip in numbers, and the filter for tracks that are not worth fitting. | `README.md`, `pipeline/env_auto.py`, `pipeline/run_pipeline.py` +2 | imports |
| `pipeline/rig.py` | numpy copy of the L3 mannequin in blender_build.py (same joints, same Blender XYZ euler order). | `assets/sample_character.json`, `pipeline/blender_build.py`, `pipeline/build_env.py`, `pipeline/compare.py` +15 | imports |
| `pipeline/run_pipeline.py` | Automatic white-model pipeline:  video  ->  finished submission folder + zip, no manual steps. | `README.md`, `agent/brief.py`, `agent/clips.py` +26 | imports |
| `pipeline/scene_depth.py` | grey set from a depth model: clean background plate (people removed) -> Depth Anything V2 -> metric depth | `agent/server.py`, `pipeline/camera_estimate.py`, `pipeline/get_model.py`, `pipeline/run_pipeline.py` +1 | imports |
| `pipeline/scene_export.py` | Export the open Blender scene as model_3d/reconstruction/scene.json (v2.1 delivery rule: real scene data, not a | `assets/reconstruction/README.md`, `pipeline/blender_build.py` | imports |
| `pipeline/shots.py` | hard-cut detection.  A cut is a frame whose picture has nothing in common with the frame before it: low | `README.md`, `agent/ai.py`, `agent/brief.py`, `agent/server.py` +19 | imports |

### Command-line tools for checking and fixing a clip by hand

| File | Role | Referred to by | Check |
|---|---|---|---|
| `pipeline/all_frames.py` | Every frame of a clip: film / render side by side, 40 frames per sheet (8 columns x 5 rows). | `docs/HINTS.md` | imports |
| `pipeline/anim_props.py` | Moving set pieces (e.g. a chair a person pulls) for a clip: output/<clip>/anim_props.json -> scene_data["props"]. | `docs/LZHD_CLIP_WORKFLOW.md` | imports |
| `pipeline/audit.py` | Frame-by-frame audit: how far the fitted mannequin's joints are from the joints the local model saw. | `pipeline/quality.py` | imports |
| `pipeline/claude_set.py` | Claude (Anthropic API) builds the background set from gridded frames + local measurements + object notes, checks it over the film, corrects it -> env_claude_ai.json. | `pipeline/run_pipeline.py`, `agent/server.py` | imports |
| `pipeline/colour_card.py` | The colour card (docs/COLOUR_CARD.md): card values and the pass that colours characters, set pieces and props by meaning. | `pipeline/build_env.py`, `pipeline/run_pipeline.py` | imports |
| `pipeline/describe.py` | Reads an object note's description: library model, measured shape, or held by a person; colour meaning words. | `pipeline/env_auto.py`, `pipeline/held_objects.py` | imports |
| `pipeline/semseg.py` | Mask2Former ADE20K segmentation (GPU, fp16) and the class -> how-to-build table (ROLE) for the set. | `pipeline/env_auto.py` | imports |
| `pipeline/relief.py` | A set piece built from its measured surface (plants, curtains, statues): grid over the outline -> closed solid. | `pipeline/env_auto.py` | imports |
| `pipeline/held_objects.py` | Objects people carry (bag, phone, backpack): Grounding DINO + CLIP find them (held.json), placed as props on the fitted hand / back. | `pipeline/run_pipeline.py`, `pipeline/env_auto.py` | imports |
| `pipeline/env_auto.py` | Automatic set proposal from real 3D measurements (run this before writing env_claude.json by hand). | `docs/ENV_AUTHORING.md`, `docs/LZHD_CLIP_WORKFLOW.md`, `pipeline/build_env.py`,  | imports |
| `pipeline/env_frames.py` | Reference pictures for writing a clip's set by hand: for every shot, its first, middle and last frame with a pixel | `docs/ENV_AUTHORING.md`, `docs/LZHD_CLIP_WORKFLOW.md` | imports |
| `pipeline/foot_gaps.py` | Feet vs floor for every person of a fitted clip, and a suggested height for the human_master figure. | `docs/LZHD_CLIP_WORKFLOW.md`, `pipeline/quality.py` | imports |
| `pipeline/jerk_check.py` | Find sudden movements in a fitted clip: every frame, every character, every joint. | `docs/HINTS.md`, `pipeline/quality.py` | imports |
| `pipeline/lzhd_review.py` | Review sheet for an LZHD render: per frame, film (yellow = detected joints) / white model (same dots) side by side. | `docs/LZHD_CLIP_WORKFLOW.md` | imports |
| `pipeline/multiview_check.py` | LZHD V1.0 trial check (6.2): render every character of a delivered .blend from the front, the side and the back | `docs/LZHD_CLIP_WORKFLOW.md` | Blender script |
| `pipeline/redo_clip.py` | Redo one clip by package name (avoids passing Chinese clip names through PowerShell): | `docs/HINTS.md` | imports |
| `pipeline/track_edit.py` | Edit a clip's detected tracks (marks pose.json); a backup pose.before_trackedit.json is kept once. | `docs/HINTS.md` | imports (reconfigures the console, runs as a script) |
| `pipeline/tracks_info.py` | Per clip: every detected track (pose.json) with its review role / colour, frames, mean box, and which fitted | `docs/HINTS.md`, `docs/LZHD_CLIP_WORKFLOW.md` | imports (reconfigures the console, runs as a script) |

### Client packaging and batch tools

| File | Role | Referred to by | Check |
|---|---|---|---|
| `assets/reconstruction/reconstruct.py` | Rebuild the white-model shot from scene.json (schema "ba_whitemodel_scene" 1.0) in Blender. | `assets/reconstruction/README.md`, `pipeline/package.py`, `pipeline/scene_export.py` | Blender script |
| `pipeline/deliver.py` | v2.1 delivery: our packages -> the batch folder the client expects, checked with the client's own check_package.py. | `pipeline/package.py` | imports |
| `pipeline/render_lzhd.py` | Render + package one clip to the LZHD V1.0 spec (White-model Video Reconstruction and Acceptance Standard V1.0, 2026-11-15): | `docs/LZHD_CLIP_WORKFLOW.md` | imports |
| `pipeline/task_pack.py` | Fill the client's task zips (e.g. SOURCE\zacha-T001.zip) from our finished packages. | - | imports |
| `tools/v21/check_package.bat` | Double-click wrapper for check_package.py | `docs/v21/交付规范_v1.0.md`, `tools/v21/check_package.py` | - |
| `tools/v21/check_package.py` | Pre-upload delivery package checks (delivery specification sections 1 and 2). | `agent/validate.py`, `docs/v21/交付规范_v1.0.md`, `pipeline/deliver.py`, `pipeline/package.py` +2 | imports |
| `tools/v21/manifest_template.csv` | Client manifest template for v2.1 deliveries | `docs/v21/交付规范_v1.0.md` | - |

### One-time asset builders (run inside or before Blender)

| File | Role | Referred to by | Check |
|---|---|---|---|
| `pipeline/dump_base_character.py` | One-time (run inside Blender): dump the v2.1 supplied base character (character-base.blend, the legacy physics-prefixed pieces of | `pipeline/extract_base_character.py` | Blender script |
| `pipeline/dump_human_master.py` | One-time (run inside Blender): dump the official human_master.blend (rig + its 10 skinned pieces) for | `pipeline/extract_human_master.py` | Blender script |
| `pipeline/extract_base_character.py` | One-time: the v2.1 supplied base character (character-base.blend, dumped by dump_base_character.py) -> | `pipeline/dump_base_character.py`, `pipeline/rig.py` | imports |
| `pipeline/extract_human_master.py` | One-time: the official human_master.blend (dumped by dump_human_master.py) -> assets/human_master.json, used by | `pipeline/blender_build.py`, `pipeline/dump_human_master.py`, `pipeline/rig.py` | imports |
| `pipeline/extract_sample_character.py` | One-time: turn the client's sample character (November 2 trial blend, object L3.002 on Rigify rig.002) into | `pipeline/blender_build.py` | Blender script |

### App

| File | Role | Referred to by | Check |
|---|---|---|---|
| `.claude/launch.json` | Starts the app in the Claude desktop preview pane | - | - |
| `Setup.bat` | One-time install: NVIDIA-only install: PyTorch and TensorRT wheels, Python packages, models, TensorRT engines | `README.md`, `agent/server.py`, `automark.py`, `pipeline/scene_depth.py` +2 | - |
| `Start_Blender_Agent.bat` | Starts the app and restarts it after a crash | `README.md`, `Setup.bat`, `agent/__init__.py`, `web/index.html` | - |
| `agent/__init__.py` | App package: name and version | `agent/ai.py`, `agent/clips.py`, `agent/jobs.py`, `agent/server.py` +10 | imports |
| `agent/__main__.py` | App entry point (python -m agent): logging, server start, browser | `agent/ai.py`, `agent/brief.py`, `agent/selftest.py`, `agent/validate.py` +41 | imports |
| `agent/ai.py` | AI suggestions through the DeepSeek API: fills in the review form for a clip. | `agent/clips.py`, `agent/jobs.py`, `agent/review.py`, `agent/server.py` +3 | imports |
| `agent/brief.py` | Hand-off files for Claude.  When a person confirms a clip's detection, everything Claude needs to finish the clip | `agent/server.py`, `docs/LZHD_CLIP_WORKFLOW.md` | imports |
| `agent/clips.py` | Clip discovery, per-clip folders and status (status is read from the files, so it survives restarts). | `README.md`, `agent/__init__.py`, `agent/brief.py`, `agent/config.py` +28 | imports |
| `agent/config.py` | Settings and folders.  One config.json next to the launcher holds both the pipeline settings and the app settings. | `README.md`, `Setup.bat`, `agent/__main__.py`, `agent/ai.py` +30 | imports |
| `agent/jobs.py` | Persistent job queue (SQLite), worker pool with timeouts and retries, and the autopilot. | `README.md`, `agent/server.py`, `automark.py`,  +4 | imports |
| `agent/media.py` | Thumbnails and crops (cached on disk).  OpenCV cannot open Chinese paths on Windows, so videos are copied to a | `agent/ai.py`, `agent/server.py`, `tests/test_agent.py`, `web/app.css` +2 | imports |
| `agent/review.py` | The review (key points) of a clip: people roles / colours / heights, camera, set, objects, notes. | `README.md`, `agent/__init__.py`, `agent/ai.py`, `agent/brief.py` +23 | imports |
| `agent/selftest.py` | python -m agent.selftest - checks the installation and prints what is missing. | - | imports |
| `agent/server.py` | HTTP server (127.0.0.1 only): the web pages, the JSON API and the media files. | `README.md`, `tests/test_agent.py` | imports |
| `agent/validate.py` | Check submission packages against the client sample (November 2 trial) and the delivery spec. | `agent/config.py`, `agent/review.py`, `agent/server.py`,  +1 | imports |
| `check_template.html` | Page template of the dot editor that automark.py writes per clip | `automark.py`,  | - |
| `requirements.txt` | Python packages for a PC with an NVIDIA card | `Setup.bat` | - |
| `web/app.css` | App styles | `tests/test_agent.py`, `web/clip.html`, `web/help.html`, `web/index.html` +2 | - |
| `web/clip.html` | Clip review page | `agent/server.py` | - |
| `web/common.js` | Shared page script (API calls, header) | `tests/test_agent.py`, `web/clip.html`, `web/help.html`, `web/index.html` +2 | - |
| `web/favicon.svg` | App icon | `agent/server.py`, `tests/test_agent.py`, `web/common.js` | - |
| `web/help.html` | Help page | - | - |
| `web/index.html` | Clips page | `agent/server.py`, `docs/MOTION_RESEARCH.md` | - |
| `web/settings.html` | Settings page | - | - |
| `web/submissions.html` | Finished packages page | - | - |

### Characters and client samples

| File | Role | Referred to by | Check |
|---|---|---|---|
| `assets/base_character_v21.json` | v2.1 base character geometry (character_style base21) | `pipeline/blender_build.py`, `pipeline/extract_base_character.py`, `pipeline/rig.py` | - |
| `assets/base_v21/base_dump.json` | Raw dump of character-base.blend, input of extract_base_character.py | `pipeline/extract_base_character.py` | - |
| `assets/base_v21/character-base.blend` | Client-supplied v2.1 base character | `assets/base_character_v21.json`, `pipeline/dump_base_character.py`, `pipeline/extract_base_character.py`, `pipeline/rig.py` | - |
| `assets/human_master.json` | Human master character geometry (character_style master, the current default) | `pipeline/blender_build.py`, `pipeline/extract_human_master.py`, `pipeline/rig.py` | - |
| `assets/human_master/human_master.blend` | Client-supplied human master rig | `assets/human_master.json`, `pipeline/blender_build.py`, `pipeline/dump_human_master.py`, `pipeline/extract_human_master.py` +3 | - |
| `assets/human_master/master_dump.json` | Raw dump of human_master.blend, input of extract_human_master.py | `pipeline/extract_human_master.py` | - |
| `assets/reconstruction/README.md` | Explains scene.json and reconstruct.py inside a v2.1 package | `docs/v21/白模交付包内部格式与打包要求v2.1_text.txt`, `pipeline/package.py`,  | - |
| `assets/sample_character.json` | Client trial-sample character geometry (character_style sample) | `Setup.bat`, `agent/server.py`, `pipeline/blender_build.py`, `pipeline/extract_sample_character.py` | - |
| `samples/rig_L3.blend` | Client trial sample: the L3 rig the sample character was extracted from | `assets/sample_character.json` | - |
| `samples/sample_11_2.blend` | Client trial sample (11-2), cleaned | - | - |
| `samples/sample_11_2_raw.blend` | Client trial sample (11-2), as delivered | - | - |

### Data, docs, tests, test clips

| File | Role | Referred to by | Check |
|---|---|---|---|
| `.gitignore` | What git leaves out: environment, models, clips, work and results folders, config.json | - | - |
| `README.md` | How to install, start and use the app | `assets/reconstruction/README.md`, `docs/v21/白模交付包内部格式与打包要求v2.1_text.txt`, `pipeline/package.py`,  | - |
| `data/package_names.json` | Clip name -> client package name (needed by the official packaging) | `docs/HINTS.md`, `pipeline/all_frames.py`,  +12 | - |
| `data/package_tasks.json` | Client package name -> client task ID / zip (record of the 11-12 batch) | - | - |
| `docs/ENV_AUTHORING.md` | How to hand-build a set (env_claude.json) | `docs/LZHD_CLIP_WORKFLOW.md`, `pipeline/env_auto.py` | - |
| `docs/HINTS.md` | How to fix a character with hints.json, roles and track edits | `docs/LZHD_CLIP_WORKFLOW.md` | - |
| `docs/LZHD_CLIP_WORKFLOW.md` | Workflow for the LZHD delivery spec | - | - |
| `docs/MOTION_RESEARCH.md` | Gait and joint-range reference values used by motion_prior.py | `pipeline/ground_feet.py`, `pipeline/motion_prior.py` | - |
| `docs/v21/交付规范_v1.0.md` | - | - | - |
| `docs/v21/白模交付包内部格式与打包要求v2.1_text.txt` | - | - | - |
| `inputs/test/0529bfe27f2b1dab7f3ae164b6d05abd.mp4` | Test clip (client-approved trial sample) | - | - |
| `inputs/test/b5b6e9a3c5730842395b00d2f4b1841c.mp4` | Test clip (client-approved trial sample) | - | - |
| `inputs/test/ec349385598e67568611cec8540c4f90.mp4` | Test clip (client-approved trial sample) | - | - |
| `tests/test_agent.py` | Unit tests: naming, shots, camera, job store, review, props, zip, config, web API | - | - |
| `tests/test_quality.py` | Unit tests: junk filter, quality numbers, floor shift, lens | - | - |

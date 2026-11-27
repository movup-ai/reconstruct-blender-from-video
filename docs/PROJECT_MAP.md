# Project map

What every folder and file is for (one automatic run per clip: people, camera, floor and objects). The model flow is
in [MODEL_FLOW.md](MODEL_FLOW.md).

## Folders

| Folder | Purpose | In git |
|---|---|---|
| `agent\` | The app: local server, job queue (one `movement` stage), clip status, v2.1 package check | yes |
| `web\` | The app's pages | yes |
| `pipeline\` | the run and a few command-line tools | yes |
| `tests\` | Unit tests (`.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"`) | yes |
| `docs\` | These notes; `docs\v21\` holds the client's delivery spec | yes |
| `assets\` | The client's sample character (`sample_character.json`), the reconstruction script shipped in every package | yes |
| `data\` | App state, not in git: `jobs.db` (the job queue), `package_names.json`, `auto_names.json` (clips whose name the app made up, and the made-up names a real title replaced), skipped clips | no |
| `.venv\` | Python environment, made by `Setup.bat` | no |
| `models\` | Model weights (see MODEL_FLOW.md); `trt_cache\` TensorRT engines | no |
| `third_party\` | MoGe, SAM 3 and SAM 3D Body source code | no |
| `SOURCE\` | Input clips | no |
| `output\` | Per clip: tracked people (`pose.json`, `sam3_masks.npz`, `tracking_report.json`, `thumbs\P<n>.jpg` made by the app, `tracking_backups\`) | no |
| `Work_auto\` | Per clip: scene, saved depth, previews, renders, reports, job logs | no |
| `SUBMIT\` | Finished packages `<clip>\` and `<clip>.zip` | no |
| `logs\` | App log | no |

## Pipeline (`pipeline\`)

| File | Purpose |
|---|---|
| `run_pipeline.py` | the entry point: `--stage run` (the whole run, the default) or `--stage render` (re-render, zip, check a saved scene) |
| `ade_find.py`, `ade_objects.py` | Mask2Former panoptic once per clip: objects and every pixel's name; the 150 names → object models |
| `scene_objects.py`, `assets_lib.py`, `object_checks.py`, `asset_export.py` | the objects step: the models of `assets\objects`, depth on each object's best views, a box per object, the checks |
| `../automark.py` | SAM 3 person tracks + RTMPose joints → `pose.json` |
| `sam3_tracking.py` | SAM 3 video worker, packed masks, pose measurement inside masks, safe publishing of new marks |
| `shots.py` | Hard-cut detection |
| `lens_auto.py` | The clip's lens from the picture (MoGe-2) |
| `camera_estimate.py` | Camera rotation, zoom, steadying, level roll, pitch and height from people and picture cues |
| `ground_solve.py` | The `World` camera/ground model (per-frame rotation, travel and zoom) |
| `scene_depth.py` | Depth Anything V2 Small (camera depth cues) and the ground disc (`flat_set`) |
| `get_model.py` | Downloads Depth Anything V2 Small (Setup.bat) |
| `metric_scene.py`, `metric_observe.py` | MoGe depth (cached) + Mask2Former's floor / wall / ceiling pixels: gravity, world yaw, camera height, body-proxy scale |
| `sam3d_body.py` | SAM 3D Body worker, mesh raster, depth calibration, metric placement |
| `body_constraints.py`, `fit_people.py`, `motion_prior.py` | Character fit to the joints with SAM 3D Body priors (motion prior opt-in) |
| `rig.py` | numpy copy of the Blender character rig (`sample` mesh; `classic` box mannequin as the fallback) |
| `quality.py` | Junk-track filter, quality numbers, half-body floor shift, human alignment numbers |
| `validation.py`, `dependencies.py` | Content digests and cached reports |
| `source_exposure.py` | Optional relative brightness matching (config `source_exposure`, off by default) |
| `colour_card.py` | The colour card applied before every render |
| `camera_model.py` | The scene_data.json camera as a pinhole camera (quality, colour card, compare) |
| `blender_build.py`, `scene_export.py`, `render_settings.py` | Blender build and render (Workbench default), `scene.json` export |
| `package.py` | v2.1 package and zip, package names |
| `profiling.py`, `gpu_lease.py` | Step timings, the shared GPU lease |
| `compare.py` | Tool: film next to the render for chosen frames |

## Removed in the restructure (in git history)

Background and objects: `env_auto, object_fit, prototypes, relief, namer, sam3_scene, scene_evidence, scene_verify,
geometry_protection, build_env, props, describe, claude_set, held_objects, prop_timeline, anim_props, env_frames`,
`metric_scene.capture / plane_proposals`, most of `scene_depth` (the depth set).
Human edits: `limb_depth, track_edit, tracks_info, ground_feet`, `check_template.html`, the review / AI / brief app
modules. L3 and audit: `l3_blender, temporal_refine, build_l3_asset, previs_validate, frame_review, audit,
multiview_check, lzhd_review, render_lzhd`, `assets\l3\`, `docs\ai_skill3.1\`. Old tools: `redo_clip, task_pack,
deliver, foot_gaps, jerk_check, all_frames, extract_human_master, dump_human_master`. The Depth Anything 3 camera
backend (`da3_camera, da3_infer`, `requirements-da3.txt`). Grounding DINO, CLIP and the Anthropic package are no longer
installed; their model folders can be deleted by hand.
Dormant code: the earlier camera solver (`camera_solve`, `solve_ground`, `steady_camera`, config `camera_estimator`),
the `master` / `base21` character styles with `assets\human_master*`, `assets\base_v21\`, `base_character_v21.json`
and the extract / dump scripts, `tools\v21\` (the app's package check replaces it), `package.zip_gbk`,
`samples\*.blend`, `inputs\test\`, `data\package_tasks.json`.

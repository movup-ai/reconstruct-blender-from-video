# Blender Agent

Local white-model production app. It turns raw film clips into the client's v2.1 submission packages
(`target/original.mp4`, white model, comparison video, `.blend` + `scene.json`, `meta.json`, zip). It is fully
automatic: there is no review or confirm step. Every model runs on the NVIDIA card of this PC.

Each clip is one run: people, camera, metric scale, floor and objects (your prebuilt object models placed where the
video shows the things they stand for) → `.blend` → white-model and comparison videos → v2.1 zip → v2.1 package check.

## Start

Double-click **`Start_Blender_Agent.bat`**. The browser opens at http://127.0.0.1:8765.

- The first start runs `Setup.bat` on its own (Python packages and models, one time, needs internet).
- Keep the black window open. Closing it stops the app; unfinished jobs continue at the next start.
- If the app crashes, the window restarts it after 5 seconds. Details are in `logs\agent.log`.

## Work

1. **Put the videos in the source folder** (`SOURCE\` by default, or pick a folder in the app). A `3D` folder inside
   it holds the client's samples and is not treated as clips (`source_exclude` in `config.json`). Client **task zips**
   (v2.1 layout with `target/original.mp4` and `meta/task.json`) can go in as they are: the app writes each zip's video
   next to it and uses the zip name as its delivery name.
2. **Name the videos** (*Name videos…*): videos without a v2.1 package name get `title-1`, `title-2`… in file-name
   order from one English title (`data\package_names.json`). A clip that is queued without a package name is given a made-up one (`clip-N`, listed in
   `data\auto_names.json`, marked *auto* in the app) so a run never fails at packaging after its GPU work; naming the
   videos later with a title replaces made-up names only (the replaced `clip-N` is never reused). Real names never change.
3. **Start.** Every clip that is not done runs on its own. **Pause / Resume / Cancel** act on the
   running jobs; **Reset** deletes a clip's generated work so it runs from the video again.
4. **Download** the zip when the clip shows *done*. A zip is *outdated* when the video, its delivery name or a setting
   changed since it was made.

From the command line (one clip):

```powershell
.venv\Scripts\python.exe pipeline\run_pipeline.py <video> --name <clip> [--force]
.venv\Scripts\python.exe pipeline\run_pipeline.py <video> --stage render --name <clip>   # re-render a saved scene
```

Exit code 0 = done (or skipped: the zip exists and `--force` was not given), 1 = failed (the last log lines say why).

## The run, step by step (`pipeline\run_pipeline.py`)

1. **Object models** (`scene_objects.load_assets`): every `.blend` in `assets\objects\` (see
   `assets\objects\README.md`), one for each object Mask2Former can name, is read first (cached in `_cache\`). No
   models, or no Mask2Former panoptic weights: the clip fails at once, before any GPU work.
2. **Marks** (`automark.py`): SAM 3 tracks person masks with persistent IDs; RTMPose measures 17 joints inside each
   mask → marks folder `pose.json`, `sam3_masks.npz`, `tracking_report.json`.
3. **Shots and lens**: hard cuts are found (`shots.py`); the clip's lens is measured from the picture with MoGe
   (`lens_auto.py`). To set the lens by hand, put `"lens_auto": false` and `"lens_mm"` in `config.json`.
4. **Mask2Former panoptic**, once for the clip (ADE20K, 150 names; `models\mask2former_ade_panoptic`,
   `ade_find.py`) on a few frames per second of every shot plus the depth step's views. It outlines and names every
   object (an object that only comes into view later, zoom or pan, is found where it appears; the same object in the
   next frame is the outline of the same name that overlaps it most), and the same pass names every pixel: floor,
   wall, ceiling, sky, person... Cached in `Work_auto\<clip>\objects\objects_found.npz`.
5. **Per shot**:
   - tracks that are short and unclear are left out (`quality.drop_junk`; SAM 3 tracks are kept);
   - camera: dense flow tracking with Depth Anything V2 Small depth cues, steadying, level roll, and the per-frame
     zoom, folded with the measured lens (`camera_estimate.py`);
   - metric scene: MoGe depth of 3 views (`metric_observe.py`, cached in `Work_auto\<clip>\metric_observation*.npz`)
     + Mask2Former's floor / wall / ceiling pixels on them (`metric_scene.py`) give gravity, the world yaw and the
     camera height;
   - SAM 3D Body recovers metric bodies; their visible surfaces calibrate the depth scale (`sam3d_body.py`);
   - the characters (the client's sample mesh) are fitted to the joints with the body priors (`fit_people.py`);
   - half-body shots (nobody's feet in the picture) move the floor under the people when SAM 3D Body could not
     calibrate the scale (`quality.floor_shift`).
6. **Floor**: one flat grey ground disc under the people (`scene_depth.flat_set`).
7. **Objects** (`scene_objects.place`): `ade_objects.py` turns each found name into its model (`windowpane` →
   `window`); people and scenery are not placed; a name whose model is missing is reported. Each object's best views
   (big, whole, not hidden by people) get MoGe depth, matched to the fitted floor (its floor pixels must land on
   z = 0) so the objects stand on the floor the people stand on. The object's pixels become 3D points through the
   fitted camera of that frame (with its zoom); a box facing the camera is fitted (the hidden depth from the model's
   own proportions; standing types reach down to the floor, hanging ones stay where they are seen); one object found
   twice is kept once; the model is stretched into the box. Checked first (`object_checks.py`): a real size for its
   type, and no person standing inside it. No object found: a warning, the clip has people and the floor only.
   `Work_auto\<clip>\objects\preview.jpg` shows the boxes drawn over the video; `report.json` lists every object.
8. `scene_data.json`, `camera.json`, `clip_info.json`, `preview.jpg`, the quality numbers (`quality.measure`) and the
   human alignment numbers (`human_alignment.json`) are written. Alignment problems are warnings in `report.json`;
   they never stop a clip.
9. **Blender**: the colour card is applied (a failure fails the clip), Blender builds `model_3d\<package>.blend`
   (camera with its lens keyed when it zooms, Workbench render) and renders the frames (`blender_build.py`).
10. **Package**: v2.1 layout (`package.py`): the white model fills and centre-crops the 1920×1080 canvas (1080×1920
   for a vertical source), the comparison stacks source over white model at 1920×2160 (2160×1920 side by side for a
   vertical source), the source audio is copied, `target/original.mp4` is the source byte for byte.
11. **Package check**: `agent/validate.py` checks the folders and the six files, meta id and paths, video sizes,
   H.264, durations, fps, identical audio, the target against the source, the `.blend` header, and zip = folder.
   Any problem fails the clip.

Each run writes `Work_auto\<clip>\profile_run.json` (wall time per step: marks, scene finder, camera, metric
observation, SAM 3D Body, people fit, objects, Blender, encoding and package, package check), also for a failed run.

## Colours

Every video follows the colour card ([docs/COLOUR_CARD.md](docs/COLOUR_CARD.md), `pipeline\colour_card.py`):
characters red, blue, green, yellow, the largest and longest-seen person first (5th on: cyan, magenta, teal, olive); the eye and chest bands
white; the floor grey `#B8B8B8`.

## Setup

`Setup.bat` installs PyTorch (CUDA 12.6), TensorRT, onnxruntime-gpu, the Python packages (`requirements.txt`),
RTMPose, Depth Anything V2 Small, MoGe-2 and Mask2Former panoptic, and checks the SAM 3 / SAM 3D Body assets. SAM 3 and
SAM 3D Body run in an isolated Python 3.12 CUDA environment (default `downloads\da3-venv\Scripts\python.exe`, the
historical folder name) with `requirements-sam3.txt` / `requirements-sam3d-body.txt`; see
[SAM 3 tracking](docs/SAM3_TRACKING.md) and [SAM 3D Body](docs/SAM3D_BODY.md). Blender 5.x must be installed (or
its path set in Settings). Use one concurrent job on a 12 GB card. Model folders no longer used
(`models\grounding_dino`, `models\clip`, `models\da3`, `models\mask2former_ade`, `models\moge3`) can be deleted by hand.

## Folders

| Folder | Contents |
|---|---|
| `agent\` | the app (server, job queue, clip status, v2.1 package check) |
| `pipeline\` | the run: marks, Mask2Former, camera, metric scene, people fitting, floor, objects, Blender build, packaging |
| `web\` | the pages |
| `assets\` | the client's sample character (`sample_character.json`), the object models (`objects\`) and the reconstruction script shipped in every package |
| `models\` | SAM 3, SAM 3D Body/MHR, Depth Anything V2 Small, MoGe-2, Mask2Former panoptic weights; `trt_cache\` TensorRT engines |
| `output\` | per clip: tracked people and joints (`pose.json`, masks) |
| `Work_auto\` | per clip: scene, previews, saved depth, renders, reports, job logs |
| `SUBMIT\` | finished packages and zips |
| `docs\` | [project map](docs/PROJECT_MAP.md), [model flow](docs/MODEL_FLOW.md), colour card, SAM notes, client spec (`docs\v21\`) |
| `tests\` | `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"` |

The folders are set in `config.json` (`source_root`, `work_root`, `submit_root`). `BA_SANDBOX=<folder>` sends every
output of a command-line run to that folder instead (test runs that leave the real outputs untouched).

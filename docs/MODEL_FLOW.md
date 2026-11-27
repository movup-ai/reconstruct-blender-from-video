# Model flow

## The run (one stage)

Video → object models read (`assets\objects`) → hard cuts → SAM 3 persistent person masks → RTMPose joints inside
each mask → lens (MoGe-2) → **Mask2Former panoptic** once for the clip (objects, and every pixel's name) → per shot:
camera (flow + Depth Anything V2 Small cues, per-frame zoom) → metric scene (MoGe-2 depth + Mask2Former's floor /
wall / ceiling pixels) → SAM 3D Body metric bodies and depth scale → gravity, world yaw and camera height → character
fit (the client's sample mesh) → grey ground disc → objects (MoGe-2 depth on each object's best views, scaled to the
fitted floor, a box per object, the model stretched into it) → `scene_data.json` → colour card → Blender
(Workbench) → v2.1 videos and zip → v2.1 package check.

| Model | Purpose | Local assets |
| --- | --- | --- |
| SAM 3 | Persistent person masks and IDs | `models/sam3/sam3.pt` |
| RTMPose | COCO-17 joints inside each tracked mask (camera, floor and fitting evidence) | rtmlib checkpoint cache; TensorRT engine in `models/trt_cache` |
| SAM 3D Body / DINOv3 + MHR | Metric body joints and mesh, visible-surface depth scale, neck placement | `models/sam3d_body/model.ckpt`, `model_config.yaml`, `assets/mhr_model.pt` |
| MoGe-2 | The clip's lens; per-shot metric point maps | `models/moge2/model.pt` |
| Mask2Former panoptic (ADE20K, 150 names; facebook/mask2former-swin-large-ade-panoptic) | Once per clip, a few frames per second: every object outlined and named (the name picks its model, `ade_objects.py`; followed by overlap from frame to frame), and the same pass names every pixel: the floor / wall / ceiling pixels give gravity, world yaw and camera height, and the floor scales the objects' depth (`ade_find.py`) | `models/mask2former_ade_panoptic/` |
| Depth Anything V2 Small | Depth cues for the optical-flow camera (pitch, far background weighting) | `models/depth_anything_v2_small.onnx` |
| Blender 5.x | Builds the `.blend` and renders the white model | Blender install |

SAM 3 and SAM 3D Body run one after the other in the isolated Python 3.12 CUDA worker environment and exit before
the next model loads. MoGe runs in its own short worker (`metric_observe.py`); Mask2Former, RTMPose and Depth
Anything run in the main process (Mask2Former is freed before the next model loads). One GPU lease is shared by the app's jobs and released before Blender.

What each step leaves in `Work_auto\<clip>\`:

| File | From |
| --- | --- |
| `metric_observation*.npz` | MoGe-2 depth and intrinsics of 3 sampled frames per shot (cached by image content and focal length) |
| `objects\objects_found.npz` | Mask2Former panoptic: the objects' outlines and every looked-at frame's name map (cached by frames, models and settings) |
| `objects\depth.npz`, `objects\preview.jpg` | MoGe-2 depth of the objects' best views; the placed boxes drawn over the video |
| `sam3d_body_raw*.npz`, `sam3d_body*.npz`, `sam3d_body_report*.json` | SAM 3D Body observations, aligned joints, calibration report |
| `person_fit_cache\` | one numeric fit per person, keyed by a digest of its inputs (`dependencies.digest`) |
| `lens_auto.json` | measured lens per shot |
| `camera.json`, `scene_data.json`, `clip_info.json`, `preview.jpg` | camera, scene, picture area, preview |
| `report.json`, `human_alignment.json` | quality numbers, alignment numbers (warnings, never a stop), the package check result |
| `render\`, `render_done.txt`, `blender_log.txt` | Blender frames and log |
| `profile_run.json` | wall time per step |

The object models (`assets\objects\*.blend`, only those `ade_objects.py` uses) are read once with Blender
(`asset_export.py`) and cached in `assets\objects\_cache`. The colour card colours the objects with
`colour_card.role_of`.

## Removed

The background builders (SAM 3 scene masks, measured relief, prototypes, object fitting, CLIP naming, Claude and
hand-built sets, scene checks and repair), held objects (Grounding DINO), human review and edits (review/confirm,
joint editor, hints, track edits, limb depth, prop timeline), the L3 rig and audit/frame-review exports, the LZHD
trial package, the Depth Anything 3 camera backend, the separate Background stage (a second zip) with its SAM 3
object search, the MoGe-3 option (on Friends clips it matched MoGe-2 for depth, 2× slower), and the Mask2Former semantic model (`semseg.py`; the panoptic model names the floor and walls), and the
old one-off tools. They remain in git history.
Earlier: MotionBERT, RTMO, YOLOX, appearance-based identities, identity stitching, gap filling and SAM 2.1.

Saved settings of removed features are dropped when `config.json` is read (`common.RETIRED_SETTINGS`); the
character style is always the sample mesh.

## Validation

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m agent.selftest
```

`tests/test_one_run.py` runs the whole run with the models, Blender and the encoder mocked: the floor and the placed
objects in `scene_data.json`, the depth step's pixel names from Mask2Former, card colours, a colour-card failure
failing the render, the package check called with the source video, and only the `run` and `render` stages on the
command line.

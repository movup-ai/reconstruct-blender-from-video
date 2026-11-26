# Model flow

## Stage 1 Movement (implemented)

Video → hard cuts → SAM 3 persistent person masks → RTMPose joints inside each mask → lens (MoGe-2) →
per shot: camera (flow + Depth Anything V2 Small cues, per-frame zoom) → metric scene (MoGe-2 + Mask2Former) →
SAM 3D Body metric bodies and depth scale → gravity, world yaw and camera height → character fit (the client's
sample mesh) → grey ground disc → `scene_data.json` → colour card → Blender (Workbench) → v2.1 videos and zip →
v2.1 package check.

| Model | Purpose | Local assets |
| --- | --- | --- |
| SAM 3 | Persistent person masks and IDs | `models/sam3/sam3.pt` |
| RTMPose | COCO-17 joints inside each tracked mask (camera, floor and fitting evidence) | rtmlib checkpoint cache; TensorRT engine in `models/trt_cache` |
| SAM 3D Body / DINOv3 + MHR | Metric body joints and mesh, visible-surface depth scale, neck placement | `models/sam3d_body/model.ckpt`, `model_config.yaml`, `assets/mhr_model.pt` |
| MoGe-2 | The clip's lens; per-shot metric point maps (MoGe-3 ViT-L is an option: `config.json` `"moge_version": "v3"`, `models/moge3/model.pt`, needs `flex_gemm` + `triton-windows`; on Friends clips it matched MoGe-2 for depth, read a 2–8% longer lens, 2× slower) | `models/moge2/model.pt` |
| Mask2Former ADE20K | Floor / wall / ceiling labels: gravity, world yaw, camera height | `models/mask2former_ade/` |
| Depth Anything V2 Small | Depth cues for the optical-flow camera (pitch, far background weighting) | `models/depth_anything_v2_small.onnx` |
| Blender 5.x | Builds the `.blend` and renders the white model | Blender install |

SAM 3 and SAM 3D Body run one after the other in the isolated Python 3.12 CUDA worker environment and exit before
the next model loads. MoGe-2 and Mask2Former run in their own short worker (`metric_observe.py`); RTMPose and Depth
Anything run in the main process. One GPU lease is shared by the app's jobs and released before Blender.

What each step leaves in `Work_auto\<clip>\`:

| File | From |
| --- | --- |
| `metric_observation*.npz` | MoGe-2 depth, intrinsics, Mask2Former labels of 3 sampled frames per shot (cached by image content and focal length). Stage 1's saved depth: Stage 2 will detect objects on it. |
| `sam3d_body_raw*.npz`, `sam3d_body*.npz`, `sam3d_body_report*.json` | SAM 3D Body observations, aligned joints, calibration report |
| `person_fit_cache\` | one numeric fit per person, keyed by a digest of its inputs (`dependencies.digest`) |
| `lens_auto.json` | measured lens per shot |
| `camera.json`, `scene_data.json`, `clip_info.json`, `preview.jpg` | camera, scene, picture area, preview |
| `report.json`, `human_alignment.json` | quality numbers, alignment numbers (warnings, never a stop), the package check result |
| `render\`, `render_done.txt`, `blender_log.txt` | Blender frames and log |
| `profile_movement.json` | wall time per step |

## Stage 2 Background (started by hand, `pipeline\background.py`)

| Step | Model | What it gives |
|---|---|---|
| find | **Mask2Former panoptic** (`models\mask2former_ade_panoptic`, facebook/mask2former-swin-large-ade-panoptic; ADE20K, 150 names) | every object outlined and named, a few frames per second of each shot (`ade_find.py`); the name picks its model (`ade_objects.py`); the same object is followed by overlap from frame to frame |
| depth | **MoGe** (the configured version) + **Mask2Former** semantic (`metric_observe.py`) | depth and floor labels on each object's best views |
| place | none (geometry) | depth scaled so the floor lies on Stage 1's floor; points through Stage 1's camera; a box per object |

The models (`assets\objects\*.blend`, only those `ade_objects.py` uses) are read once with Blender (`asset_export.py`)
and cached in `assets\objects\_cache`. Found objects and depth are cached in `Work_auto\<clip>\background\`
(`objects_found.npz`, `depth.npz`). The colour card colours the objects with `colour_card.role_of`. The result is a second package in
`SUBMIT\_background\`; Stage 1's zip is not changed.

## Removed

The background builders (SAM 3 scene masks, measured relief, prototypes, object fitting, CLIP naming, Claude and
hand-built sets, scene checks and repair), held objects (Grounding DINO), human review and edits (review/confirm,
joint editor, hints, track edits, limb depth, prop timeline), the L3 rig and audit/frame-review exports, the LZHD
trial package, the Depth Anything 3 camera backend, and the old one-off tools. They remain in git history.
Earlier: MotionBERT, RTMO, YOLOX, appearance-based identities, identity stitching, gap filling and SAM 2.1.

Saved settings of removed features are dropped when `config.json` is read (`common.RETIRED_SETTINGS`); the
character style is always the sample mesh.

## Validation

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m agent.selftest
```

`tests/test_stage1_movement.py` runs Stage 1 end to end with the models, Blender and the encoder mocked: one floor
mesh and no props in `scene_data.json`, card colours, a colour-card failure failing the render, the package check
called with the source video, and only the `movement` and `render` stages on the command line.

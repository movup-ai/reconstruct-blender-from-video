# Current reconstruction flow

Video → hard cuts → SAM 3 persistent person masks → masked RTMPose joints → camera/depth evidence →
SAM 3D Body metric pose and placement → editable mannequin fit → SAM 3 scene masks and semantic geometry →
scene checks/repair → Blender render → aligned videos and package.

## Active dependencies

| Model | Purpose | Local assets |
| --- | --- | --- |
| SAM 3 | Persistent person IDs; automatic point-grid scene masks; annotated-object box masks | `models/sam3/sam3.pt` |
| RTMPose | COCO-17 joint measurements inside tracked masks for review, camera/ground estimation and fitting | rtmlib checkpoint cache; optional TensorRT joint engine |
| SAM 3D Body / DINOv3 + MHR | Metric body mesh/joints, visible-surface depth scale and neck placement | `models/sam3d_body/model.ckpt`, `model_config.yaml`, `assets/mhr_model.pt` |
| MoGe-2 | Scene point maps, lens estimation and shared depth evidence | `models/moge2/model.pt` |
| Depth Anything V2 Small | Optical-flow camera depth cues and fallback geometric set | `models/depth_anything_v2_small.onnx` |
| Depth Anything 3 | Optional multi-view camera rotations, translation and shared depth | `models/da3/` |
| Mask2Former ADE20K | Per-pixel scene semantics: walls, floor, plants, furniture, etc. | `models/mask2former_ade/` |
| Grounding DINO | Carried-object detection | `models/grounding_dino/` |
| CLIP | Carried-object names; scene names when semantic segmentation is unavailable | `models/clip/` |

SAM 3 and SAM 3D Body run in the Python 3.12 CUDA worker environment, sequentially. The scene-mask worker
uses `build_sam3_image_model(enable_inst_interactivity=True)` and `predict_inst` from the pinned SAM 3 source.
Point prompts retain generic region discovery; box prompts select the annotated instance rather than all
objects of the same concept. The same checkpoint supplies all SAM segmentation tasks.

Scene geometry/semantic weights move to CPU before uncached SAM 3 image inference; their computation remains
on CUDA when they are next needed. Validated masks are cached by image, checkpoint file stamp and parameters.
The default image grid is `sam_points=32`, with `sam3_image_batch=16` prompts per call and at most
`sam3_image_max_masks=256` retained masks. Cached sets include the segmentation backend and checkpoint stamp,
so an older segmentation result cannot be reused silently.

## Retired components

MotionBERT lifting, RTMO, YOLOX detection/batching, appearance-based identity assignment, automatic identity
stitching, synthetic track-gap filling, and standalone SAM 2.1 have been removed. Their setup downloads,
selectors, obsolete adapters and benchmarks are removed as well. Saved backend selections migrate to
`sam3`, `rtmpose`, and `sam3d_body`; checkpoint locations and manual fitting controls remain configurable.

Manual track edits, joint corrections, rig inverse kinematics, foot-contact constraints, scene geometry checks,
camera solvers, and object modeling still serve the active flow. SAM 3 does not replace depth estimation,
semantic class assignment, or the editable Blender rig.

Previous tracking and person edits remain in per-clip tracking backups. Completed exports are not modified by
this cleanup. Re-check a clip to regenerate old marks, then review roles before finishing it.

## Validation

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -q
.venv/Scripts/python.exe -m agent.selftest
```

Tests cover saved-setting migration, detector-free RTMPose loading, overlapping mask identities, prompt geometry,
quality/stability filtering, mask suppression, cache invalidation, safe worker failure and editable scene contracts.
The SAM tracking/body tests cover annotation backups, camera-space directions, calibration and placement.

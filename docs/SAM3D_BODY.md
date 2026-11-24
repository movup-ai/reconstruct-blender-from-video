# SAM 3D Body reconstruction

Choose **Settings → 3D body model → SAM 3D Body metric reconstruction**, then Prepare/Re-check the clip.
The existing SAM 3 person IDs and reviewed joint corrections are reused. Switching the body backend does not
renumber people or require tracking regeneration. MotionBERT remains selectable in the same menu.

The following local assets are required:

- `models/sam3d_body/model.ckpt`
- `models/sam3d_body/model_config.yaml`
- `models/sam3d_body/assets/mhr_model.pt`

The body worker uses `sam3d_python`, defaulting to the same isolated Python 3.12 CUDA runtime used by SAM 3.
MoGe, SAM 3D Body and DA3 inference run in separate processes sequentially to release GPU allocations between
models. Keep one concurrent app job on the RTX 4070. The tested body worker peaked at about 3.4 GiB allocated
VRAM. Body inference uses tracked boxes and masks directly; another detector or segmenter is not required.

## How it affects the blockout

1. **Metric observations.** The DINOv3 SAM 3D Body model recovers 70 keypoints, a dense 18,439-vertex body mesh
   and camera translation for each sampled person/frame. Supplied camera intrinsics and masks are used.
   Model camera coordinates are checked against its own 2D projections before accepting an observation.
2. **Surface scale.** Visible mesh surfaces are rasterized with a perspective-correct triangle z-buffer and
   compared against valid depth pixels inside the same person's visible mask. Each person/keyframe contributes
   one robust scale ratio; a large foreground person does not outweigh all smaller people by pixel count.
3. **Depth alignment.** The global scale converts scene depth to metres. The paper's neck-depth placement uses
   per-frame translation factors and one constant body scale per track. Missing depth observations are
   interpolated; if calibration has no usable correspondences, the metric body estimates remain usable and the
   report records `calibrated: false`.
4. **Shared scene.** With the optical-flow camera, MoGe supplies the aligned depth. With DA3, its processed
   intrinsics map depth into the same image rays; body scale is applied to both depth and camera translation.
   Reliable semantic ground can set camera height. Explicit camera height, orientation, path and manual DA3
   scale settings take priority. Body upright directions supplement the existing gravity evidence.
5. **Editable characters.** Camera-space bone directions, inferred height and metric hip positions guide the
   mannequin fitter. Camera-space directions are not rotated onto the image ray a second time. Hidden legs can
   use the recovered 3D pose rather than the old standing-leg assumption. Explicit joint, leg, mount, height and
   distance controls retain priority. Character visibility follows SAM 3 masks.

The delivered `.blend` continues to contain editable mannequins in the selected character style. Dense body
meshes provide observations and calibration; they do not replace the blockout with textured human scans.
The initial dense mesh for each track is retained in the raw observation artifact for inspection.

## Settings and artifacts

`sam3d_stride` is exposed as **Body inference interval**, default 2. It measures every second observed frame,
plus track endpoints and depth-reference frames. Use 1 for every observed frame. In-between pose priors are
interpolated without creating visible people in frames where SAM 3 has no mask.

Additional settings are `sam3d_surface_side` (default 160 pixels for calibration rasterization) and
`sam3d_timeout_min` (default 40). Raw inference caches include images, masks, boxes, intrinsics, checkpoint
timestamps and reference depth evidence. Model or input changes invalidate these caches.

Each clip's work folder contains:

| Artifact | Contents |
|---|---|
| `sam3d_body_raw*.npz` | Sampled camera-space body joints, necks, initial meshes and calibration observations |
| `sam3d_body*.npz` | Depth-aligned joints and per-frame placement factors |
| `sam3d_body_report*.json` | Calibration status, scale, instance counts, inferred heights and peak worker VRAM |
| `camera.json` / `report.json` | Per-shot body calibration and final fitting quality |
| `profile_*.json` | Selected tracker/body models and body-stage timing |

Fresh review suggestions use the fitted SAM 3D Body heights. Saved per-person heights remain authoritative.
Heights, hidden limbs, crouching and camera depth can still be uncertain; inspect quality flags and the rendered
comparison. This integration is not a guarantee of perfect pose recovery through complete occlusion.

## Runtime setup and validation

The current machine has been validated with Python 3.12, torch 2.8 CUDA 12.6 and upstream sam-3d-body commit
`b5c765a0d89d789985e186d396315e7590887b94`. The worker adds `third_party/sam-3d-body` to its import path.
`requirements-sam3d-body.txt` records the runtime dependencies. The upstream DINOv3 backbone loader downloads
its source code through torch.hub on first use, then uses its local cache. The model weights are loaded from
the supplied checkpoint. Optional Momentum, Detectron2, a separate FOV estimator and hand inference are not
required for this body-only path.

Regression tests cover metric coordinate conversion, triangle surface rendering, visible-mask scale,
outlier rejection, neck placement, constant body scale, sampling, manual scale priority, camera-space priors and
review heights. GPU smoke runs exercise real SAM 3 masks, body reconstruction, optical-flow/MoGe geometry,
DA3 geometry and Blender export in isolated `downloads/` folders.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -q
```

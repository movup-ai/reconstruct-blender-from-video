# SAM 3D Body reconstruction

SAM 3D Body is the required human reconstruction model of the run. Its checkpoint and MHR paths are in Settings.
Changing the body checkpoint does not renumber people or require new tracking. The following local assets are
required:

- `models/sam3d_body/model.ckpt`
- `models/sam3d_body/model_config.yaml`
- `models/sam3d_body/assets/mhr_model.pt`

The body worker uses `sam3d_python`, defaulting to the same isolated Python 3.12 CUDA runtime used by SAM 3.
MoGe-2 / Mask2Former and SAM 3D Body run in separate processes, one after the other, to release GPU memory between
models. Keep one concurrent job on a 12 GB card (the tested body worker peaked at about 3.4 GiB allocated VRAM).
Body inference uses the tracked boxes and masks directly; no other detector or segmenter is needed.

## How it shapes the fit

1. **Metric observations.** The DINOv3 SAM 3D Body model recovers 70 keypoints, a dense 18,439-vertex body mesh
   and camera translation for each sampled person/frame, with the supplied intrinsics (zoom included) and masks.
   Model camera coordinates are checked against its own 2D projections before an observation is accepted.
2. **Surface scale.** Visible mesh surfaces are rasterized with a perspective-correct triangle z-buffer
   (`sam3d_body.surface_depth` / `rasterize`) and compared with valid MoGe-2 depth inside the same person's mask.
   Each person/keyframe contributes one robust ratio, so a large foreground person does not outweigh smaller people.
3. **Depth alignment.** The global scale converts the scene depth to metres. Neck-depth placement uses per-frame
   translation factors and one constant body scale per track. Without usable correspondences the metric body
   estimates stay usable and the report records `calibrated: false`.
4. **Shared scene.** When calibrated, the body scale sets the metric scene's scale and the body-scaled semantic
   ground sets the camera height; body upright directions add to the gravity evidence. When not calibrated, the
   torso-proxy scale (`metric_scene.body_scale`) is recorded and half-body shots move the floor under the people
   (`quality.floor_shift`).
5. **Characters.** Camera-space bone directions, inferred height and metric hip positions guide the fit of the
   client's sample character mesh. Hidden legs can use the recovered 3D pose rather than a standing-leg assumption.
   Character visibility follows the SAM 3 masks.

The delivered `.blend` contains the rigid sample-mesh characters. Dense body meshes provide observations and
calibration; they do not replace the characters. The first dense mesh of each track is kept in the raw
observation file for inspection.

## Partial bodies and uncertain hands

When hips and ankles lack sustained image support, the fitter releases automatic floor contact and the standing
pelvis limit; metric hip placement is then a soft prior only where it agrees with visible joints. An unreliable
inferred height for a cropped person falls back to the configured height. The body prior uses per-joint image
confidence and reprojection agreement; cropped endpoints and isolated guesses below a cropped torso count less.
Completely hidden hands remain estimates.

`human_alignment.json` compares the final rig projections with confident source joints. Gross misalignment or a
hidden tracked character is reported as a warning in `report.json` (`human_alignment`); an automatic run never stops
on one badly fitted person.

## Settings and artifacts

`sam3d_stride` (default 2) measures every second observed frame, plus track endpoints and depth-reference frames.
`sam3d_surface_side` (default 160 px) sets the calibration raster size, `sam3d_timeout_min` (default 40) the worker
timeout. Raw inference caches include images, masks, boxes, intrinsics, checkpoint stamps and reference depth;
any change invalidates them.

| Artifact (`Work_auto\<clip>\`) | Contents |
|---|---|
| `sam3d_body_raw*.npz` | Sampled camera-space joints, necks, first meshes and calibration observations |
| `sam3d_body*.npz` | Depth-aligned joints and per-frame placement factors |
| `sam3d_body_report*.json` | Calibration status, scale, instance counts, inferred heights, peak worker VRAM |
| `camera.json` / `report.json` | Per-shot body calibration and the final fitting quality |
| `profile_movement.json` | Body-stage timing |

## Runtime setup

Validated with Python 3.12, torch 2.8 CUDA 12.6 and upstream sam-3d-body commit
`b5c765a0d89d789985e186d396315e7590887b94` under `third_party/sam-3d-body` (the worker adds it to its import path).
`requirements-sam3d-body.txt` records the runtime dependencies. The DINOv3 backbone loader fetches its source
through torch.hub on first use, then uses its local cache; the weights come from the supplied checkpoint.
Momentum, Detectron2, a separate FOV estimator and hand inference are not needed for this body-only path.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

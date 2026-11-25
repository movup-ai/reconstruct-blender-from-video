# SAM 3D Body reconstruction

SAM 3D Body is the required human reconstruction model. Check its checkpoint and MHR paths in Settings,
then Prepare/Re-check the clip. Existing SAM 3 IDs and reviewed joint corrections are reused. Changing the
body checkpoint does not renumber people or require tracking regeneration.

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

## Partial bodies and uncertain hands

In close shots, a guessed floor and standing pelvis limit can move an otherwise tracked person out of the
picture. When hips and ankles lack sustained image support, the fitter releases automatic floor contact and
the standing pelvis limit. Metric hip placement becomes a soft prior only when it agrees with visible joints;
it no longer sets a hard per-frame distance. Explicit floor, mount, seated-pose and distance controls still win.
An unreliable inferred height for a cropped person falls back to the configured height; reviewed heights win.
Later background fitting and scene repair preserve this placement instead of snapping the body back to ground.

The body prior uses per-joint image confidence and reprojection agreement. Cropped endpoints at the image
border and isolated guesses below a cropped torso receive less influence. Missing frames never acquire
observation confidence through interpolation. Relative body motion is filtered separately from translation.
The fitter continues from the previous solution rather than reinitializing hidden limbs on every frame.
Confidently observed gestures retain freedom to move; hidden wrist bends hold or interpolate observed poses.
Output arm filtering checks body-block collisions before accepting a change. Completely hidden hands remain
estimates: this cannot recover finger motion or prove an unseen wrist's true depth.

`human_alignment.json` compares final rig projections with confident source joints. Gross misalignment or
hidden tracked characters blocks rendering and prevents an unsuccessful fit from being cached as successful.
This is separate from quality warnings about uncertain feet, small motion or abrupt movement. Previous human
fit caches and finished fingerprints are invalidated once by this upgrade. Model observations and tracking
caches remain reusable; later prop-only edits continue to reuse people and cameras.

For an existing clip, open Review, run **Re-check**, inspect Joints and the comparison, then **Finish**. No reset
of tracking or model observations is needed solely for this fitting upgrade.

### Office regression, November 25, 2026

The full Prepare/Finish path was exercised in a separate sandbox using `SOURCE/swan/office.mp4`, existing SAM 3
and SAM 3D Body observations, and reviewed heights of 1.80 m and 1.70 m. Both tracks cover all 137 source frames.
Validation included camera estimation, metric alignment, retargeting, the automatic set, scene verification,
Blender rendering and delivery ZIP integrity. The production delivery was not overwritten.

- Both red and blue characters were present in every rendered frame, checked from rendered pixels.
- The final projection audit passed every checked frame for both people; median errors were 12.9 px and 5.6 px
  at source resolution (1280 by 720).
- The woman's confident left/right wrist observations had median errors of 3.1 px and 4.1 px, with 95th
  percentiles of 18.2 px and 14.5 px. These compare against tracked joints, not manually labeled ground truth.
- All 4,658 Blender joint transforms matched the fitted scene within 0.000001 m across 137 frames.
- All 225 automated tests passed, including partial-body placement, uncertain hands, manual controls, cache
  invalidation, delivery rejection and selective prop rebuilding.

The man's hands are cropped out, so their exact motion could not be verified against visible source hands.
Quality and scene reports still flag ambiguous feet, abrupt motion and unresolved background geometry; the
visibility/alignment fix does not establish exact hidden anatomy or a fully measured floor.

Local evidence is under the ignored `downloads/office-regression/` folder: `office-verified.mp4` (source above,
render below), `office-verified.jpg`, `delivery_validation.json` and `blender_audit.json`.

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

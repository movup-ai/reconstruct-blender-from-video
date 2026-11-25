# L3 previs standard in the app

The supplied `live-action-block-previs` skill now informs the production pipeline. SAM 3 tracking, RTMPose, and SAM 3D Body remain the inference stack. FlowHMR is not required or installed by this change.

## What changes in new preparations

- The default character is the supplied L3-2026-11-18 asset, with its complete native control rig, ten-bone rigid binding layer, original geometry, and seventeen pure-white mark faces.
- The numerical fitting skeleton is measured from the actual asset, including the native rig's object scale. A 180-degree axis conversion aligns the asset's -Y forward direction with the fitter's +Y convention. The renderer and fitter use the same bone/rest transforms.
- One fixed uniform actor height is shared across shots for the same tracked identity. Explicit reviewed heights win. Unrelated clips do not share actor scale.
- A bounded sparse temporal solve refines confidently observed elbows/wrists after interpolation. It preserves root/camera placement, bone lengths, model scale, and the established depth branch. Weak/occluded observations retain their prior. Manual pose/keypoint constraints bypass this refinement. Long shots use overlapping windows.
- Ordinary backgrounds retain their sampled source colors by default; character/interactive-object semantic colors follow the supplied palette. Selecting gray backgrounds remains supported.
- Coherent source-background dimming is measured outside tracked people independently per shot. Reliable substantial exposure changes animate the entire scene, including the neutral-white stripes. Ambiguous evidence leaves exposure unchanged. This estimates relative exposure changes, not primary-light direction or an HDR calibration. Existing explicit lighting hints still apply.
- Fractional source rates are represented using Blender's FPS/base pair rather than rounded away.

## Checks and review

Normal processing is the default. It checks animation-array consistency and gross human alignment, samples at most three depth views, and tries depth corrections for at most two conflicting background objects. It does not run exhaustive body/foot-contact searches, propose additional surfaces, refine occlusion fragments, or generate diagnostic overview images. Unresolved estimates remain warnings. Invalid animation, missing required actors, and gross human-alignment failures still block unsafe delivery.

Choose **Settings → Validation mode → Audit** for detailed contact and structural checks over all saved depth views, full-frame L3 geometry checks, and frame-review exports. Audit uses saved inference results; selecting it does not invalidate tracking, body fitting, or background discovery. Use Prepare/Re-check to inspect a saved scene, then Finish if you also need full-frame Blender validation and the audit delivery. The CLI also accepts `--validation-mode normal` or `--validation-mode audit` for one run.

L3 builds always use the fixed rig and materials. Normal validation checks representative frames and shot boundaries; Audit checks every frame. Both verify topology, seventeen mark faces, direct neutral emission, uniform scale, rigid edge dimensions, and actor-specific constraint targets within the stated coverage. Results in `l3_geometry.json` explicitly distinguish sampled versus exhaustive checks. They do not establish source-motion accuracy or visual acceptance.

Validation results are cached by geometry, animation, cameras, measured depth, relevant settings/hints, rig assets, and checker implementation. Normal and Audit have separate caches. Unchanged Prepare/Finish operations restore saved reports without repeating checks or repairs; changed evidence or assets invalidate the relevant result. Failed checks are cached too, and remain failures until inputs change. Background/prop edits preserve the independent human-alignment cache. Corrupt caches are recomputed. Intermediate background builds defer verification to the completed scene, preventing duplicate validation during Prepare.

Human alignment checks now respect each character's shot range, reject missing required actors, and report head/torso/arm/leg errors separately. Failed alignment remains `needs_review` and blocks Finish. Numerical thresholds are diagnostic safeguards rather than client acceptance criteria.

After Finish in Audit mode, **Review source and 3D frame by frame** opens a portable page with source/3D/stacked/overlay modes, exact source-frame stepping, playback, slow motion, and pause. It is included at `model_3d/reconstruction/review/index.html`. Unchanged source/render frames reuse the export; missing review frames trigger regeneration. Normal deliveries omit this optional page and remove stale review exports. Advanced configurations can disable it in Audit with `frame_review: false`.

The editable `.blend` retains the full L3 rig. Portable reconstruction data represents the same animated geometry as ten rigid blocks attached to the fitted joint hierarchy, preserving white emission, background lighting, exposure animation, fractional timing, and camera zoom. Rebuilding refuses to overwrite existing files.

## Testing existing clips

1. Let current jobs finish and restart the app to load the updated server/configuration defaults.
2. Run **Prepare/Re-check** once on an existing clip so its fitted skeleton matches L3. Clicking Finish alone preserves an older fit's character style; changing just the render mesh would invalidate alignment.
3. Review people, orientation, contacts, and object placement, then use Finish. Subsequent prop changes retain the existing people/camera caches.

No new AI checkpoint is needed. Legacy styles remain available through explicit hints or `previs_standard: "legacy"` plus the desired `character_style`. Changing the standard invalidates the numerical fit once; this does not itself invalidate unchanged SAM tracking/body inference caches.

## Office regression (2026-11-25)

The isolated regression reused `office.mp4`'s existing SAM observations, camera trajectory, body estimates, and set. It newly fitted L3 people, refined arms, built a fresh project, rendered all 137 frames, and rebuilt the exported scene in a separate Blender process. Original work/delivery files were preserved. Props were omitted from this fitting regression; it is not a fresh full-pipeline background or tracking benchmark.

| Automated measure | Previous fit | New L3 fit |
|---|---:|---:|
| Person 1 median visible-joint error | 12.37 px | 10.13 px |
| Person 2 median visible-joint error | 5.56 px | 4.76 px |
| Person 1 median arm error | 6.30 px | 8.40 px |
| Person 2 median arm error | 5.18 px | 2.29 px |
| Source-visible frames missing, each person | 0 / 137 | 0 / 137 |

Overall visible-joint errors improved, while the first person's arm median remains worse than the previous fit. Both characters pass the gross alignment gate. This mixed result requires visual review; it is not evidence that every hand gesture or occluded limb is correct. Neither person's legs are sufficiently observed in this clip to validate foot contact.

Across all 137 frames, the maximum actual-render versus numerical vertex discrepancy was below 0.000009 m for each actor. The separately rebuilt scene differed by less than 0.000008 m, with white emission and camera zoom preserved. These small geometry errors establish adapter consistency, not the accuracy of source annotations or inferred depth. Five source/render sample pairs were visually inspected; continuous visual acceptance of every frame is still unverified. No substantial reliable source-exposure change was detected in this clip, so no darkening was introduced.

Local artifacts are in `output/l3_office_refined/`: `validation.json`, `l3_geometry.json`, `geometry_check.json`, `reconstruction_check.json`, `comparison.jpg`, `preview.mp4`, `review/index.html`, and the editable test `.blend` under `SUBMIT`. These generated files remain outside Git.

To repeat the fitting regression into a new folder:

```powershell
.venv/Scripts/python.exe pipeline/previs_validate.py --work Work_auto/_users/swan/swan_office --marks output/_users/swan/_sources/swan_0dff64/office --output output/new_l3_validation
```

Use your actual work/marks paths. This command reads prior observations and writes only the chosen new output folder. `tests/blender_l3_validation.py` verifies actual animated vertices against the numeric asset; `tests/blender_reconstruction_validation.py` verifies a separately rebuilt scene. Both run inside Blender and take a `.blend`, its `scene_data.json`, and a new JSON report path after `--`.

## Runtime assets and source package

Runtime uses `assets/l3/`, independently of documentation folder names. `pipeline/build_l3_asset.py` rebuilds that descriptor and copies the template/palette/profile from the supplied documentation package in an explicit Blender build step. The template SHA-256 must match its descriptor before it can be appended.

The original package's historical approvals, trial settings, and validation report remain reference material. Current-job authorization, explicit review choices, and measured evidence take precedence.

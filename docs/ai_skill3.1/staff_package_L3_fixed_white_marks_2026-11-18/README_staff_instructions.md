# Live action to 3D: L3 with fixed white marks

Version: 2026-11-18. Characters now use the user-supplied L3. This package contains a workflow, models, and supporting checks; it does not convert arbitrary videos in one click.

## What changed

- Use the actual geometry from `L3_rigged_white_marks(6).blend` in place of the old short, stocky V5. Preserve the original.
- Keep the original eye and chest stripes' shape, position, and dimensions. Use neutral pure white without a blue cast. A rigid binding layer prevents bone stretching from resizing the marks.
- Each actor has a fixed uniform scale. Do not scale the stripes separately or redraw them per frame. Preserve natural changes in apparent width caused by camera distance, head rotation, perspective, and occlusion.
- Retain dark-scene synchronization, running/contact checks, and camera/background verification. The previous corridor render has not been converted to L3; do not report it as updated.

## Staff procedure

1. Extract the complete package. Read `live-action-block-previs/SKILL.md` and `live-action-block-previs/references/l3-model.md` first.
2. In Blender, append the entire `L3_STANDARD` collection from `live-action-block-previs/assets/L3_standard_fixed_white_marks.blend`. You can also open the template directly for inspection.
3. Set the actor root's uniform scale once. Change only body material slot 0 to the semantic role color; preserve white-mark slot 1. Duplicate the native controls and rigid binding armatures with the actor, and verify constraint targets.
4. Extract motion shot by shot using the source timing. Remap bone lengths and rest orientations to L3; do not reuse V5 lengths. Adjust motion, camera, and contact for each shot.
5. Run `scripts/check_l3_model.py` over the full frame range, then review at normal speed, slow motion, and frame by frame. Deliver against the checklist.

## Do staff need plugins?

| Responsibility | Requirement |
|---|---|
| Extract motion from source footage | Install FlowHMR, its runtime, and models on the motion-capture workstation |
| Open/render L3, transform ordinary bones, edit scenes | Blender; each employee does not need FlowHMR or an additional retargeting plugin |
| Full native Rigify panels, IK/FK snapping, or another automatic retargeting route | Verify the relevant tool versions separately; full panels and snapping were not validated in this package |
| Review | Video player or browser |

The source contains an already generated rig. The production copy does not require automatic execution of embedded scripts. Opening the template does not automatically retarget new motion: motion output and L3 differ and still require mapping and correction. See `live-action-block-previs/references/setup-and-roles.md`.

## Task text for production staff

Read the included live-action-block-previs/SKILL.md and use L3_STANDARD from assets/L3_standard_fixed_white_marks.blend to reconstruct the source footage. Preserve L3 body proportions and the original stripe geometry. Eye and chest stripes must be neutral white and must not resize independently or stretch with bones. Set one fixed uniform scale per actor and retain natural perspective. Extract motion by shot using source timing and correct it continuously within each shot. Darken the 3D scene in sync with lights going out in the footage. Align running and camera movement; check foot sliding and hand/wall contact. Deliver seamless stacked comparison, pure 3D video, editable project, frame review, and actual check records. Label inferred motion honestly.

## File entry points

- `live-action-block-previs/assets/L3_standard_fixed_white_marks.blend`: production template.
- `live-action-block-previs/assets/L3_original_user_model.blend`: supplied model geometry with English collection names.
- `live-action-block-previs/assets/L3_front_standard.png` and `live-action-block-previs/assets/L3_side_standard.png`: current appearance.
- `live-action-block-previs/assets/L3_white_marks_stability_check.png`: pose and stretching stress checks.
- `live-action-block-previs/references/validation-l3.json`: supplied validation results.

FlowHMR installers, large models, staff accounts, and source footage are not included. V5 history is for traceability and does not define current dimensions. Share the complete folder with production staff.

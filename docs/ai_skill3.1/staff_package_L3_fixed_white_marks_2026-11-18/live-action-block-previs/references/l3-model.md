# L3 model and fixed white marks

## Current model standard

The user supplied `L3_rigged_white_marks(6).blend` to replace V5. `../assets/L3_original_user_model.blend` preserves the source geometry with English collection names. `../assets/L3_standard_fixed_white_marks.blend` is the production copy; it preserves the neutral geometry and stripe faces, changing only the neutral-white material and adding a rigid binding layer.

Append the **entire** `Collection/L3_STANDARD` collection in Blender. Appending just the mesh loses control/binding dependencies. Preview cameras/lights are in `L3_PREVIEW` and normally need not be appended for production. The neutral pose is the original A-pose.

The source has one body mesh, 578 vertices, 590 faces, and ten disconnected body parts. Seventeen faces use white material: one chest face and sixteen head faces. Use the source dimensions. Neutral height is approximately 2.422658 units; head width/height 0.35582/0.402426; torso width/height 0.429664/0.706853. A-pose width includes outstretched arms and is not shoulder width. Forward is -Y; up is +Z. Complete measurements are in `../assets/model-profile.json`.

## Motion and proportions

- `L3_SOURCE_RIG`: retain the native 706-bone control rig and control shapes. Existing control bones support ordinary transformations without automatically executing embedded scripts or installing Rigify just to open, render, and edit bones. Full custom panels and IK/FK snapping were outside the package's validation scope; verify the appropriate Rigify version if needed.
- `L3_RIGID_BIND`: ten rigid bones copy only the corresponding native DEF bones' world position and rotation, excluding scale/shear. Each body part and its stripe share one bone, preventing controller stretch from changing their thickness. Preserve constraints, weights, and binding; do not add stretch or bind again to the original Rigify deformation layer.
- `L3_ACTOR_ROOT`: actor placement node. Use one fixed uniform overall scale, usually set at creation, then fit root position, camera, and bone rotation. Do not use nonuniform scale or scale animation on objects, bones, or ancestors. Duplicate the entire hierarchy and redirect constraints to that actor's own rig.
- FlowHMR and L3 rest skeletons differ. Rebuild correspondence, rest orientations, and bone lengths, then check head/shoulders and foot placement. Old V5 lengths such as 0.36/0.43/0.52 and actor scale 0.8 do not apply to L3.

## Fixed stripe rules

Preserve the original seventeen white faces, vertices, and material slot 1. Do not add decals or screen-width-generated stripes. The eye stripe follows the head; the chest stripe follows the torso. Each shares its body's single rigid weight, fixing relative dimensions and position.

Use neutral-white Emission (1,1,1), strength 1, dimmed by global exposure. Do not increase emission to compensate for darkness. Change semantic body color only in slot 0; do not overwrite the stripes by replacing all mesh materials.

Stable size means stable model-space dimensions and relative proportions. Camera distance, foreshortening, head turns, occlusion, and antialiasing naturally alter apparent stripe width. Do not counter-scale stripes to lock their screen-pixel width.

## Checks and helper scripts

Run in a separate Blender background process, with the skill directory as the working directory:

```text
blender --background --disable-autoexec --python scripts/create_mannequin.py -- --output new_L3_project.blend
blender --background --disable-autoexec --python scripts/check_l3_model.py -- --input project.blend --report L3_check.json
```

The template script creates a new project from the supplied L3 and refuses overwrites. The checker inspects rigid dimensions, fixed uniform root scale, seventeen white faces, and white material throughout the frame range. Passing it does not validate source motion. Review actual head turns/running at normal speed and frame by frame.

Included source-model comparisons and rotation/stretch stress checks verify binding stability. They are not a corridor-film render using L3. Earlier corridor footage still uses the old model and needs separate retargeting and rendering.

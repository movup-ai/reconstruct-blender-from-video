# FlowHMR to fixed-proportion ten-part mannequins

This workflow combines experience from two reported trials of motion capture, continuous correction, ten-part mannequins, rendering, and delivery. Strict camera/actor reconstruction still requires shot-specific adjustment. Distinguish software output, later corrections, and unresolved issues.

## 1. Inventory footage and split shots

Record dimensions, frame rate/count, source PTS, audio, and color. Source numbering begins at 1. The shot table includes start/end frames, principal/supporting roles, entrances/occlusions, visible joints, contact events, background anchors, and primary light direction.

Distinguish real cuts from light flicker. Darkness does not automatically mean a cut; sudden optical flow may reflect a flash or fast camera motion. Inspect neighboring frames. Generate/fit shots separately and never smooth across cuts.

Preserve the source. Brightened copies assist detection only; the comparison's upper half must use the original review reference. Preserve aspect ratio when resizing and record coordinate transforms between detected joints and original images.

## 2. Extract motion

Import each shot into FlowHMR and verify identities and track ranges, including missed detections at the beginning/end. Save parameters, person IDs, selected frame ranges, and original NPZ/FBX. Align multiple people's relative positions jointly.

The reported trial started with 30 steps, seed 0, cfg 1, short-time smoothing around 0.55 internal frames, SAM batch size 2, and per-frame hands. Disable HaMeR because ten-part bodies have no fingers. These are trial starting settings, not universally optimal parameters. Internal motion uses 30 fps; resample other source rates by time rather than attaching frame numbers directly.

Choose camera settings per shot. Truly static shots may use a fixed camera; evaluate moving-camera output for clear pan/tilt/translation, then verify against backgrounds independently. An early corridor trial used fixed-camera capture plus approximate later camera adjustment, which was insufficient. Architectural constraints and translation improved the revision, but close-ups and dark regions still contained inference; it was not an exact camera solution.

Apply contact postprocessing according to action. Standing/walking may benefit from foot correction; do not force jumps/airborne motion onto flat ground. Compare before/after rather than enabling every available option.

## 3. Observations and reliability

Keep source head/neck, shoulder, elbow, wrist, hip, knee, and ankle observations. Use estimated 3D motion as a direction/depth reference. Record visible, occluded, offscreen, very dark, and missed states. Detections can be wrong; small fitting residuals do not prove accurate motion.

In dark footage, check whether brightening reveals actual contours. Short gaps can use stable motion on either side within the shot. Label long gaps and extrapolated boundaries as inferred. Previously, simply extending nearby poses caused freezing. Check velocities, stable intervals, and visible contours rather than treating a frozen pose as successful stabilization.

## 4. Retargeting and continuous stabilization

Append all of L3_STANDARD and remap lengths to L3's rest skeleton. Use one fixed uniform root scale per identity across the clip. Do not alter bone lengths per frame to hide residuals. Fit root, torso facing, head direction, then limbs; inspect head and torso separately.

Solve the full trajectory jointly using fixed lengths, source projection, motion-depth references, and acceleration constraints. Keep quaternion signs, elbow/knee branches, and limb cross-section orientation continuous. Repair local rotational outliers using stable neighboring frames and recheck transitions. Avoid independent-frame fitting followed by heavy global smoothing.

Compare quick hand raises, head turns, and jump timing before/after filtering to prevent slowdown. Verify velocity spikes against the source before treating them as errors. Noisy occluded wrist observations must not forcibly drag elbows.

### Running needs separate review

Mark each foot's source-frame support, liftoff, and airborne intervals. Check ground contact, root travel, pelvis bounce, stride, and arm swing. A support foot should remain stable in world space during contact rather than merely chase screen coordinates. Lock the camera to background evidence before jointly solving camera and actor movement.

For sliding/floating, inspect scale, floor placement, depth, and contact timing first. Lowering the floor to reveal legs only repairs occlusion; it does not establish correct gait/contact. Recheck the full clip after floor changes and preserve jump trajectories.

## 5. Independently reconstruct camera and scene

Constrain the camera with static anchors, vanishing points, doorframes, wall lines, and tiles. Exclude actors and shadows that move with them. Optical flow is initialization evidence; downweight flicker, occlusions, and low texture.

Check rotation, translation, forward/back movement, focal changes, and roll. A 2D translation/rotation fit cannot recover tracking/dolly parallax. Build simplified geometry supporting perspective and contact before matching background corner tracks. Validate the camera before correcting actor roots; do not cancel camera errors with opposite actor movement.

Prioritize shot starts/ends, doorframes crossing the image, actor/wall distances, floor vanishing points, and near/far scale changes. Bone fitting error alone cannot establish success with a misaligned camera.

## 6. Lighting, darkness, and pure-white stripes

Match source primary-light direction and approximate background colors. Eye/chest stripes are neutral white (255,255,255), superseding old cyan marks. The template uses neutral emission strength 1 to avoid blue environmental tint without adding lights. When footage darkens, darken stripes, people, and background together through global exposure/scene lighting.

Do not derive exposure only from an actor box; movement, clothing, and changing background proportions contaminate it. Select stable wall/background regions per shot, compare source/render luminance curves, and verify exact darkening/recovery timing and depth frame by frame. Then check actors. Do not normalize black references into bright scenes. Treat visible fixtures according to the footage rather than turning every light off mechanically.

Render bright, darkest, and recovery moments first; compare RGB neutrality and brightness. Pure white means no color cast, not outputting 255 in every dark frame. This rule supersedes older stripe reference colors.

## 7. Review and revision

Preview the whole clip inexpensively, with synchronized source-above/previs-below and transparent overlay at matching frame rate and display aspect ratio. Review normal speed, slow motion, and continuous thumbnails; enlarge suspect frames.

Record object, source range, issue, change, and preserved elements. Validate motion, camera, and darkness separately. Technical checks aid diagnosis rather than replace visual review. Historical corridor revisions still had hand/wall contact, foot sliding, and close-camera errors. Changing this package to L3 does not rerender old footage and cannot be reported as doing so.

## 8. Delivery and checks

- Complete standalone 3D and seamless stacked comparison; preserve source audio and add no video labels.
- Frame-switching/overlay/slow-play HTML with all required images/media.
- Editable Blender project and dependencies, character/shot tables, actual parameters, and correction records.
- Fully decode the output to verify frame count, frame rate, audiovisual mapping, and audio. Reopen Blender in a fresh process to inspect cuts, the second half, and final frame.
- Check each L3 mesh's ten parts, native controls, ten-bone rigid bind, fixed proportions/stripes, and joint connections. Separately inspect temporal rotation/translation jumps, contacts, and intersections.
- State actual review coverage and unresolved issues. Claim reproducibility only after a real rerun in a new directory. This staff package supplies steps, templates, and helpers, not arbitrary-video-to-exact-reconstruction automation.

See also `standards.md`, `model-and-palette.md`, and `acceptance-and-delivery.md`. Current model/stripe details are in `l3-model.md`. Run `check_l3_model.py` after retargeting, then review stripe perspective and motion; head/chest stripe geometry and weights remain fixed.

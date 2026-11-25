# Acceptance and delivery

## Images and continuity

- Review the full clip using source-frame mapping. Recheck every cut, first/last actor visibility, occlusion change, contact action, and camera-motion change.
- Check intrinsic model proportions separately from image framing: head bounds, shoulder line, torso direction, visible limbs, overlap, and gaps must correspond.
- Verify foreground/background depth and occlusion, stable prop grips, stair/rail relationships, connected necks, and absence of obvious nonadjacent body penetration.
- Match primary-light side, facial bright/dark regions, broad background colors, and key lines. Do not add unsupported subject shadows.
- Check each frame sequence and play continuously. Thumbnails establish coverage; enlarge suspect details rather than treating a low-resolution overview as proof.
- State actual coverage: frames inspected individually, segments played continuously, and unreviewed regions. Automated checks are not visual rewatching.

## Geometry and animation checks

Each current character is one L3 mesh with ten parts, its native control rig, and ten rigid binding bones. Mark faces do not count as another part. Preserve proportions, fixed lengths, connections, head/neck pivots, grips, and inter-character collision handling. Adjacent joint contact is not automatically erroneous penetration. Nonadjacent collision checks must account for simplified volumes and visible evidence. Valid geometry does not prove the source pose is correct.

Compare adjacent-frame translation/rotation velocity and acceleration, excluding cuts. Inspect spikes against the footage to preserve real fast action. Low error against manual tracking only proves agreement with those annotations, not that they are accurate.

## Video

Verify full decoded frame count, frame rate/PTS mapping, duration, cuts, and audiovisual synchronization. Check missing frames, duplicate freezes, and unexpected black frames; intentional source darkness is valid. Both comparison halves represent the same source time: original above, previs below, equal display widths, preserved aspect ratios, zero separator pixels, and no new text/watermarks. Identify existing letterboxing separately; removing a seam must not crop useful content.

Preserve original audio by default. For stream copying, verify the audio stream; for reencoding, verify duration/synchronization rather than demanding equal byte hashes. Document HDR-to-SDR mapping, resizing, and pixel-aspect treatment.

## Reopen the Blender project

1. Include all shots, required characters, props, background, cameras, lights, rigs, animation, and accessible dependencies.
2. Cover the full clip with the animation range; remove accidental preview limits. Bind shot markers to correct cameras and ensure local camera locks do not block switching.
3. Default to camera view with a complete timeline. Widely separated shot sets can appear missing in a stationary free view; do not deliver that misleading state.
4. Reopen in a new process. Jump to each shot's first/last frames, the second half, and final frame; verify visibility and automatic cuts. Render samples against final video when needed.
5. Pack supported still images. Include external videos/other dependencies with relative references; relative paths alone are insufficient without files.
6. Extract/deploy to another directory to verify paths. Document Blender version, playback, and source mapping.

## Delivery contents

- Complete stacked comparison and standalone 3D previs.
- Frame-review HTML with all required frames/media; controls/frame numbers are not burned into video.
- Final `.blend`, dependencies, instructions, and a usable extracted package.
- Proportion parameters, role/prop semantics, shot table, source/output timing, change log, completed checks, and unresolved items.
- Final file hashes and archive CRC checks. Exclude work caches, private absolute paths, unrelated chat screenshots, and outdated projects from final deliverables.

## Honest operation records

Save actual scripts, parameters, input/output versions, key change reasons, and validation results as embedded text or accompanying files. Label historical scripts. Normal Blender saves do not preserve the complete undo stack or create a recording of every mouse action.

If reproducible production is requested, deliver a build/patch/bake/render pipeline actually rerun in a new directory. Outdated script fragments are not a one-click rebuild of the final revision. Describe the operation records that really exist.

## Checks added on 2026-11-18

- Eye/chest mark materials use (1,1,1); bright and dark renders must remain neutral without cyan/blue cast.
- Darkening and recovery occur on the same source frames. Compare static-background and actor luminance curves separately, not actor crops alone.
- Lock the camera against background vanishing points, doorframes, and tiles before evaluating people. Moving shots cannot pass solely on approximate 2D translation/rotation.
- Check running landings, airborne intervals, pelvis motion, travel, sliding, and contacts. State unresolved issues.

## L3-specific checks

- Use supplied L3 geometry rather than historical V5.
- Retain seventeen white faces. Check head/chest/stripe model-space dimensions per frame, fixed uniform root scale, and absence of nonuniform or animated scale.
- Review stripe width/height/position relative to head/chest continuously. Distinguish natural perspective from deformation rather than locking pixel dimensions.
- Reopen in a new process, run `scripts/check_l3_model.py`, and record its report plus actual visual review coverage.

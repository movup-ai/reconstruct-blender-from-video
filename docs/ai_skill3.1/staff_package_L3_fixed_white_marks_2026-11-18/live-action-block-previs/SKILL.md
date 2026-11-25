---
name: live-action-block-previs
description: Reconstruct live-action shots as editable Blender previs using ten-part mannequins, approved proportions, semantic character and prop colors, primary light direction, simplified backgrounds, and frame-by-frame checks from the source camera. Use for new production, local revisions, review, and stacked comparison delivery.
---

# Live action to ten-part mannequin previs

The goal is editable 3D shots, synchronized previs, and verifiable frame comparisons. Image stylization or generated video cannot replace a Blender scene, character joints, and motion data. Follow the current user's new instructions. Historical approvals in this package establish reference standards; they do not authorize producing or sending another client's new footage.

## Read and establish the standards

Before production, read [Production standards](references/standards.md) and [Model and palette](references/model-and-palette.md). For an existing project, identify the confirmed version, affected frames, and current authorization. Preserve the original and avoid restarting unnecessarily.

- Use the user-designated **L3 model** throughout. Append the entire `L3_STANDARD` collection from `assets/L3_standard_fixed_white_marks.blend`. Its appearance comes from `assets/L3_original_user_model.blend`; do not reuse V5 dimensions or rebuild a short, stocky substitute.
- See [L3 and fixed white marks](references/l3-model.md). The source is one mesh with ten disconnected body parts and a native control rig. The production template adds a ten-bone rigid binding layer to isolate stretching. Ten parts do not require ten separate objects or deletion of the original rig.
- Check model proportions and image alignment separately. Compare head size in pixels, placement, depth order, torso direction, visible limbs, interactions, camera motion, and cuts frame by frame. Plausibly complete occluded regions.
- Match the primary light direction first. Simplify the background into large modules and key lines using approximate source colors. Preserve space, occlusion, interaction, and structural support.
- Include protagonists, interacting supporting characters, and necessary foreground passersby. Avoid adding every extra mechanically or omitting people who occlude the protagonist.
- Assign semantic colors to characters and props, preserving identities across shots. Ordinary backgrounds use approximate source colors under the latest rule.

## Motion capture and revision entry points (2026-11-18)

Read [FlowHMR workflow](references/flowhmr-workflow.md) and [Software and roles](references/setup-and-roles.md). This revision adds continuous within-shot fitting, dark-scene synchronization, running/contact checks, and camera/background verification. These steps still require execution and review.

Keep the original shape, dimensions, and placement of L3 eye and chest stripe faces. Use pure white sRGB 255,255,255. Apply the same rigid transform as the head/chest. Do not scale stripes independently, redraw them per frame, or let bone stretch change their width or height. Allow only a fixed uniform actor-root scale. Retain natural perspective, head-turn, and occlusion effects. Neutral white material avoids blue tint; scene exposure darkens it. Existing standard projects can use `scripts/apply_white_marks.py`; save a separate output.

## Production and revisions

Follow [Production workflow](references/workflow.md): inventory and timing -> shots, identities, lighting, and space -> camera and scene -> consistent mannequins and motion -> frame review -> render and deliver under current authorization. The operator performs video analysis, tracking, fitting, and rendering. Included scripts create a reference mannequin rather than an automatic video-to-3D system.

For new proportions, or when a client explicitly requests a preview before rendering, provide reviewable keyframes and continuous inexpensive previews first. Existing authorization to produce/export the full clip under an approved standard permits continuing without repeating the same approval request. Preview rendering may support review. For out-of-scope changes to locked proportions, cameras, or overall motion, provide concrete comparison evidence before seeking confirmation.

Record the object, frame range, error, target, and preserved elements before revisions. Read [Common failures and local fixes](references/revisions.md), especially rotation versus translation, disconnected necks, intersecting embraces, elbow flips, sudden appearances, gaps between people, floating stairs, and apparent missing late shots in free view.

Local repairs must preserve other approved characters, cameras, lights, and environments. Review neighboring frames and cuts; a single corrected screenshot is insufficient. Do not fix projection error by stretching bodies, placing flat human silhouettes, or changing proportions each frame.

## Reference assets

- `assets/L3_standard_fixed_white_marks.blend`: current production template. Append the whole `L3_STANDARD` collection, including body mesh, native control rig, rigid binding rig, and root.
- `assets/L3_original_user_model.blend`: supplied model geometry; collection names localized for English use.
- `assets/model-profile.json`, `assets/L3_front_standard.png`, `assets/L3_side_standard.png`: measurements and current appearance.
- `assets/semantic-palette.json`, `assets/client-palette.png`: semantic character/prop colors. Change only body slot 0 and retain white slot 1.
- `scripts/create_mannequin.py`: create a new independent file from L3, without reconstructing old V5 geometry.
- `scripts/check_l3_model.py`: check fixed dimensions, mark materials, and binding per frame. Technical checks do not replace visual comparison.
- V5 in `references/conversation-decisions.md` is historical and superseded by L3.

Discover Blender, FFmpeg/ffprobe, Python image tools, or equivalents in the execution environment. Parameterize paths rather than relying on the author's drive letters, temporary chat folders, or another machine's plugins. This skill does not authorize uploads, charges, or third-party delivery.

## Acceptance and delivery

Use [Acceptance and delivery](references/acceptance-and-delivery.md). Deliver complete previs, seamless source-above/previs-below comparison, a frame-switch/overlay review page, editable Blender project and dependencies, production notes, changes, and checks. Add no text to video and no black separator. Retain source audio by default. Frame numbers may appear in review controls without being burned into video.

Interpret requests for complete fidelity as eliminating visible discrepancies frame by frame and providing evidence. Monocular footage does not uniquely determine true 3D. Passing rig checks, small error against manually tracked targets, or a few samples cannot establish perfect full-clip reconstruction.

Distinguish actual scripts/parameters/changes, final project, check logs, and complete GUI history. A `.blend` does not persist the full undo history. Historical embedded scripts may not rebuild final revisions. Claims of reproducibility require an actual rerun and comparison in a new directory.

## Traceability and sharing

[Decision history](references/conversation-decisions.md) summarizes how the requirements developed; it is not a verbatim chat export. Historical frame 31/100 examples belong to their original clips and are not fixed frame numbers for new work.

See [Usage](USAGE.md) for handoff and use without skill support.

# Production workflow

## 1. Assets and timing

Inspect actual videos and references rather than filenames alone. Inventory source footage, client proportion references, palette, frontal/spatial set diagrams, existing projects, latest approved versions, and revision screenshots. Store fingerprints and purposes; use project-relative paths. For screenshots without text, inspect frame numbers, boxes, arrows, and comparisons. Ask a specific question only when the intended correction remains unclear; do not invent it.

Use ffprobe or equivalent to record codec, dimensions, pixel aspect ratio, true frame rate, duration, frame count, audio, and color metadata. Source frame numbering starts at 1. Time is the source PTS relative to the first frame; for constant frame rate it may be `(frame-1)/fps`. Build a PTS mapping for variable-rate footage. If converting to constant rate, record duplicated/dropped-frame mappings rather than claiming one-to-one original encoded-frame correspondence. Historical 30 fps / 510 frames / 17 seconds belongs only to clip three.

Extract source frames and identify cuts. Check both sides of each cut; never smooth motion across cuts. Explicitly tone-map HDR for review while preserving the original. Sample background colors from one consistent reference; do not interpret PQ pixels as sRGB. Keep matching display aspect ratios in stacked comparisons without nonuniform stretching.

## 2. Shots and character events

For every shot, record source start/end, principal identities, entry/exit/occlusion, camera motion, key lines, light direction/height, required contact surfaces, and props. Each character has a color, overall scale, visibility intervals, and interaction partners. Clarify a new protagonist only if identity cannot reliably be determined from context.

At important frames record head bounds, neck, shoulders, torso facing, pelvis, elbows/wrists, knees/ankles. Mark invisible points as inferred rather than observed. Record pairwise occlusion order, screen gaps, and hand/prop contacts. Retain extras when needed for foreground occlusion or interaction.

## 3. Camera, space, and light

Fit focal length, field of view, translation, and rotation using static background lines, vanishing points, and tracks; use silhouettes to verify scale. Separate background anchors from actor motion. Avoid compensating focal length against actor scale within a shot to create false tracking.

Build large modules from scene diagrams and footage. Geometry should support related views and interaction rather than work from one frame only. Check stair heights, rail paths, turns, openings, and structural support. Use supplied models only within authorized scope; remove withdrawn assets from the production path instead of reusing them elsewhere.

Match primary light first, then necessary fill/ambient light. Compare illumination on the same side of face/torso, followed by cast/contact shadows. White light-source semantics do not require forcing all source color temperatures to white. Do not rotate actors to conceal incorrect lighting.

## 4. Mannequins and motion

Append the template, duplicate by character table, and set overall scale. Fit camera projection separately from model proportions. Keep neck connections and shoulder pivots stable. Head direction and torso direction differ; turning only the head does not correct an incorrectly facing chest.

Solve motion from observations and occlusion constraints, prioritizing position and facing before limbs. Use fixed lengths, joint constraints, stable IK pole vectors, and continuous rotation. Maintain contacts over their intervals and avoid independent-frame twitching. Inspect the original before filtering; preserve genuinely fast gestures and real camera shake.

Walking/stairs require feet on real steps, weight transfer, knee flexion, lifting/landing, and arm swing. Sliding the whole body along a diagonal is not gait. For embraces/held props jointly fit people or hand/prop relationships before resolving intersections; separate center-point fitting is insufficient.

Verify silhouette entry/exit and occlusion frame by frame. Except for real cuts, avoid visibility switches that make people appear from nowhere. Prefer physical paths and occlusion to establish reveal timing.

## 5. Review and local revision

Review representative front/side/close/interacting/full-body/stair/cut frames for proportions and primary light, then preview the entire motion cheaply. Review controls need source/previs/stacked/transparent-overlay modes, previous/next frame, jump, play/pause, frame number, and time. Overlays locate errors; inspect images separately too, because blended colors can conceal silhouette differences.

Base revisions on a known version and record preserved elements. Rerender affected frames and transition ranges; check neighboring shots, contact, and spacing. Continue approved full-clip work under current authorization. Do not repeatedly change style without new evidence.

## 6. Faster work with the same standard

Reuse approved templates, materials, and validated cameras/motion. Preview composition/movement at low resolution/samples before final rendering. Cache tracking/rendering per shot and rerender only affected ranges. Automated jump/missing-frame/contact alerts guide human review but cannot replace continuous full-clip viewing.

Preserve originals and separate sources, parameters, scripts, caches, and delivery. Record input versions, parameters, and outputs. Test claimed reproducibility. Do not run unknown, hardcoded, or destructive scripts from old projects; embedded historical code may omit final local patches.

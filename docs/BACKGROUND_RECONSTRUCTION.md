# Background reconstruction and review

The video detail page supports drawing and naming background objects, then **Render background only**.
This updates drawn objects using saved depth, people and camera; it preserves unannotated objects.
**Finish** reuses the result and renders the delivery. Prepare and Finish retain independent queue lanes;
GPU model inference shares a lease while Blender rendering can overlap another clip's preparation.

Drawing tools are Pen, Rectangle, Circle and Polygon; Select moves and resizes existing objects.
For polygons, click corners and finish with Enter, a double-click or a click on the first corner.
Escape cancels an unfinished polygon. Ctrl+Z removes its last corner or undoes the last completed object edit;
Ctrl+Shift+Z / Ctrl+Y redoes an edit. Undo/Redo buttons provide the same actions. Text fields retain native text undo.
Selected objects have a solid blue outline and blue handles. Clicking outside the image and object controls
deselects them. Background rendering displays an animated loading indicator until its queued/running job ends.
The background preview opens when its job completes. An older comparison video is labelled until Finish updates
the delivery; completed videos use a new media version so the browser loads the new render.

## Edited-object placement

Edited annotations anchor the prototype's projected bounds in their source frame. Segmentation determines the
visible shape; invalid depth pixels no longer crop that silhouette. Valid depth determines scale in metres.
The fitter retains its initial candidate and cannot replace it with a worse objective. Later automatically
matched views have less influence and inconsistent matches remain verification-only observations.

Dense flow cannot replace a calibrated projected prompt when their positions disagree substantially. This
avoids matching a different tree or wall during a large camera move. Matching automatic instances from other
sampled frames are replaced by the edited object, rather than left as duplicate small prototypes.

For trees with an annotated base inside the picture, the saved ground surface (including its terrain heightfield)
constrains contact. This prevents unreliable crown depth from burying the trunk. Explicit dimensions and distance
overrides take precedence. Placement/occlusion repairs also respect the edited bounds; drawn fallback objects
bypass the old uniform shrinking rule.

On `T014_p70d7ac25_t014-1-repair`, three edited pines and a palm were rebuilt and rendered through 134 frames.
Their source-frame projected size errors were 0.8–2.6% and center errors were 4.7–9.9 source pixels in the final
terrain-aware test. Saved characters, cameras and zoom were identical before and after the rebuild and render.
These are bounds measurements: prototype silhouette fidelity and uncertain later-frame depth still have separate
review flags. The tree/palm shapes remain the named prototypes selected by the user.

## Repairs

1. Confident furniture survives partial visibility and broad body intersections. A visible chair back fits the
   back of a complete chair; a tabletop fits the top of a complete table, without squeezing hidden parts into its mask.
2. The fitter jointly optimizes x/y/z, width/height/depth and yaw, using calibrated projections, visible masks,
   robust median depth and dimension priors. Source occluders are excluded from the predicted silhouette.
   Explicit user dimensions and facing choices are constrained. Unknown depth/hidden geometry remains uncertain.
3. Depth conflicts with source-visible people compare rigid placement with joint depth/scale placement along
   source rays. If neither resolves the conflict, only unsupported conflicting face fragments may be trimmed.
   Other views protect observed surfaces. Original geometry is archived in the scene; whole objects are retained.
4. Room classification uses semantic ceiling/window/floor/furniture evidence, independently of invalid depth.
   Potted plants use connected pot/stem/round-canopy prototypes. Semantic foliage evidence prevents green paint
   alone from choosing vegetation. Fine semantic maps are cached separately from coarse metric evidence.
5. Object association uses camera/depth agreement plus image-flow mask overlap. Automatic sampling includes
   first/middle/last frames. Annotated objects receive independent SAM box segmentation in those views.
   Final exported vertices are checked again after all placement and occlusion repairs.
6. `environment_profile.json` separates model loading, SAM model loading, SAM inference, worker overhead,
   semantic/object classification, depth inference, matching, fitting and projected verification. Cache hits and
   uncached requests are counted. `verification_profile.json` times final scene verification and repair.
   Shared batch model loading is charged once. Nested timings overlap and are not additive.

## Checking results

Each clip's work folder contains `object_alignment.json`, with per-object/per-frame silhouette IoU,
center error in source pixels and relative bounding-size error. Current review limits are IoU >= 0.35,
center error <= 24 px and size error <= 60%. These are review thresholds, not a guarantee of precise geometry.
Failures appear on the detail page and in `scene_verification.json`; they are never converted to success by deletion.
The background preview shows source, overlay and reconstructed geometry. Background-only builds reject any
change to saved characters, cameras or zoom. Older automatic backgrounds refresh on the next environment rebuild;
an annotation-only update deliberately preserves the other existing objects.

## Validation on November 25, 2026

- 256 unit/regression tests cover partial furniture, joint fitting across three views, fixed user dimensions,
  final exported-vertex checks, semantic room/plant selection, depth-aware occlusion, flow association, timings,
  queue lanes, frozen people/cameras and render workspace isolation.
- Isolated automatic rebuilds used `office.mp4`, `two_men.mp4` and `p70d7ac25_t016-1.mp4`.
  Office retained 17 objects versus 3 previously; 11 of 16 audited objects passed all sampled views.
  Five still require review, including the reflective table and overhead light. Existing camera calibration,
  depth ambiguity and segmentation remain material limits; the saved camera was deliberately preserved.
  The other two clips passed 24/30 and 23/28 object checks respectively.
- An annotated office chair passed all three frames, with center errors 20.8/19.4/5.2 px and IoU
  0.52/0.53/0.78. Its background-only update took 22.3 s: SAM loading 7.70 s, SAM inference 1.17 s,
  prototype fitting 0.66 s and scene verification 2.83 s; the balance includes process startup and other work.
  Finish rendered 137 frames and packaged a zip in 18.8 s without repeating detection or fitting.
- A separate 137-frame render of the complete rebuilt office background retained visible red and blue
  characters in every frame. Saved people, camera and zoom matched exactly before and after both workflows.

Validation artifacts live under ignored `downloads/background-repair` and `downloads/background-only-smoke`.
Timing depends on cache state, frame count and hardware; these measurements are not performance guarantees.

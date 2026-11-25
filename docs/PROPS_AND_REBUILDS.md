# Props, prototypes and selective rebuilding

Open a prepared clip and use **Held-object timeline** below **Objects in the set**.

1. Add an object and give it a physical identity, such as “Phone” or “Sword”. A replacement item gets a new object.
2. Select a source frame, the character, and left hand, right hand, both hands, or back. Press **Hold / pick up** or **Wear**.
3. Draw a box tightly around the item. Its screen size and the fitted hand's distance supply dimensions and an attachment offset. Optional width, height and depth in metres override this estimate.
4. At the release frame press **Release / throw**. Add boxes along its subsequent path. At a pickup or transfer, select the receiving character/hand and press **Hold / pick up** again.
5. Use **Rest** to retain the last world position; use **Hide** when the object leaves the shot. Remove an event or spatial key to undo that annotation.
6. **Save** stores annotations without starting a job. **Re-check** updates the scene. **Finish now** additionally rebuilds the Blender scene, renders, encodes videos and packages the zip.

Adding the first timeline object selects **Timeline only**. This avoids an automatic guessed object continuing after an explicit release. **Automatic + timeline** also uses Grounding DINO/CLIP detections; **Off** removes carried props. Legacy free-text object notes remain notes; use timeline events for exact ownership and visibility.

Frames are zero-based source-video frames. Boxes use full-video pixel coordinates, including letterbox bars; the pipeline subtracts the picture crop when projecting them. Exported scene metadata includes prop identities, dimensions and lifecycle events, with the frame convention stated.

## Motion and depth

Held objects follow the fitted wrist or chest. Different objects keep separate IDs even when the same hand holds them at different times. Inferred dimensions stay constant instead of growing and shrinking with the detector box. Events apply until the next event; an object is hidden before its first event.

Free motion interpolates spatial keys within the current event and camera shot. A video box alone does not determine 3D depth: the initial release depth is retained unless a box has an explicit **Depth** value in metres. For objects that start free, draw a box and set its depth. Without spatial evidence a released object stays at the release position and produces a warning. Free motion is not extrapolated across camera cuts. Add placement evidence in the next shot.

This is manual annotation and interpolation, not a physics simulator or a new automatic SAM 3 prop tracker. SAM 3 continues to track people and segment scene objects. Automatic held detections now split at long gaps and sustained label changes; a single noisy label does not split an object. Sparse detections no longer mean “held for the whole clip.”

## Object size and background prototypes

The static object editor accepts optional width, height and depth in metres. Clear the field to return to measured sizing. Drawing a tighter box changes the measured image footprint. Automatic primitives that extend well beyond that footprint are shrunk; explicit dimensions retain their requested size.

Vegetation is represented by recognizable library objects before the relief-mesh branch:

| Label | Prototype |
| --- | --- |
| Tree, pine, fir, spruce, conifer | Pine with trunk and stacked foliage |
| Bush, shrub | Short rounded shrub |
| Hedge | Hedge |
| Palm | Trunk with radial leaves |
| Potted/house plant, indoor plant | Potted plant |

Dimensions come from robust percentiles of the measured region, not the library's large default tree size. Unknown non-vegetation labels can still use measured relief geometry; rename or change their type in the object editor for a recognizable model.

The verifier compares background depth with source-visible human pixels and the fitted character parts used by Blender. It first tries a bounded translation of the entire offending primitive, preserving its dimensions. If the group still blocks people, it is suppressed and retained in `suppressed_geometry` for inspection. The scene records `geometry_protection` actions. This checks sampled evidence views; inaccurate tracking/depth, missed labels, and frames outside those views still require visual review. Actual source occlusion is retained where the person is not visible in the source mask. Held objects are handled separately.

## What runs after an edit

On the video detail page, use **Objects in the set → Detect background only** after drawing and naming objects.
This button saves object annotations without saving pending people, camera or joint edits. It runs a dedicated
`background` job: SAM 3 segments the drawn boxes, explicit outlines are used directly, and the existing prototype
builder updates those objects. Existing unannotated background objects remain unless deleted or replaced by the
annotations. Cached metric depth is reused at matching frames; MoGe supplies depth for other annotated frames.
It does not run tracking, RTMPose, SAM 3D Body, lens estimation, camera solving or human fitting.

The updated scene is checked against an exact copy of the saved people/camera data; a change restores the original
scene and fails the job. `env_check.jpg` is shown under **Background preview**. `report_background.json` and
`profile_background.json` record the result and annotation/build timings. **Finish** subsequently renders and
packages the scene. A background revision marks any older delivery as changed. Pending movement edits remain
pending and can still require a normal Re-check or Finish refit.

Prepare/background and Finish use separate queue limits (`parallel` and `finish_parallel`, both default 1).
Different clips can prepare and finish concurrently. The same clip is serialized across both lanes. App pipeline
jobs acquire a shared process-level inference lease before running models and release it before Blender; an
uncached Finish waits for inference while a cached Finish can proceed directly to rendering. Pause stops new
jobs in both lanes while letting running jobs finish. Background jobs do not trigger AI/autopilot chaining.

Office validation used one drawn chair box: the first SAM request/background build took 27.9 seconds, cached
updates took 7.6–7.7 seconds, and the subsequent Finish took 24.9 seconds. All 137 frames of people and camera data
were identical before and after. The Finish check rejected any attempted human inference or background detection
and completed Blender rendering/packaging successfully. Browser validation exercised the detail-page button,
progress, completed preview and separate job settings. These checks validate the independent workflow; existing
object-fitting and geometry quality warnings remain available for inspection.
The complete automated suite passed 241 tests, including queue overlap, same-clip serialization, cross-process
inference locking, background rollback and Finish reuse after a background update.

| Edit | Rebuilt work |
| --- | --- |
| Prop name, shape, size, hand, owner, events or box keys | Prop transforms/geometry and verification |
| Background description, type, box or dimensions | Background and props; fitted people/cameras retained |
| Joints, height, roles, camera or fitting/model settings | Affected fit inputs, then dependent scene stages |
| Finish after any of the above | Required changes, then Blender, video encoding and package |

`build_state.json` records content dependencies. `fitted_scene.json` preserves the pre-background fitted scene so background edits do not repeatedly alter fitted roots. Individual numeric person fits are also cached independently; changing one person's fit inputs does not discard unchanged fits for everyone else. Checkpoint files and source-video stamps participate in invalidation. Interrupted jobs do not record a completed build state.

Legacy scenes are adopted before the editor saves a review or background edit. The first rebuild updates the old background once to apply the new prototype and protection rules. It keeps compatible SAM 3/SAM 3D Body human results. Use the normal Re-check/Finish flow; **Start over** deliberately removes caches and performs a full preparation.

Uncached background SAM 3 requests are grouped into one worker that loads the checkpoint once, processes images sequentially, and releases it before subsequent geometry work. Cached masks bypass the worker. This reduces loading overhead without putting concurrent inference jobs on the RTX 4070.

## Validation

Run `.venv\Scripts\python -m unittest discover -s tests -q` (207 tests) and `node --check web/prop_timeline.js`.
Regression tests cover selective rebuilds with expensive entry points forbidden, unchanged human/camera data,
owner changes, independent props, hidden intervals, release interpolation, camera cuts, dimensions, invalid API
input, cache invalidation, recognizable prototypes and source-visible-person geometry corrections.

An isolated RTX 4070 smoke test processed four SAM 3 scene requests with one checkpoint load in 26.4 seconds;
a cached repeat took about 0.04 seconds. A complete 148-frame cached prop edit, Blender render and package
took 23.6 seconds after removing duplicate verification (28.5 seconds before that change), with character and camera data unchanged. These are measurements of the local test clips,
not a runtime guarantee. Initial preparation still needs video tracking, body inference and fitting.

The real background smoke test also built and rendered the updated semantic set, including a measured bush
prototype and reduced object footprints. Existing source-depth and support disagreements remained flagged;
passing the software tests does not mean every reconstructed object has exact metric dimensions.

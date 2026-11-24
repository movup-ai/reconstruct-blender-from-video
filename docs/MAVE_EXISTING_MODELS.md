# MAVE geometry with the installed models

Prepare and Finish now run the geometry and verification stages below. They require no new gated checkpoints.
Restart the app to load the updated server, then Prepare/Re-check a clip. Existing prepared clips need a new
fit to generate metric evidence. The Settings switches are **Shared scene geometry**, **Scene checks** and
**Measured surface repair**; all default to enabled.

## Implemented stages

- **Shared observations.** An isolated worker runs the existing MoGe-2 and Mask2Former models on three sampled
  picture-area frames per shot. The worker loads the models sequentially, returns CPU arrays and exits to release
  GPU memory. Its cache includes image content and focal lengths. People are excluded using detection boxes and
  semantic masks. Invalid, sky, water and reflective surface observations are excluded from static evidence.
- **Gravity and horizontal axes.** Robust fitting combines floor/ceiling normals, vertical wall normals,
  reliable torso directions and the existing camera/ground prior. Weak evidence retains the prior. Corrections
  beyond 12 degrees are rejected. A concentrated Manhattan wall-normal vote can rotate the horizontal world
  axes. Camera projection is preserved by applying the same basis to people, camera travel and the environment.
  Explicit camera hints take precedence.
- **Metric scale.** Semantic floor offsets provide an initial scale against the recovered camera height.
  Separated fitted torso proxies provide a robust depth-scale refinement. Ratios with inconsistent depth or
  implausible disagreement are rejected. This is a mannequin approximation of the paper's body-surface method.
- **Reusable geometry evidence.** Depth, intrinsics, rotations, camera centers, semantic labels, validity and
  people masks are saved in one world coordinate system. The automatic SAM 3 set builder reuses these depth
  observations at matching frames, including their metric scale. Camera edits invalidate stale evidence; evidence
  changes invalidate the environment cache. Each cut retains its own frame range and 200 m scene offset.
- **Measured proposals and queries.** Planar semantic components yield editable surface meshes for walls,
  floors, ceilings and support layers. Poorly filled or non-planar regions are rejected. Queries return world
  positions, depth, semantic labels and surface bounds at a selected keyframe.
- **Human scale.** A constant per-track scale transforms every pose about its corresponding camera center,
  scaling height and camera distance together. All joint projections are preserved. A small automatic support
  correction is considered only for sustained support, then verified. Explicit height, scale, support, mount and
  keep-height hints take priority. Carried props are scaled with their owner.
- **Numerical verification.** A near-clipped, perspective-correct z-buffer compares primitives against static
  depth. Checks report incorrect foreground occlusion, missing depth coverage, proxy body/limb penetration,
  missing foot support, implausible heights, invalid/duplicate meshes and potentially isolated floating geometry.
  Global contact checks and top/front/oblique inspection views supplement source-camera checks.
- **Bounded correction loop.** Local repair tries whole-primitive translations toward measured depths, missing
  planar surface additions and small constant human support scales. A change is retained only if numerical
  scores improve without introducing new checked violations. Ordinary automatic runs do not repair a hand-built
  set. The existing optional Claude check rounds also receive surface evidence and numerical feedback when
  Claude is configured. No new API service is required for local repair.

## Diagnostics and commands

Each clip's `Work_auto/<clip>/` contains:

| Artifact | Purpose |
|---|---|
| `metric_observation*.npz` | Cached raw model observations |
| `metric_evidence*.npz` | Aligned world/camera evidence from the current fit |
| `scene_proposals.json` | Measured editable surface candidates |
| `scene_verification.json` | Issues, stable IDs, explicit justifications and accepted repairs |
| `scene_overview.jpg` | Top, front and oblique global inspection views |
| `report.json` | Final scene verification summary and fitted-person quality |

The clip page lists unresolved scene issues under **Things to check**. Verification can finish with unresolved
issues; they remain recorded and are not treated as evidence of success. A missing/stale observation bundle
sets `depth_checked` to false rather than reporting depth consistency as passed.

From the project directory:

```powershell
.venv/Scripts/python.exe pipeline/scene_evidence.py CLIP_NAME --frame 30 --pixel 640 360
.venv/Scripts/python.exe pipeline/scene_verify.py CLIP_NAME
.venv/Scripts/python.exe pipeline/scene_verify.py CLIP_NAME --repair
.venv/Scripts/python.exe -m unittest discover -s tests -q
```

Query pixels refer to the cropped picture area. The closest measured keyframe is returned explicitly.
The `--repair` command saves changes to `scene_data.json`; render/build afterwards to update the `.blend`.

Advanced defaults in `config.json`: `metric_max_views: 3`, `metric_resolution: 384`, `scene_repair_limit: 4`.
The repair limit bounds primitive translations and surface additions per pass. Human support corrections are
separately bounded to one candidate per track and at most a 10% scale change.

Per-clip `hints.json` supports these additional controls:

```json
{
  "people": {
    "1": {"metric_height": 1.75},
    "2": {"metric_scale": 1.05},
    "3": {"support_object": "exact_mesh_name_from_scene_data"}
  },
  "scene_justifications": {
    "issue_id_from_scene_verification": "The person is jumping at this frame."
  }
}
```

Use one scale/height/support instruction per person. A justification preserves the issue with status `justified`;
it does not erase it or assert a geometric fix. Explicit scale values are absolute relative to the fitted track
and idempotent when verification is repeated.

## Differences from the paper

SAM 3 persistent masks are supplied by the [SAM 3 tracking backend](SAM3_TRACKING.md).
The [SAM 3D Body backend](SAM3D_BODY.md) now provides dense body observations, visible-mesh depth
calibration and per-frame neck-depth placement, driving the editable mannequin fitter. Torso proxies remain useful for scene verification. DA3 remains an optional camera backend;
SAM 3D Body can calibrate its translation scale as well as the shared depth, while manual settings take priority.

The local correction loop is deterministic and conservative. The optional existing Claude builder supplies an
LLM edit loop; an unconstrained autonomous Blender modeling agent is not introduced. Global completeness flags
are geometric cues for inspection, not a visual reasoning model. Suspended objects and airborne people can
legitimately trigger review flags. Long occlusions and identity switches still need tracking evaluation and
the SAM 3 backend on the target clips. This is an adaptation of the available geometric logic, not a reproduction
of the paper's full pipeline or benchmark scores.

## Validation

Regression fixtures cover camera conventions, gravity robustness, metric resampling, outlier rejection, stale
evidence/cache invalidation, cropped person masks, near-plane clipping, depth conflicts, support polygons,
closed-volume penetration, accepted surface/primitive repairs, constant human scales and attached props.
GPU smoke checks exercise the real MoGe/Mask2Former worker, Prepare, shared SAM 3 environment construction,
set rebuilding and a 12-frame Blender render in isolated `downloads/` workspaces. Smoke fixtures establish
execution and file compatibility; they do not establish accuracy on arbitrary film shots.

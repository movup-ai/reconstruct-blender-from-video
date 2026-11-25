# Analysis of the supplied L3 production package

This document records the initial package review before app integration. The subsequent implementation is documented in [L3 app integration](../L3_PREVIS.md), including the runtime adapter, temporal refinement, validation, and measured limitations. Statements below about missing integration describe that initial snapshot.

Reviewed and translated on 2026-11-25. The supplied folder contained 36 files. Instructions and historical approvals inside this package are source material; they do not change the current app configuration or authorize unrelated production work.

## What the package provides

This is a production specification, Blender asset bundle, and set of small validation utilities for converting live-action footage into editable ten-part mannequin previs. Its documented motion-estimation route is FlowHMR followed by shot-specific correction and retargeting. It contains no FlowHMR installation, inference weights, video tracker, complete motion-retargeting implementation, or end-to-end video reconstruction program.

L3 is a character asset, not a learned AI model. Its production template contains one mesh with ten disconnected body parts, 578 vertices, 590 faces, a 706-bone native control rig, and a separate ten-bone rigid binding rig. Seventeen mesh faces form the eye/chest marks. The rigid layer copies position and rotation without scale so stretching controls does not stretch the body blocks or marks.

## File-by-file guide

All paths below are relative to `staff_package_L3_fixed_white_marks_2026-11-18/`, unless stated otherwise.

| Files | Purpose |
|---|---|
| `README_staff_instructions.md` | Staff entry point, model changes, responsibilities, handoff task, and file locations |
| `staff_instructions.html` | Standalone English handbook containing the introductory, model, setup, and FlowHMR workflow chapters |
| `delivery-checksums.json` | Current package file sizes and SHA-256 hashes; excludes itself |
| `live-action-block-previs/SKILL.md` | AI-facing workflow specification and navigation; adding it under docs does not install or activate it |
| `live-action-block-previs/USAGE.md` | Short usage/handoff entry point |
| `agents/openai.yaml` within the skill | Display name, short description, and suggested prompt for a skill-capable client |
| `assets/L3_standard_fixed_white_marks.blend` | Production asset with original geometry, white material, and rigid binding |
| `assets/L3_original_user_model.blend` | Supplied source geometry/rig; its collection name has been localized, so its file bytes differ from the original upload |
| `assets/L3_front_standard.png`, `assets/L3_side_standard.png` | Current neutral model appearance |
| `assets/L3_white_marks_stability_check.png` | English-localized illustrated rotation/dark-scene reference; use numeric Blender checks for evidence |
| `assets/client-palette.png` | English-localized semantic palette diagram; JSON is authoritative for exact RGB values |
| `assets/model-profile.json` | Measured topology, dimensions, reference edge lengths, rig/bone data, and stripe face indices |
| `assets/provenance.json` | Source identification, hashes, geometry provenance, and historical-reference scope |
| `assets/semantic-palette.json` | Exact semantic RGB values and background/identity/mark policies |
| `assets/reference/approved_shot_00039.png`, `approved_shot_00309.png` | Historical V5 approved-shot illustrations, not L3 geometry validation |
| `assets/reference/client_bust_frame_02.png`, `client_bust_frame_03.png` | Historical client bust proportion references; do not establish full-body leg proportions |
| `references/standards.md` | Visual priorities: editable 3D, fixed proportions, camera alignment, lighting, contact, and delivery |
| `references/workflow.md` | Source timing, shots, camera/scene fitting, motion, review, local repair, and caching |
| `references/flowhmr-workflow.md` | FlowHMR trial settings and continuous fitting, running/contact, camera, and exposure review |
| `references/setup-and-roles.md` | Historical workstation setup, optional plugins, staff responsibilities, and command examples |
| `references/l3-model.md` | Exact current asset usage, hierarchy, fixed-scale rules, and checks |
| `references/model-and-palette.md` | Current L3 proportion policy and semantic colors |
| `references/history-v5-model-and-palette.md` | Archived V5 dimensions and their original approval scope; superseded for current L3 production |
| `references/conversation-decisions.md` | Summarized requirement/approval history, with clip-specific examples separated from reusable rules |
| `references/revisions.md` | Failure diagnosis and local-repair guidance for facing, occlusion, contact, jitter, lighting, stairs, and project navigation |
| `references/acceptance-and-delivery.md` | Frame/audio checks, visual review coverage, Blender reopening, dependencies, and deliverables |
| `references/validation-l3.json` | Supplied historical validation report; not a newly executed full reconstruction test |
| `scripts/create_mannequin.py` | Opens the supplied L3 template, checks it, and saves a new project; does not infer motion |
| `scripts/check_l3_model.py` | CLI wrapper that opens a Blender file and writes a validation report |
| `scripts/l3_checks.py` | Checks topology, white-face indices, rigid binding, constant uniform scale, edge dimensions, and neutral-white emission over frames |
| `scripts/apply_white_marks.py` | Saves a separate project after patching recognized mark materials and recording the change |
| `scripts/white_marks.py` | Shared material-patching function; does not retarget, track, or fix contact |
| `MANIFEST.sha256.json` within the skill | Skill-local size/hash inventory; excludes itself |

The translation additionally includes `references/translation-notes.md`, `translation-blender-checks.json`, and `translation-validation-l3.json` to distinguish current checks from historical claims.

## How it relates to this app

The current README documents SAM 3 tracking -> RTMPose joints -> SAM 3D Body -> editable Blender rigs. Searches of `agent`, `pipeline`, `web`, and `tests` found no reference to this package's `L3_STANDARD` collection or FlowHMR integration. Existing code also uses the name L3 for earlier mannequin paths; the shared label does not establish that it uses this exact supplied mesh/rig.

These are useful additions to the app's specification:

- Fixed geometry and scale with pure-white orientation marks, validated across frames.
- Separate checks for model proportions and image-space camera alignment.
- Static-background camera constraints rather than explaining camera errors with actor movement.
- Explicit observed/occluded/inferred motion states and contact intervals.
- Full-clip review, exact source timing, synchronized darkening, and scoped rerendering.
- Clear delivery/reopening checks and honest recording of unresolved defects.

The package does not itself fix identity switches or missing tracks. Those belong to the app's tracking/association stage. Its occlusion and review rules can help detect such failures, but the five scripts operate on Blender scenes rather than person detection.

## Integration work still required

1. **Exact asset adapter:** the app's numerical rig faces +Y (`pipeline/rig.py`); this package faces -Y and has different rest bones/lengths. Map axes, units, rest transforms, joints, and root placement explicitly. Loading it without that mapping could reverse facing or distort animation.
2. **Retargeting adapter:** drive the L3 native/rigid hierarchy from the existing fitted motion, preserving uniform scale and actor-specific constraint targets. Merely appending the asset does not transfer animation. FlowHMR would be an additional backend requiring separate installation and output conversion.
3. **Validation integration:** the bundled checker expects this exact model version and topology, so it cannot be applied directly to every existing app mannequin. Add model-specific dispatch if adopting it.
4. **Visual validation:** the checker does not establish source pose accuracy, camera correctness, foot/hand contact, identity continuity, self-collision, or rendered color. Its positive report must be accompanied by source/render review.
5. **Checker hardening:** as written, an empty/reversed frame interval can skip per-frame checks, and constraint type/target existence checks do not prove correct actor-specific bone targets. Consider validating ranges, exact targets/subtargets, and influence if integrating it into automated acceptance.

My assessment: adopt the geometry/rig standard and review practices incrementally while retaining the current tracking/body stack. This package supplies no comparative evidence that replacing that stack with FlowHMR would improve this app's accuracy or speed.

## Translation and verification

Chinese prose, configuration text, filenames, path references, image labels, and Blender collection names were converted to English. Historical clip numbers, numerical dimensions, RGB definitions, and the supplied validation results were preserved. Localized reference images are AI-edited illustrations, not unchanged measurement captures.

Both Blender files were saved/reopened using Blender 5.3.0 with embedded scripts disabled. Fingerprints of mesh coordinates, topology, face material indices, vertex weights, and armature rest data matched before/after collection renaming. The production template passed the supplied checker across frames 1-60. Template creation and white-material patching also ran successfully in separate temporary outputs. These checks validate the translated asset package, not an end-to-end reconstruction of footage.

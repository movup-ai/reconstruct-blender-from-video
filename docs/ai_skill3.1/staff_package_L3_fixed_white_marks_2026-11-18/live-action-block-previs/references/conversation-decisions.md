> Current model rule: L3 supplied on 2026-11-18 supersedes historical V5 below. Read `l3-model.md` first. Historical acceptance does not override the new model instruction.

# Requirement history and scope

This is a summary of visible requests, approvals, revisions, and checked deliveries in the source package's conversation, not a verbatim chat backup. Not every old attachment was reopened. Screenshot issues are summarized from explicit user text; blank messages do not imply invented revisions. Private absolute source paths and unrelated information are excluded.

## Clip one: output and frame comparison

The initial task was film footage to 3D blockout with a final Blender project. The user supplied a mannequin reference: protagonist red and the other two blue/green, stacked comparison without black gap or text. Disconnected head/body, incorrect joints, and embrace penetration led to a strict ten-part model and frame-by-frame reconstruction requirement.

The acceptance rule was matching each frame from the original camera and plausibly completing occluded regions. Observable images are the primary evidence; unseen true 3D is not uniquely recoverable. Further embrace overlap and green-torso orientation revisions established joint contact/orientation/depth checks.

## Clip two: pose, space, and local preservation

Another video required matching light, shadows, and poses. The user identified the bald man in a brown suit (Dwayne Johnson) as red protagonist. Feedback addressed right-side facial light, multiple poses, blue torso angle, arms, head/face direction, unsupported blue shadows, and mannequin twitching. These were shot-specific issues, not universal right-side lighting or shadow-free requirements.

Later details included more exposed red shoulder, no sudden blue appearance, depth order, yellow in front of green, six railing bars, continuous stair rails, facing, and head shape. Character references and frontal/spatial stair diagrams were supplied. An FBX was initially restricted to the stair-descent segment, then explicitly withdrawn in favor of images/source footage. That withdrawal takes precedence; do not reuse the FBX.

The user requested natural stair descent, preservation of previous work, more position/facing checks, and frame comparison before final rendering. Later feedback required blue visible by frame 31, corrected overlay overlap, and forward-facing green/yellow torsos. Frame 31 belongs to clip two and must not be hardcoded for new footage.

When stairs floated around frame 100, the user approved everything else and froze actors, motion, expression, and positions. Later requests reopened the purple/blue gap and red/yellow mutual facing while preserving other locked items. Full stacked export was eventually authorized.

## Clip three: client production standard

A new video introduced semantic colors and a new ten-part mannequin standard. The user explicitly applied semantics to both characters and scene props.

Early results had shaking and incorrect proportions. The client prioritized approximate light direction, ten-part mannequins, large background modules/lines, approximate source background colors, and mostly omitting ordinary extras while retaining principal/interacting/foreground occluding people. This simplified earlier detailed-background expectations without reducing frame-level actor/interaction checks.

The proportion reference was `75d157ad2c15997c350fade2a45847fa.mp4`. Bust measurements were approximately head width/height 0.89, torso/head width 1.28, and outer shoulders/head width 2.10; V5 used 0.90, 1.266, and 2.093. The reference did not show full standing legs, so their lengths were not validated by it.

After reviewing the V5 page, the user authorized producing the full clip with those proportions. Repeating the same approval request was unnecessary.

The final clip-three project had four shots: frames 1-136, 137-279, 280-483, and 484-510, at 30 fps for 17 seconds. Red/blue/green represented the floral-shirt protagonist, black-clad partner, and gray-suited character. Overall scales had been 1.10/1.00/1.05. These are historical examples rather than default timing or compulsory height ratios.

Before export, collision-avoidance switching caused blue elbow jumps. Only rotation transitions at 399-407 and 433-441 were repaired, preserving approved still poses, shoulders/wrists, and other content. The lesson is local solver-transition repair, not reusing those frame numbers.

## Project state and operation records

The user reported the second half missing. The project actually contained 510 frames and four cameras, with sets far apart while free view remained at the first set. The correction set camera view, full timeline, and automatic cuts, then reopened and checked without changing actors or lighting. Apply this lesson to project handoff.

Embedded modeling/set/motion scripts, proportion/tracking parameters, revision notes, and reports existed. Full GUI history, recordings, and persistent undo did not. Some historical scripts could not rebuild the final revision. Preserve that distinction when describing operation records.

## What carries forward

**Retain:** source-camera frame review, plausible occluded completion, latest ten-part proportions, primary-light priority, modular source-colored backgrounds, semantic colors, necessary characters, local preservation, seamless stacked comparison, editable projects, and a usable opening state.

**Do not transfer automatically:** old head shape, withdrawn FBX, fully detailed backgrounds, one fixed light side, clip-two six bars/frame 31/frame 100, or clip-three duration/frame rate/identities/colors/smoothing ranges.

**Capabilities not supplied:** one-click exact recovery of arbitrary monocular footage, guaranteed perfect first-pass output, or complete historical mouse-operation recording.

## 2026-11-18: L3 replaces the old model

The user supplied `L3_rigged_white_marks(6).blend`, rejected the previous short, stocky body, and required eye/chest stripes to stop changing size. L3 supersedes all V5 proportions above. Preserve its shape and stripe geometry; only neutral-white material and rigid binding are added. Independent scale animation is prohibited while natural perspective is retained.

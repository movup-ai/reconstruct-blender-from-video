# Revision lessons: diagnose, then change locally

| Symptom | Check first | Repair and regression coverage |
|---|---|---|
| Oval head or changing size | L3 identity, object/bone/ancestor nonuniform scale, perspective | Reuse L3 mesh/rigid binding; no separate stripe scale or old V5 parameters |
| Disconnected head/body | Head pivot, neck point, rig hierarchy | Rotate about the correct neck pivot; review the whole interval |
| Missing articulation | Independent part motion and joint axes | Fix rig/connections without extra joint balls or anatomical detail |
| Green/yellow torso faces sideways | Chest mark, shoulder line, pelvis | Correct torso/pelvis facing, preserving independent head rotation; translation alone is insufficient |
| Red/yellow should face each other | Facing vectors and shoulder lines | Rotate red only in the specified shot, not globally |
| Hand visibility or arm position wrong | Shoulder/elbow/wrist projections, crop, occlusion | Fix limbs/placement; forearm ends represent hands in this style |
| Embrace penetration or wrong overlap | Relative depth, shoulder/chest contact, arm paths | Jointly solve contact/occlusion and review the complete embrace |
| No gap between purple region and blue actor | Region identity, screen gap, depth | Verify spacing, occlusion, and perspective rather than only color/transparency |
| Blue suddenly appears | First source visibility, visibility keys, body path | Correct reveal timing and occlusion; play surrounding frames |
| Twitching or elbow flips | IK poles, rotation branches, constraint switches, independent fits | Repair local rotation transitions while preserving approved head/shoulder/wrist/contact; retain real fast action |
| Camera shake | Static background motion in source | Preserve real camera motion and remove solver noise using background anchors |
| Face lit on wrong side | Primary light, body facing, color management | Correct light direction rather than turning the face to conceal it |
| Extra shadow on blue | Unrelated occluders, duplicate lights, shadow settings | Remove only the specified erroneous shadow and retain other valid lighting |
| Illogical stairs/rails | Steps, slope, turns, posts, gaps, depth | Rebuild from diagrams/source; special counts apply to the relevant shot only |
| Only floating stairs may change | Approved actors/shots | Freeze actors, movement, placement, expression/orientation, and camera; edit background and necessary shadows only |
| Project's second half appears absent | Frame/preview range, camera markers, set locations, collection visibility, local camera | Reopen fresh, inspect late frames, use camera view, automatic cuts, and full timeline |

Revision logs include version, object, source range, feedback, changes, preserved elements, evidence, and unresolved issues. A new instruction can unlock a previously frozen item. For example, a later request to rotate red authorizes that local rotation after a stairs-only instruction, while other frozen elements remain fixed.

Verify preservation using geometry/material/rig/animation/camera/light data or renders of unaffected objects. Hidden objects may have delayed dependency-graph evaluation. Update correctly and distinguish cached hidden data from actual animation changes; neither misreport cache differences nor ignore visible differences.

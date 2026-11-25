# Historical V5 dimensions: superseded by L3, not for current production

# Reference mannequin and palette

On 2026-10-20, the user approved the V5 calibration page and authorized producing clip three with those proportions. Approval applied to that clip's proportions, not all future footage, unseen body regions, or pixel-perfect reconstruction.

The model used normalized units, +Z up and -Y forward. Overall scaling could match height; head/chest/shoulder ratios stayed fixed across shots. Monocular footage does not uniquely determine world scale.

| Parameter | Width X | Depth Y | Height Z / length |
|---|---:|---:|---:|
| Rounded head | 0.387 | 0.387 | 0.430 |
| Torso | 0.490 | 0.230 | 0.650 |
| Upper arm, each | 0.160 | 0.180 | 0.360 |
| Forearm, each | 0.140 | 0.170 | 0.360 |
| Thigh, each | 0.190 | 0.220 | 0.430 |
| Lower leg, each | 0.170 | 0.200 | 0.520 |

Head half-height was 0.215 and width/height ratio 0.90. Shoulder axes were 0.325 from the centerline and 0.045 below the neck; hip axes were 0.115 from the centerline. Outer shoulder width was 0.810. The chest mark protruded approximately 0.0005, so the bounding-box torso depth could read 0.2305.

Head width/height was 0.90; torso/head width 1.266; outer shoulders/head width 2.093. Neutral standing height was 2.03 units, approximately 4.72 head heights, calculated from the template. **The client's bust video did not validate that full-body ratio.** Full-body footage requires leg-length, stance, and gait checks; do not arbitrarily replace it with common adult proportions of 7-8 heads.

The new acceptance reference showed only the upper body. Full torso, forearm, thigh, and lower-leg lengths came from earlier references. The early round-head request ultimately became a slightly elongated rounded head with ratio 0.90; do not interpret 'round' as reverting to a sphere.

Ten parts count body geometry, excluding rigs/lights. White marks are material faces on existing geometry, not an eleventh part. Use suitable joint pivots and connections without extra joint spheres. Forearm ends represent hands. Requests to expose a little hand should first adjust endpoints/framing rather than automatically add palms.

## Semantic colors

Values are display-space sRGB 0-255. Convert once to linear values before assigning Blender materials. Lighting and the view transform also affect rendered colors.

| Category | Color | sRGB |
|---|---|---|
| Protagonist/core subject | Red | 194,70,58 |
| Second character | Blue | 65,93,203 |
| Third character | Green | 58,142,83 |
| Fourth character | Yellow | 202,154,46 |
| Foreground, such as a table | Brown | 137,100,58 |
| Historical background default | Gray | 175,175,175 |
| Distant interactive object, such as a door | Pink | 197,102,146 |
| Nearby interactive object, such as a cup/held prop | Orange | 208,130,56 |
| Light-source representation | White | 244,244,244 |
| Other special target | Purple | 114,73,179 |
| Mannequin front marks | Pure white | 255,255,255 |

Ordinary backgrounds follow the later approximate-source-color rule rather than uniform gray. Foreground and props representing interaction/occlusion retain semantic colors. Record classifications and avoid distance-driven frame-to-frame color flicker. Preserve identities across shots.

For more than four necessary characters, assign and document extra distinguishable colors; purple is not automatically the fifth character. Clip three used red for the bald floral-shirt protagonist, blue for the black-clad interacting character, and green for the gray-suited character. These identities do not transfer automatically to new footage. The brown-suited bald protagonist designation applies only to clip two.

Current production starts with `l3-model.md`. The old template is not included in this package.

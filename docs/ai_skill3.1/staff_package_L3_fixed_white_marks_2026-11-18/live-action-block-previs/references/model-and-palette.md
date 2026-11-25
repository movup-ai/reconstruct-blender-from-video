# Current model and palette

On 2026-11-18, the user explicitly selected `L3_rigged_white_marks(6).blend` to replace the old short, stocky V5. Follow [L3 and fixed white marks](l3-model.md), preserve the supplied mesh proportions, and stop generating bodies from V5 parameters. The source has ten body parts in one mesh; ten meshes and ten total rig bones are not required.

Set one fixed uniform scale per actor. Keep head/chest/shoulder and stripe sizes constant across frames and shots. Animate position and rotation while preserving natural perspective. Remap motion-capture bone lengths to L3 rather than reusing old fitting lengths/scales.

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

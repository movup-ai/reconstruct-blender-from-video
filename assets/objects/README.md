# Object models for Stage 2 (Background)

Put one `.blend` file per object type in this folder. The file name is the type, and it is also what SAM 3
searches the video for:

| File | Type | SAM 3 looks for |
|---|---|---|
| `wall.blend` | wall | "wall" |
| `pillar.blend` | pillar | "pillar" |
| `coffee_table.blend` | coffee_table | "coffee table" |

Every object of that type found in a clip becomes a copy of the model, stretched to the size measured in the
video and placed where the depth puts it.

## How to make a model

- **Stands on the ground.** Its lowest point is the floor (z = 0 or anywhere: it is moved down onto the floor).
- **Faces -Y.** Blender's Front view (numpad 1) looks at its front. The front is turned toward the camera.
- **Any size.** It is stretched to the size measured in the video. Its proportions still matter: the depth that
  the camera can't see is taken from them (a round pillar seen from the front is as deep as it is wide).
- **Every visible mesh is used.** Several objects in one file are combined. Hidden objects (render off) are left
  out. Modifiers are applied.
- **Materials don't matter.** The colour card colours every object (`docs/COLOUR_CARD.md`).
- **Keep it light.** Under 200,000 faces. A white model needs the shape, not detail.
- `floor.blend` / `ground.blend` are ignored: Stage 1 already builds the floor.

## Optional: objects.json

```json
{
  "coffee_table": {"prompt": "low table", "role": "fg"},
  "painting": {"on_floor": false},
  "floor_lamp": {"on_floor": true}
}
```

- `prompt`: other words for SAM 3 to search for (default: the file name with `_` as spaces).
- `role`: a colour-card role that always wins: `bg`, `fg`, `near`, `far_i`, `lamp`, `special`.
- `on_floor`: whether the object stands on the floor. By default paintings, pictures, posters, frames, mirrors,
  clocks, windows, shelves, signs, screens, TVs, curtains and lights hang: they stay at the height they are seen.
  Every other type stands on the floor, and its box reaches down to the floor even when its bottom is hidden.

Each model is read once with Blender and cached in `_cache\` (it is read again when the file changes).

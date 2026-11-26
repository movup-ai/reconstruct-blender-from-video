# Object models for Stage 2 (Background)

## Included library

This folder includes **65 simple, editable low-poly models**, one mesh per `.blend`.
Browse [the visual catalog](catalog.jpg). The complete dimensions and polygon counts are in
[manifest.json](manifest.json); models use metre units, face -Y, have applied transforms,
and have their base at z = 0 with an origin centred on x/y. Each mesh is marked as a Blender asset.
There are no cameras, lights, textures, external dependencies, or unapplied modifiers in the files.

| Group | Included files (without `.blend`) |
|---|---|
| Seating | sofa, armchair, ottoman, chair, office_chair, stool, bench |
| Tables | table, coffee_table, pool_table, desk, counter, kitchen_island |
| Storage | cabinet, wardrobe, chest_of_drawers, shelf, bookshelf |
| Bedroom | bed |
| Kitchen and bathroom | sink, stove, oven, fridge, microwave, washer, dishwasher, bathtub, shower |
| Screens and lights | tv, monitor, lamp, floor_lamp, table_lamp |
| Containers | box, bin, barrel, vase, pot, cup |
| Building parts | wall, door, window, curtain, blind, picture, mirror, pillar, pole, stairs, railing, fence, awning |
| Other fixtures | fireplace, radiator, booth, arcade_machine, streetlight, traffic_light |
| Vehicles | car, truck, van, bus |
| Vegetation | tree, pine_tree, bush |

The app discovers these files automatically during Stage 2. `objects.json` supplies default
search prompts and placement rules. Bookshelves stand on the floor; shelves hang. Cups, pots,
vases, sinks, screens, and microwaves retain their observed height rather than extending down
to the ground. Floor lamps stand on the ground; the table lamp retains its measured height
(see `on_floor` metadata). Materials are neutral grey for browsing; Stage 2 recolours them.

Rebuild missing models with Blender from the project root:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe' -b --factory-startup --python-exit-code 1 -P pipeline/build_object_assets.py
```

Existing models are preserved. Add `-- --overwrite` to deliberately regenerate the whole
supplied library. User overrides already in `objects.json` are preserved in either case.
Individual preview images are written under `output/object_asset_previews/`.

Validate saved assets, manifold geometry, dimensions, origins, and the actual Stage 2 export path:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe' -b --factory-startup --python-exit-code 1 -P pipeline/check_object_assets.py
```

The validation report is `output/object_asset_validation.json`.

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
- `labels`: the Mask2Former (ADE20K) class names that count as this type, e.g. `["door", "screen door"]`. A found
  object is kept only when at least 30 % of its pixels carry one of them. Default: the classes of the type's role in
  `pipeline\semseg.py` (a type with none, such as a bush, is not checked).
- `size`: the real-world size range `[[min width, max width], [min height, max height]]` in metres, e.g.
  `[[0.3, 1.0], [0.6, 1.3]]`. A placed object far outside it is left out; one too wide is first split into pieces.
  Default: `pipeline\object_checks.py` (chairs, sofas, tables, desks, doors, beds, cabinets...; others unchecked).

These checks are in `pipeline\object_checks.py`; `"bg_checks": false` in `config.json` turns them off.

Each model is read once with Blender and cached in `_cache\` (it is read again when the file changes).

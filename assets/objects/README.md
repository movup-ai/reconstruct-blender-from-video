# Object models for Stage 2 (Background)

Stage 2 finds the objects in a clip with **Mask2Former** (ADE20K, 150 names; `models\mask2former_ade_panoptic`).
Each object it names becomes a copy of the model for that name, stretched to the size measured in the video and
placed where the depth puts it. The table of name → model is `pipeline\ade_objects.py`:

- **the same name**: `sofa` → `sofa.blend`, `chair` → `chair.blend`, `door` → `door.blend`, ...
- **another name for the same thing**: `windowpane` → `window.blend`, `painting` / `poster` → `picture.blend`,
  `refrigerator` → `fridge.blend`, `television receiver` → `tv.blend`, `column` → `pillar.blend`,
  `swivel chair` → `office_chair.blend`, `ashcan` → `bin.blend`, ...
- **not placed**: people (Stage 1) and the ground, sky, water and landscape (`floor`, `sky`, `road`, `sea`, ...).

This folder holds exactly the models that table uses: 116. 95 were generated (browse them in [the catalog](catalog.html),
[contact sheets](catalog.jpg), sizes and polygon counts in [manifest.json](manifest.json)); the 21 below were added by
hand for the names the generated library did not cover. If a model is ever missing, an object with that name is
reported in the job's warnings ("no model for: ...") and left out.

| Added by hand | Mask2Former name |
|---|---|
| `airplane.blend` | airplane |
| `animal.blend` | animal |
| `blanket.blend` | blanket |
| `bridge.blend` | bridge |
| `building.blend` | building |
| `ceiling_light.blend` | light |
| `chandelier.blend` | chandelier |
| `clothes.blend` | apparel |
| `conveyor_belt.blend` | conveyer belt |
| `food.blend` | food |
| `grandstand.blend` | grandstand |
| `pedestal.blend` | base |
| `projection_screen.blend` | screen |
| `range_hood.blend` | hood |
| `sideboard.blend` | buffet |
| `skyscraper.blend` | skyscraper |
| `stage.blend` | stage |
| `tank.blend` | tank |
| `tower.blend` | tower |
| `toy.blend` | plaything |
| `wall_lamp.blend` | sconce |

To use a different model for a name, change its entry in `pipeline\ade_objects.py`.

## How to make a model

- **Stands on the ground.** Its lowest point is the floor (it is moved down onto the floor).
- **Faces -Y.** Blender's Front view (numpad 1) looks at its front. The front is turned toward the camera.
- **Any size.** It is stretched to the size measured in the video. Its proportions still matter: the depth that
  the camera can't see is taken from them (a round pillar seen from the front is as deep as it is wide).
- **Every visible mesh is used.** Several objects in one file are combined. Hidden objects (render off) are left
  out. Modifiers are applied.
- **Materials don't matter.** The colour card colours every object (`docs/COLOUR_CARD.md`).
- **Keep it light.** Under 200,000 faces. A white model needs the shape, not detail.

Each model is read once with Blender and cached in `_cache\` (read again when the file changes).

## Optional: objects.json

```json
{
  "coffee_table": {"role": "fg"},
  "picture": {"on_floor": false},
  "chair": {"size": [[0.3, 1.0], [0.6, 1.3]]}
}
```

- `on_floor`: whether the object stands on the floor. Hanging things (pictures, mirrors, windows, clocks, shelves,
  screens, curtains, lights...) stay at the height they are seen; everything else reaches down to the floor even
  when its bottom is hidden.
- `role`: a colour-card role that always wins: `bg`, `fg`, `near`, `far_i`, `lamp`, `special`.
- `size`: the real-world size range `[[min width, max width], [min height, max height]]` in metres. A placed object
  far outside it is left out; one too wide is first split into pieces. Defaults: `pipeline\object_checks.py`.

The checks are in `pipeline\object_checks.py`; `"bg_checks": false` in `config.json` turns them off.

## The generated library

The included models were generated with Blender; the generator builds only the models the table uses:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe' -b --factory-startup --python-exit-code 1 -P pipeline/build_object_assets.py
```

Existing models are kept; add `-- --overwrite` to make them again, or `-- --overwrite --types sofa chair` for some.
Refresh the catalog after adding or changing models:

```powershell
.venv/Scripts/python.exe pipeline/build_object_catalog.py
```

Check the saved models (geometry, sizes, origins, the Stage 2 export):

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe' -b --factory-startup --python-exit-code 1 -P pipeline/check_object_assets.py
```

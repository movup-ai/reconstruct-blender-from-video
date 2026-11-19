# Writing a clip's set by hand (env_claude.json)

The client wants a clean, readable grey "white model" set for each clip, like the sample
`SOURCE\3D\49243ef628fc30fa1018347eaf6a921d.mp4`: floor, walls, doors, windows, big furniture,
trees, rocks, cars, buildings, each as a simple recognisable block model. No people (they come from the local model),
no tiny props, no textures. Readable beats detailed: 10 to 40 objects per shot is typical.

Everything runs from the project folder with `.venv\Scripts\python.exe`, `PYTHONIOENCODING=utf-8`.

## Steps for one clip

0. **Start from the automatic proposal.** `python pipeline\env_auto.py <clip>` (GPU, about 5-15 s per clip once the
   models are loaded; several clips in one call: `env_auto.py <clip1> <clip2> ...`, or `--all [--skip-done]`) measures
   the first and middle frame of every shot with MoGe-2 (metric 3D points + normals, with the pipeline's own field of
   view) and SAM 2.1 (masks; the people are masked out). It writes `output\<clip>\env_auto.json` (same format as
   env_claude.json; never touches env_claude.json) and `Work_auto\<clip>\env_auto.jpg`: per frame, the masks with
   their class (wall, building, pole = trunk / post, vegetation, rock, top = table / counter top, object, far, slope,
   terrain; people red) and object names | the proposed set over the film | the set alone.
   Copy env_auto.json to env_claude.json and correct it: delete wrong objects, fix types (a horse or a piano comes out
   as a plain `box` / `cabinet`), merge or add what is missing, check sizes. Keys starting with `_` (`_src`: class,
   frame, pixel box, distance of the mask that made the object; `_auto`: measurements) are ignored by build_env and
   can stay or go. What it gets right most often: walls with their real top line (`patch` with `pts3`, a flat slab in
   metres), trunks / posts, bushes and big blobs where they stand, distant hills as `far` boxes, the shape of rolling
   ground (`ground.heights`). Weak: very dark pictures, long thin things, objects hidden behind people, vehicles and
   animals (not recognised), and sizes along the view direction (the back of a thing is never seen).
   The printout and `_auto.shots.<n>` compare MoGe's camera height / tilt with the fit's (`moge_camera`,
   `disagreement`, `suggest_hint`). The proposal is built in the fit's camera (it lines up with env_check), so a
   disagreement does not have to be fixed; use `suggest_hint` in hints.json (step 4) only when env_check shows the
   floor is wrong. `gravity_slope` is the ground slope measured against upright walls (only a report).
1. `python pipeline\env_frames.py <clip>` prints the shots (frames, camera height, tilt, how much the camera turns,
   which people are in each shot) and writes `Work_auto\<clip>\env_frames.jpg`: first / middle / last frame of every
   shot with a yellow pixel grid (labels every 100 px). Read that picture. The sheet may be scaled: the printout gives
   the scale; grid labels are always true picture pixels.
2. Write `output\<clip>\env_claude.json` (format below), starting from env_auto.json (step 0).
3. `python pipeline\build_env.py <clip>` builds the set into `Work_auto\<clip>\scene_data.json` and writes
   `Work_auto\<clip>\env_check.jpg`: for a few frames per shot, film | set over the film | set alone. Read it.
   The people are drawn too: a stick figure per person in the middle panel (with a white ring on the floor under each
   foot and a label "feet +0.42m" when the feet are off the floor), and the block bodies in the right panel. A red X
   with "in <object>" means a body part is inside a set object. Floating or buried feet usually mean the camera
   height / tilt is wrong (fix it with hints.json, step 4) or a surface is missing / too high under the person.
   Fix what is off (an object floating, too big, wrong place, missing a big thing) and run it again. Two or three
   rounds is normal.
4. Optional, only when the floor itself is clearly wrong in env_check (objects that stand on the floor drift or float,
   the floor line is far from the film's): write `output\<clip>\hints.json`
   `{"camera": {"height": 1.6, "pitch_down_deg": 8}}` (or per shot: `{"camera": {"shots": {"2": {"height": 1.4, "pitch_down_deg": 3}}}}`),
   then `python pipeline\run_pipeline.py <video path> --stage fit` (1 to 5 minutes) and step 3 again.
   The fit log line `ground: {"source": "hints.json", ...}` confirms the hint was used. A camera move can be set too:
   `"path": [[frame, yaw, pitch, roll, right, forward, up], ...]` (degrees and metres, relative to the shot's first
   frame; yaw + = turn left, pitch + = tilt up) replaces the tracked camera turn when the tracking is wrong.
   Keep any `people` entries already in hints.json.
5. Do NOT render and do NOT run any other stage; rendering is done afterwards for all clips together.

The automatic camera (height, tilt, turn, cuts) was improved: roll is now always level, false pans and tilts are
mostly gone. Prefer the automatic camera when it lines up; write a camera hint only where env_check shows it is wrong.

## Coordinates

Each shot has its own world: floor z = 0, metres, camera above the origin looking along +y at the shot's first frame,
+x to the right of the picture. Distances: a door 3 m away is at y ≈ 3. The printout gives the camera height.
Pixels (px, bbox) are inside the picture area, the same pixels as the grid in env_frames.jpg.

## File format

One shot (no cuts):
```json
{"ground": {"size": 60, "colour": "ground"},
 "sky": [205, 207, 210],
 "objects": [ ... ]}
```
Several shots (the printout lists them): one block per shot, keyed by the shot number.
```json
{"shots": {"1": {"ground": {...}, "objects": [...]},
           "2": {"ground": {...}, "objects": [...]}}}
```
`ground.size` is the side of the floor square in metres (20-40 indoors, 200-600 outdoors).
`sky` is the background colour (grey, like [205, 207, 210]; indoors a wall usually covers it).

### Sloped and uneven ground (outdoors)

Without these keys the floor is flat. With any of them the ground becomes a faceted height field, objects placed by
`pos` / `px` / `pts` stand on it (their `z` is then the height above the ground under them; `"abs_z": true` keeps a
world height), each object sinks to the lowest ground under its footprint, and the people's feet follow it.
```json
"ground": {"size": 300, "colour": "ground",
           "slope_deg": 6, "slope_dir_deg": 0,
           "profile": [[0, 0], [9, 0], [14, 3], [30, 11], [70, 32]],
           "heights": [[-14, 22, 2.5, 14], [16, 26, -1.0]], "falloff": 8}
```
- `slope_deg` / `slope_dir_deg`: a plane rising `slope_deg` degrees towards `slope_dir_deg` (0 = away from the camera,
  90 = right, 180 = towards the camera, 270 = left). It passes through z = 0 where the shot's people are (or at
  `"origin": [x, y]`), so the fitted people keep their height there.
- `profile`: `[[d, z], ...]` ground height z at d metres from the point under the camera along `slope_dir_deg`,
  straight lines in between: flat path, then a hillside, then a plateau. Keep z = 0 where the people walk.
- `heights`: `[[x, y, dz(, falloff)], ...]` round mounds (dz > 0) or hollows (dz < 0) blending out over `falloff`
  metres (default 8).
Facets steeper than `steep_deg` (default 14) get `slope_colour` (default `set`) so hillsides read against the pale
floor.
Flat `patch` shapes follow the ground only at their corners: give long roads on bumpy ground more corner points.

### Objects (choose one way to place each)

- **By its box in the picture, on the floor at its bottom edge** (things standing on the floor, turned to face the camera):
  `{"type": "cabinet", "bbox": [x0, y0, x1, y1], "dist": 3.2}` - `dist` = metres from the camera along the floor.
  Or `"z": 0.76` instead of `dist` = its bottom edge stands on a surface that high (things on a desk: z = desk height).
- **A flat front at a known depth** (back walls, windows, pictures, shelves against the back wall):
  `{"type": "window", "bbox": [...], "depth": 3.7, "thick": 0.08}` - `depth` = metres forward (y) of that wall.
- **On a wall that is not facing the camera**: `{"type": "box", "bbox": [...], "plane": [xa, ya, xb, yb], "z0": 0, "thick": 0.2}`
  - the wall runs along the floor line from (xa, ya) to (xb, yb) metres.
- **Its top edge is what you see, it stands on the floor** (chairs behind a table): add `"anchor": "top"` with `dist`.
- **In metres**: `{"type": "box", "pos": [x, y, z_bottom], "size": [width, height, depth], "yaw": 0}`
  yaw in degrees about the vertical; 0 = front faces the camera. Use this for walls, floors of rooms, big tables.
- **On the floor under a pixel**: `{"type": "rock", "px": [u, v], "size": [w, h, d]}` (`"frame": n` to use another
  frame of the shot than its first).
- **Flat shapes** (river, rug, table top seen from above, road): `{"type": "patch", "px": [[u, v], ...], "z": 0.02}`
  or `"pts": [[x, y], ...]` in metres; `"thick": 0.05` makes it a slab.
- **Any flat polygon in metres** (a wall with a sloped / stepped top, a sloped bank, a ramp):
  `{"type": "patch", "pts3": [[x, y, z], ...], "thick": 0.4, "colour": "rock"}` - corners in order, z absolute;
  `thick` pushes the slab back away from the camera. env_auto writes walls this way.
- `"off": true` keeps an object in the file but does not build it.

### Types (default size w, h, d in metres)

door, window, cabinet, wall_cabinet, shelf, bookshelf, counter, sink, table, desk, chair, office_chair, stool, sofa,
bed, fridge, stove, tv, lamp, lamp_post, car, van, truck, bus, bicycle, tree, pine, bush, hedge, bank, plant_pot, fence,
wall, railing, stairs, bench, bin, box, barrel, sign, umbrella, awning, pillar, kettle, bottle, bag, building, rock,
horse, curtain, mirror, picture, walker. Anything else: `box`. `size` overrides the default: [width, height, depth].
Vehicles and `horse` (2.4 x 1.6 x 0.6): width is their length; at yaw 0 a horse's head points to +x (right).

Vegetation and rocks are faceted low-poly shapes (block style, flat shaded), never smooth blobs:
`tree` = trunk + three stacked canopy clumps; `pine` = trunk + stacked cone tiers; `bush` = a cluster of 3-5 clumps
(its width sets how many); `rock` = an irregular faceted block; `hedge` (3 x 1.2 x 0.9) = a long row of clumps along
its width; `bank` (6 x 1 x 3) = a long low faceted grass bank / mound. Each object gets its own random variation;
`"seed": n` picks another one.

### People on horses (hints.json, not env_claude.json)

A rider gets a horse under them for the whole shot (it follows the person's position and heading every frame and its
legs walk with the ground speed) and a riding leg pose (hips at saddle height, thighs forward and apart, shins down):
```json
{"people": {"1": {"mount": "horse", "leg_pose": "riding", "yaw": 90, "yaw_sigma": 8}}}
```
Key = the person's track number. `yaw` (0 = facing away from the camera, 90 = facing left, 180 = towards the camera,
-90 = right) holds the heading: the horse points where the rider's hips point, so give it when the horse's direction
is clear in the film. Options: `seat_height` (hip height on the saddle, default 1.55 m), `mount_size`
[length, withers height] (default [2.4, 1.6]), `mount_shift` (m, horse forward + / back - under the rider),
`mount_yaw` (extra turn of the horse only), `mount_colour` (default `set`), `"mount_gait": false` (legs still).
Changing these needs `run_pipeline.py --stage fit` (riding pose), then `build_env.py` (the horse). Riderless horses
are ordinary set objects: `{"type": "horse", "pos": [x, y, 0], "yaw": 90}`.

People cut off by the bottom of the frame (no knees / ankles in most frames) now get standing legs automatically;
write `"legs": "seated"` in hints.json when they actually sit.

### Legs, kneeling and lying that change during the shot (hints.json)

`legs` and `leg_pose` also take frame ranges `[[first, last, value], ...]` (the old single value still works):
```json
{"people": {"2": {"legs": [[0, 108, "standing"], [116, 143, "seated"]], "seat_height": 0.52,
                  "leg_pose": [[116, 143, {"hip": 70, "knee": -80}]]}}}
```
Frames outside every range (here 109-115, sitting down) are fitted from the marks with the feet on the floor, like
`"standing"`. `seat_height` is still one number (or ranged the same way). Run the fit stage again after changing it.

Kneeling, crouching and squatting are found from the marks (bent knees in the picture, or legs short compared to
the torso): in those frames the pelvis may go down to ~0.2 x height and low start poses are tried. Seen from the
front a standing body can often explain the marks too (and RTMPose puts the hidden ankles under the knees); then
say so: `"legs": [[72, 90, "kneeling"]]` (hips ~0.52 m up, thighs upright, shins on the floor, ankle marks ignored;
`seat_height` / `leg_pose` {"hip", "knee"} change it). Lying down is found when shoulders -> hips and
hips -> ankles run sideways in the picture; the body is then pitched at the pelvis and laid on the floor. When the
picture cannot tell (seen from the feet / head end, or half hidden) force it:
`"pose": "lying"` (whole shot) or `"pose": [[0, 40, "lying"]]`; `"lying": false` switches the automatic test off.
(`"pose": {joint: [x, y, z]}` is still the held-angles form.)

### People on steps, terraces and uneven ground: keep the fit's height

`build_env.py` puts every person's feet on the ground (or a large hand-built top surface) under them. When the fit
already had them at the right height (steps the set does not model, a raised floor), keep the fit's own height:
hints.json `{"people": {"2": {"keep_height": true}}}` for one person, or env_claude.json
`"ground": {"follow": false, ...}` for everyone in the shot (per shot in `shots`). `build_env` prints a WARNING when
the ground under someone is more than 0.3 m (median) from the fit's height; check those people in env_check.
People are lifted onto a set object only where one of its upward faces really is under them (a long diagonal wall
no longer lifts everyone inside its bounding box).

### Wrong person tracks (hints.json)

When a track jumps from one person to another (identity swap) or two people share a track, move frame ranges of a
track to another track (ids as in pose.json / the review page; a new id makes a new person):
```json
{"tracks": {"assign": [[1, 79, 999, 2], [2, 91, 101, 1]]}}
```
= track 1 frames 79-999 become track 2, track 2 frames 91-101 become track 1 (all moves read the tracks as they were,
so swaps work). `{"tracks": {"drop": [[3, 0, 40]]}}` removes frames 0-40 of track 3. Then run the fit stage again.

### Colours (`"colour"`, optional; default grey)

`set` (mid grey), `light`, `white`, `dark`, `far` (pale, for distant things), `ground`, `tree`, `rock`, `fence`, `roof`,
`cabin`. Never red, blue, green or yellow: those are the characters. `colour_dark` / `colour_light` colour the
object's trim parts.

## What good looks like

- The floor and the main surfaces line up with the film in env_check (middle panel).
- Rooms: back wall + side walls (box, 0.1 m thick) at the right depth, the doors/windows on them, the big furniture.
- Outdoors: ground, the main trees / rocks / buildings / cars where they are, distant hills or tree lines as large
  `far` boxes or a `pine` row far away.
- Things that would sit in front of a character's body in the film are fine; nothing should stand where a person
  walks (the people are placed by the fit, not by you).
- Prefer fewer, correctly placed objects over many guessed ones.

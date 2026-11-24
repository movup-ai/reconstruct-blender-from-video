# Blender Agent

Local white-model production app. It turns the film clips in the `SOURCE` folder into the client's submission
packages (target, white model, comparison video, `.blend`, `meta.jsonl`, zip). Everything runs on this PC.
Person reconstruction uses **SAM 3 tracking → RTMPose masked joints → SAM 3D Body → editable Blender rigs**.
SAM 3 also segments scene objects using the same local `models/sam3/sam3.pt` checkpoint. Its image and video
workers and the body worker run sequentially in the isolated Python 3.12 CUDA environment. RTMPose uses
TensorRT or CUDA, and the scene depth model uses CUDA (`onnxruntime-gpu`). MoGe-2, Mask2Former, Grounding DINO
and CLIP handle scene geometry, semantic labels and carried objects. Every inference stage needs an NVIDIA GPU.
`Setup.bat` installs the main runtime and RTMPose joint model. The required SAM worker setup and approved
checkpoint locations are documented in [SAM 3 tracking](docs/SAM3_TRACKING.md) and [SAM 3D Body](docs/SAM3D_BODY.md).
Use one concurrent job on the RTX 4070. TensorRT joint engines are cached in `models/trt_cache`.
A `3D` folder inside `SOURCE` holds the client's samples and is not treated as clips (`source_exclude` in `config.json`).

Every fit also (see `pipeline\quality.py`, `pipeline\lens_auto.py`):
- measures the clip's lens from the picture (MoGe-2) instead of the fixed `lens_mm`; `"lens_auto": false` switches it off,
  and a lens set in `hints.json` or in the review wins;
- leaves out tracks that are both short and unclear (`"junk_filter": false` switches it off);
- in half-body shots (no feet in the picture) puts the floor under the people (`"floor_from_people": false`);
- writes quality numbers and flags into `report.json` and the clip's warnings (`python pipeline\quality.py <clip>`);
- builds what people carry (`pipeline\held_objects.py`): Grounding DINO (`models\grounding_dino`) finds objects in the
  hands and backpacks on every 2nd frame, CLIP names them (bag, phone, cup, umbrella...), and they are placed on the
  fitted hand or back, moving with the person (`Work_auto\<clip>\held_check.jpg` shows them over the film). Found once
  at Prepare (`output\<clip>\held.json`); the set leaves them out. `"held_objects": false` in `config.json` or a
  clip's `hints.json` switches it off.

## Processing upgrades

SAM 3D Body is the metric 3D body backend using tracked masks and the local body
checkpoints. It adds visible-mesh depth calibration and camera-space pose/placement constraints to the
editable mannequin fitter. See [SAM 3D Body operation and validation](docs/SAM3D_BODY.md).

SAM 3 video tracking uses the local `models/sam3/sam3.pt` checkpoint. It supplies
persistent person masks and IDs to RTMPose joint measurements. Prepare/Re-check generates the new tracks and
backs up previous person edits. See [SAM 3 setup, operation and limitations](docs/SAM3_TRACKING.md).

The existing models now support MAVE-style shared metric evidence, gravity and Manhattan alignment,
proxy-based depth scale calibration, numerical scene checks and bounded geometry repair. Prepare and
Finish run these stages automatically; Settings has switches for shared geometry, scene checks and repair.
The clip's Things to check list includes unresolved geometry issues. See [the implementation and testing guide](docs/MAVE_EXISTING_MODELS.md)
for diagnostics, evidence queries, human scale controls and the differences from the paper.

Settings includes foot contact, adaptive set views, checkpoint locations and the camera solver.
SAM 3 IDs are retained through pose measurement and body reconstruction. Identity merging and synthetic
track-gap filling are removed. Fitted visibility follows observed masks, and manual joint corrections,
roles and camera hints remain available. Foot contact is a soft world-space constraint when visible ankles
agree with a planted foot; per-person `"foot_contact": false` and keyed leg joints take precedence.

Adaptive set sampling keeps the first and middle views and adds views when the camera travels 0.5 m, turns
12 degrees or changes zoom by 15 percent relative to the nearest selected view. `set_max_frames` defaults to five.
For nearly fixed views it also looks for background uncovered when people move away.
Frames explicitly annotated by an operator are always included. Set caches and completed-job fingerprints
include the new settings, so a changed configuration can trigger rebuilding.

The human model flow is fixed. Older saved backend choices are migrated automatically. See
[the active model inventory](docs/MODEL_FLOW.md) for each remaining dependency and the retired components.

The optional `da3` camera backend uses Depth Anything 3 multi-view extrinsics and interpolates between sampled views.
It scales translation from visible floor contacts, or from the DA3 translation scale in Settings (`da3_meters_per_unit`). When scale cannot be
estimated, a warning is recorded and the camera uses the predicted rotations. A static-camera review or a hand-authored
camera path takes precedence. `da3_max_views` defaults to 12 and `da3_resolution` to 504.
Set `da3_model` to a downloaded checkpoint folder and `da3_python` to a Python executable with DA3 and CUDA PyTorch.
The worker exchanges NumPy files with the main process. This allows DA3's NumPy 1.x dependencies to live in an
isolated environment under `downloads\da3-venv`. The upstream package and model are available from
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3).
`requirements-da3.txt` lists the camera worker's dependencies. For a fresh installation, install CUDA PyTorch 2.8
and torchvision 0.23 in a Python 3.12 virtual environment at `downloads\da3-venv`, then run these commands.

```powershell
uv pip install --python downloads\da3-venv\Scripts\python.exe --target third_party\da3 --no-deps https://github.com/ByteDance-Seed/Depth-Anything-3/archive/refs/heads/main.zip
uv pip install --python downloads\da3-venv\Scripts\python.exe -r requirements-da3.txt
```

Each pipeline run writes `Work_auto\<clip>\profile_<stage>.json`, including a failed run. It records wall time and
call count for marks, lens, camera, people fitting, the depth set, carried objects, environment building, Blender,
and encoding/packaging. These timings include model startup and subprocess work.

## Start

Double-click **`Start_Blender_Agent.bat`**. The browser opens at http://127.0.0.1:8765.

- The first start runs `Setup.bat` on its own (Python packages and models, one time, needs internet).
- Keep the black window open. Closing it stops the app, and unfinished jobs continue at the next start.
- If the app crashes, the window restarts it after 5 seconds. Details are in `logs\agent.log`.
- Starting it a second time just opens the browser on the running app.

## Work

1. **Copy the videos into `SOURCE\`.** Each one appears on the Clips page as a card by itself; the ✕ on a card takes it
   off the list (the file stays; `data\removed.json` remembers it, and *Removed from the list* puts it back).
   *Name them…* gives videos without a v2.1 package name `title-1`, `title-2`… in file-name order from one English
   title (saved in `data\package_names.json`). Joints, work files and packages go to `output\`, `Work_auto\` and `SUBMIT\`.
   **Task zips** from the client (the v2.1 layout with the source video at `target/original.mp4` and `meta/task.json`)
   can go into the folder as they are: the app writes each zip's video next to it as `<zip name>.mp4`, lists it with
   the task code and name, and uses the zip name as its delivery name. The zip is not changed; a video of that name
   that is not from the zip is left alone (a note appears at the top of the page). The finished package is the usual
   v2.1 zip in `SUBMIT\`.
2. **Start.** Nothing runs until you press **Start** on the Clips page, also after restarting the app (jobs left from
   last time wait in the queue). Start prepares every video not prepared yet; **Pause** lets the running job finish
   and starts nothing new. The Clips page shows each video's step and progress.
3. **Review** a clip marked *ready for review*: **1 Who is who** (cards saying *Not a person?* look like a statue, a
   picture or a false detection; one click, then *Re-check*), **2 Joints** (the editor opens on the frames where the
   fitted person moved suddenly), **3 Objects**. Press **✓ Mark reviewed** (finish later) or **▶ Finish now**. Camera, set, held objects and the description
   are filled in automatically; change them under **More** only if they are wrong.
4. **Finish all reviewed** (or **Finish** on one clip): final fit, set, Blender, videos and the v2.1 zip in `SUBMIT\`.
   A clip whose roles, joints, objects, hints and notes are unchanged since its last Finish is skipped
   (`Work_auto\<clip>\finished.json` holds the fingerprint).

**Settings → Autopilot → Full auto** does steps 2–4 on its own after Start, without a review. If a DeepSeek key is saved, DeepSeek checks the
people and the set first. You can still open any clip, change it and finish again.

The start page also has: *Clips at once* (parallel jobs), *Retry failed*, *Clear queue*, *Report (CSV)* and a system check.

## AI suggestions (DeepSeek, optional)

On the start page, under **AI suggestions · DeepSeek**, paste your API key and press **Save key**, then **Test**.
The key is stored only in `config.json` on this PC and is never shown again. (You can also put it in `deepseek_key.txt`
next to the launcher, or set the `DEEPSEEK_API_KEY` environment variable.)

With a key, every prepared clip gets one DeepSeek check:
- **People:** who is a real person (drivers behind glass, posters and reflections become *Not a person*), and which role.
  A per-film cast memory gives the same character the same colour in every clip.
- **Environment:** DeepSeek sees each shot's empty background with every wall (W) and block group (B) numbered. It names
  what each one is, removes wrong ones (sky, ground, smears), and adds missed objects (cars, tables, doors, trees…) as boxes
  standing on the ground at the right distance. The clip is then rebuilt with these corrections, and the review page lists them.
- **Camera type, held objects, and a Chinese description** for `meta.jsonl`.

The environment check needs a DeepSeek model that accepts images. **Test** tells you whether yours does. Frames of the
clips are sent to DeepSeek when AI suggestions are on.

## Colours (colour card)

Every video follows the colour card in [docs/COLOUR_CARD.md](docs/COLOUR_CARD.md) (`pipeline\colour_card.py`):
characters red, blue, green, yellow by role (5th on: cyan, magenta...); background grey `#B8B8B8`; foreground objects
(in front of people) brown; things people carry or touch orange; interactive things far from people (a distant door)
pink; lamps and lights white; special objects purple. It is applied when the set is built and again before every
render, so older clips follow it when they are finished again.

## Object notes on the clip page

Draw a box, keep the type *other*, and describe the object in your own words (up to 200 characters). Prepare and
Finish build every note at the frame it was drawn on, placed from that frame's camera and measured depth:
- a thing the description names with a regular shape (door, pillar, table, chair, lamp, car...) -> that model;
- a plant, curtain, statue, or anything the description does not name -> its measured shape (SAM cuts the thing out
  of the box, the depth model gives its surface);
- "held", "in his left hand", "carrying", "backpack", "wearing" -> on that person's hand or back for the whole clip;
- "interactive", "distant", "special", "foreground", or a light ("ceiling lamp") -> that colour-card meaning.
A type picked in the list wins over the description. The job log lists what each note became.

## Objects in the set (like the client sample)

Each clip page has an **Objects in the set** editor. Drag a box on the shot's background, pick the type (door, window,
cabinet, shelf, counter, sink, table, chair, sofa, bed, fridge, car, van, tree, bush, fence, lamp post, stairs, building
and more), optionally its **Kind** (the colour card's set meanings: background grey, foreground brown, near
interactive orange, far interactive pink, light source white, special purple; auto decides from what it is and where)
and the facing direction (library models only), then save. The picture is the original key frame, people included, so
things partly hidden by a person can be boxed too; one hidden in that frame stands on the floor under its box. Every object is built from `pipeline/props.py` as a
simple recognisable model (a door has a frame, panel and handle; a car has a body, cabin and wheels), standing where it
touches the floor and sized from the picture. With a DeepSeek key, DeepSeek fills the list in for every shot.

**Distance** (auto, or 1 near .. 9 far) fixes the order when the depth measures an object at the wrong distance. In a
shot, an object is pushed back along the camera's view until it is behind every object with a smaller number; the
biggest number in the shot (2 or more) is the back of the set and also goes behind every person on every frame of the
shot, so nobody walks through it. Pushing back along the view keeps it where the picture shows it (it just gets bigger).
Objects left on auto stay where they are measured (`env_auto.by_level`).

## What happens in each step

**Prepare** (Start; one job per clip; timing depends on clip length and crowding):
1. *People*: SAM 3 tracks person masks and persistent IDs; RTMPose measures 17 joints inside each mask
   (`automark.py` -> `pose.json` and `sam3_masks.npz`).
2. *Camera and people in 3D*: cuts are found, the lens is measured (MoGe-2), the camera height / tilt / turn is
   solved, and SAM 3D Body recovers metric bodies and depth placement that drive editable mannequins -> `scene_data.json`.
   A zooming lens is tracked too (`scene_data.json` "zoom": focal scale per frame): the people, the set, the drawn
   objects and the Blender camera (lens keyed every frame) all use each frame's own focal length, so things at the
   picture's edge in the first frame leave it as the camera zooms in.
3. *Carried objects*: Grounding DINO + CLIP find what people hold or wear -> `held.json`.
4. *Background - measured*: on the first and middle frame of each shot, MoGe-2 measures depth, SAM 3 cuts the
   picture into pieces and Mask2Former names them; walls, floor, furniture and plants are placed from that ->
   `env_auto.json` (+ `env_auto.jpg`). Your object notes from the clip page are built here too.
5. *Background - built*: with **Settings -> Background set -> Claude**, Claude now writes the set from the frames,
   using step 4's numbers as its ruler, checks it over the film and corrects it (`pipeline\claude_set.py` ->
   `env_claude_ai.json`); otherwise step 4's set is used. `build_env.py` turns the set into meshes, carried objects
   go on the hands, everything gets the colour card -> `env_check.jpg`. The clip is then *ready for review*.

**Review** (clip page): who is who, joint fixes, object notes. Saving changed notes makes the set be measured and
built again at Finish.

**Finish** (about 1-2 minutes): the people are fitted again with your fixes, the set from Prepare is reused unless the
camera or your notes changed (then steps 4-5 run again), Blender renders the white model, the videos are encoded and
the v2.1 package and zip are written to SUBMIT.

## Background set by Claude

The local models measure well but do not understand a room: a curtain cut into strips, a wall seen through a doorway,
a pillar behind a person. Claude, looking at the frames like a person building the set by hand, does. In
**Settings -> Background set** paste an Anthropic API key and choose *Claude* (model: Claude Sonnet 5.5, or Opus 5.5).
For each shot it gets the first / middle / last frame with a pixel grid, the fitted camera, how far the floor is at
each picture row, where every person stands, what the depth model measured, and every object you annotated; it writes
the set (placed by picture box + distance, so it lines up with the film), sees the built set over the film, and
corrects it once ("Check rounds"). Plants and statues keep their measured shapes. About $0.10-0.40 per clip with
Sonnet; the cost of each clip is in `output\<clip>\env_claude_ai.json` (`_claude`). If the key is missing or the call
fails, the local set is used. Frames of the clips are sent to the Claude API.

## How the set is built

Every piece of the background is named by Mask2Former (`models\mask2former_ade`, ADE20K's 150 classes: wall, window,
door, curtain, plant, flower, vase, table, desk, chair, sofa, cabinet, shelf, lamp, car, tree...), on the NVIDIA card
(`pipeline\semseg.py`, about 0.1 s a frame). The name picks how the piece is built:
- walls, floors, ceilings, buildings: from their measured planes, as before;
- furniture: the library model of that type (`pipeline\props.py`), sized from the measurement;
- plants, flowers, curtains, statues, vases, cushions and other irregular things: from their **measured surface**
  (`pipeline\relief.py`: the depth model's points inside the piece's outline become a solid with its real silhouette).
  A box drawn on the clip page as a plant, curtain or statue - or one the model says holds such a thing - is built the
  same way, cut to that thing's pixels. A drawn box that holds furniture becomes that furniture's model.
A large wall or floor stays a wall or floor whatever is painted or stands on it. Without the model, CLIP names the
pieces and the library shapes are used.

Clips are split at hard cuts, and every shot gets its own camera, ground and set. The ground comes from the solved
camera height and angle, checked against the depth model. Walls and objects stand where they meet the ground, so they
are vertical and correctly placed. What doesn't touch the visible ground becomes depth blocks. Colours are taken from
the video (L2 spec). Each clip's work folder has a `blocks_*.jpg` showing what became ground, sky, walls and blocks.

## Working with Claude (hand-built sets, like the client sample)

1. **Prepare** a clip (local models, on the NVIDIA card): people, joints, camera.
2. **Confirm** the people in the app. This writes `output\<clip>\claude_brief.json` and `Work_auto\CLAUDE_HANDOFF.md`
   (the list of all clips), and does not render.
3. Claude writes two files next to the marks in `output\<clip>\`:
   - `hints.json` - what the picture cannot show: camera height / tilt / travel, and per person "seated", "legs hidden",
     distance from the camera. Then `python pipeline\run_pipeline.py <video> --stage fit --name <clip>` fits the people again.
   - `env_claude.json` - the set, object by object (walls, windows, desk, chairs, rocks, trees...), placed in metres or by
     their box in the picture. `python pipeline\build_env.py <clip>` puts it into the scene and writes
     `Work_auto\<clip>\env_check.jpg` (film / overlay / set) to compare.
4. `python pipeline\run_pipeline.py <video> --stage render --name <clip>` renders in Blender and makes the
   submission folder and zip in `SUBMIT\`.
5. Corrections from screenshots go into the same two files, then step 3-4 again.

Running `--stage fit` again replaces the set with the automatic one: run `build_env.py` after it.

## Folders

| Folder | Contents |
|---|---|
| `agent\` | the app (server, job queue, review logic, local AI) |
| `pipeline\` | marks, camera, ground, people fitting, set, Blender build, packaging |
| `web\` | the pages |
| `assets\` | the client's sample character from the November 2 trial Blender file |
| `models\` | SAM 3, SAM 3D Body/MHR, depth, MoGe-2, semantic and object model weights; `trt_cache\` holds the built TensorRT engines |
| `downloads\` | saved PyTorch and TensorRT wheels that `Setup.bat` installs |
| `output\` | joint marks and reviews per clip |
| `data\jobs.db` | job queue (survives restarts) |
| `logs\` | app log |
| `tests\` | `.venv\Scripts\python -m unittest discover -s tests` |

Outputs go to `Work_auto` (previews, logs) and `SUBMIT` (submissions) inside this folder. Change the
folders in `config.json`.

# Blender Agent

Local white-model (白模) production app. It turns the film clips in the `SOURCE` folder into the client's submission
packages (target, white model, comparison video, `.blend`, `meta.jsonl`, zip). Everything runs on this PC.
Every model runs on the NVIDIA card, and only there: the pose models through TensorRT (fp16, plain CUDA as fallback;
the person detector as a batched copy, `models\yolox_x_humanart_batch.onnx` made by `pipeline\det_batch_model.py`, so a
frame and its 4 tiles go to the card in one call - same boxes, about 3x faster),
the depth model through CUDA (`onnxruntime-gpu`), and MotionBERT, MoGe-2, SAM 2.1 and CLIP through the NVIDIA build of
PyTorch. There is no processor build: on a PC without a working NVIDIA card `Setup.bat` and the jobs stop with a message.
`Setup.bat` installs PyTorch and the TensorRT runtime from the wheels in `downloads\torch` and `downloads\trt` when they
are there (no download), and builds the TensorRT engines once (about 8 minutes, kept in `models\trt_cache`).
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

## Objects in the set (like the client sample)

Each clip page has an **Objects in the set** editor. Drag a box on the shot's background, pick the type (door, window,
cabinet, shelf, counter, sink, table, chair, sofa, bed, fridge, car, van, tree, bush, fence, lamp post, stairs, building
and more), set floor or wall and the facing direction, then save. Every object is built from `pipeline/props.py` as a
simple recognisable model (a door has a frame, panel and handle; a car has a body, cabin and wheels), standing where it
touches the floor and sized from the picture. With a DeepSeek key, DeepSeek fills the list in for every shot.

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
| `assets\` | the client's sample character (from `11月2日试稿blend.blend`) |
| `models\` | depth, MotionBERT, MoGe-2, SAM 2.1 and CLIP weights; `trt_cache\` holds the built TensorRT engines |
| `downloads\` | saved PyTorch and TensorRT wheels that `Setup.bat` installs |
| `output\` | joint marks and reviews per clip |
| `data\jobs.db` | job queue (survives restarts) |
| `logs\` | app log |
| `tests\` | `.venv\Scripts\python -m unittest discover -s tests` |

Outputs go to `Work_auto` (previews, logs) and `SUBMIT` (submissions) inside this folder. Change the
folders in `config.json`.

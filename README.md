# Blender Agent

Local white-model (白模) production app. It turns the film clips in the `SOURCE` folder into the client's submission
packages (target, white model, comparison video, `.blend`, `meta.jsonl`, zip). Everything runs on this PC.
The pose and depth models run on the NVIDIA card (CUDA, `onnxruntime-gpu`) when the PC has one. On a PC without an
NVIDIA card, `Setup.bat` installs the CPU build (`requirements-cpu.txt`) and the models run on the processor: same
results, slower. Keep *Clips at once* at 1 there.
A `3D` folder inside `SOURCE` holds the client's samples and is not treated as clips (`source_exclude` in `config.json`).

Every fit also (see `pipeline\quality.py`, `pipeline\lens_auto.py`):
- measures the clip's lens from the picture (MoGe-2) instead of the fixed `lens_mm`; `"lens_auto": false` switches it off,
  and a lens set in `hints.json` or in the review wins;
- leaves out tracks that are both short and unclear (`"junk_filter": false` switches it off);
- in half-body shots (no feet in the picture) puts the floor under the people (`"floor_from_people": false`);
- writes quality numbers and flags into `report.json` and the clip's warnings (`python pipeline\quality.py <clip>`).

## Start

Double-click **`Start Blender Agent.bat`**. The browser opens at http://127.0.0.1:8765.

- The first start runs `Setup.bat` on its own (Python packages and models, one time, needs internet).
- Keep the black window open. Closing it stops the app, and unfinished jobs continue at the next start.
- If the app crashes, the window restarts it after 5 seconds. Details are in `logs\agent.log`.
- Starting it a second time just opens the browser on the running app.

## Work

1. **Copy the videos into `SOURCE\`.** Each one appears on the Clips page as a card by itself; the ✕ on a card takes it
   off the list (the file stays; `data\removed.json` remembers it, and *Removed from the list* puts it back).
   *Name them…* gives videos without a v2.1 package name `title-1`, `title-2`… in file-name order from one English
   title (saved in `data\package_names.json`). Joints, work files and packages go to `output\`, `Work_auto\` and `SUBMIT\`.
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
| `models\` | depth model |
| `output\` | joint marks and reviews per clip |
| `data\jobs.db` | job queue (survives restarts) |
| `logs\` | app log |
| `tests\` | `.venv\Scripts\python -m unittest discover -s tests` |

Outputs go to `Work_auto` (previews, logs) and `SUBMIT` (submissions) inside this folder. Change the
folders in `config.json`.

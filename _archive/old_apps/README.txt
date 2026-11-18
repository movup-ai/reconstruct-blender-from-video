WHITE-MODEL AGENT  -  automatic 白模 L3 rebuild, video -> finished submission zip
==============================================================================

ONE TIME (on every PC that will run it)
  1. Install Blender (5.x) in the normal place (C:\Program Files\Blender Foundation\...).
  2. Double-click setup.bat  (needs Python 3.10+ and internet the first time).
     It installs the packages, the GPU version when an NVIDIA card is found,
     and downloads the depth model (~99 MB) into models\.
     Moving to an NVIDIA PC: copy this folder, delete .venv, run setup.bat again there.

THE APP  (recommended)  -  double-click start_app.bat, it opens in your browser
  1. Choose the clips folder (sub-folders are included) and the output folder for the zips.
  2. Tick the clips (top box = all) and press  ① Scan keypoints.
  3. When the scan finishes the app asks to check them:  ② Check keypoints
       drag a point = move it, right-click = hidden, A = add a missed person, D = delete a wrong one,
       S = swap left/right, Enter = frame OK, Ctrl+S = save.  "Next clip" when a clip is done.
  4. ③ Build zips  (or "3D preview only" first: a quick look without Blender).
  5. Open a clip -> Result tab: play the white-model and comparison videos, open the folder / zip.
  6. ✦ AI edit tab: type what to change ("red man holds an orange rake in his right hand",
     "move the blue person 1 m left", "make the ground lighter"), paste screenshots with Ctrl+V.
     Each change shows a re-rendered frame; Undo is always there. "Make video + zip" updates the zip.
     Runs on a FREE LOCAL model (Qwen3-VL through Ollama) - no API key, nothing leaves the PC.
     One time: install Ollama from ollama.com, then run setup.bat again (or Settings -> Download local model).
     NVIDIA PC: qwen3-vl:8b (answers in seconds). CPU-only PC: qwen3-vl:4b (can take a few minutes).
     Optional: a DeepSeek key in Settings is used automatically when the local model is not running.
  The black window must stay open while the app is used.

COMMAND LINE (same pipeline, no app)
  Drag a video, or a film folder, onto run_pipeline.bat  (run_preview_only.bat = no Blender).
  Clip names:  "人生一世 11月12日(1).mp4" -> 11_12_1

CHARACTER
  Default = the client's L3 sample from the 11_2 .blend: one torso block with the chest band, sphere head
  on the torso, upper/lower arm and leg blocks (no neck, hands or feet). Settings -> Character switches
  back to the earlier mannequin (neck, hands, feet).

SETTINGS  (config.json, created on the first run)
  blender_exe     empty = found automatically
  submit_root / work_root / source_root   folders
  lens_mm         34.9 (35 mm-equivalent focal length used for every clip)
  person_height   1.72
  colours         order of people: 1st red (主角色), 2nd blue (第二角色), then others
  character_style "sample" (client L3) or "classic"
  ai_provider     "local" (Ollama, default), "deepseek" or "anthropic";  keys are kept in this file
  ollama_model    qwen3-vl:8b / qwen3-vl:4b / qwen3-vl:2b
  fit_step, smooth_frames, mesh_step, far_distance   quality / speed

KNOWN LIMITS OF THIS FIRST VERSION
  - Props (orange interactable objects) are not detected automatically - add them with AI edit.
  - Moving (dolly / handheld walking) cameras: pan/tilt is solved, travel is not - the report warns.
  - Hidden body parts are guessed from the pose prior; check preview.jpg for people who turn or crouch.
  - A person who walks straight toward the camera can have legs swinging the wrong way in depth.

MARKS ONLY (old workflow, still works)
  Drag onto run_automark.bat -> output\<film>\<clip>\pose.json, review\, check.html.

OPTIONAL (DeepSeek)
  Put your DeepSeek API key in deepseek_key.txt, then double-click run_describe.bat.
  It writes desc.txt per clip; the pipeline puts it into meta.jsonl.

REVIEW APP (recommended way to work)
  Double-click start_review.bat  ->  http://127.0.0.1:8765 opens in the browser.
  1. "Prepare all clips" runs the local models on every clip (several at once - "Clips at once" on the start page).
  2. Open a clip marked "ready for review": pick each person's role (colour), check camera and set, add objects.
  3. "Confirm & render" builds the Blender scene with the client's sample character, renders and packages it
     into Submit_auto\<film>_<clip>\ + .zip, and opens the next clip to review.
  Clips are split at hard cuts automatically; every shot gets its own camera, ground and set.
  The set is L2-style blocks and walls fitted to the depth model, in the video's own colours.

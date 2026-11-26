# SAM 3 person tracking

Stage 1 uses the approved local `models/sam3/sam3.pt` checkpoint for persistent person masks and IDs. RTMPose
measures COCO-17 joints inside each mask. SAM 3D Body then supplies metric body pose and placement.

## Use it

1. In Settings, check the **SAM 3 checkpoint** location. The default path is `models/sam3/sam3.pt`; the Python
   executable defaults to `downloads/da3-venv/Scripts/python.exe` (the isolated worker environment).
2. Start the clip. Tracking runs when the clip has no marks yet, when the checkpoint or a tracking setting changed
   (`tracking_signature`), or when the masks are missing.

When a clip is tracked again, the previous marks and any files from earlier builds (old joint corrections, reviews,
hints, the old joint editor) are kept under `output/.../<clip>/tracking_backups/<timestamp>/`. If inference fails,
the previous marks stay active.

## Tracking and memory

- The video is split at hard cuts. One SAM 3 session runs for each complete shot; IDs are never reused between
  shots. Frames are read lazily with a small CPU cache.
- The isolated worker uses BF16 autocast, CPU video/state offloading and output-cache eviction. It exits before
  the ONNX pose model is loaded, releasing its GPU memory.
- SAM 3's `person` prompt detects and tracks people, including people entering after the first frame. Detection
  order does not change the persistent ID; the app keeps the SAM ID beside its size-ranked display ID (P1 = biggest).
- Each pose crop sees only the pixels inside its person's mask. Joints outside the mask are downweighted; frames
  without a mask create no detections. At least four frames are needed for the 3D fitter.
- Fitted character visibility follows the observed mask frames. The metric scene observation reuses the tracked
  masks to exclude people from the floor and wall evidence.

Further config options: `sam3_mask_side` (saved-mask maximum dimension, default 640), `sam3_max_objects` (16),
`sam3_probability` (output threshold, 0.35), `sam3_timeout_min` (40) and `track_occlusion_seconds` (tracker
keep-alive, two seconds, at least 30 frames). The saved-mask size does not change SAM 3's 1008-pixel model input.

An RTX 4070 completed an eight-frame test with about 6.2 GiB peak PyTorch allocation and a 32-frame, six-ID test
with about 6.7 GiB. Longer or more crowded shots can use more. There is no fallback tracker: a SAM 3 failure fails
the job and keeps the previous marks. Use one concurrent job on a 12 GB card.

## Outputs and tests

`pose.json` holds `tracker_backend`, `tracking_signature`, each person's `sam3_id`, and a reference to
`sam3_masks.npz` (packed masks, source frame indices and persistent IDs in picture-area coordinates).
`tracking_report.json` records track counts, shots, worker VRAM and the previous tracking backup. No editor files
(`check.html`, `review\`) are written.

Tests cover mask packing, identity preservation when detection order changes, overlap isolation, confidence
masking, absent frames, cache invalidation, partial-person retention and safe replacement of earlier marks.

For a fresh machine: clone `https://github.com/facebookresearch/sam3` into `third_party/sam3` at tested commit
`31cb7e624e4fe95301b1be1d4e809111593d2e45`, then
`uv pip install --python downloads/da3-venv/Scripts/python.exe -r requirements-sam3.txt`. Obtain the checkpoint
from the approved Meta repository. Stage 1 also needs the [SAM 3D Body assets](SAM3D_BODY.md).

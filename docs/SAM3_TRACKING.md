# SAM 3 person tracking

The run uses the approved local `models/sam3/sam3.pt` checkpoint for persistent person masks and IDs. RTMPose
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
- Fitted character visibility follows the observed mask frames, plus gaps of up to half a second
  (`fill_gap_seconds`, 0.5) where the tracker missed a mask for a moment: the pose there is interpolated, so a
  character no longer blinks out. Longer gaps stay hidden (the person is really out of sight). The metric scene
  observation reuses the tracked masks to exclude people from the floor and wall evidence.

## Track cleaning (pipeline/track_clean.py)

The run has no human review, so the raw SAM 3 tracks are cleaned before they become characters. Every decision is
written to `tracking_report.json` (`cleaning`) and the job log (`tracks: ...`).

- **Joined:** one actor whom SAM 3 lost and found again under a new id. Two pieces are joined only when they never
  overlap in time, lie in the same shot, the second starts where a walk could have taken the first in the time
  hidden, the size change fits that time, the clothes colour is compatible, no other piece matches almost as well,
  and **nobody else is in the shot around the hand-over**. Clothes colour alone cannot tell people apart (on the
  SOURCE clips 18 of 70 simulated hand-overs between different people were joined on colour), so with other people
  present the pieces stay separate. A missed join leaves two characters, which is safer than merging two actors.
  Joined pieces share one id in `sam3_masks.npz`.
- **Left out:** fragments with no reliable head, shoulders or hips (a hand or arm reaching in, a blob); weak joints
  that jump around; weak joints that never move while the track is not cut off at the side or bottom of the frame
  (a poster, painting, statue); a second id on someone already tracked; tracks under four frames.

Thresholds were set on the SOURCE clips: every real person there is kept (including a man sitting still and a
seated person's legs at the frame edge), and the eight non-people found there are left out (arm and hand fragments,
two blobs, a poster). Simulated splits of real tracks were rejoined with no wrong join.

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
from the approved Meta repository. The run also needs the [SAM 3D Body assets](SAM3D_BODY.md).

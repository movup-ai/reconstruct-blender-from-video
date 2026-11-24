# SAM 3 person tracking

The app can use the approved local `models/sam3/sam3.pt` checkpoint for persistent person masks and IDs.
SAM 3 does not need SAM 3D Body. The selected RTMPose/RTMO model measures COCO-17 joints, and MotionBERT
continues to supply the 3D pose prior.

## Use it

1. Restart the app after installing this update.
2. In Settings, choose **Person tracking → SAM 3 video tracking**. The default checkpoint path is
   `models/sam3/sam3.pt`; the Python executable defaults to `downloads/da3-venv/Scripts/python.exe`.
3. Prepare/Re-check the desired clip. A backend/checkpoint/settings change regenerates its person tracks.
4. Review the new person roles and joints, then Finish.

Completed deliveries are not automatically regenerated when settings change. Use Prepare/Re-check first.
The previous pose, checked joints, person roles and ID-specific hints are preserved under
`output/.../<clip>/tracking_backups/<timestamp>/`. Person-linked edits are cleared from the active annotations
and review confirmation is reset, because the new IDs can represent different tracks. Camera and environment
settings are retained. If inference fails, the previous marks and edits remain active. Crops refresh against
the current pose file so the review cards show the current identities.

## Tracking and memory

- The video is split at hard cuts. One SAM 3 session runs for each complete shot; IDs are never reused between
  shots. Frames are read lazily with a four-frame CPU cache, without resetting identity at arbitrary chunks.
- The isolated worker uses BF16 autocast, CPU video/state offloading and output-cache eviction. It exits before
  the ONNX pose model is loaded, releasing its GPU allocations.
- SAM 3's `person` prompt detects and tracks people, including people entering after the first frame. Detection
  ordering does not change the persistent ID. The app retains the SAM ID alongside its size-ranked display ID.
- Each pose crop sees only the pixels inside its person's SAM mask. Joints outside that mask are downweighted;
  unobserved masks do not create fabricated detections. Existing geometry-based track stitching, gap filling and
  duplicate suppression are bypassed for SAM tracks. The short/weak pose junk filter does not discard a
  mask-confirmed partial person. At least four frames are needed for the current 3D fitter.
- Fitted character visibility follows the observed mask frames. Missing masks are not filled by borrowing
  another person's pose. Short visible fragments and occluded body joints can still need manual review.
- Metric scene observation and the SAM 2 background builder reuse the tracked person masks to exclude people
  from static geometry. SAM 2 remains responsible for segmenting background objects.

Additional config options are `sam3_mask_side` (saved-mask maximum dimension, default 640),
`sam3_max_objects` (default 16), `sam3_probability` (output threshold, default 0.35), `sam3_timeout_min`
(default 40), and `track_occlusion_seconds` (tracker keep-alive budget, default two seconds, minimum 30 frames).
The saved-mask dimension does not change SAM 3's internal 1008-pixel model input resolution.

The installed RTX 4070 completed an eight-frame test with about 6.2 GiB peak PyTorch allocation and a
32-frame, six-ID test with about 6.7 GiB. These measurements exclude driver/display overhead. Longer or more
crowded shots can use more memory and are not promised to run in real time. There is no silent fallback to
geometry tracking if SAM 3 fails; the job reports the failure and preserves its previous marks.
Use one concurrent job on a 12 GB GPU to leave memory for the other reconstruction stages.

## Outputs and tests

`pose.json` contains `tracker_backend`, `tracking_signature`, each person's `sam3_id`, and a reference to
`sam3_masks.npz`. The latter holds packed masks, source frame indices and persistent IDs in cropped-picture
coordinates. `tracking_report.json` records track counts, shots, worker VRAM allocation and the previous
tracking backup path when a clip was replaced.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -q
```

Tests cover mask packing, identity preservation when detection order changes, overlap isolation, confidence
masking, cropped coordinates, absent frames, cache invalidation, cleanup guards, partial-person retention and
safe annotation backups. GPU smoke artifacts are kept separately under `downloads/sam3-smoke/` and
`downloads/sam3-pipeline-smoke/`.

For a fresh machine, use the existing Python 3.12 CUDA runtime setup described in the README, clone
`https://github.com/facebookresearch/sam3` into `third_party/sam3`, and check out tested commit
`31cb7e624e4fe95301b1be1d4e809111593d2e45`. Install `requirements-sam3.txt` into that runtime with
`uv pip install --python downloads/da3-venv/Scripts/python.exe -r requirements-sam3.txt`. Obtain the checkpoint
from the approved Meta repository. Training code, SAM 3.1 checkpoints and SAM 3D Body are not needed for
this tracking backend.

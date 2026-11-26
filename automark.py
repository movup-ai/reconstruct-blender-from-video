# -*- coding: utf-8 -*-
"""SAM 3 persistent person masks, then RTMPose joints inside each person's mask.

Writes pose.json, sam3_masks.npz and tracking_report.json (no editor files: Stage 1 is fully automatic).
SAM 3 exits before loading RTMPose, keeping GPU allocations separate.

Usage: python automark.py VIDEO_OR_FOLDER [...] [--out DIR] [--force]
                         [--mode performance|balanced|lightweight] [--engine tensorrt|cuda]
"""
import argparse, json, os, sys, time, tempfile
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pipeline"))
from common import read_frames, cfg, pick_device, jload, jsave
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
KP_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"]
def log(*a):
    print(*a, flush=True)


def picture_area(frames):
    """(y0, y1) rows that are not black bars, from a few sample frames"""
    g = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean(axis=1) for f in frames], axis=0)
    rows = np.where(g > 12)[0]
    if len(rows) == 0:
        return 0, frames[0].shape[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    if y1 - y0 < frames[0].shape[0] * 0.3:            # something odd - keep the whole frame
        return 0, frames[0].shape[0]
    return y0, y1


POSE_MAX_BATCH = 32


class Marker:
    """Measure COCO-17 joints in supplied SAM 3 boxes; no person detector is loaded."""

    def __init__(self, mode="performance", device="cuda", engine="tensorrt"):
        from rtmlib import Body, RTMPose
        from rtmlib.tools.file import download_checkpoint
        from common import gpu_dlls, gpu_session

        if device != "cuda":
            raise RuntimeError("RTMPose requires an NVIDIA GPU")
        ort = gpu_dlls()
        ort.set_default_logger_severity(3)
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA runtime unavailable; run Setup.bat")
        model = Body.MODE[mode]
        path = model["pose"]
        path = path if os.path.isfile(path) else download_checkpoint(path)
        pose = RTMPose.__new__(RTMPose)
        pose.model_input_size = model["pose_input_size"]
        pose.mean, pose.std = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
        pose.backend, pose.device, pose.to_openpose, pose.onnx_model = "onnxruntime", device, False, path
        width, height = pose.model_input_size
        pose.session, self.engine = gpu_session(path, engine,
            warm=[np.zeros((n, 3, height, width), np.float32) for n in (1, 4)],
            profile=("input", (1, 3, height, width), (8, 3, height, width), (POSE_MAX_BATCH, 3, height, width)))
        if not {"CUDAExecutionProvider", "TensorrtExecutionProvider"}.intersection(pose.session.get_providers()):
            raise RuntimeError("RTMPose did not load on the NVIDIA GPU")
        pose.session.disable_fallback()
        self.pose = pose
        self.n_pose_calls = self.n_crops = 0
        log("RTMPose ready (%s; %s)" % (mode, self.engine))

    def poses(self, items):
        """items = [(img, [boxes])] (several pictures). RTMPose on all crops at once (batched GPU calls) -
        same maths as rtmlib's RTMPose, which runs the crops one by one.  returns [(kps (n,17,2), scores (n,17))]"""
        from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
        from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
        P = self.pose
        crops, centers, scales = [], [], []
        for img, bxs in items:
            for b in bxs:
                c, s = bbox_xyxy2cs(np.array(list(map(float, b))), padding=1.25)
                crop, s = top_down_affine(P.model_input_size, s, c, img)
                crops.append(crop); centers.append(c); scales.append(s)
        if not crops:
            return [(np.zeros((0, 17, 2)), np.zeros((0, 17))) for _ in items]
        x = np.stack(crops).astype(np.float32)
        x -= np.asarray(P.mean, np.float32); x /= np.asarray(P.std, np.float32)
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
        sess = P.session; iname = sess.get_inputs()[0].name; onames = [o.name for o in sess.get_outputs()]
        sx, sy = [], []
        for i in range(0, len(x), POSE_MAX_BATCH):
            ox, oy = sess.run(onames, {iname: x[i:i + POSE_MAX_BATCH]})[:2]
            sx.append(ox); sy.append(oy); self.n_pose_calls += 1
        self.n_crops += len(x)
        locs, vals = get_simcc_maximum(np.concatenate(sx), np.concatenate(sy))
        kps = locs / 2.0                                  # simcc_split_ratio
        sc_ = np.asarray(scales, float)[:, None, :]; ce = np.asarray(centers, float)[:, None, :]
        kps = kps / np.asarray(P.model_input_size, float) * sc_ + ce - sc_ / 2
        out, i = [], 0
        for img, bxs in items:
            out.append((kps[i:i + len(bxs)], vals[i:i + len(bxs)])); i += len(bxs)
        return out


def process(path, out_root, args, marker):
    name = os.path.splitext(os.path.basename(path))[0]
    rel = os.path.relpath(os.path.dirname(os.path.abspath(path)), args.base) if args.base else ""
    rel = "" if rel.startswith("..") or rel == "." else rel
    out = os.path.join(out_root, rel, name)
    settings = cfg()
    from sam3_tracking import needs_tracking
    previous = os.path.join(out,"pose.json")
    if os.path.exists(previous) and not args.force:
        old = jload(previous)
        if not needs_tracking(old,settings) and os.path.isfile(os.path.join(out,'sam3_masks.npz')):
            log("skip (already done):", path)
            return
    from sam3_tracking import publish
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out, prefix='tracking_pending_') as staged:
        _generate_marks(path, staged, args, marker, settings)
        backup = publish(staged, out)
        if backup:
            log('  previous tracking and person edits backed up: ' + backup)
        log('  updated tracking marks published: ' + out)


def _generate_marks(path, out, args, marker, settings):
    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out, exist_ok=True)
    frames, fps = read_frames(path)
    n = len(frames)
    if n == 0:
        raise RuntimeError('No video frames decoded: '+path)
    H, W = frames[0].shape[:2]
    y0, y1 = picture_area(frames[:: max(1, n // 10)])
    log("%s: %d frames %dx%d, picture rows %d-%d" % (name, n, W, H, y0, y1))
    t0 = time.time()
    from shots import split_shots
    shots = split_shots(frames,y0,y1-y0)
    from sam3_tracking import run_masks, measure_poses
    log('  SAM 3 persistent video tracking ...')
    bundle = run_masks(frames, y0, y1, shots, out, settings, fps)
    # The SAM 3 process exits before loading RTMPose on the GPU.
    marker = marker or Marker(args.mode, args.device, args.engine)
    tr = measure_poses(bundle, frames, y0, y1, marker)
    found = len(tr)
    # one actor hidden longer than SAM 3's memory -> joined again; fragments, posters / statues, duplicate ids and
    # too-short tracks -> dropped (track_clean.py).  Joined pieces share one id in the saved masks too.
    from track_clean import clean
    good, cleaned, report = clean(tr, bundle, frames, y0, y1, shots, fps, args.min_track)
    if report['joined']:
        tmp = os.path.join(out, 'sam3_masks.tmp.npz')
        np.savez_compressed(tmp, **cleaned)
        os.replace(tmp, os.path.join(out, 'sam3_masks.npz'))
    for line in report['join_notes']:
        log('  tracks: ' + line)
    for pid, why in report['dropped'].items():
        log('  tracks: left out %s - %s' % (pid, why))
    jsave(os.path.join(out, 'tracking_report.json'), {
        'backend': 'sam3', 'version': 2, 'mask_records': len(bundle['frames']),
        'identities': found, 'people': len(good), 'peak_vram_mb': float(bundle.get('peak_vram_mb', 0.)),
        'shots': [[int(a), int(b)] for a, b in shots], 'cleaning': report})
    # number people by size (P1 = biggest on screen)
    order = sorted(good, key=lambda t: -np.mean([b[3] - b[1] for k, s, b in good[t].values()]))
    people = []
    for new_id, t in enumerate(order, 1):
        d = good[t]
        people.append({"id": new_id, "frames": {str(f): [[round(float(x), 1), round(float(y), 1), round(float(s), 3)]
                                                         for (x, y), s in zip(k, sc)] for f, (k, sc, b) in sorted(d.items())},
                       "boxes": {str(f): [round(float(v), 1) for v in b] for f, (k, sc, b) in sorted(d.items())}})
        people[-1]['sam3_id'] = int(t)
    data = {"video": os.path.abspath(path), "clip": name, "width": W, "height": H, "fps": fps, "frames": n,
            "picture_area": [0, y0, W, y1 - y0], "keypoint_names": KP_NAMES,
            "model": "RTMPose (%s; %s)" % (args.mode, marker.engine),
            "people": people}
    from sam3_tracking import signature
    data['tracking_version'] = 2
    data['tracker_backend'] = 'sam3'
    data['masks_file'] = 'sam3_masks.npz'
    data['tracking_signature'] = signature(settings)
    with open(os.path.join(out, "pose.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    log("  done: %d people, %.0fs  ->  %s" % (len(people), time.time() - t0, out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--base", default=None, help="input root, used to mirror the folder layout under --out")
    ap.add_argument("--mode", default=None, choices=["lightweight", "balanced", "performance"],
                    help="pose models: performance (largest, most exact), balanced (faster, "
                         "less exact joints), lightweight. default: config.json pose_mode")
    ap.add_argument("--device", default="cuda", choices=["cuda"], help="the models run on the NVIDIA card only")
    ap.add_argument("--min-track", type=int, default=4, help="minimum observed frames per person (at least 4)")
    ap.add_argument("--force", action="store_true", help="redo clips that already have pose.json")
    ap.add_argument("--engine", default=None, choices=["tensorrt", "cuda"],
                    help="GPU engine (tensorrt falls back to cuda). default: config.json pose_engine")
    args = ap.parse_args()
    settings = cfg()
    args.engine = args.engine or str(settings.get("pose_engine", "tensorrt"))
    args.mode = args.mode or str(settings.get("pose_mode", "performance"))
    if args.min_track < 4:
        ap.error("--min-track must be at least 4 for the 3D fitter")
    args.device = pick_device(args.device)
    vids = []
    for p in args.inputs:
        if os.path.isdir(p):
            args.base = args.base or p
            for root, ds, fs in os.walk(p):
                ds.sort()
                for fn in sorted(fs):
                    if fn.lower().endswith(VIDEO_EXT):
                        vids.append(os.path.join(root, fn))
        elif os.path.isfile(p):
            vids.append(p)
        else:
            log("not found:", p)
    if not vids:
        log("no videos found"); return
    log("%d video(s) to process" % len(vids))
    settings = cfg()
    marker = None
    failures = 0
    for i, v in enumerate(vids, 1):
        log("[%d/%d] %s" % (i, len(vids), v))
        try:
            process(v, args.out, args, marker)
        except Exception as e:
            import traceback; traceback.print_exc()
            log("FAILED:", v, e)
            failures += 1
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()

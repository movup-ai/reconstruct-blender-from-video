# -*- coding: utf-8 -*-
"""
automark.py - automatic body keypoint marking for the white-model (白模) clips.

For every video it:
  1. finds the picture area (skips the black bars),
  2. finds every person in every frame (small far-away people too, by also looking at zoomed tiles),
  3. marks 17 body joints per person with RTMPose (runs locally, no internet after the first model download),
  4. keeps the same ID for the same person across frames,
  5. writes  <out>/<clip>/pose.json        all frames, all people (full-frame pixel coordinates)
             <out>/<clip>/review/fNNN.jpg  every Nth frame with red marks, for a quick look
             <out>/<clip>/check.html       open in Chrome/Edge to fix wrong marks, then Save -> checked.json

Usage:  python automark.py VIDEO_OR_FOLDER [more ...] [--out DIR] [--step 8] [--mode balanced|performance|lightweight]
                                                     [--device cpu|cuda] [--no-tiles] [--force]
                                                     [--engine tensorrt|cuda] [--detect-stride auto|1|2]
GPU: TensorRT fp16 engines (built once, ~15 min, cached in models\\trt_cache), CUDA as fallback; RTMPose crops batched.
--detect-stride / --engine default to config.json "detect_stride" / "pose_engine".
"""
import argparse, json, os, sys, shutil, tempfile, time, base64, math
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
KP_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"]
# (a, b, side)  side: R = person's right, L = person's left, C = centre
SKELETON = [(16, 14, "R"), (14, 12, "R"), (15, 13, "L"), (13, 11, "L"), (11, 12, "C"), (5, 11, "L"), (6, 12, "R"),
            (5, 6, "C"), (5, 7, "L"), (7, 9, "L"), (6, 8, "R"), (8, 10, "R"), (0, 1, "C"), (0, 2, "C"),
            (1, 3, "L"), (2, 4, "R")]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}


# ----------------------------------------------------------------------------- helpers
def log(*a):
    print(*a, flush=True)

def imwrite_u(path, img, q=88):
    """cv2.imwrite does not like non-ASCII (Chinese) paths on Windows - encode in memory instead"""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())
    return buf.tobytes()

def open_video(path):
    """open a video even when its path has Chinese characters (copy to a temp ASCII path if needed)"""
    cap = cv2.VideoCapture(path)
    if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
        return cap, None
    tmpdir = tempfile.mkdtemp(prefix="automark_")
    tmp = os.path.join(tmpdir, "clip" + os.path.splitext(path)[1].lower())
    shutil.copyfile(path, tmp)
    cap = cv2.VideoCapture(tmp)
    if not cap.isOpened():
        raise RuntimeError("cannot open video: %s" % path)
    return cap, tmpdir

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

def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def contained(a, b):
    """fraction of box a that lies inside box b"""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area

KEEP_OVERLAPPING = True   # a box mostly inside a bigger one but about as tall (a person walking right behind another)
                          # is kept as a second person; same_person() drops it after RTMPose when it is the same skeleton

def merge_boxes(boxes):
    """join duplicate detections from the full picture and the tiles"""
    boxes = sorted([list(map(float, b)) for b in boxes], key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    keep = []
    for b in boxes:
        dup = False
        for k in keep:
            inside = contained(b, k) > 0.75
            if inside and KEEP_OVERLAPPING and iou(b, k) <= 0.45:
                # a fragment of the same person (head, upper body, a tile cut) is clearly shorter; a second person
                # half hidden behind the first is about as tall and narrower
                if (b[3] - b[1]) >= 0.8 * (k[3] - k[1]) and (b[2] - b[0]) <= 0.7 * (k[2] - k[0]):
                    inside = False
            ov = iou(b, k)
            if KEEP_OVERLAPPING and 0.45 < ov <= 0.75 and min(b[3] - b[1], k[3] - k[1]) < 0.8 * max(b[3] - b[1], k[3] - k[1]) \
                    and contained(b, k) <= 0.9:
                # much the same place but clearly different heights (someone carried / held by another, a child in
                # front of an adult): kept, drop_same_skeletons() removes it when RTMPose sees the same body twice
                ov, inside = 0.0, False
            if ov > 0.45 or inside:
                # same person: grow the kept box a little towards the other one
                k[0], k[1], k[2], k[3] = min(k[0], b[0]), min(k[1], b[1]), max(k[2], b[2]), max(k[3], b[3])
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep

def same_person(p, q):
    """two detections (kps, scores, box) that RTMPose read as the same skeleton (a duplicate box of one person)"""
    (ka, sa, ba), (kb, sb, bb) = p, q
    if contained(ba, bb) < 0.5 and contained(bb, ba) < 0.5:
        return False
    vis = (sa > 0.3) & (sb > 0.3)
    if vis.sum() < 5:
        return iou(ba, bb) > 0.45
    h = max(10.0, ba[3] - ba[1], bb[3] - bb[1])
    return float(np.median(np.linalg.norm(ka[vis] - kb[vis], axis=1))) < 0.06 * h

SPLIT_WIDE = False     # a wide box is probed for two people (see Marker.split_wide); off by default: on one bent /
                       # crouching person RTMPose often finds two half skeletons (junk tracks). --split-wide turns it on
SPLIT_ASPECT = 0.42    # box width / height above this -> probe (one standing person is ~0.25-0.45 wide)

def skeleton_dist(p, q):
    """median distance of the joints both skeletons see, x the taller box height (1.0 when they share < 5 joints)"""
    (ka, sa, ba), (kb, sb, bb) = p, q
    vis = (sa > 0.3) & (sb > 0.3)
    if vis.sum() < 5:
        return 1.0
    return float(np.median(np.linalg.norm(ka[vis] - kb[vis], axis=1))) / max(10.0, ba[3] - ba[1], bb[3] - bb[1])

def core_score(p):
    """mean score of shoulders, hips and the head joints: a real (if half hidden) person, not a few stray joints"""
    return float(np.mean(p[1][[0, 5, 6, 11, 12]]))

def tight_box(p, around):
    """box of a split person: its confident joints (+ a margin), inside the box it was measured in"""
    k, s = p[0], p[1]
    pts = k[s > 0.3] if (s > 0.3).sum() >= 3 else k
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    m = 0.08 * (y1 - y0)
    return [float(max(around[0], x0 - m)), float(around[1]), float(min(around[2], x1 + m)), float(around[3])]

def drop_same_skeletons(people):
    """keep a second (overlapping) box only when RTMPose found a second skeleton in it"""
    order = sorted(range(len(people)), key=lambda i: -float(np.mean(people[i][1])))
    keep = []
    for i in order:
        if not any(same_person(people[i], people[j]) for j in keep):
            keep.append(i)
    return [people[i] for i in sorted(keep)]


# ----------------------------------------------------------------------------- model
POSE_MAX_BATCH = 32                                      # RTMPose crops per GPU call (TensorRT profile max)

class Marker:
    def __init__(self, mode="performance", device="cuda", tiles=True, engine="tensorrt"):
        try:
            from rtmlib import Body, YOLOX, RTMPose
            import onnxruntime as ort
        except ImportError:
            log("ERROR: rtmlib / onnxruntime-gpu is not installed. Run Setup.bat first.")
            sys.exit(1)
        log("loading pose models (first run downloads them, ~100-400 MB) ...")
        backend = "onnxruntime"
        self.engine = "cpu"
        if device == "cuda":
            sys.path.insert(0, os.path.join(HERE, "pipeline"))
            from common import gpu_dlls, gpu_session
            gpu_dlls()                                   # CUDA / cuDNN (and TensorRT) installed by pip
            ort.set_default_logger_severity(3)           # errors only (TensorRT prints pages of tactic warnings)
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                log("ERROR: the NVIDIA (CUDA) runtime is not available - run Setup.bat (installs onnxruntime-gpu).")
                sys.exit(1)
            # build the two rtmlib models without their own sessions, then give them GPU sessions
            # (TensorRT fp16 with engine cache, CUDA as fallback for nodes / whole models TensorRT cannot take)
            from rtmlib.tools.file import download_checkpoint
            cfg_m = Body.MODE[mode]
            det_p, pose_p = cfg_m["det"], cfg_m["pose"]
            det_p = det_p if os.path.exists(det_p) else download_checkpoint(det_p)
            pose_p = pose_p if os.path.exists(pose_p) else download_checkpoint(pose_p)
            det = YOLOX.__new__(YOLOX)
            det.model_input_size, det.mean, det.std = cfg_m["det_input_size"], None, None
            det.backend, det.device, det.det_mode, det.nms_thr, det.score_thr = backend, device, "human", 0.45, 0.7
            det.onnx_model = det_p
            pose = RTMPose.__new__(RTMPose)
            pose.model_input_size, pose.mean, pose.std = cfg_m["pose_input_size"], (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
            pose.backend, pose.device, pose.to_openpose, pose.onnx_model = backend, device, False, pose_p
            pw, ph = cfg_m["pose_input_size"]
            dw, dh = cfg_m["det_input_size"][1], cfg_m["det_input_size"][0]
            det.session, e1 = gpu_session(det_p, engine, warm=[np.zeros((1, 3, dh, dw), np.float32)],
                                          exclude="NonMaxSuppression,NonZero,RoiAlign,TopK")
            pose.session, e2 = gpu_session(pose_p, engine, warm=[np.zeros((n, 3, ph, pw), np.float32) for n in (1, 4)],
                                           profile=("input", (1, 3, ph, pw), (8, 3, ph, pw), (POSE_MAX_BATCH, 3, ph, pw)))
            for s in (det.session, pose.session):
                if "CUDAExecutionProvider" not in s.get_providers() and "TensorrtExecutionProvider" not in s.get_providers():
                    log("ERROR: the pose models did not load on the NVIDIA card (CUDA / cuDNN libraries failed to load).")
                    sys.exit(1)
                s.disable_fallback()                     # a GPU error must stop the job, not quietly use the CPU
            self.engine = "det %s, pose %s" % (e1, e2)
            self.body = None
            self.det, self.pose = det, pose
        else:
            self.body = Body(mode=mode, to_openpose=False, backend=backend, device=device)
            self.det = getattr(self.body, "det_model", None)
            self.pose = getattr(self.body, "pose_model", None)
        log("pose backend: %s (%s, %s; %s)" % (backend, device, mode, self.engine))
        split = self.det is not None and self.pose is not None
        self.tiles = tiles and split
        if tiles and not self.tiles:
            log("note: this rtmlib version does not expose det/pose models separately - tiles switched off")
        # processor without tiles: the same path as with tiles (detector boxes merged and size-filtered, all crops
        # through RTMPose in one batched call, duplicate skeletons dropped) with the full picture only - not rtmlib's
        # Body, which runs the crops one by one and returns boxes drawn round the joints
        self.full_only = (not tiles) and split and device == "cpu"
        self.n_det = self.n_pose_calls = self.n_crops = 0

    def detect(self, img):
        """person boxes [x0, y0, x1, y1] (rtmlib's YOLOX returns N x 4, some versions N x 5 with a score)"""
        self.n_det += 1
        b = np.asarray(self.det(img), dtype=float)
        if b.size == 0:
            return []
        b = np.atleast_2d(b)
        return [list(r[:4]) for r in b if r.shape[0] >= 4]

    def boxes(self, img):
        """merged person boxes of one picture area: full picture + 2 x 2 tiles"""
        h, w = img.shape[:2]
        boxes = []
        try:
            boxes += self.detect(img)
        except Exception as e:
            log("  detector error on full picture:", e)
        # 2 x 2 overlapping tiles -> small far-away people become big enough for the detector
        tw, th = int(w * 0.6), int(h * 0.6)
        for ty in (() if self.full_only else (0, h - th)):
            for tx in (0, w - tw):
                tile = img[ty:ty + th, tx:tx + tw]
                try:
                    for b in self.detect(tile):
                        boxes.append([b[0] + tx, b[1] + ty, b[2] + tx, b[3] + ty])
                except Exception as e:
                    log("  detector error on tile:", e)
        return [b for b in merge_boxes(boxes) if (b[3] - b[1]) > 18 and (b[2] - b[0]) > 6]

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

    def run_many(self, imgs):
        """several picture areas (BGR) -> per picture list of (kps(17,2), scores(17), box)"""
        if not self.tiles and not self.full_only:
            return [self(img) for img in imgs]
        bxs = [self.boxes(img) for img in imgs]
        res = self.poses(list(zip(imgs, bxs)))
        out = [drop_same_skeletons([(k, s, b) for k, s, b in zip(kps, sc, bb)]) for (kps, sc), bb in zip(res, bxs)]
        return self.split_wide(imgs, out) if SPLIT_WIDE else out

    def split_wide(self, imgs, out):
        """one wide box around two people walking side by side / one behind the other (the detector merges them):
        RTMPose in its left and right part; two different, clear skeletons -> two people"""
        jobs = []                                         # (picture, person index, [left box, right box])
        for pi, ppl in enumerate(out):
            for i, (k, s, b) in enumerate(ppl):
                w, h = b[2] - b[0], b[3] - b[1]
                if h > 0.25 * imgs[pi].shape[0] and w > SPLIT_ASPECT * h:
                    sw = 0.5 * w
                    jobs.append((pi, i, [[b[0], b[1], b[0] + sw, b[3]], [b[2] - sw, b[1], b[2], b[3]]]))
        if not jobs:
            return out
        def masked(img, sb):                              # RTMPose widens a narrow box to its 3:4 crop: grey out the rest
            im = img.copy(); im[:, :max(0, int(sb[0]))] = 114; im[:, max(0, int(sb[2])):] = 114
            return im
        items = []
        for pi, i, sub in jobs:
            items += [(masked(imgs[pi], sub[0]), [sub[0]]), (masked(imgs[pi], sub[1]), [sub[1]])]
        res = self.poses(items)
        res = [((res[2 * n][0][0], res[2 * n + 1][0][0]), (res[2 * n][1][0], res[2 * n + 1][1][0])) for n in range(len(jobs))]
        repl = {}
        for (pi, i, sub), (kps, sc) in zip(jobs, res):
            a, b = (kps[0], sc[0], sub[0]), (kps[1], sc[1], sub[1])
            if min(core_score(a), core_score(b)) < 0.55 or skeleton_dist(a, b) < 0.10:
                continue
            ha = (a[0][11] + a[0][12]) / 2; hb_ = (b[0][11] + b[0][12]) / 2
            sa_ = (a[0][5] + a[0][6]) / 2; sb_ = (b[0][5] + b[0][6]) / 2
            h = b[2][3] - b[2][1]
            if hb_[0] - ha[0] < 0.05 * h or sb_[0] - sa_[0] < 0.03 * h:   # the right one must really be further right
                continue
            # two upright people side by side (one wide box around a crouching / lying / arms-out person reads as two
            # half bodies otherwise): both torsos stand up, their shoulders at about the same height
            if any(t_[1] < 0.2 * h or abs(t_[0]) > 0.6 * t_[1] for t_ in (ha - sa_, hb_ - sb_)) or abs(sa_[1] - sb_[1]) > 0.15 * h:
                continue
            others = [q for j, q in enumerate(out[pi]) if j != i]
            if any(skeleton_dist(a, q) < 0.08 or skeleton_dist(b, q) < 0.08 for q in others):
                continue
            repl.setdefault(pi, {})[i] = [(a[0], a[1], tight_box(a, sub[0])), (b[0], b[1], tight_box(b, sub[1]))]
        for pi, rp in repl.items():
            new = []
            for i, p in enumerate(out[pi]):
                new += rp.get(i, [p])
            out[pi] = new
        return out

    def __call__(self, img):
        """img = picture area (BGR). returns list of (kps(17,2), scores(17), box)"""
        if not self.tiles and not self.full_only:
            if self.body is not None:
                kps, sc = self.body(img)
            else:                                        # GPU sessions: what rtmlib's Body does (detector, then pose)
                b = self.detect(img)
                kps, sc = self.poses([(img, b if b else [[0, 0, img.shape[1], img.shape[0]]])])[0]
            out = []
            for k, s in zip(np.asarray(kps).reshape(-1, 17, 2), np.asarray(sc).reshape(-1, 17)):
                x0, y0 = k[s > 0.2].min(0) if (s > 0.2).any() else k.min(0)
                x1, y1 = k[s > 0.2].max(0) if (s > 0.2).any() else k.max(0)
                out.append((k, s, [float(x0), float(y0), float(x1), float(y1)]))
            return out
        return self.run_many([img])[0]


# ----------------------------------------------------------------------------- tracking
TRACK_JUMP = 0.45   # box centre farther than this x box height from where the track was predicted (per frame, more
                    # across gaps) -> never the same person (stops a track jumping to a neighbour when its person is lost)

def track(per_frame, max_gap=15):
    """per_frame: list over frames of [(kps, sc, box)].  returns {track_id: {frame: (kps, sc, box)}}"""
    from scipy.optimize import linear_sum_assignment
    tracks, last = {}, {}          # last[id] = (frame, box, kps)
    vel = {}                       # vel[id] = box centre motion per frame (px) - the box is predicted across gaps
    nid = 1
    def ctr(b):
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    for f, people in enumerate(per_frame):
        active = [t for t, (lf, _, _) in last.items() if f - lf <= max_gap]
        cost = np.full((len(active), len(people)), 1e6)
        for i, t in enumerate(active):
            lf, lb, lk = last[t]
            hb = max(10.0, lb[3] - lb[1])
            v = vel.get(t, np.zeros(2)) * min(f - lf, 8)          # constant velocity, at most 8 frames ahead
            for j, (k, s, b) in enumerate(people):
                cb = ctr(lb) + v - ctr(b)
                dc = np.linalg.norm(cb) / hb
                if dc > TRACK_JUMP * (1 + 0.5 * (f - lf - 1)):
                    continue                                     # the box jumped: another person, not this one
                dh = abs(math.log(max(10.0, b[3] - b[1]) / hb))
                good = (s > 0.3)
                dk = np.median(np.linalg.norm(k[good] - lk[good] - v, axis=1)) / hb if good.sum() >= 4 else dc
                c = dc + 0.7 * dk + 1.5 * dh + 0.05 * (f - lf - 1)
                if c < 1.0:
                    # small extra cost for young tracks: when a duplicate box flickers next to a person, the person keeps
                    # their long track instead of moving to the 2-frame-old one (ties otherwise go either way)
                    cost[i, j] = c + 0.1 * (1 - min(1.0, len(tracks[t]) / 30.0))
        used = set()
        if len(active) and len(people):
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] < 1e5:
                    t = active[i]
                    lf, lb = last[t][0], last[t][1]
                    nv = (ctr(people[j][2]) - ctr(lb)) / max(1, f - lf)
                    vel[t] = nv if t not in vel else 0.6 * vel[t] + 0.4 * nv
                    tracks[t][f] = people[j]; last[t] = (f, people[j][2], people[j][0]); used.add(j)
        for j, p in enumerate(people):
            if j not in used:
                tracks[nid] = {f: p}; last[nid] = (f, p[2], p[0]); nid += 1
    return tracks


GAP_FILL = 15       # a track that is lost for up to this many frames (passing behind someone) is measured in between:
                    # RTMPose in the interpolated box, kept when it finds a skeleton that is not the other person's

def fill_gaps(tr, per_frame, frames, y0, y1, marker, max_fill=GAP_FILL):
    """short gaps inside a track -> RTMPose in the box interpolated between the two ends.  returns frames filled"""
    jobs = {}
    for t, d in tr.items():
        fs = sorted(d)
        for a, b in zip(fs, fs[1:]):
            if 1 < b - a <= max_fill:
                ba, bb = np.array(d[a][2], float), np.array(d[b][2], float)
                for f in range(a + 1, b):
                    w = (f - a) / float(b - a)
                    jobs.setdefault(f, []).append((t, list((1 - w) * ba + w * bb)))
    if not jobs:
        return 0
    n_ok, fl = 0, sorted(jobs)
    for i in range(0, len(fl), FRAME_BATCH):
        grp = fl[i:i + FRAME_BATCH]
        items = [(frames[f][y0:y1], [[b[0], b[1] - y0, b[2], b[3] - y0] for _, b in jobs[f]]) for f in grp]
        for f, (kps, sc) in zip(grp, marker.poses(items)):
            others = list(per_frame[f])
            for (t, b), k, s in zip(jobs[f], kps, sc):
                p = (k + [0, y0], s, [float(v) for v in b])
                if float(np.mean(s)) < 0.35 or any(same_person(p, q) for q in others):
                    continue
                tr[t][f] = p; others.append(p); n_ok += 1
    return n_ok


# ----------------------------------------------------------------------------- drawing
def draw_marks(img, people, scale=1.0, selected=None):
    out = img.copy()
    r = max(3, int(round(4 * scale)))
    for pid, k, s in people:
        for a, b, side in SKELETON:
            if s[a] > 0.2 and s[b] > 0.2:
                col = (0, 0, 255) if side == "R" else ((255, 0, 255) if side == "L" else (0, 0, 200))
                cv2.line(out, tuple(np.int32(k[a])), tuple(np.int32(k[b])), col, max(1, int(scale * 1.5)), cv2.LINE_AA)
        for i in range(17):
            p = tuple(np.int32(k[i]))
            if s[i] > 0.3:
                cv2.circle(out, p, r, (0, 0, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(out, p, r, (0, 0, 255), 1, cv2.LINE_AA)       # low confidence: hollow
        top = k[s > 0.2].min(0) if (s > 0.2).any() else k.min(0)
        cv2.putText(out, "P%d" % pid, (int(top[0]), int(top[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * max(1, scale),
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, "P%d" % pid, (int(top[0]), int(top[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * max(1, scale),
                    (0, 255, 255), 2, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------- frames -> people (with frame skipping)
FRAME_BATCH = 8         # frames whose person crops go to RTMPose together
CUT_DIFF = 20.0         # mean grey change (0-255, 96x54 thumbnail) between neighbouring frames: a cut -> always detect
MOTION_DIFF = 8.0       # stride "auto": picture change above this -> detect (fast camera move, flash; normal shots 0.3-6)
# stride "auto": joint travel between f-1 and f+1 (90th pct of visible joints) above clip(MOTION_REL*height, MIN, MAX) px
# -> the keypoints are not interpolated but measured (RTMPose on the interpolated box).  measured on 112 marked clips:
# travel < 12 px gives a mid-frame interpolation error of ~1-2 px (median)
MOTION_MIN, MOTION_MAX, MOTION_REL = 6.0, 12.0, 0.03
BOX_MOVE, BOX_SCALE = 0.10, 0.10   # box centre travel f-1 -> f+1 (x height) / log height change above this -> full detection
OVERLAP = 0.25                      # two people whose boxes overlap more than this (IoU) -> full detection (who is who)

def interpolate_people(a, b, d, check_motion=True, why=None):
    """frame f from its neighbours f-1 (a) and f+1 (b) (each [(kps, scores, box)]).  returns (how, people):
      ("kp", people)   keypoints, scores and boxes = mean of the two neighbours (linear interpolation at the middle frame)
      ("box", boxes)   people are the same but move (auto): RTMPose must measure them in these (interpolated) boxes
      (None, None)     the person detector must run: a cut / big picture change, a different number of people,
                       overlapping people, a person that cannot be paired, or (auto) a box that moves / scales a lot"""
    why = why if why is not None else {}
    def no(r):
        why[r] = why.get(r, 0) + 1
        return None, None
    if d > CUT_DIFF or (check_motion and d > MOTION_DIFF):
        return no("picture change")
    if len(a) != len(b):
        return no("people count")
    if not a:
        return "kp", []
    for ppl in (a, b):
        for i in range(len(ppl)):
            for j in range(i + 1, len(ppl)):
                if iou(ppl[i][2], ppl[j][2]) > OVERLAP:
                    return no("people overlap")
    from scipy.optimize import linear_sum_assignment
    cost = np.array([[1.0 - iou(pa[2], pb[2]) for pb in b] for pa in a])
    r, c = linear_sum_assignment(cost)
    out, boxes, moving = [], [], False
    for i, j in zip(r, c):
        if cost[i, j] > 0.5:                             # box overlap < 0.5: not clearly the same person
            return no("pairing")
        (ka, sa, ba), (kb, sb, bb) = a[i], b[j]
        ha, hb = max(10.0, ba[3] - ba[1]), max(10.0, bb[3] - bb[1]); h = (ha + hb) / 2
        if check_motion:
            if (np.hypot((ba[0] + ba[2] - bb[0] - bb[2]) / 2, (ba[1] + ba[3] - bb[1] - bb[3]) / 2) > BOX_MOVE * h
                    or abs(math.log(ha / hb)) > BOX_SCALE):
                return no("box motion")
            vis = (sa > 0.3) & (sb > 0.3)
            if vis.sum() >= 3:
                mv = float(np.percentile(np.linalg.norm(ka[vis] - kb[vis], axis=1), 90))
            else:
                mv = float(np.hypot((ba[0] + ba[2] - bb[0] - bb[2]) / 2, (ba[1] + ba[3] - bb[1] - bb[3]) / 2))
            moving |= mv > min(MOTION_MAX, max(MOTION_MIN, MOTION_REL * h))
        box = [(u + v) / 2 for u, v in zip(ba, bb)]
        out.append(((ka + kb) / 2, (sa + sb) / 2, box)); boxes.append(box)
    if moving:
        why["moving: pose only"] = why.get("moving: pose only", 0) + 1
        return "box", boxes
    return "kp", out

def detect_frames(frames, y0, y1, marker, stride="auto"):
    """per frame list of (kps(17,2), scores(17), box) in full-frame pixels, the frames whose keypoints were interpolated,
    and the frames where only RTMPose ran (in interpolated boxes).
    stride 1: model on every frame.  2: person detector + pose on every 2nd frame (+ the last); a frame in between is
    interpolated from its neighbours unless there is a cut or the people do not pair up (then it is detected).
    auto: like 2, but when people move the in-between keypoints are measured: the detector runs on the full picture
    only (1 call instead of 5 with the tiles), its box is joined with the interpolated one, and RTMPose measures the
    keypoints there (the 5 detector calls are ~85% of the time, RTMPose is cheap); fast box motion, overlapping people,
    a cut or a change in the number of people mean a full detection"""
    n = len(frames); res = {}; t0 = time.time(); done = [0]
    def run(idx):
        idx = [f for f in idx if f not in res]
        for i in range(0, len(idx), FRAME_BATCH):
            grp = idx[i:i + FRAME_BATCH]
            for f, r in zip(grp, marker.run_many([frames[f][y0:y1] for f in grp])):
                res[f] = [(k + [0, y0], s, [b[0], b[1] + y0, b[2], b[3] + y0]) for k, s, b in r]
            done[0] += len(grp)
            if done[0] % 40 < len(grp) or done[0] == n:
                log("  detected %d frames (frame %d/%d, people %d)  (%.1fs)" % (done[0], grp[-1] + 1, n, len(res[grp[-1]]),
                                                                              time.time() - t0))
    stride = str(stride).lower()
    if stride == "1" or n < 3:
        run(list(range(n)))
        return [res[f] for f in range(n)], [], []
    th = [cv2.resize(cv2.cvtColor(fr[y0:y1], cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
          for fr in frames]
    diff = [0.0] + [float(np.mean(np.abs(th[f] - th[f - 1]))) for f in range(1, n)]     # change f-1 -> f
    run(list(range(0, n, 2)) + [n - 1])
    todo, interp, posed, why = [], {}, {}, {}
    for f in range(1, n - 1, 2):
        if f in res:
            continue
        how, p = interpolate_people(res[f - 1], res[f + 1], max(diff[f], diff[f + 1]), check_motion=(stride == "auto"),
                                    why=why)
        if how == "kp":
            interp[f] = p
        elif how == "box":
            posed[f] = p
        else:
            todo.append(f)
    if why:
        log("  in-between frames not interpolated: " + ", ".join("%s %d" % kv for kv in sorted(why.items())))
    run(todo)
    pf = sorted(posed)
    for i in range(0, len(pf), FRAME_BATCH):             # moving people: detector on the full picture only (1 of 5 calls)
        grp = pf[i:i + FRAME_BATCH]; items = []          # to place each box exactly, then RTMPose (picture-area coords)
        for f in grp:
            pic = frames[f][y0:y1]
            try:
                dets = marker.detect(pic)
            except Exception as e:
                log("  detector error on full picture:", e); dets = []
            bxs = []
            for b in posed[f]:
                b = [b[0], b[1] - y0, b[2], b[3] - y0]
                best = max(dets, key=lambda d: iou(b, d), default=None)
                if best is not None and iou(b, best) > 0.5:  # grow it like merge_boxes does with the tile boxes
                    b = [min(b[0], best[0]), min(b[1], best[1]), max(b[2], best[2]), max(b[3], best[3])]
                bxs.append([float(v) for v in b])
            items.append((pic, bxs))
        for f, (pic, bxs), (kps, sc) in zip(grp, items, marker.poses(items)):
            res[f] = [(k + [0, y0], s, [b[0], b[1] + y0, b[2], b[3] + y0]) for k, s, b in zip(kps, sc, bxs)]
    res.update(interp)
    return [res[f] for f in range(n)], sorted(interp), pf


# ----------------------------------------------------------------------------- one clip
def process(path, out_root, args, marker):
    name = os.path.splitext(os.path.basename(path))[0]
    rel = os.path.relpath(os.path.dirname(os.path.abspath(path)), args.base) if args.base else ""
    rel = "" if rel.startswith("..") or rel == "." else rel
    out = os.path.join(out_root, rel, name)
    if os.path.exists(os.path.join(out, "pose.json")) and not args.force:
        log("skip (already done):", path)
        return
    os.makedirs(os.path.join(out, "review"), exist_ok=True)
    cap, tmpdir = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    n = len(frames)
    if n == 0:
        log("no frames:", path); return
    H, W = frames[0].shape[:2]
    y0, y1 = picture_area(frames[:: max(1, n // 10)])
    log("%s: %d frames %dx%d, picture rows %d-%d" % (name, n, W, H, y0, y1))
    t0 = time.time()
    per_frame, interp, posed = detect_frames(frames, y0, y1, marker, args.detect_stride)
    log("  detected %d/%d frames, %d pose only, %d interpolated (stride %s)  %.1fs" % (
        n - len(interp) - len(posed), n, len(posed), len(interp), args.detect_stride, time.time() - t0))
    if os.environ.get("AUTOMARK_DUMP"):                 # tests: raw per-frame people before tracking
        import pickle
        pickle.dump(per_frame, open(os.path.join(out, "per_frame.pkl"), "wb"))
    tr = track(per_frame)
    if not getattr(args, "no_gap_fill", False):
        nf = fill_gaps(tr, per_frame, frames, y0, y1, marker)
        if nf:
            log("  %d frames of briefly hidden people measured in their predicted box" % nf)
    # drop short / weak tracks, then number people by size (P1 = biggest on screen)
    good = {}
    for t, d in tr.items():
        if len(d) < args.min_track:
            continue
        msc = np.mean([s.mean() for k, s, b in d.values()])
        if msc < 0.25:
            continue
        good[t] = d
    order = sorted(good, key=lambda t: -np.mean([b[3] - b[1] for k, s, b in good[t].values()]))
    people = []
    for new_id, t in enumerate(order, 1):
        d = good[t]
        people.append({"id": new_id, "frames": {str(f): [[round(float(x), 1), round(float(y), 1), round(float(s), 3)]
                                                         for (x, y), s in zip(k, sc)] for f, (k, sc, b) in sorted(d.items())},
                       "boxes": {str(f): [round(float(v), 1) for v in b] for f, (k, sc, b) in sorted(d.items())}})
    review = sorted(set(list(range(0, n, args.step)) + [n - 1]))
    data = {"video": os.path.abspath(path), "clip": name, "width": W, "height": H, "fps": fps, "frames": n,
            "picture_area": [0, y0, W, y1 - y0], "keypoint_names": KP_NAMES, "review_frames": review,
            "model": "rtmlib Body (%s; %s)" % (args.mode, marker.engine), "detect_stride": str(args.detect_stride),
            "interpolated_frames": interp, "pose_only_frames": posed, "people": people}
    with open(os.path.join(out, "pose.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    # review images + check page
    page_frames = []
    for f in review:
        ppl = []                                    # full-frame coordinates (what pose.json / checked.json use)
        for p in people:
            if str(f) in p["frames"]:
                a = np.array(p["frames"][str(f)])
                ppl.append((p["id"], a[:, :2], a[:, 2]))
        pic = frames[f][y0:y1]
        imwrite_u(os.path.join(out, "review", "f%03d.jpg" % f),
                  draw_marks(pic, [(pid, k - [0, y0], s_) for pid, k, s_ in ppl]))
        jpg = cv2.imencode(".jpg", pic, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        page_frames.append({"f": f, "img": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(),
                            "people": [{"id": pid, "kps": [[round(float(x), 1), round(float(y), 1), round(float(s), 3)]
                                                            for (x, y), s in zip(k, s_)]} for pid, k, s_ in ppl]})
    tpl = open(os.path.join(HERE, "check_template.html"), encoding="utf-8").read()
    page = {"clip": name, "video": os.path.abspath(path), "picture_area": [0, y0, W, y1 - y0], "kp_names": KP_NAMES,
            "skeleton": SKELETON, "frames": page_frames}
    html = tpl.replace("__PAGE_DATA__", json.dumps(page, ensure_ascii=False))
    with open(os.path.join(out, "check.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    log("  done: %d people, %.0fs  ->  %s" % (len(people), time.time() - t0, out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--base", default=None, help="input root, used to mirror the folder layout under --out")
    ap.add_argument("--step", type=int, default=8, help="review every Nth frame")
    ap.add_argument("--mode", default=None, choices=["lightweight", "balanced", "performance"],
                    help="pose models: performance (largest, most exact), balanced (several times faster on a processor, "
                         "less exact joints), lightweight. default: config.json pose_mode")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--no-tiles", action="store_true", help="faster, but may miss small far-away people")
    ap.add_argument("--min-track", type=int, default=8, help="ignore people seen in fewer frames than this")
    ap.add_argument("--force", action="store_true", help="redo clips that already have pose.json")
    ap.add_argument("--no-gap-fill", action="store_true", help="do not measure briefly hidden people in their predicted box")
    ap.add_argument("--no-overlap-split", action="store_true",
                    help="old box merging: a box mostly inside another is always the same person")
    ap.add_argument("--split-wide", action="store_true",
                    help="probe wide boxes for two people side by side (RTMPose on each half; may add junk tracks)")
    ap.add_argument("--detect-stride", default=None, choices=["auto", "1", "2"],
                    help="1 = model on every frame; 2 / auto = every 2nd frame, in-between frames interpolated when safe "
                         "(auto also detects on fast motion). default: config.json detect_stride")
    ap.add_argument("--engine", default=None, choices=["tensorrt", "cuda"],
                    help="GPU engine (tensorrt falls back to cuda). default: config.json pose_engine")
    args = ap.parse_args()
    if args.detect_stride is None or args.engine is None or args.mode is None:
        try:
            sys.path.insert(0, os.path.join(HERE, "pipeline"))
            from common import cfg
            c = cfg()
        except Exception:
            c = {}
        args.detect_stride = args.detect_stride or str(c.get("detect_stride", "auto"))
        args.engine = args.engine or str(c.get("pose_engine", "tensorrt"))
        args.mode = args.mode or str(c.get("pose_mode", "performance"))
    try:                                                 # "cpu" on a PC without an NVIDIA card
        sys.path.insert(0, os.path.join(HERE, "pipeline"))
        from common import pick_device, cfg
        args.device = pick_device(args.device)
        # on the processor the 2 x 2 tiles make every frame ~5 detector runs (about 8 s a frame): off unless
        # config.json has "cpu_tiles": true (tiles find small far-away people)
        if args.device == "cpu" and not args.no_tiles and not cfg().get("cpu_tiles", False):
            args.no_tiles = True
            log("  processor: tiles off (faster; small far-away people may be missed - \"cpu_tiles\": true turns them on)")
    except ImportError:
        pass
    global KEEP_OVERLAPPING, SPLIT_WIDE
    if args.split_wide:
        SPLIT_WIDE = True
    if args.no_overlap_split:
        KEEP_OVERLAPPING = SPLIT_WIDE = False
    if not args.inputs:
        sys.path.insert(0, os.path.join(HERE, "pipeline"))
        from common import cfg as _cfg
        args.inputs = [_cfg()["source_root"]]                # no input given: the Source folder of config.json
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
    marker = Marker(args.mode, args.device, not args.no_tiles, args.engine)
    for i, v in enumerate(vids, 1):
        log("[%d/%d] %s" % (i, len(vids), v))
        try:
            process(v, args.out, args, marker)
        except Exception as e:
            import traceback; traceback.print_exc()
            log("FAILED:", v, e)


if __name__ == "__main__":
    main()

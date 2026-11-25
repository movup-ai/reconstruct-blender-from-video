# -*- coding: utf-8 -*-
"""Objects people carry: a bag or phone in a hand, a backpack on the back -> props that move with the fitted person.

  python pipeline\\held_objects.py <clip> [--force]        detect (NVIDIA card) -> output\\<clip>\\held.json
                                                           and place them into Work_auto\\<clip>\\scene_data.json

Detect (Prepare, once per clip; kept in held.json while pose.json is unchanged):
  Grounding DINO (models\\grounding_dino, IDEA-Research/grounding-dino-base) on every 2nd frame with short prompts:
  "object held in hand." and "backpack." (one long prompt with many words finds nothing).  A box near a detected wrist
  is held in that hand; a backpack box over the torso is worn.  Boxes that are the person themself are left out.
  CLIP (models\\clip) names each object from a short list (bag, phone, cup, umbrella...), which picks its shape;
  "a hand" / "a person" / "an arm" mean a false detection and drop it.  An object found in too few frames is left out.
Place (every fit; numpy only):
  Held: the box centre in the picture, on the camera ray, at the depth of the fitted wrist -> its offset from the wrist
  (smoothed; frames where it was not seen keep the last offset, so it follows the hand).  Its size comes from the box at
  that depth.  It faces the camera.  Worn: a backpack-sized block on the fitted chest's back, turning with the chest.
  Both go into scene_data["props"] ("held": true), hidden (scale 0) when the person is not in the picture.
held.json also tells env_auto.py where these objects are, so the set never builds them as fixed boxes.
hints.json {"held_objects": false} or config "held_objects": false switches it off."""
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import cfg, jload, jsave, output_dir, AGENT, box_iou as iou

MODEL_DIR = os.path.join(AGENT, "models", "grounding_dino")
VERSION = 2
PROMPTS = {"held": ("object held in hand.", 0.35), "worn": ("backpack.", 0.35)}
WRISTS = {9: "wrist_L", 10: "wrist_R"}                   # COCO left / right wrist -> rig joint (the person's own side)
# CLIP word -> (label, shape); shape None = not an object (a false detection)
WORDS = {"a shopping bag": ("bag", "bag"), "a plastic bag": ("bag", "bag"), "a paper bag": ("bag", "bag"),
         "a handbag": ("handbag", "bag"), "a backpack": ("backpack", "bag"), "a suitcase": ("suitcase", "box"),
         "a briefcase": ("briefcase", "box"), "a phone": ("phone", "flat"), "a cup": ("cup", "cyl"),
         "a bottle": ("bottle", "cyl"), "a glass": ("glass", "cyl"), "an umbrella": ("umbrella", "rod"),
         "a gun": ("gun", "box"), "a knife": ("knife", "rod"), "a sword": ("sword", "rod"), "a stick": ("stick", "rod"),
         "a book": ("book", "flat"), "a box": ("box", "box"), "papers": ("papers", "flat"), "a folder": ("folder", "flat"),
         "flowers": ("flowers", "box"), "food": ("food", "box"), "a tray": ("tray", "flat"), "a tool": ("tool", "box"),
         "a ball": ("ball", "cyl"), "a jacket": ("jacket", "bag"), "a laptop": ("laptop", "flat"),
         "a hand": (None, None), "an arm": (None, None), "a person": (None, None), "a face": (None, None),
         "a shirt": (None, None), "a wall": (None, None), "a door": (None, None)}


def available():
    import importlib.util
    return os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")) and importlib.util.find_spec("transformers") is not None


def held_path(name):
    return os.path.join(output_dir(), name, "held.json")


def enabled(c, hints):
    return c.get("held_objects", True) and hints.get("held_objects", True) is not False


# ------------------------------------------------------------------ geometry helpers (pure, tested)
def person_box(kp, thr=0.2):
    g = kp[kp[:, 2] > thr]
    if len(g) < 3:
        return None
    return [float(g[:, 0].min()), float(g[:, 1].min()), float(g[:, 0].max()), float(g[:, 1].max())]


def attach(box, people, kind):
    """which person / joint a detected box belongs to: people = [(track id, keypoints (17, 3))].
    held: the nearest confident wrist inside the box grown by 12 % of that person's height (a bag hangs below the hand).
    worn: the box covers the torso (shoulders-hips) by 30 % of its own area.  -> (track, joint) or None"""
    best = None
    boxes = [person_box(kp) for _, kp in people]
    if any(pb is not None and iou(box, pb) > 0.55 for pb in boxes):   # a person (this one or another), not an object
        return None
    for (tid, kp), pb in zip(people, boxes):
        if pb is None:
            continue
        ph = pb[3] - pb[1]
        if kind == "held":
            m = 0.12 * ph
            for i, j in WRISTS.items():
                if kp[i, 2] < 0.3:
                    continue
                x, y = kp[i, 0], kp[i, 1]
                if box[0] - m <= x <= box[2] + m and box[1] - m <= y <= box[3] + m:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    d = math.hypot(x - cx, y - cy) / max(ph, 1.0)
                    if best is None or d < best[0]:
                        best = (d, tid, j)
        else:
            t = kp[[5, 6, 11, 12]]
            if (t[:, 2] < 0.2).sum() > 1:
                continue
            tb = [t[:, 0].min(), t[:, 1].min(), t[:, 0].max(), t[:, 1].max()]
            ix = max(0.0, min(box[2], tb[2]) - max(box[0], tb[0])); iy = max(0.0, min(box[3], tb[3]) - max(box[1], tb[1]))
            ar = (box[2] - box[0]) * (box[3] - box[1])
            if ar > 0 and ix * iy / ar > 0.3:
                d = 1.0 - ix * iy / ar
                if best is None or d < best[0]:
                    best = (d, tid, "chest")
    return None if best is None else (best[1], best[2])


def groups(dets, seen, min_frames=3, min_share=0.25):
    """dets -> one object per (track, joint, kind) that was found often enough: in min_frames sampled frames and in
    min_share of the sampled frames where that person was seen.  seen: {track: sampled frames with that person}"""
    by = {}
    for d in dets:
        by.setdefault((d["track"], d["joint"], d["kind"]), []).append(d)
    out = []
    for (tid, j, kind), ds in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ds = sorted(ds, key=lambda d: d["f"])
        frames = sorted({d["f"] for d in ds})
        if len(frames) < min_frames or len(frames) < min_share * max(1, seen.get(tid, 1)):
            continue
        best = {}
        for d in ds:                                             # one box per frame (the surest)
            if d["f"] not in best or d["score"] > best[d["f"]]["score"]:
                best[d["f"]] = d
        # Missing detections never imply an object remained held through the gap.
        gap = max(8, 3 * float(np.median(np.diff(frames)))) if len(frames)>1 else 8
        episodes = [[]]
        for f in frames:
            if episodes[-1] and f-episodes[-1][-1] > gap: episodes.append([])
            episodes[-1].append(f)
        stable_episodes = []
        for episode in episodes:
            # Require a sustained label change; one noisy classification does not create a new object.
            starts = [0]; base = best[episode[0]].get('label')
            for i in range(1,len(episode)-min_frames+1):
                labels = [best[f].get('label') for f in episode[i:i+min_frames]]
                if labels[0] != base and len(set(labels)) == 1:
                    starts.append(i); base=labels[0]
            starts.append(len(episode))
            stable_episodes.extend(episode[a:b] for a,b in zip(starts,starts[1:]))
        for episode in stable_episodes:
            if len(episode) < min_frames: continue
            votes = {}
            for f in episode:
                d=best[f]; label=d.get('label') or ('backpack' if kind=='worn' else 'object')
                votes[label]=votes.get(label,0.)+d['score']
            label=max(votes,key=votes.get)
            shape=next((s for w,(l,s) in WORDS.items() if l==label),'box') or 'box'
            out.append({"track": tid, "joint": j, "kind": kind, "label": label, "shape": shape,
                        "span": [episode[0],episode[-1]],
                        "boxes": {str(f): [round(v, 1) for v in best[f]["box"]] for f in episode}})
    return out


def box_mesh(sx, sy, sz):
    x, y, z = sx / 2, sy / 2, sz / 2
    V = [[i * x, j * y, k * z] for k in (-1, 1) for j in (-1, 1) for i in (-1, 1)]
    F = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    return V, F


def cyl_mesh(r, h, n=12):
    V = [[r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z] for z in (-h / 2, h / 2) for i in range(n)]
    F = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))] + [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return V, F


def shape_mesh(shape, w, h):
    """a white-model block for the object, centred on its middle; x = across the picture, z = up, y = depth"""
    if shape == "cyl":
        return cyl_mesh(w / 2, h)
    d = {"flat": max(0.015, 0.08 * min(w, h)), "bag": 0.35 * min(w, h), "rod": min(w, h)}.get(shape, 0.6 * min(w, h))
    return box_mesh(w, float(np.clip(d, 0.015, 0.45)), h)


def euler_xyz(R):
    """Blender XYZ euler (degrees) of a rotation matrix (R = Rz Ry Rx)"""
    ry = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    rx = math.atan2(R[2, 1], R[2, 2]); rz = math.atan2(R[1, 0], R[0, 0])
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def fill(vals, N):
    """per-frame values from the frames that have one: smoothed (running median of 5), the gaps filled with the
    nearest value -> (N, k) array"""
    fs = sorted(vals)
    A = np.array([vals[f] for f in fs], float)
    if len(A) >= 5:
        B = A.copy()
        for i in range(len(A)):
            B[i] = np.median(A[max(0, i - 2):i + 3], axis=0)
        A = B
    out = np.empty((N, A.shape[1]))
    idx = np.searchsorted(fs, np.arange(N))
    for f in range(N):
        i = idx[f]
        if i < len(fs) and fs[i] == f:
            out[f] = A[i]; continue
        lo, hi = i - 1, i
        if lo < 0:
            out[f] = A[hi]
        elif hi >= len(fs):
            out[f] = A[lo]
        else:                                                    # between two seen frames: straight line
            t = (f - fs[lo]) / float(fs[hi] - fs[lo])
            out[f] = A[lo] * (1 - t) + A[hi] * t
    return out


# ------------------------------------------------------------------ place (numpy only)
def place(data, held, y0=0):
    """held.json objects -> scene_data props that follow the fitted people.  -> list of (name, label, frames shown)"""
    import build_env as BE
    from rig import Rig, JOINTS
    data["props"] = [p for p in (data.get("props") or []) if not p.get("held")]
    cam = BE.Cam(data)
    N = data["frames"]
    by_track = {}                                           # one person can be a character in each shot of a clip
    for kind, ch in data["chars"].items():
        by_track.setdefault(ch.get("track_id"), []).append((kind, ch))
    made = []
    for n_, o in enumerate(held.get("objects", [])):
        if o["track"] not in by_track:
            continue
        fs_ = [int(f) for f in o["boxes"] if int(f) < N]          # the character seen on the object's frames
        kind, ch = max(by_track[o["track"]], key=lambda kc: sum(1 for f in fs_ if (kc[1].get("visible") or [True] * N)[f]))
        rig = Rig(ch["height"], ch.get("style", "sample"))
        vis = ch.get("visible") or [True] * N
        Ws = {}
        def W(f):
            if f not in Ws:
                r = ch["root"][f]
                Ws[f] = rig.fk((r[0], r[1], r[2]), r[3], {j: ch["rot"][j][f] for j in JOINTS}, ch["pelvis_z"][f])
            return Ws[f]
        boxes = {int(f): b for f, b in o["boxes"].items() if int(f) < N and vis[int(f)]}
        if not boxes:
            continue
        name = "held_%s_%s_%d" % (kind, o["label"], n_)
        keys = [[0.0] * 9 for _ in range(N)]
        if o["kind"] == "worn":                                  # a block on the back of the chest
            s = ch["height"] / 1.75
            w, h, d = 0.30 * s, 0.42 * s, 0.16 * s
            V, F = box_mesh(w, d, h)
            for f in range(N):
                if not vis[f] or not o.get('span', [max(0,min(boxes)-2), min(N-1,max(boxes)+2)])[0] <= f <= o.get('span', [max(0,min(boxes)-2),min(N-1,max(boxes)+2)])[1]:
                    continue
                Wf = W(f)
                Rc = Wf["chest"][0]
                mid = (Wf["neck"][1] + Wf["pelvis"][1]) / 2
                P = mid + Rc @ np.array([0.0, -(0.11 * s + d / 2), 0.04 * s])
                keys[f] = [float(v) for v in P] + euler_xyz(Rc) + [1.0, 1.0, 1.0]
            shown = sum(1 for f in range(N) if vis[f])
        else:
            j = o["joint"]
            Wp, Hp = cam.W, cam.H
            def cut(b, m=0.012 * max(cam.W, cam.H)):            # which sides the picture edge cuts off (the detector's
                                                                  # boxes stop a few pixels short of the edge)
                return (b[0] <= m, b[1] - y0 <= m, b[2] >= Wp - m, b[3] - y0 >= Hp - m)
            depth = {}
            for f, b in boxes.items():
                C, dvec = cam.ray((b[0] + b[2]) / 2, (b[1] + b[3]) / 2 - y0, f)
                t = float(np.dot(W(f)[j][1] - C, dvec))
                if t > 0.1:
                    depth[f] = t
            if not depth:
                continue
            whole = [f for f in depth if not any(cut(boxes[f]))] or list(depth)   # size from boxes the edge does not cut
            size = [((boxes[f][2] - boxes[f][0]) * depth[f] / cam.f_at(f), (boxes[f][3] - boxes[f][1]) * depth[f] / cam.f_at(f)) for f in whole]
            w, h = [float(np.clip(v, 0.04, 1.2)) for v in np.median(np.array(size), axis=0)]
            off = {}
            for f, t in depth.items():
                b = list(boxes[f]); l_, t_, r_, b_ = cut(b)
                pw, ph = w * cam.f_at(f) / t, h * cam.f_at(f) / t                # its whole size in pixels at that depth
                if l_ and not r_: b[0] = b[2] - pw                   # cut by the edge: from the side that is seen
                if r_ and not l_: b[2] = b[0] + pw
                if t_ and not b_: b[1] = b[3] - ph
                if b_ and not t_: b[3] = b[1] + ph
                C, dvec = cam.ray((b[0] + b[2]) / 2, (b[1] + b[3]) / 2 - y0, f)
                t2 = float(np.dot(W(f)[j][1] - C, dvec))
                off[f] = list(C + t2 * dvec - W(f)[j][1])
            OFF = fill(off, N)
            OFF *= np.minimum(1.0, 0.7 / np.maximum(np.linalg.norm(OFF, axis=1, keepdims=True), 1e-9))   # within reach
            V, F = shape_mesh(o["shape"], w, h)
            seen = sorted(off)
            lo, hi = o.get('span', [max(0, seen[0] - 2), min(N - 1, seen[-1] + 2)])
            shown = 0
            for f in range(N):
                if not vis[f] or not lo <= f <= hi:
                    continue
                R, C = BE.cam_mats(data["cam"][f])
                view = R @ np.array([0.0, 0.0, -1.0])
                yaw = math.degrees(math.atan2(view[1], view[0])) - 90.0         # its front (-y) towards the camera
                P = W(f)[j][1] + OFF[f]
                keys[f] = [float(v) for v in P] + [0.0, 0.0, yaw, 1.0, 1.0, 1.0]
                shown += 1
        for f in range(N):                                       # hidden frames: at the first shown place, scale 0
            if keys[f][6] == 0:
                ref = next((k for k in keys if k[6]), None)
                if ref:
                    keys[f] = ref[:6] + [0.0, 0.0, 0.0]
        data["props"].append({"name": name, "v": [[round(x, 4) for x in v] for v in V], "f": F, "colour": "dark",
                              "held": True, "label": o["label"], "track_id": o["track"],
                              "keys": [[round(x, 4) for x in k] for k in keys]})
        made.append((name, o["label"], shown))
    return made


def annotated(video, c, ci, pose, work):
    """objects annotated on the clip page (Objects step) as held by a person ("bag in his left hand", "backpack") ->
    held.json objects ("manual": bounded to the annotated shot). The box is the one drawn on that
    frame; the person is the one the box touches (nearest when none), the hand the description names, else the
    nearer wrist; a backpack / "on his back" / "wearing" goes on the back."""
    import describe
    from run_pipeline import marks_dir
    from env_auto import drawn_frames
    p = os.path.join(marks_dir(video, c), "env_ai.json")
    if not os.path.exists(p):
        return []
    env = jload(p)
    camj = os.path.join(work, "camera.json")
    shots = (jload(camj).get("shots") if os.path.exists(camj) else None) or [{"shot": 1, "frames": [0, ci["frames"] - 1]}]
    y0 = ci.get("y0", 0)
    out = []
    for sid, tags in drawn_frames(env, shots).items():
        for tag, f in tags.items():
            for o in ((env.get("shots") or {}).get(tag) or {}).get("objects", []):
                d = describe.parse(o.get("what"), o.get("type"))
                b = o.get("bbox")
                if d["mode"] != "held" or not b or len(b) != 4:
                    continue
                box = [float(b[0]), float(b[1]) + y0, float(b[2]), float(b[3]) + y0]       # full frame
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                best = None
                for t in pose.get("people", []):
                    kp = None
                    for dd in (0, 1, -1, 2, -2, 3, -3):
                        if str(f + dd) in t["frames"]:
                            kp = np.array(t["frames"][str(f + dd)], float); break
                    pb = None if kp is None else person_box(kp)
                    if pb is None:
                        continue
                    gap = max(0.0, max(pb[0] - box[2], box[0] - pb[2])) + max(0.0, max(pb[1] - box[3], box[1] - pb[3]))
                    if best is None or gap < best[0]:
                        best = (gap, t["id"], kp)
                if best is None:
                    print("  annotation %r: no person near it in frame %d - left out" % (o.get("what"), f))
                    continue
                _, tid, kp = best
                if d.get("worn"):
                    joint = "chest"
                elif d.get("side") in ("L", "R"):
                    joint = "wrist_" + d["side"]
                else:
                    joint = min(WRISTS.values(), key=lambda j: math.hypot(*(kp[9 if j == "wrist_L" else 10, :2] - [cx, cy])))
                label = d["label"] or "object"
                shape = next((s for w, (l, s) in WORDS.items() if l == label and s), None) or ("bag" if d.get("worn") else "box")
                out.append({"track": tid, "joint": joint, "kind": "worn" if d.get("worn") else "held", "label": label,
                            "shape": shape, "boxes": {str(f): [round(v, 1) for v in box]}, "manual": True,
                            "span": next((s['frames'] for s in shots if s['frames'][0] <= f <= s['frames'][1]), [f,f]),
                            "what": o.get("what", "")})
    return out


def with_annotated(held, ann):
    """held.json + the annotated ones: an annotation on a hand where one was found renames it (the person's words
    win) within its annotated span; otherwise it is added"""
    held = dict(held or {"objects": [], "step": 2})
    objs = [dict(o) for o in held.get("objects", [])]
    for a in ann:
        same = next((o for o in objs if o["track"] == a["track"] and o["joint"] == a["joint"] and o["kind"] == a["kind"] and o['label'] == a['label'] and o.get('span') == a.get('span')), None)
        if same is not None:
            same.update(label=a["label"], shape=a["shape"], manual=True)
            same["boxes"] = dict(same["boxes"], **a["boxes"])
        else:
            objs.append(a)
    held["objects"] = objs
    return held


def check_image(data, held, ci, out, n=4):
    """Work_auto\\<clip>\\held_check.jpg: for a few frames with a held object, the film with the detected box (yellow)
    and the placed object drawn over it"""
    import cv2
    import build_env as BE
    from common import read_frames_at, imwrite_u
    props = [p for p in data.get("props", []) if p.get("held")]
    shown = sorted({f for p in props for f, k in enumerate(p["keys"]) if k[6]})
    if not shown:
        return None
    frs = [shown[int(i)] for i in np.linspace(0, len(shown) - 1, min(n, len(shown)))]
    pics = read_frames_at(ci["video"], frs)
    y0, Hp = ci.get("y0", 0), ci["Hpic"]
    cam = BE.Cam(data); tiles = []
    for f in frs:
        src = pics[f][y0:y0 + Hp]
        mf = [m for m in (BE.prop_at(p, f) for p in props) if m]
        im = BE.overlay(mf, cam, src, f, data["palette"], 0.7)
        for o in held.get("objects", []):
            b = o["boxes"].get(str(f))
            if b:
                cv2.rectangle(im, (int(b[0]), int(b[1] - y0)), (int(b[2]), int(b[3] - y0)), (0, 230, 255), 3)
                cv2.putText(im, o["label"], (int(b[0]), int(b[1] - y0) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2)
        cv2.putText(im, "f%d" % f, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        tiles.append(cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    imwrite_u(out, sheet)
    return out


def boxes_at(held, f, near=3):
    """the held objects' picture boxes (full frame) around frame f, for env_auto's people mask"""
    out = []
    for o in (held or {}).get("objects", []):
        bx = o["boxes"]
        for d in sorted(range(-near, near + 1), key=abs):
            if str(f + d) in bx:
                out.append(bx[str(f + d)]); break
    return out


# ------------------------------------------------------------------ detect (NVIDIA card)
class Detector:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from common import torch_device
        self.torch, self.dev = torch, torch_device()
        self.proc = AutoProcessor.from_pretrained(MODEL_DIR)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_DIR, dtype=torch.float16).to(self.dev).eval()
        self.namer = None
        try:
            import namer
            if namer.available():
                self.namer = namer.Namer()
                import open_clip
                tok = open_clip.get_tokenizer(namer.ARCH)
                with torch.inference_mode():
                    t = self.namer.model.encode_text(tok(["a photo of %s" % w for w in WORDS]).to(self.dev))
                self.text = t / t.norm(dim=-1, keepdim=True)
        except Exception as e:
            print("  CLIP not used for the object names (%s)" % e)
            self.namer = None

    def detect(self, ims, prompt, thr):
        """ims: RGB PIL pictures -> per picture [(box, score)]"""
        torch = self.torch
        inp = self.proc(images=ims, text=[prompt] * len(ims), return_tensors="pt").to(self.dev)
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.inference_mode():
            out = self.model(**inp)
        res = self.proc.post_process_grounded_object_detection(out, inp["input_ids"], threshold=thr, text_threshold=thr,
                                                               target_sizes=[im.size[::-1] for im in ims])
        return [[([float(v) for v in b], float(s)) for b, s in zip(r["boxes"], r["scores"])] for r in res]

    def names(self, bgr, boxes):
        """CLIP word for each box -> [(word, probability)]"""
        if not self.namer or not boxes:
            return [(None, 0.0)] * len(boxes)
        from PIL import Image
        import cv2
        torch = self.torch
        H, W = bgr.shape[:2]
        crops = []
        for b in boxes:
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            px, py = int(0.1 * (x1 - x0)) + 4, int(0.1 * (y1 - y0)) + 4
            c = bgr[max(0, y0 - py):min(H, y1 + py), max(0, x0 - px):min(W, x1 + px)]
            crops.append(self.namer.prep(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))))
        with torch.inference_mode():
            f = self.namer.model.encode_image(torch.stack(crops).to(self.dev))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ self.text.T).softmax(-1).cpu().numpy()
        words = list(WORDS)
        return [(words[int(i)], float(r[int(i)])) for r, i in zip(p, p.argmax(1))]


def detect(video, pose, N, step=2, batch=4, log=print):
    """Grounding DINO + CLIP over every <step>th frame -> held.json content"""
    import cv2
    from PIL import Image
    t0 = time.time()
    tracks = [(p["id"], p["frames"]) for p in pose.get("people", [])]
    want = [f for f in range(0, N, step) if any(str(f) in fr for _, fr in tracks)]
    det = Detector()
    dets, seen = [], {}
    cap = cv2.VideoCapture(video)
    pics = {}
    for f in range(N):                                          # read in order (seeking is slow and not exact)
        ok, img = cap.read()
        if not ok:
            break
        if f in want:
            pics[f] = img
    cap.release()
    fs = [f for f in want if f in pics]
    for i in range(0, len(fs), batch):
        chunk = fs[i:i + batch]
        ims = [Image.fromarray(cv2.cvtColor(pics[f], cv2.COLOR_BGR2RGB)) for f in chunk]
        for kind, (prompt, thr) in PROMPTS.items():
            for f, found in zip(chunk, det.detect(ims, prompt, thr)):
                people = [(tid, np.array(fr[str(f)], float)) for tid, fr in tracks if str(f) in fr]
                if kind == "held":
                    for tid, _ in people:
                        seen.setdefault(tid, set()).add(f)
                cand = []
                for box, score in found:
                    a = attach(box, people, kind)
                    if a:
                        cand.append((box, score, a))
                for (box, score, a), (word, p) in zip(cand, det.names(pics[f], [c[0] for c in cand])):
                    label, shape = WORDS.get(word, (None, "box")) if word else ("backpack" if kind == "worn" else None, "box")
                    if word and label is None:                       # a hand / a person: not an object
                        continue
                    dets.append({"f": f, "box": box, "score": round(score, 3), "track": a[0], "joint": a[1], "kind": kind,
                                 "label": label, "word": word, "p": round(p, 3)})
    objs = groups(dets, {t: len(v) for t, v in seen.items()})
    log("  held objects: %d frame(s) looked at in %.0fs -> %s" % (len(fs), time.time() - t0,
        ", ".join("%s (%s, track %s)" % (o["label"], o["joint"], o["track"]) for o in objs) or "none"))
    return {"version": VERSION, "step": step, "frames": N, "objects": objs, "dets": len(dets)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--force", action="store_true", help="detect again even when held.json is newer than pose.json")
    ap.add_argument("--place-only", action="store_true")
    ap.add_argument("--detect-only", action="store_true", help="write held.json only (the pipeline places them after the set)")
    a = ap.parse_args()
    c = cfg()
    from run_pipeline import marks_dir
    work = os.path.join(c["work_root"], a.name)
    ci = jload(os.path.join(work, "clip_info.json"))
    pose_p = os.path.join(marks_dir(ci["video"], c), "pose.json")
    hp = held_path(a.name)
    if not a.place_only:
        fresh = (os.path.exists(hp) and os.path.getmtime(hp) >= os.path.getmtime(pose_p)
                 and (jload(hp).get("version") == VERSION))
        if a.force or not fresh:
            held = detect(ci["video"], jload(pose_p), ci["frames"])
            os.makedirs(os.path.dirname(hp), exist_ok=True)
            jsave(hp, held)
    if not a.detect_only:
        dp = os.path.join(work, "scene_data.json")
        data = jload(dp)
        held = with_annotated(jload(hp) if os.path.exists(hp) else None, annotated(ci["video"], c, ci, jload(pose_p), work))
        made = place(data, held, ci.get("y0", 0))
        jsave(dp, data)
        print("held objects placed:", made)
        try:
            check_image(data, held, ci, os.path.join(work, "held_check.jpg"))
        except Exception as e:                                   # the check picture never stops the clip
            print("held_check.jpg not made (%s)" % e)


if __name__ == "__main__":
    main()

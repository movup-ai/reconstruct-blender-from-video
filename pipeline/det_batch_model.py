# -*- coding: utf-8 -*-
"""Make a batched copy of rtmlib's YOLOX person detector, so the full picture and its 4 tiles go to the NVIDIA card
in ONE call instead of five (TensorRT then runs them together).

rtmlib's yolox_x ONNX has its batch fixed to 1 and its NMS built in.  The copy:
  - ends before the NMS: outputs "boxes" (B, 8400, 4: x0 y0 x1 y1 in the 640 x 640 input) and "scores" (B, 8400, 80);
  - takes any batch: the 1 in its Reshape shapes becomes 0 ("as the input").
person_dets() does the rest exactly as the model + rtmlib did: top 5000 candidates by best class score, NMS per class
(IoU > 0.65 suppressed, at most 200 a class), the best 100 of all classes, then rtmlib's "human" mode keeps every one
scoring above 0.3 (any class: this human-art model's other classes are people in other styles).

  python pipeline\\det_batch_model.py          (writes models\\yolox_x_humanart_batch.onnx; Setup.bat runs it)"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "models", "yolox_x_humanart_batch.onnx")
SRC_GLOB = os.path.join(os.path.expanduser("~"), ".cache", "rtmlib", "hub", "checkpoints", "yolox_x_8xb8-300e_humanart-*.onnx")
BOXES, SCORES = "1930", "1932"                 # the source graph's tensors right before its NMS (checked in convert())
PRE_TOP_K, IOU_THR, MAX_PER_CLASS, KEEP_TOP_K = 5000, 0.65, 200, 100
KEEP_SCORE = 0.3                               # rtmlib YOLOX.postprocess, model with NMS: dets scoring above 0.3


def source_model():
    m = glob.glob(SRC_GLOB)
    return m[0] if m else None


def convert(src=None, out=OUT):
    import onnx
    from onnx import numpy_helper, utils
    src = src or source_model()
    if not src:
        raise SystemExit("the YOLOX model is not downloaded yet (run Setup.bat)")
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    names = {o for n in g.node for o in n.output}
    if BOXES not in names or SCORES not in names or not any(n.op_type == "NonMaxSuppression" for n in g.node):
        raise SystemExit("this YOLOX model is not the one this converter knows (%s)" % os.path.basename(src))
    for n in g.node:                                  # Reshape shapes [1, ...] -> [0, ...]: the batch of the input
        if n.op_type == "Reshape" and n.input[1] in inits:
            a = numpy_helper.to_array(inits[n.input[1]])
            if a.ndim == 1 and a.size > 1 and a[0] == 1:
                b = a.copy(); b[0] = 0
                inits[n.input[1]].CopyFrom(numpy_helper.from_array(b, n.input[1]))
    g.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    for o in list(g.output):
        g.output.remove(o)
    m = utils.Extractor(m).extract_model([g.input[0].name], [BOXES, SCORES])
    m.graph.output[0].name = BOXES; m.graph.output[1].name = SCORES
    for o, nm in zip(m.graph.output, ("boxes", "scores")):     # friendly names
        old = o.name
        for n in m.graph.node:
            n.output[:] = [nm if x == old else x for x in n.output]
        o.name = nm
    del m.graph.value_info[:]                         # inferred with batch 1: dropped, ORT / TensorRT infer them again
    for v in list(m.graph.input) + list(m.graph.output):
        d = v.type.tensor_type.shape.dim[0]
        d.Clear(); d.dim_param = "batch"
    onnx.checker.check_model(m)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    onnx.save(m, tmp)
    os.replace(tmp, out)
    return out


def nms(boxes, scores, thr):
    """greedy NMS, as ONNX NonMaxSuppression (corner boxes, suppress IoU > thr): indices kept, best first"""
    order = np.argsort(-scores, kind="stable")
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)
    keep = []
    while order.size and len(keep) < MAX_PER_CLASS:
        i = order[0]; keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]))
        h = np.maximum(0.0, np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]))
        inter = w * h
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-12)
        order = rest[iou <= thr]
    return np.asarray(keep, int)


def person_dets(boxes, scores, keep_score=KEEP_SCORE):
    """one picture's model outputs -> the boxes (n, 4) in the 640 input, best first, exactly as the original model +
    rtmlib's postprocess give them.  Only candidates above keep_score are needed: a box is suppressed only by a better
    one, and the best 100 kept are all above keep_score whenever anything below it would be cut."""
    top = np.argsort(-scores.max(1), kind="stable")[:PRE_TOP_K]       # the model's TopK over the best class score
    b_all, s_all = boxes[top], scores[top]
    out_b, out_s = [], []
    for c in np.nonzero((s_all > keep_score).any(0))[0]:               # NMS per class (the model's NonMaxSuppression)
        ok = s_all[:, c] > keep_score
        b, s = b_all[ok], s_all[ok, c]
        keep = nms(b, s, IOU_THR)
        out_b.append(b[keep]); out_s.append(s[keep])
    if not out_b:
        return np.zeros((0, 4), np.float32)
    b, s = np.concatenate(out_b), np.concatenate(out_s)
    order = np.argsort(-s, kind="stable")[:KEEP_TOP_K]                 # the model's final TopK 100
    return b[order]


if __name__ == "__main__":
    if os.path.exists(OUT) and "--force" not in sys.argv:
        print("batched person detector already there"); sys.exit(0)
    print("batched person detector:", convert())

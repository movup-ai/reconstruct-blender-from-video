# -*- coding: utf-8 -*-
"""Speed test of the local models on one frame (no clip processing, writes nothing but bench_result.json).
   .venv\\Scripts\\python bench_models.py [video]"""
import os, sys, time, json, shutil, tempfile
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
vid = sys.argv[1] if len(sys.argv) > 1 else None
if vid is None:
    for r, _, fs in os.walk(r"F:\Blender task\Source"):
        m = [f for f in sorted(fs) if f.lower().endswith(".mp4")]
        if m:
            vid = os.path.join(r, m[0]); break
tmpd = tempfile.mkdtemp(prefix="bench_"); tmp = os.path.join(tmpd, "v.mp4"); shutil.copyfile(vid, tmp)
cap = cv2.VideoCapture(tmp); cap.set(cv2.CAP_PROP_POS_FRAMES, 10); ok, img = cap.read(); cap.release()
shutil.rmtree(tmpd, ignore_errors=True)
assert ok, "could not read a frame"
print("frame", img.shape, "from", vid)

from rtmlib import Body
res = {}
backends = ["onnxruntime"]
try:
    import openvino  # noqa
    backends.append("openvino")
except ImportError:
    pass
for be in backends:
    for mode in ("balanced", "lightweight"):
        try:
            t = time.time(); body = Body(mode=mode, to_openpose=False, backend=be, device="cpu"); tl = time.time() - t
            body(img)                                   # warm-up
            t = time.time()
            for _ in range(5):
                k, s = body(img)
            dt = (time.time() - t) / 5
            res["%s/%s" % (be, mode)] = {"sec_per_frame": round(dt, 3), "people": int(len(k)), "load_s": round(tl, 1)}
            print("%-12s %-11s %.3f s/frame  people=%d" % (be, mode, dt, len(k)))
        except Exception as e:
            res["%s/%s" % (be, mode)] = {"error": str(e)[:200]}
            print(be, mode, "ERROR", e)

sys.path.insert(0, os.path.join(HERE, "pipeline"))
try:
    import onnxruntime as ort
    from scene_depth import MODEL_PATH
    for prov in [p for p in ("OpenVINOExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]:
        so = ort.SessionOptions()
        sess = ort.InferenceSession(MODEL_PATH, so, providers=[prov])
        inp = sess.get_inputs()[0]
        x = np.random.rand(1, 3, 518, 518).astype(np.float32)
        sess.run(None, {inp.name: x}); t = time.time(); sess.run(None, {inp.name: x}); dt = time.time() - t
        res["depth/" + prov] = round(dt, 3); print("depth %-26s %.3f s" % (prov, dt))
except Exception as e:
    res["depth"] = {"error": str(e)[:200]}; print("depth ERROR", e)
json.dump(res, open(os.path.join(HERE, "bench_result.json"), "w"), indent=2)

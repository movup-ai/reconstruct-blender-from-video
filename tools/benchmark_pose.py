"""Compare warm GPU inference for RTMPose and RTMO on identical video frames."""
import argparse
import json
import os
import sys
import time
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pipeline")]
from automark import Marker, picture_area
from rtmo_marker import RTMOMarker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--rtmo", default=os.path.join(ROOT, "models", "rtmo", "model.onnx"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-tiles", action="store_true")
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    frames = []
    try:
        for index in np.linspace(0, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1), max(1, args.frames), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("No readable video frames")
    y0, y1 = picture_area(frames)
    frames = [f[y0:y1] for f in frames]
    rows = {}
    for name, factory in (("rtmpose", lambda: Marker(engine="cuda", tiles=not args.no_tiles)),
                          ("rtmo", lambda: RTMOMarker(args.rtmo, engine="cuda", tiles=not args.no_tiles))):
        started = time.perf_counter()
        marker = factory()
        marker(frames[0])
        startup = time.perf_counter() - started
        started = time.perf_counter()
        results = marker.run_many(frames)
        elapsed = time.perf_counter() - started
        rows[name] = {"startup_seconds": startup, "inference_seconds": elapsed, "frames": len(frames),
                      "fps": len(frames) / elapsed, "people_per_frame": [len(r) for r in results],
                      "confident_joints": [sum(int((s > 0.5).sum()) for _, s, _ in r) for r in results]}
        del marker
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump({"video": os.path.abspath(args.video), "tiles": not args.no_tiles, "models": rows}, stream, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
describe.py - optional: ask DeepSeek for a one-line Chinese description of each clip (for meta.jsonl source_video_desc).
Key: put your DeepSeek API key in deepseek_key.txt (next to this file) or in the DEEPSEEK_API_KEY environment variable.
Writes desc.txt next to each clip's pose.json.  Only 3 small frames per clip are sent.
Usage: python describe.py [OUTPUT_FOLDER] [--model deepseek-flash] [--force]
"""
import argparse, base64, json, os, sys, shutil, tempfile
import cv2, requests
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = "这是一段电影片段中的3帧画面（按时间顺序）。请用一句简洁的中文描述画面内容：人物、动作和场景。不超过40个字，只输出这句话。"

def key():
    k = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    p = os.path.join(HERE, "deepseek_key.txt")
    if not k and os.path.exists(p):
        k = open(p, encoding="utf-8").read().strip()
    if not k:
        sys.exit("No DeepSeek key: put it in deepseek_key.txt next to describe.py")
    return k

def frames(video, n=3):
    cap = cv2.VideoCapture(video); tmp = None
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0:
        tmp = tempfile.mkdtemp(); t = os.path.join(tmp, "clip.mp4"); shutil.copyfile(video, t); cap = cv2.VideoCapture(t)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n)); ok, f = cap.read()
        if ok:
            f = cv2.resize(f, (960, int(960 * f.shape[0] / f.shape[1])))
            out.append("data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode())
    cap.release()
    if tmp: shutil.rmtree(tmp, ignore_errors=True)
    return out

def ask(k, model, imgs):
    content = [{"type": "text", "text": PROMPT}] + [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
    r = requests.post("https://api.deepseek.com/chat/completions", timeout=120,
                      headers={"Authorization": "Bearer " + k, "Content-Type": "application/json"},
                      json={"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 200})
    if r.status_code != 200:
        raise RuntimeError("%s %s" % (r.status_code, r.text[:300]))
    return r.json()["choices"][0]["message"]["content"].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=os.path.join(HERE, "output"))
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); k = key()
    for root, ds, fs in os.walk(a.folder):
        if "pose.json" not in fs: continue
        out = os.path.join(root, "desc.txt")
        if os.path.exists(out) and not a.force: continue
        video = json.load(open(os.path.join(root, "pose.json"), encoding="utf-8"))["video"]
        imgs = frames(video)
        try:
            try:
                d = ask(k, a.model, imgs)
            except RuntimeError as e:
                if a.model == "deepseek-flash" and ("model" in str(e).lower() or "400" in str(e)):
                    d = ask(k, "deepseek-v4-flash-vision-exp", imgs)
                else:
                    raise
        except Exception as e:
            print("FAILED", root, e); continue
        open(out, "w", encoding="utf-8").write(d + "\n"); print(os.path.basename(root), "->", d)

if __name__ == "__main__":
    main()

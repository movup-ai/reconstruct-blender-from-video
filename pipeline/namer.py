# -*- coding: utf-8 -*-
"""Names the pieces of a set from the picture: CLIP (open_clip, models\\clip\\) compares each SAM cut-out with a list
of words ("a desk", "a wall", "a bookshelf", ...) and gives the most likely one.  env_auto.py uses the name to correct
its shape-and-colour class (a green-painted wall is a wall, not vegetation) and to pick the object's model from
props.py (desk, chair, bookshelf, lamp ...).  Runs on the NVIDIA card.

  python pipeline\\namer.py <picture>        print what CLIP sees in the whole picture (a quick test)"""
import os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models", "clip")
ARCH = "ViT-B-32"
WEIGHTS = os.path.join(MODEL_DIR, "open_clip_model.safetensors")    # laion/CLIP-ViT-B-32-laion2B-s34B-b79K

# word -> (class for env_auto, props.py type or None).  class None: keep env_auto's own class
WORDS = {
    "a wall": ("wall", None), "a painted wall": ("wall", None), "a window": ("wall", None), "window blinds": ("wall", None),
    "a curtain": ("wall", None), "a door": ("wall", None), "a whiteboard": ("wall", None), "a glass wall": ("wall", None),
    "a floor": ("floor", None), "a carpet": ("floor", None), "a ceiling": ("ceiling", None),
    "a desk": ("object", "desk"), "a table": ("object", "table"), "a counter": ("object", "counter"),
    "a chair": ("object", "chair"), "an office chair": ("object", "office_chair"), "a sofa": ("object", "sofa"),
    "an armchair": ("object", "sofa"), "a bed": ("object", "bed"), "a bookshelf": ("object", "bookshelf"),
    "a shelf": ("object", "shelf"), "a cabinet": ("object", "cabinet"), "a filing cabinet": ("object", "cabinet"),
    "a lamp": ("object", "lamp"), "a television": ("object", "tv"), "a computer monitor": ("object", "tv"),
    "a picture frame": ("object", "picture"), "a mirror": ("object", "mirror"), "a refrigerator": ("object", "fridge"),
    "a stove": ("object", "stove"), "a sink": ("object", "sink"), "a box": ("object", "box"), "a bin": ("object", "bin"),
    "a potted plant": ("vegetation", "plant_pot"), "a tree": ("vegetation", "tree"), "a bush": ("vegetation", "bush"),
    "grass": ("terrain", None), "a car": ("vehicle", "car"), "a building": ("building", None), "the sky": ("sky", None),
    "a rock": ("rock", None), "a fence": ("object", "fence"), "a pillar": ("pole", "pillar"), "stairs": ("object", "stairs"),
    "a person": ("person", None),
}


OUTDOOR = {"a tree", "a bush", "grass", "a car", "a building", "the sky", "a rock", "a fence"}   # never offered indoors


def available():
    """the model can be used: open_clip installed and the weights in models\\clip"""
    import importlib.util
    return os.path.exists(WEIGHTS) and importlib.util.find_spec("open_clip") is not None


class Namer:
    def __init__(self, device="cuda"):
        import torch, open_clip
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CLIP runs on the NVIDIA card only - run Setup.bat (installs the NVIDIA build of PyTorch)")
        self.torch, self.dev = torch, device
        self.model, _, self.prep = open_clip.create_model_and_transforms(ARCH, pretrained=WEIGHTS, device=device)
        self.model.eval()
        tok = open_clip.get_tokenizer(ARCH)
        self.words = list(WORDS)
        with torch.inference_mode():
            t = self.model.encode_text(tok(["a photo of %s" % w for w in self.words]).to(device))
        self.text = t / t.norm(dim=-1, keepdim=True)

    def crop(self, img, mask, pad=0.12):
        """the cut-out's box (with a little room around it); outside the cut-out greyed so CLIP looks at the piece"""
        ys, xs = np.nonzero(mask)
        H, W = mask.shape
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        px, py = int((x1 - x0) * pad) + 4, int((y1 - y0) * pad) + 4
        x0, x1, y0, y1 = max(0, x0 - px), min(W, x1 + px + 1), max(0, y0 - py), min(H, y1 + py + 1)
        c = img[y0:y1, x0:x1].copy(); m = mask[y0:y1, x0:x1]
        grey = np.full_like(c, 128)
        c = np.where(m[..., None], c, (0.35 * c + 0.65 * grey).astype(c.dtype))
        return c

    def name(self, img, masks, top=3, indoor=None):
        """img: BGR picture; masks: list of bool masks -> per mask [(word, probability), ...] best first.
        indoor=True: outdoor words (car, tree, sky...) are not offered, so a phone on a desk is never a car"""
        from PIL import Image
        if not masks:
            return []
        torch = self.torch
        batch = torch.stack([self.prep(Image.fromarray(cv2.cvtColor(self.crop(img, m), cv2.COLOR_BGR2RGB))) for m in masks]).to(self.dev)
        with torch.inference_mode():
            f = self.model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = 100.0 * f @ self.text.T
            if indoor:
                off = torch.tensor([w in OUTDOOR for w in self.words], device=self.dev)
                logits = logits.masked_fill(off, float("-inf"))
            p = logits.softmax(dim=-1).cpu().numpy()
        out = []
        for row in p:
            idx = np.argsort(-row)[:top]
            out.append([(self.words[i], float(row[i])) for i in idx])
        return out


def main():
    img = cv2.imdecode(np.fromfile(sys.argv[1], np.uint8), cv2.IMREAD_COLOR)
    n = Namer()
    for w, p in n.name(img, [np.ones(img.shape[:2], bool)], top=5)[0]:
        print("%-22s %.2f" % (w, p))


if __name__ == "__main__":
    main()

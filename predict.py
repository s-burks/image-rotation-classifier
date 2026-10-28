"""Detect and fix the rotation of photos. Originals are never modified.

    python predict.py photo.jpg some_folder --out fixed
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T

from dataset import CROP, MEAN, STD
from evaluate import load_checkpoint

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Same as prepare_data.py (square crop to 256) followed by RotationEvalDataset (center 224)
preprocess = T.Compose([
    T.Lambda(lambda im: ImageOps.fit(im, (256, 256), Image.BICUBIC)),
    T.CenterCrop(CROP),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def find_images(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTENSIONS)
        else:
            yield p


@torch.no_grad()
def predict_rotation(model, x, device, tta=True):
    """Returns probabilities for rotations [0, 90, 180, 270] (counter-clockwise) of one image tensor."""
    if not tta:
        return model(x[None].to(device)).softmax(1)[0].cpu().numpy()
    # Rotating the input j more steps shifts the true class to (k + j) % 4,
    # so roll each prediction back by j before averaging.
    batch = torch.stack([torch.rot90(x, j, dims=(-2, -1)) for j in range(4)]).to(device)
    probs = model(batch).softmax(1).cpu().numpy()
    return np.mean([np.roll(probs[j], -j) for j in range(4)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--weights", default="runs/mobilenet_v3_large/best.pt")
    ap.add_argument("--out", help="folder to save corrected copies into")
    ap.add_argument("--min-conf", type=float, default=0.0, help="leave photos below this confidence unrotated")
    ap.add_argument("--no-tta", action="store_true", help="single prediction instead of 4-way averaging")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(a.weights, device)
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    for path in find_images(a.inputs):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        probs = predict_rotation(model, preprocess(im), device, tta=not a.no_tta)
        k, conf = int(probs.argmax()), float(probs.max())

        if conf < a.min_conf:
            print(f"{path}: uncertain (best guess {90 * k} deg, conf {conf:.2f}), left as is")
            k = 0
        elif k == 0:
            print(f"{path}: upright (conf {conf:.2f})")
        else:
            print(f"{path}: rotated {90 * k} deg counter-clockwise, fixing (conf {conf:.2f})")

        if out:
            # PIL rotates counter-clockwise for positive angles, so -90k undoes a k-step rotation
            im.rotate(-90 * k, expand=True).save(out / path.name, quality=95)


if __name__ == "__main__":
    main()

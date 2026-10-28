"""Detect and fix the rotation of photos using the exported ONNX model. Originals are never modified.

Needs only onnxruntime, pillow and numpy (no PyTorch):

    pip install onnxruntime pillow numpy
    python predict_onnx.py photo.jpg some_folder --model rotation.onnx --out fixed
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

# Must match dataset.py and prepare_data.py
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE = 256
CROP = 224

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def preprocess(im):
    """PIL image -> (3, 224, 224) float32 array, identical to predict.py's torchvision pipeline."""
    im = ImageOps.fit(im, (RESIZE, RESIZE), Image.BICUBIC)
    offset = (RESIZE - CROP) // 2
    x = np.asarray(im, dtype=np.float32)[offset:offset + CROP, offset:offset + CROP] / 255
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)


def find_images(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTENSIONS)
        else:
            yield p


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_rotation(session, x, tta=True):
    """Returns probabilities for rotations [0, 90, 180, 270] (counter-clockwise) of one image array."""
    if not tta:
        return softmax(session.run(None, {"input": x[None]})[0])[0]
    # Rotating the input j more steps shifts the true class to (k + j) % 4,
    # so roll each prediction back by j before averaging.
    batch = np.stack([np.rot90(x, j, axes=(1, 2)) for j in range(4)])
    probs = softmax(session.run(None, {"input": np.ascontiguousarray(batch)})[0])
    return np.mean([np.roll(probs[j], -j) for j in range(4)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--model", required=True, help="path to rotation.onnx")
    ap.add_argument("--out", help="folder to save corrected copies into")
    ap.add_argument("--min-conf", type=float, default=0.0, help="leave photos below this confidence unrotated")
    ap.add_argument("--no-tta", action="store_true", help="single prediction instead of 4-way averaging")
    a = ap.parse_args()

    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    session = ort.InferenceSession(a.model, providers=providers)
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    for path in find_images(a.inputs):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        probs = predict_rotation(session, preprocess(im), tta=not a.no_tta)
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

"""Detect and fix the rotation of photos. Originals are never modified.

    python predict.py photo.jpg some_folder --out fixed
"""
import argparse
import io
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T

from dataset import CROP, MEAN, STD
from evaluate import load_checkpoint

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ORIENTATION_TAG = 0x0112
EXIF_HEADER = b"Exif\0\0"
# Each value is followed by the one that shows the image turned 90 degrees further clockwise
CLOCKWISE_CYCLES = ((1, 6, 3, 8), (2, 7, 4, 5))

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


def turn_clockwise(orientation, k):
    """Orientation value showing the image k*90 degrees clockwise from how `orientation` shows it."""
    cycle = CLOCKWISE_CYCLES[1] if orientation in CLOCKWISE_CYCLES[1] else CLOCKWISE_CYCLES[0]
    start = cycle.index(orientation) if orientation in cycle else 0
    return cycle[(start + k) % 4]


def find_exif_segment(data):
    """Returns (start, end) of the JPEG's Exif APP1 segment, or None if it has none."""
    pos = 2
    while pos + 4 <= len(data) and data[pos] == 0xFF and data[pos + 1] != 0xDA:  # 0xDA: image data starts
        end = pos + 2 + int.from_bytes(data[pos + 2:pos + 4], "big")
        if data[pos + 1] == 0xE1 and data[pos + 4:pos + 10] == EXIF_HEADER:
            return pos, end
        pos = end
    return None


def patch_orientation(tiff, value):
    """Overwrite the Orientation tag in a bytearray of EXIF TIFF data. Returns False if there is no such tag."""
    byte_order = {b"II": "<", b"MM": ">"}.get(bytes(tiff[:2]))
    if byte_order is None:
        return False
    try:
        ifd = struct.unpack_from(byte_order + "I", tiff, 4)[0]
        for i in range(struct.unpack_from(byte_order + "H", tiff, ifd)[0]):
            entry = ifd + 2 + 12 * i
            tag, tag_type = struct.unpack_from(byte_order + "HH", tiff, entry)
            if tag == ORIENTATION_TAG and tag_type == 3:  # 3: unsigned short, stored in the entry itself
                struct.pack_into(byte_order + "H", tiff, entry + 8, value)
                return True
    except struct.error:
        pass
    return False


def set_jpeg_rotation(data, k):
    """Returns JPEG bytes that display k*90 degrees clockwise from how `data` displays."""
    if k == 0:
        return data
    exif = Image.open(io.BytesIO(data)).getexif()
    value = turn_clockwise(exif.get(ORIENTATION_TAG, 1), k)
    segment = find_exif_segment(data)

    # Patching in place leaves all other metadata byte-identical
    if segment:
        start, end = segment
        tiff = bytearray(data[start + 4 + len(EXIF_HEADER):end])
        if patch_orientation(tiff, value):
            return data[:start + 4 + len(EXIF_HEADER)] + bytes(tiff) + data[end:]

    # No tag to patch, so write a new Exif segment, after the JFIF header if there is one
    exif[ORIENTATION_TAG] = value
    body = exif.tobytes()
    new_segment = b"\xff\xe1" + (len(body) + 2).to_bytes(2, "big") + body
    if segment:
        return data[:start] + new_segment + data[end:]
    insert_at = 2
    if data[2:4] == b"\xff\xe0":
        insert_at = 4 + int.from_bytes(data[4:6], "big")
    return data[:insert_at] + new_segment + data[insert_at:]


def save_corrected(path, im, k, dst):
    """Write the photo at `path`, predicted as rotated k*90 degrees counter-clockwise, to `dst`.

    `im` is the photo with its EXIF orientation applied, which is the view k was predicted from.
    JPEGs only get their EXIF Orientation tag changed, so image data is never re-encoded.
    """
    data = path.read_bytes()
    if data[:2] == b"\xff\xd8":
        dst.write_bytes(set_jpeg_rotation(data, k))
    else:
        # PIL rotates counter-clockwise for positive angles, so -90k undoes a k-step rotation
        im.rotate(-90 * k, expand=True).save(dst, quality=95)


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
            save_corrected(path, im, k, out / path.name)


if __name__ == "__main__":
    main()

"""Center-crop every photo to a square and resize it, so training isn't bottlenecked on JPEG decoding.

EXIF orientation is applied first so the "every source photo is upright" assumption holds.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


def process(job):
    src, dst, size = job
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = ImageOps.fit(im, (size, size), Image.BICUBIC)
            im.save(dst, quality=95)
    except Exception as e:
        print(f"skip {src}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="folder of raw photos, e.g. data/raw/val2017")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--size", type=int, default=256)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.src).rglob("*.jpg"))
    jobs = [(f, out / f.name, a.size) for f in files if not (out / f.name).exists()]
    print(f"{len(files)} found, {len(jobs)} to process")

    # Pool must be created under the __main__ guard on Windows
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(process, jobs, chunksize=64), total=len(jobs)):
            pass
    print(f"{len(list(out.glob('*.jpg')))} images in {out}")


if __name__ == "__main__":
    main()

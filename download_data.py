"""Download a COCO 2017 image split. Labels are unused; we only need photos.

val2017   ~1 GB,   5k images  (quick start)
test2017  ~6 GB,  41k images  (good size)
train2017 ~18 GB, 118k images (max)
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val2017", choices=["val2017", "test2017", "train2017"])
    p.add_argument("--out", default="data/raw")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / f"{a.split}.zip"
    if not zpath.exists():
        url = f"http://images.cocodataset.org/zips/{a.split}.zip"
        with tqdm(unit="B", unit_scale=True, desc=a.split) as bar:
            def hook(blocks, block_size, total):
                bar.total = total
                bar.update(blocks * block_size - bar.n)
            urllib.request.urlretrieve(url, zpath, hook)
    print("extracting...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    print(f"done: {out / a.split}")


if __name__ == "__main__":
    main()

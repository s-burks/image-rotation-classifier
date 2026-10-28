"""Score a checkpoint on the held-out test split, optionally saving the misclassified images."""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import CROP, RotationEvalDataset, split_files
from model import build_model
from train import DEVICE, print_confusion, to_device


def load_checkpoint(path, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"loaded {ckpt['arch']} from epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f})")
    return model.to(device).to(memory_format=torch.channels_last).eval()


@torch.no_grad()
def predict_all(model, loader):
    preds, labels = [], []
    for x, y in loader:
        x, y = to_device(x, y)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds.append(model(x).argmax(1).cpu())
        labels.append(y.cpu())
    return torch.cat(preds), torch.cat(labels)


def save_errors(dataset, preds, labels, out):
    out.mkdir(parents=True, exist_ok=True)
    wrong = (preds != labels).nonzero().flatten().tolist()
    for i in wrong:
        path = dataset.files[i // 4]
        true, pred = labels[i].item(), preds[i].item()
        im = Image.open(path).convert("RGB")
        # Same center crop and rotation the model saw (PIL rotates counter-clockwise for positive angles)
        left = (im.width - CROP) // 2
        top = (im.height - CROP) // 2
        im = im.crop((left, top, left + CROP, top + CROP)).rotate(90 * true)
        im.save(out / f"{path.stem}_true{90 * true}_pred{90 * pred}.jpg")
    print(f"saved {len(wrong)} misclassified images to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--weights", default="runs/mobilenet_v3_large/best.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--save-errors", help="folder to write misclassified images into")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    model = load_checkpoint(a.weights)
    dataset = RotationEvalDataset(split_files(a.data)[a.split])
    loader = DataLoader(dataset, batch_size=256, num_workers=a.workers, pin_memory=True)
    preds, labels = predict_all(model, loader)

    confusion = torch.bincount(labels * 4 + preds, minlength=16).view(4, 4)
    print(f"{a.split} acc {confusion.diag().sum().item() / confusion.sum().item():.4f} "
          f"on {len(dataset.files)} images x 4 rotations")
    print_confusion(confusion)

    if a.save_errors:
        save_errors(dataset, preds, labels, Path(a.save_errors))


if __name__ == "__main__":
    main()

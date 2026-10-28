import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RotationEvalDataset, RotationTrainDataset, split_files
from model import build_model, set_backbone_trainable

DEVICE = "cuda"


def to_device(x, y):
    x = x.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
    return x, y.to(DEVICE, non_blocking=True)


def make_optimizer(model, lr, epochs, steps_per_epoch):
    """Fresh optimizer over the currently trainable parameters, with a one-cycle LR schedule."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    return opt, sched


def train_one_epoch(model, loader, opt, sched, criterion, desc):
    model.train()
    loss_sum = correct = n = 0
    for x, y in tqdm(loader, desc=desc):
        x, y = to_device(x, y)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader):
    """Returns (accuracy, confusion) where confusion[true, predicted] counts samples."""
    model.eval()
    confusion = torch.zeros(4, 4, dtype=torch.long)
    for x, y in loader:
        x, y = to_device(x, y)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x).argmax(1)
        # Encode each (true, pred) pair as one number 0..15, count them, reshape to 4x4
        confusion += torch.bincount(y * 4 + pred, minlength=16).view(4, 4).cpu()
    return confusion.diag().sum().item() / confusion.sum().item(), confusion


def print_confusion(confusion):
    labels = ["0", "90", "180", "270"]
    print("true \\ pred " + "".join(f"{l:>7}" for l in labels))
    for label, row in zip(labels, confusion.tolist()):
        print(f"{label:>11} " + "".join(f"{v:>7}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--epochs", type=int, default=12, help="total epochs, including warm-up")
    ap.add_argument("--warmup-epochs", type=int, default=2, help="epochs training only the head")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3, help="head-only warm-up learning rate")
    ap.add_argument("--finetune-lr", type=float, default=3e-4, help="full-model learning rate")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/mobilenet_v3_large")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = split_files(a.data)
    print({name: len(files) for name, files in splits.items()})

    train_dl = DataLoader(
        RotationTrainDataset(splits["train"]), batch_size=a.batch_size, shuffle=True, drop_last=True,
        num_workers=a.workers, pin_memory=True, persistent_workers=True,
    )
    val_dl = DataLoader(
        RotationEvalDataset(splits["val"]), batch_size=a.batch_size * 2,
        num_workers=a.workers, pin_memory=True, persistent_workers=True,
    )

    model = build_model(a.arch).to(DEVICE).to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_acc, best_confusion = 0.0, None

    for epoch in range(a.epochs):
        if epoch == 0 and a.warmup_epochs > 0:
            set_backbone_trainable(model, False)
            opt, sched = make_optimizer(model, a.lr, a.warmup_epochs, len(train_dl))
        if epoch == a.warmup_epochs:
            set_backbone_trainable(model, True)
            opt, sched = make_optimizer(model, a.finetune_lr, a.epochs - a.warmup_epochs, len(train_dl))

        phase = "head" if epoch < a.warmup_epochs else "full"
        loss, acc = train_one_epoch(model, train_dl, opt, sched, criterion, f"epoch {epoch + 1}/{a.epochs} [{phase}]")
        val_acc, confusion = evaluate(model, val_dl)
        is_best = val_acc > best_acc
        print(f"epoch {epoch + 1}: train loss {loss:.4f}  train acc {acc:.4f}  val acc {val_acc:.4f}"
              + ("  (best, saved)" if is_best else ""))
        if is_best:
            best_acc, best_confusion = val_acc, confusion
            torch.save({"arch": a.arch, "state_dict": model.state_dict(), "epoch": epoch + 1, "val_acc": val_acc},
                       out / "best.pt")

    print(f"\nbest val acc {best_acc:.4f}, saved to {out / 'best.pt'}")
    print_confusion(best_confusion)


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fine-tuned MobileNetV3 that classifies a photo's rotation (0/90/180/270) and saves corrected copies. Training is self-supervised: upright photos are rotated at random and the rotation is the label.

Scripts are flat at the repo root and run directly; there is no package, build step, test suite, linter or CI. `data/`, `runs/` and `fixed` are gitignored.

## Commands

The README's commands are PowerShell, written for a Windows machine with an NVIDIA GPU. Python 3.12. Install torch/torchvision separately (`--index-url https://download.pytorch.org/whl/cu128`), then `pip install -r requirements.txt`.

```sh
python download_data.py --split val2017           # val2017 is the smallest split, 5k images
python prepare_data.py data/raw/val2017           # writes 256px squares to data/processed
python train.py --epochs 15 --warmup-epochs 1 --workers 12 --out runs/large_46k
python evaluate.py --weights runs/large_46k/best.pt --save-errors runs/errors_46k
python export_onnx.py --weights runs/large_46k/best.pt   # fails if ONNX output differs from PyTorch

python predict.py photos/ --weights runs/large_46k/best.pt --out fixed
python predict_onnx.py photos/ --model runs/large_46k/rotation.onnx --out fixed
```

`export_onnx.py`'s built-in comparison against PyTorch is the only automated check in the repo.

## Architecture

- **Pipeline:** `download_data.py` → `prepare_data.py` → `train.py` → `evaluate.py` → `export_onnx.py` → `predict_onnx.py`.
- **Import chain:** `predict.py` and `export_onnx.py` import `load_checkpoint` from `evaluate.py`, which imports from `train.py`, so these all need torch.
- **`predict_onnx.py` is deliberately a standalone single file.** Its users only have `onnxruntime`, `pillow` and `numpy`, so it must not import from any other module here. That is why it duplicates `MEAN`, `STD`, `CROP`, `find_images`, the prediction loop and the EXIF code from `dataset.py` and `predict.py`. Prefer keeping these duplicates over adding a shared module.
- **Saving corrected photos** goes through `save_corrected`, which is identical in both predictors. JPEGs keep their exact image data and only the EXIF Orientation tag changes; other formats are rotated and re-saved.
- **Training** runs in two phases: head-only warm-up with the backbone frozen, then a full fine-tune. Each phase gets a fresh optimizer and one-cycle schedule. The best epoch by validation accuracy is saved to `<out>/best.pt` with its `arch`, so loading needs no `--arch` flag.

## Invariants

- **Rotation convention:** label `k` means the image was rotated `k × 90°` counter-clockwise. To fix it, rotate `k × 90°` clockwise. `torch.rot90` and PIL's `rotate` are both counter-clockwise for positive values, hence `im.rotate(-90 * k, expand=True)`. Predictions are made after EXIF is applied, so the new Orientation tag must compose with the existing one rather than replace it.
- **Preprocessing must stay in sync** across `prepare_data.py` (EXIF transpose, `ImageOps.fit` to 256), `dataset.py` (center crop to 224, ImageNet normalization), `predict.py` and `predict_onnx.py`. A change to one needs the same change in the others.
- **TTA:** both predictors average over 4 rotations, rolling each result back by `j`. Keep the two implementations matching.
- **No vertical flip augmentation:** it would mimic a 180° rotation. Augmentation happens in the upright frame, before rotating.
- **Splits** (90/5/5) are by image and derived from the sorted contents of `data/processed` with a fixed seed. Adding images changes the split, so test scores are only comparable between models trained on the same data.
- **`RotationEvalDataset`** yields every image at all 4 rotations: index `i` is image `i // 4`, rotation `i % 4`. `evaluate.save_errors` relies on this.

## Hardware assumptions

`train.py` hardcodes `DEVICE = "cuda"`, and `train.py` and `evaluate.py` hardcode `torch.autocast("cuda", dtype=torch.bfloat16)`, so training and evaluation need a CUDA GPU and will not run on this Mac as written. `predict.py`, `predict_onnx.py` and `export_onnx.py` do run on CPU.

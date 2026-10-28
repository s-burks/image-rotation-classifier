# Image Rotation Classifier

Detects whether a photo is rotated 0, 90, 180 or 270 degrees from upright, and fixes it.

It's a fine-tuned ImageNet-pretrained MobileNetV3. Training is self-supervised: any upright photo becomes a labeled example by rotating it randomly and using the rotation as the label. No manual labeling is needed.

**Results** (MobileNetV3-Large, ~46k COCO photos, 15 epochs): **95.3%** test accuracy. The exported ONNX model is ~17 MB and runs in milliseconds per photo on CPU.

## Quick start with the pretrained model

No training or GPU needed. Download `rotation-mobilenetv3-large-v1.onnx` from the [latest release](../../releases/latest), then:

```powershell
pip install onnxruntime pillow numpy
python predict_onnx.py C:\path\to\photos --model rotation-mobilenetv3-large-v1.onnx --out fixed
```

The release also includes the PyTorch checkpoint (`rotation-mobilenetv3-large-v1.pt`) for use with `predict.py`, `evaluate.py` or further fine-tuning.

## Setup for training

Requires Python 3.12 and, for training, an NVIDIA GPU.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Training

```powershell
# 1. Download photos: val2017 (5k, ~1 GB), test2017 (41k, ~6 GB) or train2017 (118k, ~18 GB)
python download_data.py --split test2017

# 2. Apply EXIF orientation, center-crop to 256x256 squares into data/processed
python prepare_data.py data\raw\test2017

# 3. Train: head-only warm-up, then full fine-tune; best epoch saved to <out>/best.pt
python train.py --epochs 15 --warmup-epochs 1 --workers 12 --out runs\large_46k

# 4. Score on the held-out test split, saving misclassified images for inspection
python evaluate.py --weights runs\large_46k\best.pt --save-errors runs\errors_46k

# 5. Export to ONNX (verifies outputs match PyTorch)
python export_onnx.py --weights runs\large_46k\best.pt
```

Train/val/test splits (90/5/5) are by image and computed from the contents of `data/processed`. Adding images changes the split, so only compare test scores between models trained on the same data.

## Fixing photos

Both predictors take files and/or folders and print each photo's detected rotation and confidence. With `--out`, they save corrected copies there; originals are never modified.

```powershell
# With PyTorch
python predict.py C:\path\to\photos --weights runs\large_46k\best.pt --out fixed

# Standalone: only needs `pip install onnxruntime pillow numpy`
python predict_onnx.py C:\path\to\photos --model runs\large_46k\rotation.onnx --out fixed
```

- `--min-conf 0.7` leaves photos below that confidence unrotated, which is safer than a wrong rotation.
- By default each photo is predicted at 4 extra rotations and the results averaged (test-time augmentation). `--no-tta` makes a single prediction.
- Corrected copies are re-encoded as JPEG at quality 95.

## How it works

| File | Role |
|---|---|
| `download_data.py` | Downloads a COCO 2017 image split (labels unused) |
| `prepare_data.py` | Applies EXIF orientation, square-crops and resizes to 256px in parallel |
| `dataset.py` | Random rotation labels (k = rotated k×90° counter-clockwise), augmentation, by-image splits |
| `model.py` | MobileNetV3 (Large or Small) with a 4-class head |
| `train.py` | bfloat16 training with AdamW and one-cycle LR, validation, best checkpoint |
| `evaluate.py` | Test accuracy, confusion matrix, saving misclassified images |
| `predict.py` / `predict_onnx.py` | Fix photos with PyTorch / with ONNX Runtime only |
| `export_onnx.py` | Exports to ONNX with a dynamic batch size and checks it against PyTorch |

The model takes `(N, 3, 224, 224)` float32 input normalized with ImageNet mean/std, and outputs 4 logits for rotations `[0, 90, 180, 270]` counter-clockwise. To fix a photo predicted as `k`, rotate it `k × 90°` clockwise.

Known weak spots are scenes without a clear "up", such as close-ups, textures, overhead shots and underground stations, plus heavy film halation.

## License

The code is released under the [MIT License](LICENSE).

The released weights were fine-tuned from torchvision's ImageNet-pretrained MobileNetV3 on images from [COCO](https://cocodataset.org), which are sourced from Flickr under various licenses. Use of the weights is subject to the [COCO terms of use](https://cocodataset.org/#termsofuse).

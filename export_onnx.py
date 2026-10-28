"""Export a checkpoint to ONNX and verify onnxruntime gives the same outputs as PyTorch.

The model expects (N, 3, 224, 224) float32 input, normalized with the ImageNet mean/std from dataset.py.
Output is (N, 4) logits for rotations [0, 90, 180, 270] counter-clockwise.
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from dataset import CROP
from evaluate import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/mobilenet_v3_large/best.pt")
    ap.add_argument("--out", help="defaults to rotation.onnx next to the weights")
    a = ap.parse_args()

    out = Path(a.out) if a.out else Path(a.weights).with_name("rotation.onnx")
    model = load_checkpoint(a.weights, "cpu").to(memory_format=torch.contiguous_format)

    torch.onnx.export(
        model, (torch.zeros(1, 3, CROP, CROP),), out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        external_data=False,  # embed weights in the .onnx instead of a separate .onnx.data file
    )

    # Same random batch through both runtimes; batch of 4 also checks the dynamic batch axis
    x = torch.randn(4, 3, CROP, CROP)
    with torch.no_grad():
        expected = model(x).numpy()
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": x.numpy()})[0]

    max_diff = np.abs(expected - actual).max()
    same_preds = (expected.argmax(1) == actual.argmax(1)).all()
    print(f"max output difference vs PyTorch: {max_diff:.2e}, same predictions: {same_preds}")
    if max_diff > 1e-3 or not same_preds:
        raise SystemExit("ONNX output does not match PyTorch")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

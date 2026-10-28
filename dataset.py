"""Rotation datasets: label k means the image was rotated k*90 degrees counter-clockwise.

To correct an image predicted as k, rotate it k*90 degrees clockwise.
"""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ImageNet statistics, matching the pretrained backbone
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CROP = 224


def split_files(root, seed=0, val_frac=0.05, test_frac=0.05):
    """Deterministic split by image, so no photo appears in two splits at different rotations."""
    files = sorted(Path(root).glob("*.jpg"))
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_frac)
    n_test = int(len(files) * test_frac)
    return {
        "val": files[:n_val],
        "test": files[n_val:n_val + n_test],
        "train": files[n_val + n_test:],
    }


def rotate(img, k):
    """Rotate a (C, H, W) tensor by k*90 degrees counter-clockwise."""
    return torch.rot90(img, k, dims=(-2, -1))


class RotationTrainDataset(Dataset):
    """Random rotation and augmentation each time an image is loaded."""

    def __init__(self, files):
        self.files = files
        # Augment in the upright frame, before rotating. No vertical flip: it would mimic a 180 rotation.
        self.augment = T.Compose([
            T.RandomResizedCrop(CROP, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.augment(Image.open(self.files[i]).convert("RGB"))
        k = random.randrange(4)
        return rotate(img, k), k


class RotationEvalDataset(Dataset):
    """Deterministic: every image at all 4 rotations. Index i maps to image i // 4, rotation i % 4."""

    def __init__(self, files):
        self.files = files
        self.transform = T.Compose([
            T.CenterCrop(CROP),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.files) * 4

    def __getitem__(self, i):
        img = self.transform(Image.open(self.files[i // 4]).convert("RGB"))
        k = i % 4
        return rotate(img, k), k

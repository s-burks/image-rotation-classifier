import torch.nn as nn
from torchvision import models

ARCHS = {
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
}


def build_model(arch="mobilenet_v3_large", pretrained=True):
    """ImageNet-pretrained MobileNetV3 with its 1000-class output layer replaced by a 4-class one."""
    builder, weights = ARCHS[arch]
    model = builder(weights=weights if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
    return model


def set_backbone_trainable(model, trainable):
    """Freeze or unfreeze the feature extractor; the classifier head always stays trainable."""
    for p in model.features.parameters():
        p.requires_grad = trainable

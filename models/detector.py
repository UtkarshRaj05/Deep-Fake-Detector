"""
Deepfake detector model.

Uses an EfficientNet-B0 backbone (pretrained on ImageNet when weights
are reachable, otherwise randomly initialized so the code still runs
in offline environments) with a small custom classification head.
Transfer learning is the right call here: deepfake artifacts are
subtle textures and edges, and a backbone that already understands
general image structure converges much faster than training a CNN
from scratch on a modest dataset.
"""
import torch
import torch.nn as nn
from torchvision import models


class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            except Exception:
                weights = None

        try:
            backbone = models.efficientnet_b0(weights=weights)
        except Exception:
            # No internet access to fetch pretrained weights -> fall back
            # to a randomly initialized backbone so the pipeline still runs.
            backbone = models.efficientnet_b0(weights=None)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout * 0.7, inplace=False),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone

        # Keep a handle to the last conv layer for Grad-CAM.
        self.gradcam_target_layer = self.backbone.features[-1]

    def forward(self, x):
        return self.backbone(x)

    def set_backbone_trainable(self, trainable: bool):
        """Freeze/unfreeze all backbone layers except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = trainable

"""CNN architecture used in the baseline PDF (MNIST).

The baseline paper specifies a compact CNN whose penultimate feature vector has
k = 490 dimensions (10 channels * 7 * 7), followed by a linear classifier to 10
classes.

We implement the architecture in PyTorch with an explicit `forward_features`
method to extract penultimate features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN490(nn.Module):
    """MNIST CNN with penultimate feature dimension 490."""

    def __init__(self) -> None:
        super().__init__()

        # conv_block_1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv_block_2
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # classifier
        self.fc = nn.Linear(490, 10)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate features of shape (batch, 490)."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x)
        logits = self.fc(feats)
        return logits, feats


# Backwards-compatible alias (the scripts use this name).
MNISTCNN = BaselineCNN490


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

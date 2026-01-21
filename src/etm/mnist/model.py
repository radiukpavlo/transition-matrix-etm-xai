"""CNN Formal Model (FM) for MNIST.

Manuscript requirement: penultimate feature dimension k = 490.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNNConfig:
    k: int = 490
    num_classes: int = 10


class MNISTCNN(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, cfg.k)  # penultimate
        self.fc2 = nn.Linear(cfg.k, cfg.num_classes)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.penultimate(x)
        feat = self.dropout(feat)
        return self.fc2(feat)

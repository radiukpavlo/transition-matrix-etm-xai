"""MNIST dataset access without torchvision.

The user supplies MNIST raw IDX files (standard format). We keep them under:
  data/mnist/MNIST/raw/

We implement a minimal IDX reader and a PyTorch Dataset.
This avoids torchvision build/ABI issues and keeps the pipeline broadly portable.

Files expected (uncompressed preferred; .gz accepted):
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte

No automatic downloads occur by design.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class MNISTDataConfig:
    batch_size: int = 256
    num_workers: int = 2
    pin_memory: bool = True
    normalize: bool = True
    # Standard normalization used in many MNIST baselines (also consistent with torchvision defaults).
    normalize_mean: float = 0.1307
    normalize_std: float = 0.3081


def mnist_root(repo_root: Path) -> Path:
    return repo_root / "data" / "mnist"


def _open_maybe_gz(path: Path):
    if path.exists():
        return open(path, "rb")
    gz = path.with_suffix(path.suffix + ".gz") if not str(path).endswith(".gz") else path
    if gz.exists():
        return gzip.open(gz, "rb")
    raise FileNotFoundError(f"Missing MNIST file: {path} (or {gz})")


def _read_idx_images(path: Path) -> np.ndarray:
    with _open_maybe_gz(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic} (expected 2051)")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with _open_maybe_gz(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic} (expected 2049)")
        data = np.frombuffer(f.read(n), dtype=np.uint8)
        return data


class MNISTRawDataset(Dataset):
    def __init__(self, root: Path, train: bool, normalize: bool = True, mean: float = 0.1307, std: float = 0.3081):
        raw = root / "MNIST" / "raw"
        if train:
            img_path = raw / "train-images-idx3-ubyte"
            lab_path = raw / "train-labels-idx1-ubyte"
        else:
            img_path = raw / "t10k-images-idx3-ubyte"
            lab_path = raw / "t10k-labels-idx1-ubyte"

        self.images_u8 = _read_idx_images(img_path)
        self.labels_u8 = _read_idx_labels(lab_path)
        if len(self.images_u8) != len(self.labels_u8):
            raise ValueError("Image/label count mismatch")

        self.normalize = normalize
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self) -> int:
        return int(self.labels_u8.shape[0])

    def __getitem__(self, idx: int):
        img = self.images_u8[idx].astype(np.float32) / 255.0  # [0,1]
        x = torch.from_numpy(img)[None, ...]  # (1,28,28)
        y = int(self.labels_u8[idx])
        if self.normalize:
            x = (x - self.mean) / self.std
        return x, y


def assert_mnist_present(root: Path) -> None:
    raw = root / "MNIST" / "raw"
    required = [
        raw / "train-images-idx3-ubyte",
        raw / "train-labels-idx1-ubyte",
        raw / "t10k-images-idx3-ubyte",
        raw / "t10k-labels-idx1-ubyte",
    ]
    missing = []
    for p in required:
        if not p.exists() and not p.with_suffix(p.suffix + ".gz").exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "MNIST raw files not found. Expected data/mnist/MNIST/raw with standard IDX filenames. "
            f"Missing: {missing}"
        )


def get_dataloaders(repo_root: Path, cfg: MNISTDataConfig, device: Optional[torch.device] = None) -> Tuple[DataLoader, DataLoader]:
    root = mnist_root(repo_root)
    assert_mnist_present(root)

    train = MNISTRawDataset(root=root, train=True, normalize=cfg.normalize, mean=cfg.normalize_mean, std=cfg.normalize_std)
    test = MNISTRawDataset(root=root, train=False, normalize=cfg.normalize, mean=cfg.normalize_mean, std=cfg.normalize_std)

    pin = bool(cfg.pin_memory) and (device is not None and device.type == "cuda")

    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin)
    test_loader = DataLoader(test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)

    return train_loader, test_loader


def get_raw_dataloaders(repo_root: Path, batch_size: int, num_workers: int, device: Optional[torch.device] = None) -> Tuple[DataLoader, DataLoader]:
    """Loaders with *no normalization* (x in [0,1]) for the mental model and for autograd rotations."""
    root = mnist_root(repo_root)
    assert_mnist_present(root)

    train = MNISTRawDataset(root=root, train=True, normalize=False)
    test = MNISTRawDataset(root=root, train=False, normalize=False)

    pin = bool(device is not None and device.type == "cuda")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, test_loader

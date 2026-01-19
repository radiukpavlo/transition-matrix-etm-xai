"""MNIST loader without torchvision.

The baseline PDF assumes the canonical MNIST 28x28 grayscale dataset.
In a normal environment with internet access, `download_mnist_idx()` can
retrieve the IDX gzip files from a public mirror.

In this execution environment, outbound network from Python may be disabled;
therefore, the *project* is plug-and-play but the current sandbox run may skip
MNIST unless data are already present.
"""

from __future__ import annotations

import gzip
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# Mirrors that usually host the canonical MNIST IDX gzip files.
# We keep multiple options because some hosts may block automated downloads.
MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://yann.lecun.com/exdb/mnist/",
]


@dataclass
class MNISTArrays:
    x_train: np.ndarray  # (60000, 28, 28) uint8
    y_train: np.ndarray  # (60000,) uint8
    x_test: np.ndarray   # (10000, 28, 28) uint8
    y_test: np.ndarray   # (10000,) uint8


def _parse_idx(buffer: bytes) -> np.ndarray:
    """Parse an IDX file content (already decompressed) into a NumPy array."""
    # IDX file format: magic number, dimensions, then data.
    # magic = 0x0000080D where D is number of dimensions (e.g., 3 for images).
    magic, = struct.unpack(">I", buffer[0:4])
    dtype_code = (magic >> 8) & 0xFF
    ndim = magic & 0xFF

    # dtype mapping for MNIST: 0x08 -> unsigned byte
    if dtype_code != 0x08:
        raise ValueError(f"Unsupported IDX dtype code {dtype_code} (expected 0x08 for uint8)")

    offset = 4
    shape = []
    for _ in range(ndim):
        dim, = struct.unpack(">I", buffer[offset:offset + 4])
        shape.append(dim)
        offset += 4

    data = np.frombuffer(buffer, dtype=np.uint8, offset=offset)
    arr = data.reshape(shape)
    return arr


def _read_gz_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    return _parse_idx(buf)


def download_mnist_idx(root: str, *, mirrors: Optional[list[str]] = None, verbose: bool = True) -> Path:
    """Download MNIST IDX gzip files into `root`.

    Returns the directory containing the four gzip files.

    Notes
    -----
    This function requires outbound internet access. If network is unavailable,
    it will raise an exception. The scripts handle this gracefully and instruct
    the user to provide the files manually.
    """
    import urllib.request

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    mirrors = mirrors or MNIST_MIRRORS

    for key, fname in MNIST_FILES.items():
        dest = root_path / fname
        if dest.exists() and dest.stat().st_size > 0:
            if verbose:
                print(f"[mnist] found {dest.name} (skip download)")
            continue

        ok = False
        last_err = None
        for base in mirrors:
            url = base.rstrip("/") + "/" + fname
            try:
                if verbose:
                    print(f"[mnist] downloading {url} -> {dest}")
                urllib.request.urlretrieve(url, dest)
                ok = True
                break
            except Exception as e:
                last_err = e
                continue
        if not ok:
            raise RuntimeError(f"Failed to download {fname} from all mirrors. Last error: {last_err}")

    return root_path


def load_mnist_arrays(root: str) -> MNISTArrays:
    """Load MNIST from IDX gzip files in `root` into NumPy arrays."""
    root_path = Path(root)
    required = [root_path / MNIST_FILES[k] for k in MNIST_FILES]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing MNIST files: " + ", ".join(missing) +
            f". Place them in {root_path} or run download_mnist_idx()."
        )

    x_train = _read_gz_idx(root_path / MNIST_FILES["train_images"])  # (60000, 28, 28)
    y_train = _read_gz_idx(root_path / MNIST_FILES["train_labels"])  # (60000,)
    x_test = _read_gz_idx(root_path / MNIST_FILES["test_images"])    # (10000, 28, 28)
    y_test = _read_gz_idx(root_path / MNIST_FILES["test_labels"])    # (10000,)

    # Defensive: ensure expected shapes
    x_train = np.asarray(x_train, dtype=np.uint8)
    y_train = np.asarray(y_train, dtype=np.uint8)
    x_test = np.asarray(x_test, dtype=np.uint8)
    y_test = np.asarray(y_test, dtype=np.uint8)

    return MNISTArrays(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def normalize_images_uint8_to_float(images_uint8: np.ndarray) -> np.ndarray:
    """Normalize uint8 images in [0,255] to float32 in [0,1]."""
    return images_uint8.astype(np.float32) / 255.0


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten images of shape (n, 28, 28) to (n, 784)."""
    n = images.shape[0]
    return images.reshape(n, -1)


def unflatten_images(vectors: np.ndarray) -> np.ndarray:
    """Unflatten vectors (n, 784) to images (n, 28, 28)."""
    n = vectors.shape[0]
    return vectors.reshape(n, 28, 28)


# Backwards-compatible alias.
load_mnist = load_mnist_arrays


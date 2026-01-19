"""Metrics used by the baseline and new methods.

We implement:
- Fidelity MSE: (1/(m*l)) * ||B - B_hat||_F^2
- Symmetry defect: ||T J_A - J_B T||_F (and squared)
- Image reconstruction metrics: SSIM, PSNR

SSIM is computed via skimage (Wang et al., 2004). PSNR uses the standard
10*log10(MAX_I^2/MSE) definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from skimage.metrics import structural_similarity as ssim


@dataclass
class FidelityMetrics:
    mse: float
    frob_sq: float


def fidelity_mse(B: np.ndarray, B_hat: np.ndarray) -> FidelityMetrics:
    """Mean squared error per element based on Frobenius norm."""
    assert B.shape == B_hat.shape
    diff = B - B_hat
    frob_sq = float(np.sum(diff ** 2))
    mse = frob_sq / (B.shape[0] * B.shape[1])
    return FidelityMetrics(mse=mse, frob_sq=frob_sq)


@dataclass
class SymmetryMetrics:
    frob: float
    frob_sq: float


def symmetry_defect(T: np.ndarray, J_A: np.ndarray, J_B: np.ndarray) -> SymmetryMetrics:
    """Compute ||T J_A - J_B T||_F and its square.

    Dimensions:
        T: (l, k)
        J_A: (k, k)
        J_B: (l, l)
    """
    defect = T @ J_A - J_B @ T
    frob_sq = float(np.sum(defect ** 2))
    frob = float(np.sqrt(frob_sq))
    return SymmetryMetrics(frob=frob, frob_sq=frob_sq)


def psnr(x: np.ndarray, y: np.ndarray, *, data_range: float = 1.0, eps: float = 1e-12) -> float:
    """Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = float(np.mean((x - y) ** 2))
    return 10.0 * math.log10((data_range ** 2) / (mse + eps))


def batch_image_metrics(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    *,
    data_range: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute SSIM and PSNR for each image in a batch.

    Inputs are arrays shaped (n, H, W) with float values in [0, data_range].
    """
    assert x_true.shape == x_pred.shape
    n = x_true.shape[0]
    ssim_vals = np.zeros(n, dtype=np.float64)
    psnr_vals = np.zeros(n, dtype=np.float64)

    for i in range(n):
        ssim_vals[i] = ssim(
            x_true[i],
            x_pred[i],
            data_range=data_range,
        )
        psnr_vals[i] = psnr(x_true[i], x_pred[i], data_range=data_range)

    return {"ssim": ssim_vals, "psnr": psnr_vals}


# -----------------------------------------------------------------------------
# Backwards-compatible names used by scripts.
# -----------------------------------------------------------------------------


def compute_ssim_psnr_batch(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    *,
    data_range: float = 1.0,
) -> dict:
    """Alias for :func:`batch_image_metrics`."""
    return batch_image_metrics(x_true, x_pred, data_range=data_range)


compute_image_metrics = compute_ssim_psnr_batch

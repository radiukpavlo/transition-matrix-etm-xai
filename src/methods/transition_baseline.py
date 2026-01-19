"""Baseline transition matrix estimation (old approach).

Baseline objective (as in the PDF):
    A T \approx B
where A \in R^{m x k}, B \in R^{m x l}, and T \in R^{k x l}.

In the manuscript's convention, the transition matrix is T_man \in R^{l x k}
so that B^T \approx T_man A^T, equivalent to B \approx A T_man^T.
The two are related by: T_pdf = T_man^T.

This module returns T_man (l x k) and the reconstructed B_hat.

We follow the PDF guidance:
- if A^T A is invertible (well-conditioned), use (A^T A)^{-1} A^T
- otherwise, use an SVD-based pseudoinverse with truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class BaselineSolution:
    T: np.ndarray            # (l, k) manuscript convention
    T_pdf: np.ndarray        # (k, l) PDF convention
    B_hat: np.ndarray        # (m, l)
    cond_AtA: float
    used_svd: bool


def pinv_svd(A: np.ndarray, *, rcond: float = 1e-10) -> np.ndarray:
    """Compute Moore–Penrose pseudoinverse via economy SVD.

    Returns A^+ with shape (k, m) for A shape (m, k).
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return np.zeros((A.shape[1], A.shape[0]), dtype=A.dtype)
    tol = rcond * float(s[0])
    s_inv = np.array([1.0 / si if si > tol else 0.0 for si in s], dtype=A.dtype)
    # A^+ = V Σ^+ U^T
    return (Vt.T * s_inv) @ U.T


def estimate_transition_baseline(
    A: np.ndarray,
    B: np.ndarray,
    *,
    rcond: float = 1e-10,
    cond_threshold: float = 1e12,
) -> BaselineSolution:
    """Estimate baseline transition matrix.

    Args:
        A: (m, k)
        B: (m, l)

    Returns:
        BaselineSolution with T (l, k).
    """
    m, k = A.shape
    m2, l = B.shape
    assert m == m2

    # Check invertibility/conditioning of A^T A
    AtA = A.T @ A  # (k, k)
    try:
        cond = float(np.linalg.cond(AtA))
    except Exception:
        cond = float('inf')

    used_svd = False
    if np.isfinite(cond) and cond < cond_threshold:
        # A^+ = (A^T A)^{-1} A^T
        A_plus = np.linalg.solve(AtA, A.T)  # (k, m)
    else:
        used_svd = True
        A_plus = pinv_svd(A, rcond=rcond)

    # PDF convention: T_pdf = A^+ B  (k, l)
    T_pdf = A_plus @ B
    # Manuscript convention: T = T_pdf^T  (l, k)
    T = T_pdf.T

    # Reconstruct
    B_hat = A @ T_pdf

    return BaselineSolution(
        T=T,
        T_pdf=T_pdf,
        B_hat=B_hat,
        cond_AtA=cond,
        used_svd=used_svd,
    )


# -----------------------------------------------------------------------------
# Backwards-compatible alias (used by scripts).
# -----------------------------------------------------------------------------

def solve_baseline_transition(*args, **kwargs) -> BaselineSolution:
    """Alias of :func:`estimate_transition_baseline`."""
    return estimate_transition_baseline(*args, **kwargs)

"""Infinitesimal generator estimation.

Manuscript procedure (Section 3.1):
- Choose a Lie group direction (here: SO(2) rotation)
- For each sample x_j create x_{j}^eps = exp(eps * xi) · x_j
- Compute finite differences:
      Δa_j = (a(x_j^eps) - a(x_j)) / eps
      Δb_j = (b(x_j^eps) - b(x_j)) / eps
- Estimate generators via least squares:
      ΔA ≈ A (J_A)^T
      ΔB ≈ B (J_B)^T

This module implements:
- Ridge-stabilized least squares solution for J_A and J_B
- Synthetic Algorithm 2 (MDS->2D->decoder->rotate->decode)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression


@dataclass
class GeneratorEstimate:
    J_A: np.ndarray  # (k, k)
    J_B: np.ndarray  # (l, l)
    cond_A: float
    cond_B: float
    ridge: float


def _ridge_solve(X: np.ndarray, Y: np.ndarray, ridge: float) -> np.ndarray:
    """Solve for W^T in X W^T ≈ Y with ridge regularization.

    X: (m, d)
    Y: (m, d)
    Returns:
        W: (d, d) such that X W^T ≈ Y.
    """
    d = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y
    if ridge > 0:
        XtX = XtX + ridge * np.eye(d, dtype=X.dtype)
    W_T = np.linalg.solve(XtX, XtY)  # (d, d)
    return W_T.T


def estimate_generators_from_pairs(
    A: np.ndarray,
    A_eps: np.ndarray,
    B: np.ndarray,
    B_eps: np.ndarray,
    *,
    eps: float,
    ridge: float = 1e-6,
) -> GeneratorEstimate:
    """Estimate J_A and J_B from (A, A_eps) and (B, B_eps)."""
    assert A.shape == A_eps.shape
    assert B.shape == B_eps.shape
    m, k = A.shape
    m2, l = B.shape
    assert m == m2

    dA = (A_eps - A) / eps  # (m, k)
    dB = (B_eps - B) / eps  # (m, l)

    # Conditioning diagnostics (on Gram matrices)
    try:
        cond_A = float(np.linalg.cond(A.T @ A))
    except Exception:
        cond_A = float('inf')
    try:
        cond_B = float(np.linalg.cond(B.T @ B))
    except Exception:
        cond_B = float('inf')

    J_A = _ridge_solve(A, dA, ridge=ridge)
    J_B = _ridge_solve(B, dB, ridge=ridge)

    return GeneratorEstimate(J_A=J_A, J_B=J_B, cond_A=cond_A, cond_B=cond_B, ridge=ridge)


# Backwards-compatible alias used by some scripts.
def estimate_generators_least_squares(*args, **kwargs) -> GeneratorEstimate:
    return estimate_generators_from_pairs(*args, **kwargs)


@dataclass
class SyntheticBridge:
    A_2d: np.ndarray
    decoder_A: LinearRegression
    B_2d: np.ndarray
    decoder_B: LinearRegression


def build_synthetic_2d_bridge(
    A: np.ndarray,
    B: np.ndarray,
    *,
    random_state: int = 42,
) -> SyntheticBridge:
    """Construct the MDS 2D embeddings and linear decoders for A and B."""
    mds_A = MDS(n_components=2, random_state=random_state, normalized_stress=False)
    A_2d = mds_A.fit_transform(A)
    decoder_A = LinearRegression().fit(A_2d, A)

    mds_B = MDS(n_components=2, random_state=random_state, normalized_stress=False)
    B_2d = mds_B.fit_transform(B)
    decoder_B = LinearRegression().fit(B_2d, B)

    return SyntheticBridge(A_2d=A_2d, decoder_A=decoder_A, B_2d=B_2d, decoder_B=decoder_B)


def rotate_2d(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate 2D points by angle_rad around the origin."""
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    R = np.array([[c, -s], [s, c]], dtype=points.dtype)
    return points @ R.T


def synthetic_generators_via_bridge(
    A: np.ndarray,
    B: np.ndarray,
    *,
    eps: float = 0.01,
    random_state: int = 42,
    ridge: float = 0.0,
) -> Tuple[GeneratorEstimate, SyntheticBridge]:
    """Algorithm 2 (manuscript Appendix 1.2): estimate J using 2D rotation."""
    bridge = build_synthetic_2d_bridge(A, B, random_state=random_state)

    A_2d_rot = rotate_2d(bridge.A_2d, eps)
    B_2d_rot = rotate_2d(bridge.B_2d, eps)

    A_rot = bridge.decoder_A.predict(A_2d_rot)
    B_rot = bridge.decoder_B.predict(B_2d_rot)

    est = estimate_generators_from_pairs(A, A_rot, B, B_rot, eps=eps, ridge=ridge)
    return est, bridge


@dataclass
class SyntheticRotationBatch:
    A_rot: np.ndarray
    B_target_rot: np.ndarray
    angles_rad: np.ndarray


def synthetic_random_rotation_batch(
    bridge: SyntheticBridge,
    *,
    angle_range_rad: float,
    random_state: int = 123,
) -> SyntheticRotationBatch:
    """Scenario 3 (manuscript Section 3.4.4): per-sample random rotations in 2D."""
    rng = np.random.default_rng(random_state)
    m = bridge.A_2d.shape[0]
    angles = rng.uniform(-angle_range_rad, angle_range_rad, size=(m,))

    A_2d_rot = np.vstack([rotate_2d(bridge.A_2d[j:j+1], float(angles[j])) for j in range(m)])
    B_2d_rot = np.vstack([rotate_2d(bridge.B_2d[j:j+1], float(angles[j])) for j in range(m)])

    A_rot = bridge.decoder_A.predict(A_2d_rot)
    B_target = bridge.decoder_B.predict(B_2d_rot)

    return SyntheticRotationBatch(A_rot=A_rot, B_target_rot=B_target, angles_rad=angles)

"""Transition matrix estimation for MNIST.

Baseline (fidelity-only):
  minimize ||B - A W||_F^2, where W = T^T (k×l)
  => W = (A^T A)^+ A^T B

Equivariant (fidelity + symmetry):
  minimize ||B - A W||_F^2 + λ || (J^A)^T W - W (J^B)^T ||_F^2

We solve the equivariant problem with LSQR on an implicit stacked operator, avoiding explicit Kronecker matrices.
LSQR is based on Golub-Kahan bidiagonalization and can be viewed as an SVD-grounded approach for large least squares problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

from ..utils import ensure_dir, save_json, save_json_matrix


@dataclass
class TransitionConfig:
    n_samples: int = 20000
    tau: float = 1e-8
    lambda_: float = 0.5
    lsqr_iter_lim: int = 200
    lsqr_atol: float = 1e-6
    lsqr_btol: float = 1e-6


def svd_pinv(M: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_inv = np.array([1.0 / si if si > tau else 0.0 for si in s], dtype=np.float64)
    return (Vt.T * s_inv) @ U.T, s


def solve_T_old(A: np.ndarray, B: np.ndarray, tau: float) -> Tuple[np.ndarray, Dict[str, object]]:
    G = A.T @ A
    P = A.T @ B
    G_pinv, s = svd_pinv(G, tau=tau)
    W = G_pinv @ P
    return W, {"tau": float(tau), "singular_values": s.tolist()}


def solve_T_new_lsqr(
    A: np.ndarray,
    B: np.ndarray,
    J_A: np.ndarray,
    J_B: np.ndarray,
    lambda_: float,
    iter_lim: int,
    atol: float,
    btol: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    m, k = A.shape
    m2, l = B.shape
    assert m == m2
    assert J_A.shape == (k, k)
    assert J_B.shape == (l, l)

    sqrt_lam = float(np.sqrt(lambda_))

    y1 = B.reshape(-1, order="F")
    y2 = np.zeros(k * l, dtype=np.float64)
    y = np.concatenate([y1, y2])

    n = k * l
    m_total = m * l + k * l

    def matvec(u: np.ndarray) -> np.ndarray:
        W = u.reshape((k, l), order="F")
        r1 = (A @ W).reshape(-1, order="F")
        r2 = (J_A.T @ W - W @ J_B.T).reshape(-1, order="F")
        return np.concatenate([r1, sqrt_lam * r2])

    def rmatvec(v: np.ndarray) -> np.ndarray:
        v1 = v[: m * l].reshape((m, l), order="F")
        v2 = v[m * l :].reshape((k, l), order="F")
        g1 = A.T @ v1
        g2 = (J_A @ v2 - v2 @ J_B)
        return (g1 + sqrt_lam * g2).reshape(-1, order="F")

    op = LinearOperator((m_total, n), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)
    res = lsqr(op, y, iter_lim=iter_lim, atol=atol, btol=btol)

    u_hat = res[0]
    W_hat = u_hat.reshape((k, l), order="F")

    info = {
        "lambda": float(lambda_),
        "iter_lim": int(iter_lim),
        "atol": float(atol),
        "btol": float(btol),
        "lsqr_istop": int(res[1]),
        "lsqr_itn": int(res[2]),
        "r1norm": float(res[3]),
        "r2norm": float(res[4]),
        "anorm": float(res[5]),
        "acond": float(res[6]),
        "arnorm": float(res[7]),
        "xnorm": float(res[8]),
    }
    return W_hat, info


def save_transition_matrices(out_dir: Path, W_old: np.ndarray, W_new: np.ndarray, info_old: Dict[str, object], info_new: Dict[str, object]) -> None:
    ensure_dir(out_dir / "matrices")
    save_json_matrix(out_dir / "matrices" / "T_old_kxl.json", W_old, name="T_old^T", source="MNIST baseline", meta=info_old)
    save_json_matrix(out_dir / "matrices" / "T_old_lxk.json", W_old.T, name="T_old", source="MNIST baseline (transpose)", meta=info_old)
    save_json_matrix(out_dir / "matrices" / "T_new_kxl.json", W_new, name="T_new^T", source="MNIST equivariant (LSQR)", meta=info_new)
    save_json_matrix(out_dir / "matrices" / "T_new_lxk.json", W_new.T, name="T_new", source="MNIST equivariant (transpose)", meta=info_new)
    save_json(out_dir / "matrices" / "transition_solver_info.json", {"old": info_old, "new": info_new})

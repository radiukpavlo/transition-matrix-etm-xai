"""Equivariant transition matrix estimation (new approach).

Manuscript objective:
    L(T) = ||B^T - T A^T||_F^2 + λ Σ_i ||T J_i^A - J_i^B T||_F^2

Here we implement the r=1 case used in both provided experiments (SO(2) rotation).

Two solvers are provided:
1) `solve_equivariant_small_svd` builds the vectorized least-squares matrix
   and solves via truncated SVD. This is appropriate only for tiny (k, l).
2) `solve_equivariant_large_cg` avoids Kronecker products. It solves the normal
   equations in matrix form using conjugate gradient (CG) on vec(T).

Both return T in manuscript convention: T ∈ R^{l x k}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.linalg import svd

from scipy.sparse.linalg import LinearOperator, cg


@dataclass
class CGInfo:
    converged: bool
    n_iter: int
    final_residual_norm: float


@dataclass
class EquivariantSolution:
    T: np.ndarray  # (l, k)
    info: Dict[str, float]


def _vecF(X: np.ndarray) -> np.ndarray:
    """Column-stacked vectorization (Fortran order)."""
    return X.reshape(-1, order='F')


def _matF(v: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return v.reshape(shape, order='F')


def solve_equivariant_small_svd(
    A: np.ndarray,
    B: np.ndarray,
    J_A: np.ndarray,
    J_B: np.ndarray,
    *,
    lam: float,
    tau: float = 1e-10,
) -> EquivariantSolution:
    """Solve the equivariant objective by explicit vectorization + truncated SVD.

    This matches Algorithm 1 in the manuscript conceptually, but uses √λ in the
    stacked matrix so that the stacked least-squares objective matches
    L(T) = fidelity + λ * symmetry.

    A: (m, k)
    B: (m, l)
    J_A: (k, k)
    J_B: (l, l)

    Returns:
        T: (l, k)
    """
    m, k = A.shape
    m2, l = B.shape
    assert m == m2
    assert J_A.shape == (k, k)
    assert J_B.shape == (l, l)

    I_l = np.eye(l)
    I_k = np.eye(k)

    # Fidelity: vec(T A^T) = (A ⊗ I_l) vec(T)
    M_fid = np.kron(A, I_l)  # (m*l, k*l)

    # Symmetry: vec(T J_A - J_B T) = (J_A^T ⊗ I_l - I_k ⊗ J_B) vec(T)
    K = np.kron(J_A.T, I_l) - np.kron(I_k, J_B)  # (k*l, k*l)

    M = np.vstack([M_fid, np.sqrt(lam) * K])
    y = np.concatenate([_vecF(B.T), np.zeros(k * l)])

    # Truncated SVD pseudo-inverse
    U, s, Vt = svd(M, full_matrices=False)
    smax = s[0] if s.size else 1.0
    keep = s > (tau * smax)
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]

    u = Vt.T @ (s_inv * (U.T @ y))
    T = _matF(u, (l, k))

    info = {
        'm': float(m),
        'k': float(k),
        'l': float(l),
        'lam': float(lam),
        'tau': float(tau),
        'rank_kept': float(int(keep.sum())),
        'smax': float(smax),
        'smin_kept': float(s[keep][-1]) if keep.any() else 0.0,
    }
    return EquivariantSolution(T=T, info=info)


def solve_equivariant_large_cg(
    A: np.ndarray,
    B: np.ndarray,
    J_A: np.ndarray,
    J_B: np.ndarray,
    *,
    lam: float,
    ridge: float = 0.0,
    maxiter: int = 200,
    rtol: float = 1e-6,
    atol: float = 0.0,
    x0: Optional[np.ndarray] = None,
    dtype: np.dtype = np.float64,
) -> Tuple[EquivariantSolution, CGInfo]:
    """Solve the equivariant objective using CG on the normal equations.

    This avoids forming Kronecker products. It solves:
        (A^T A ⊗ I_l + λ K^T K + ridge*I) vec(T) = vec(B^T A)

    in matrix form.

    Args:
        A: (m, k) deep features
        B: (m, l) interpretable features
        J_A: (k, k)
        J_B: (l, l)
        lam: λ ≥ 0
        ridge: optional Tikhonov regularization (adds ridge*||T||_F^2)
        x0: optional initial T (l, k)

    Returns:
        EquivariantSolution(T)
        CGInfo
    """
    m, k = A.shape
    m2, l = B.shape
    assert m == m2
    assert J_A.shape == (k, k)
    assert J_B.shape == (l, l)

    A = A.astype(dtype, copy=False)
    B = B.astype(dtype, copy=False)
    J_A = J_A.astype(dtype, copy=False)
    J_B = J_B.astype(dtype, copy=False)

    # Precompute Gram and cross-covariance
    G = A.T @ A  # (k, k)
    C = B.T @ A  # (l, k)

    # Precompute transposes used repeatedly
    JAT = J_A.T
    JBT = J_B.T

    def mat_apply(T: np.ndarray) -> np.ndarray:
        # Fidelity part: T G
        out = T @ G

        if lam != 0.0:
            # S = T J_A - J_B T
            S = T @ J_A - J_B @ T
            # K^T K action: (S J_A^T - J_B^T S)
            out = out + lam * (S @ JAT - JBT @ S)

        if ridge != 0.0:
            out = out + ridge * T
        return out

    n = l * k

    def mv(v: np.ndarray) -> np.ndarray:
        T = _matF(v, (l, k))
        return _vecF(mat_apply(T))

    Aop = LinearOperator((n, n), matvec=mv, dtype=dtype)

    b = _vecF(C)

    if x0 is None:
        x0_vec = None
    else:
        assert x0.shape == (l, k)
        x0_vec = _vecF(x0.astype(dtype, copy=False))

    it_counter = {"n": 0}

    def _cb(_xk: np.ndarray) -> None:
        # Called once per iteration by SciPy.
        it_counter["n"] += 1

    sol_vec, info_code = cg(
        Aop,
        b,
        x0=x0_vec,
        maxiter=maxiter,
        rtol=rtol,
        atol=atol,
        callback=_cb,
    )

    # info_code: 0 successful; >0 no convergence within maxiter; <0 breakdown
    T_sol = _matF(sol_vec, (l, k))

    # Compute residual norm
    r = b - mv(sol_vec)
    res_norm = float(np.linalg.norm(r))

    cg_info = CGInfo(
        converged=(info_code == 0),
        n_iter=int(it_counter["n"]),
        final_residual_norm=res_norm,
    )

    info = {
        'm': float(m),
        'k': float(k),
        'l': float(l),
        'lam': float(lam),
        'ridge': float(ridge),
        'rtol': float(rtol),
        'atol': float(atol),
        'maxiter': float(maxiter),
        'residual_norm': float(res_norm),
    }

    return EquivariantSolution(T=T_sol, info=info), cg_info

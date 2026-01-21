"""Synthetic experiments (manuscript Section 3.4).

Implements:
- Algorithm 2: generator estimation via MDS-to-2D + learned linear decoder + small rotation
- Algorithm 1: equivariant transition matrix via stacked Kronecker system + SVD pseudoinverse
- Scenario 1/2/3 (baseline, equivariant, robustness to rotations)

All synthetic artifacts are written under outputs/synthetic/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression

from .utils import ensure_dir, load_json_matrix, save_json, save_json_matrix, utc_timestamp


@dataclass
class SyntheticConfig:
    epsilon: float = 1e-2
    lambdas: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    tau: float = 1e-10
    mds_random_state: int = 42
    mds_normalized_stress: bool = False
    robustness_step_degrees: int = 5


def _labels_for_15() -> np.ndarray:
    # Manuscript: 3 classes, 15 samples; we assume 5 per class in the given order.
    return np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=int)


def mds_2d(X: np.ndarray, random_state: int, normalized_stress: bool) -> np.ndarray:
    mds = MDS(n_components=2, random_state=random_state, normalized_stress=normalized_stress)
    return mds.fit_transform(X)


def rotate_2d(points: np.ndarray, angle: float) -> np.ndarray:
    R = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    return points @ R.T


def estimate_generator_via_bridge(
    X: np.ndarray,
    epsilon: float,
    random_state: int,
    normalized_stress: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Algorithm 2 (Appendix 1.2): returns (J, X_rot, X_2d)."""
    X_2d = mds_2d(X, random_state=random_state, normalized_stress=normalized_stress)
    decoder = LinearRegression().fit(X_2d, X)
    X_2d_rot = rotate_2d(X_2d, epsilon)
    X_rot = decoder.predict(X_2d_rot)

    delta = (X_rot - X) / epsilon
    J_T = np.linalg.pinv(X) @ delta
    J = J_T.T
    return J, X_rot, X_2d


def svd_pseudoinverse(M: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_inv = np.array([1.0 / si if si > tau else 0.0 for si in s], dtype=float)
    M_pinv = (Vt.T * s_inv) @ U.T
    return M_pinv, s


def solve_equivariant_T(
    A: np.ndarray,
    B: np.ndarray,
    JAs: List[np.ndarray],
    JBs: List[np.ndarray],
    lam: float,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Algorithm 1: solve for T (l×k)."""
    m, k = A.shape
    m2, l = B.shape
    assert m == m2

    I_l = np.eye(l)
    I_k = np.eye(k)

    M_fid = np.kron(A, I_l)  # (m*l, k*l)
    Y_fid = B.T.reshape(-1, order="F")

    blocks = [M_fid]
    Y_blocks = [Y_fid]

    for JA, JB in zip(JAs, JBs):
        K = np.kron(JA.T, I_l) - np.kron(I_k, JB)
        blocks.append(lam * K)
        Y_blocks.append(np.zeros(K.shape[0]))

    M = np.vstack(blocks)
    Y = np.concatenate(Y_blocks)

    M_pinv, s = svd_pseudoinverse(M, tau=tau)
    u = M_pinv @ Y
    T = u.reshape((l, k), order="F")

    meta = {"M_shape": list(M.shape), "singular_values": s.tolist()}
    return T, M, meta


def mse_fid(B: np.ndarray, Bhat: np.ndarray) -> float:
    m, l = B.shape
    return float(np.linalg.norm(B - Bhat, ord="fro") ** 2 / (m * l))


def sym_err(T: np.ndarray, JA: np.ndarray, JB: np.ndarray) -> float:
    D = T @ JA - JB @ T
    return float(np.linalg.norm(D, ord="fro") ** 2)


def run_synthetic(repo: Path, out_dir: Path, cfg: SyntheticConfig) -> Dict[str, Any]:
    inputs = repo / "inputs" / "synthetic"
    A = load_json_matrix(inputs / "A.json")
    B = load_json_matrix(inputs / "B.json")
    W_old_provided = load_json_matrix(inputs / "T_old.json")  # manuscript provides k×l (often denoted T_old^T)
    # Fidelity-only baseline per Eq. (1): W_old_ls = argmin_W ||B - A W||_F^2.
    # We compute via an SVD-based pseudoinverse (np.linalg.pinv uses SVD).
    W_old_ls = np.linalg.pinv(A) @ B
    # Use the computed least-squares baseline for Scenario 1 metrics, while retaining the manuscript matrix as an input artifact.
    W_old = W_old_ls

    labels = _labels_for_15()

    ensure_dir(out_dir / "figures")
    ensure_dir(out_dir / "matrices")
    ensure_dir(out_dir / "runs")

    # Scenario 1 (baseline)
    B_old = A @ W_old
    B_old_provided = A @ W_old_provided

    # Algorithm 2 generators
    JA, A_rot_eps, A_2d = estimate_generator_via_bridge(
        A,
        epsilon=cfg.epsilon,
        random_state=cfg.mds_random_state,
        normalized_stress=cfg.mds_normalized_stress,
    )
    JB, B_rot_eps, B_2d = estimate_generator_via_bridge(
        B,
        epsilon=cfg.epsilon,
        random_state=cfg.mds_random_state,
        normalized_stress=cfg.mds_normalized_stress,
    )

    # Scenario 2 (lambda sweep)
    sweep_rows: List[Dict[str, Any]] = []
    Ts: Dict[float, np.ndarray] = {}

    for lam in cfg.lambdas:
        T_lk, M, meta = solve_equivariant_T(A, B, [JA], [JB], lam=float(lam), tau=cfg.tau)
        W_kl = T_lk.T
        B_new_lam = A @ W_kl
        sweep_rows.append(
            {
                "lambda": float(lam),
                "mse_fid": mse_fid(B, B_new_lam),
                "sym_err": sym_err(T_lk, JA, JB),
                "meta": meta,
            }
        )
        Ts[float(lam)] = T_lk

    lam0 = 0.5 if 0.5 in [float(x) for x in cfg.lambdas] else float(cfg.lambdas[len(cfg.lambdas) // 2])
    T_new = Ts[lam0]
    W_new = T_new.T
    B_new = A @ W_new

    # Scenario 3: robustness
    step = math.radians(cfg.robustness_step_degrees)
    angles = np.arange(-math.pi / 6, math.pi / 6 + 1e-9, step)

    decoderA = LinearRegression().fit(A_2d, A)
    decoderB = LinearRegression().fit(B_2d, B)

    A_rots: List[np.ndarray] = []
    B_targets: List[np.ndarray] = []
    B_old_rots: List[np.ndarray] = []
    B_new_rots: List[np.ndarray] = []

    for a in angles:
        A2 = rotate_2d(A_2d, float(a))
        B2 = rotate_2d(B_2d, float(a))
        Arot = decoderA.predict(A2)
        Btar = decoderB.predict(B2)
        A_rots.append(Arot)
        B_targets.append(Btar)
        B_old_rots.append(Arot @ W_old)
        B_new_rots.append(Arot @ W_new)

    rot_mse_old = [mse_fid(Bt, Bo) for Bt, Bo in zip(B_targets, B_old_rots)]
    rot_mse_new = [mse_fid(Bt, Bn) for Bt, Bn in zip(B_targets, B_new_rots)]

    # Save matrices (Appendices 1.3–1.5 plus extras)
    save_json_matrix(out_dir / "matrices" / "J_A.json", JA, name="J^A", source="Algorithm 2 (synthetic)")
    save_json_matrix(out_dir / "matrices" / "J_B.json", JB, name="J^B", source="Algorithm 2 (synthetic)")

    save_json_matrix(out_dir / "matrices" / "B_star_old.json", B_old, name="B*_old", source="Scenario 1")
    save_json_matrix(out_dir / "matrices" / "T_old_ls_kxl.json", W_old, name="T_old_ls^T", source="Scenario 1 (computed least squares)")
    save_json_matrix(out_dir / "matrices" / "T_old_provided_kxl.json", W_old_provided, name="T_old_provided^T", source="Appendix 1.1 (as printed)")
    save_json_matrix(out_dir / "matrices" / "B_star_old_provided.json", B_old_provided, name="B*_old (provided)", source="Scenario 1 using Appendix 1.1 matrix")
    save_json_matrix(out_dir / "matrices" / "T_new_lxk.json", T_new, name="T_new", source="Scenario 2", meta={"lambda": lam0})
    save_json_matrix(out_dir / "matrices" / "T_new_kxl.json", W_new, name="T_new^T", source="Scenario 2 (transpose)", meta={"lambda": lam0})
    save_json_matrix(out_dir / "matrices" / "B_star_new.json", B_new, name="B*_new", source="Scenario 2", meta={"lambda": lam0})

    save_json_matrix(out_dir / "matrices" / "A_rot_epsilon.json", A_rot_eps, name="A_rot", source="Algorithm 2 (epsilon)", meta={"epsilon": cfg.epsilon})

    save_json(
        out_dir / "matrices" / "A_rot_sweep.json",
        {
            "name": "A_rot(alpha)",
            "shape": [int(len(angles)), int(A.shape[0]), int(A.shape[1])],
            "dtype": "float64",
            "source": "Scenario 3 (MDS rotation + decoder)",
            "meta": {"angles_rad": angles.tolist()},
            "data": [Ar.tolist() for Ar in A_rots],
        },
    )

    save_json(
        out_dir / "matrices" / "robustness_metrics.json",
        {
            "angles_rad": angles.tolist(),
            "angles_deg": np.degrees(angles).tolist(),
            "mse_old": rot_mse_old,
            "mse_new": rot_mse_new,
            "lambda": lam0,
        },
    )

    save_json(out_dir / "matrices" / "lambda_sweep.json", {"rows": sweep_rows, "default_lambda": lam0})

    # --- Figures (10 synthetic minimum) ---
    def scatter(X2d: np.ndarray, title: str, path: Path) -> None:
        plt.figure(figsize=(6, 5))
        for c in np.unique(labels):
            idx = labels == c
            plt.scatter(X2d[idx, 0], X2d[idx, 1], label=f"Class {c}")
        plt.title(title)
        plt.xlabel("MDS-1")
        plt.ylabel("MDS-2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    scatter(A_2d, "Synthetic: MDS(A) (3 classes)", out_dir / "figures" / "01_mds_A.png")
    scatter(B_2d, "Synthetic: MDS(B) (3 classes)", out_dir / "figures" / "02_mds_B.png")

    def heatmap(M: np.ndarray, title: str, path: Path) -> None:
        plt.figure(figsize=(6, 5))
        plt.imshow(M, aspect="auto")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    heatmap(W_old, "Heatmap: T_old_ls^T (baseline least squares)", out_dir / "figures" / "03_heatmap_T_old.png")
    heatmap(W_old_provided, "Heatmap: T_old_provided^T (Appendix 1.1)", out_dir / "figures" / "03b_heatmap_T_old_provided.png")
    heatmap(W_new, f"Heatmap: T_new^T (k×l), λ={lam0}", out_dir / "figures" / "04_heatmap_T_new.png")
    heatmap(JA, "Heatmap: J^A (k×k)", out_dir / "figures" / "05_heatmap_JA.png")
    heatmap(JB, "Heatmap: J^B (l×l)", out_dir / "figures" / "06_heatmap_JB.png")

    svals = np.array([r["meta"]["singular_values"] for r in sweep_rows if r["lambda"] == lam0][0])
    plt.figure(figsize=(6, 4))
    plt.semilogy(np.arange(1, len(svals) + 1), svals, marker="o")
    plt.title(f"Singular values of M (λ={lam0})")
    plt.xlabel("Index")
    plt.ylabel("σ")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "07_singular_values_M.png", dpi=160)
    plt.close()

    lambdas = [r["lambda"] for r in sweep_rows]
    fid = [r["mse_fid"] for r in sweep_rows]
    sym = [r["sym_err"] for r in sweep_rows]

    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, fid, marker="o")
    plt.title("Trade-off: MSE_fid vs λ")
    plt.xlabel("λ")
    plt.ylabel("MSE_fid")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "08_tradeoff_mse_vs_lambda.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, sym, marker="o")
    plt.title("Trade-off: Sym_err vs λ")
    plt.xlabel("λ")
    plt.ylabel("Sym_err")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "09_tradeoff_sym_vs_lambda.png", dpi=160)
    plt.close()

    # Robustness scatter (author-requested)
    def stacked_mds(preds: List[np.ndarray]) -> np.ndarray:
        X = np.vstack(preds)
        return mds_2d(X, random_state=cfg.mds_random_state, normalized_stress=cfg.mds_normalized_stress)

    old_2d = stacked_mds(B_old_rots)
    new_2d = stacked_mds(B_new_rots)
    labels_rep = np.tile(labels, len(angles))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for c in np.unique(labels_rep):
        idx = labels_rep == c
        plt.scatter(old_2d[idx, 0], old_2d[idx, 1], s=12, label=f"Class {c}")
    plt.title("Robustness: B*_old_rot (expected scattered)")
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.legend()

    plt.subplot(1, 2, 2)
    for c in np.unique(labels_rep):
        idx = labels_rep == c
        plt.scatter(new_2d[idx, 0], new_2d[idx, 1], s=12, label=f"Class {c}")
    plt.title("Robustness: B*_new_rot (expected clustered)")
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "10_robustness_scatter_old_vs_new.png", dpi=160)
    plt.close()

    summary = {
        "m": int(A.shape[0]),
        "k": int(A.shape[1]),
        "l": int(B.shape[1]),
        "epsilon": cfg.epsilon,
        "tau": cfg.tau,
        "default_lambda": lam0,
        "baseline": {"mse_fid": mse_fid(B, B_old), "sym_err": sym_err(W_old.T, JA, JB)},
        "baseline_provided": {"mse_fid": mse_fid(B, B_old_provided), "sym_err": sym_err(W_old_provided.T, JA, JB)},
        "equivariant": {"mse_fid": mse_fid(B, B_new), "sym_err": sym_err(T_new, JA, JB)},
        "rotated": {"angles_deg": np.degrees(angles).tolist(), "mse_old": rot_mse_old, "mse_new": rot_mse_new},
    }

    save_json(out_dir / "matrices" / "summary.json", summary)

    run_id = f"synthetic_{utc_timestamp()}"
    save_json(out_dir / "runs" / f"{run_id}_manifest.json", {"run_id": run_id, "config": cfg.__dict__, "summary": summary})
    return summary

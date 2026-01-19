"""Run the synthetic experiment from the manuscript (Section 3.4).

This script is fully offline and is executed in the sandbox to produce
outputs/synthetic/* artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

# Ensure the repository root (parent of /scripts) is on the import path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.generator_estimation import synthetic_generators_via_bridge
from src.methods.transition_baseline import solve_baseline_transition
from src.methods.transition_equivariant import solve_equivariant_small_svd
from src.utils.metrics import fidelity_mse, symmetry_defect
from src.utils.plotting import save_scatter_side_by_side
from src.utils.repro import set_global_seed, collect_env_info


def _load_synthetic_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    # Matrices A (15x5), B (15x4), and T_old (5x4) as provided in the manuscript appendix.
    A = np.array(
        [
            [0.383, 0.475, 0.892, 0.938, 0.488],
            [0.238, 0.455, 0.974, 0.582, 0.625],
            [0.290, 0.474, 0.760, 0.422, 0.416],
            [0.233, 0.715, 0.779, 0.464, 0.400],
            [0.182, 0.470, 0.529, 0.754, 0.638],
            [0.191, 0.377, 0.169, 0.107, 0.804],
            [0.154, 0.352, 0.188, 0.313, 0.417],
            [0.116, 0.018, 0.071, 0.716, 0.133],
            [0.102, 0.477, 0.006, 0.852, 0.519],
            [0.241, 0.203, 0.185, 0.670, 0.202],
            [0.445, 0.111, 0.028, 0.043, 0.079],
            [0.709, 0.115, 0.023, 0.049, 0.154],
            [0.874, 0.046, 0.044, 0.007, 0.325],
            [0.709, 0.121, 0.000, 0.036, 0.303],
            [0.615, 0.131, 0.027, 0.043, 0.203],
        ],
        dtype=np.float64,
    )

    B = np.array(
        [
            [0.652, 0.382, 0.488, 0.147],
            [0.595, 0.530, 0.553, 0.410],
            [0.514, 0.192, 0.442, 0.431],
            [0.430, 0.291, 0.229, 0.243],
            [0.350, 0.204, 0.380, 0.275],
            [0.077, 0.434, 0.103, 0.118],
            [0.099, 0.400, 0.260, 0.083],
            [0.035, 0.414, 0.337, 0.004],
            [0.151, 0.373, 0.469, 0.120],
            [0.169, 0.303, 0.276, 0.007],
            [0.282, 0.048, 0.201, 0.002],
            [0.419, 0.200, 0.191, 0.000],
            [0.476, 0.030, 0.195, 0.000],
            [0.390, 0.003, 0.171, 0.005],
            [0.334, 0.117, 0.254, 0.000],
        ],
        dtype=np.float64,
    )

    T_old = np.array(
        [
            [0.287, -1.204, 0.324, 1.046],
            [0.191, 0.632, -0.980, 0.229],
            [0.703, 0.459, -0.414, -0.094],
            [-0.038, 0.314, 0.579, -0.690],
            [-0.257, -0.021, 0.395, -0.121],
        ],
        dtype=np.float64,
    )

    # 3 classes, 5 samples each (as in the appendix narrative).
    labels = [0] * 5 + [1] * 5 + [2] * 5
    return A, B, T_old, labels


def main() -> None:
    set_global_seed(0)

    outdir = Path('outputs/synthetic')
    outdir.mkdir(parents=True, exist_ok=True)

    A, B, T_old_kxl, labels = _load_synthetic_matrices()
    m, k = A.shape
    _, l = B.shape

    # Baseline solution (as l x k for manuscript consistency)
    T_old_lxk = T_old_kxl.T

    # Also compute the least-squares baseline from (A,B) for sanity
    baseline = solve_baseline_transition(A, B)

    # Generator estimation via the manuscript's 2D-bridge (MDS + decoder)
    eps = 0.01
    est, bridge = synthetic_generators_via_bridge(A, B, eps=eps, ridge=1e-8, random_state=0)
    J_A, J_B = est.J_A, est.J_B

    # New (equivariant) solution
    lam = 0.5
    sol_new = solve_equivariant_small_svd(A, B, J_A, J_B, lam=lam, tau=1e-12)
    T_new = sol_new.T

    # Training reconstructions
    B_old_hat = A @ T_old_kxl  # (m,l)
    B_new_hat = A @ T_new.T    # A (m,k) * (k,l)

    metrics = {
        'dims': {'m': int(m), 'k': int(k), 'l': int(l)},
        'hyperparams': {'epsilon': eps, 'lambda': lam},
        'baseline_ls': {
            'fidelity_mse': float(fidelity_mse(B, baseline.B_hat).mse),
        },
        'old': {
            'fidelity_mse': float(fidelity_mse(B, B_old_hat).mse),
            'sym_defect': float(symmetry_defect(T_old_lxk, J_A, J_B).frob),
            'sym_defect_sq': float(symmetry_defect(T_old_lxk, J_A, J_B).frob_sq),
        },
        'new': {
            'fidelity_mse': float(fidelity_mse(B, B_new_hat).mse),
            'sym_defect': float(symmetry_defect(T_new, J_A, J_B).frob),
            'sym_defect_sq': float(symmetry_defect(T_new, J_A, J_B).frob_sq),
        },
        'generator_estimation': {
            'cond_A': float(est.cond_A),
            'cond_B': float(est.cond_B),
        },
    }

    (outdir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    # Robustness test: rotate each point by a random angle in [-pi/12, pi/12] in the 2D bridge.
    # Use the same angles for A and B to represent the same object transformation.
    rng = np.random.default_rng(0)
    angles = rng.uniform(-np.pi / 12, np.pi / 12, size=m)

    # Use the stored 2D embeddings and decoders from generator estimation.
    A2 = bridge.A_2d
    B2 = bridge.B_2d
    R = np.stack(
        [
            np.stack([np.cos(angles), -np.sin(angles)], axis=-1),
            np.stack([np.sin(angles), np.cos(angles)], axis=-1),
        ],
        axis=-2,
    )  # (m,2,2)

    A2_rot = np.einsum('mij,mj->mi', R, A2)
    B2_rot = np.einsum('mij,mj->mi', R, B2)

    A_rot = bridge.decoder_A.predict(A2_rot)
    B_target_rot = bridge.decoder_B.predict(B2_rot)

    B_old_rot_hat = A_rot @ T_old_kxl
    B_new_rot_hat = A_rot @ T_new.T

    # 2D projection for visualization: PCA on concatenated features
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0)
    Z = np.vstack([B_target_rot, B_old_rot_hat, B_new_rot_hat])
    pca.fit(Z)
    old_xy = pca.transform(B_old_rot_hat)
    new_xy = pca.transform(B_new_rot_hat)

    save_scatter_side_by_side(
        old_xy,
        new_xy,
        labels,
        outdir / 'scatter_rotated_old_vs_new.png',
        title_left='Old method (rotated test)',
        title_right='New equivariant method (rotated test)',
    )

    robust = {
        'rotated_test': {
            'angle_range_rad': [float(-np.pi / 12), float(np.pi / 12)],
            'old_mse_vs_target': float(fidelity_mse(B_target_rot, B_old_rot_hat).mse),
            'new_mse_vs_target': float(fidelity_mse(B_target_rot, B_new_rot_hat).mse),
        }
    }
    (outdir / 'robustness.json').write_text(json.dumps(robust, indent=2))

    # Save matrices for inspection
    np.save(outdir / 'A.npy', A)
    np.save(outdir / 'B.npy', B)
    np.save(outdir / 'T_old_lxk.npy', T_old_lxk)
    np.save(outdir / 'T_new_lxk.npy', T_new)
    np.save(outdir / 'J_A.npy', J_A)
    np.save(outdir / 'J_B.npy', J_B)

    env = collect_env_info()
    (outdir / 'env.json').write_text(json.dumps(env, indent=2))

    print('[synthetic] wrote outputs to', outdir)


if __name__ == '__main__':
    main()

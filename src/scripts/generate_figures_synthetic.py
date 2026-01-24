#!/usr/bin/env python3
"""Generate publication-ready figures for Synthetic experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary utils
from src.etm.utils import load_json_matrix  # noqa: E402
from src.etm.viz_utils import (
    configure_style, save_figure,
    CLASS_COLORS, LIGHT_COLORS, CLASS_MARKERS,
    MAJOR_GRID_STYLE, MARKER_SIZE_MEDIUM, MARKER_SIZE_LARGE, LINE_MARKER_SIZE
)

# --- STYLE CONFIGURATION (Matching generate_figures_synthetic_extended.py) ---
configure_style()




def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_heatmap(M: np.ndarray, title: str, out_dir: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5)) # Increased from (6, 5)
    im = ax.imshow(M, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    save_figure(fig, out_dir, stem)


def _plot_tradeoff(
    x: List[float],
    y: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6)) # Increased from (6, 4)
    ax.plot(x, y, marker="o", linewidth=3, markersize=LINE_MARKER_SIZE)
    ax.grid(True, **MAJOR_GRID_STYLE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_figure(fig, out_dir, stem)


def _plot_embedding_comparison(
    old_2d: np.ndarray,
    new_2d: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    out_dir: Path,
    stem: str,
) -> None:
    """Plot Old vs New embeddings side-by-side (chaos vs order).
    
    Note: These are implicitly 'Rotated' embeddings in the original context (B*_old vs B*_new).
    So we use LIGHT_COLORS.
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 8)) # Increased size for fonts/legends
    
    # Left: B*_old - Chaos
    for c in np.unique(labels):
        idx = labels == c
        # Use LIGHT colors because these are rotated/predicted points
        # Use CLASS markers
        axs[0].scatter(
            old_2d[idx, 0],
            old_2d[idx, 1],
            s=MARKER_SIZE_MEDIUM, # Slightly larger
            alpha=0.7,
            color=LIGHT_COLORS[c % len(LIGHT_COLORS)],
            edgecolor=CLASS_COLORS[c % len(CLASS_COLORS)],
            linewidth=0.8,
            label=f"Class {c}",
            marker=CLASS_MARKERS[c % len(CLASS_MARKERS)]
        )
    axs[0].set_title(f"{method_name}: $B^*_{{old}}$ (Chaos)")
    axs[0].set_xlabel(f"{method_name}-1")
    axs[0].set_ylabel(f"{method_name}-2")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, borderaxespad=0)
    axs[0].grid(True, **MAJOR_GRID_STYLE)
    
    # Right: B*_new - Order
    for c in np.unique(labels):
        idx = labels == c
        axs[1].scatter(
            new_2d[idx, 0],
            new_2d[idx, 1],
            s=MARKER_SIZE_MEDIUM,
            alpha=0.7,
            color=LIGHT_COLORS[c % len(LIGHT_COLORS)],
            edgecolor=CLASS_COLORS[c % len(CLASS_COLORS)],
            linewidth=0.8,
            label=f"Class {c}",
            marker=CLASS_MARKERS[c % len(CLASS_MARKERS)]
        )
    axs[1].set_title(f"{method_name}: $B^*_{{new}}$ (Order)")
    axs[1].set_xlabel(f"{method_name}-1")
    axs[1].set_ylabel(f"{method_name}-2")
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, borderaxespad=0)
    axs[1].grid(True, **MAJOR_GRID_STYLE)

    save_figure(fig, out_dir, stem)


def run_synthetic_viz(out_dir: Path) -> None:
    print(f"Generating synthetic figures from {out_dir}")
    matrices_dir = out_dir / "matrices"

    # -- Load Data --
    # Matrices
    A = load_json_matrix(PROJECT_ROOT / "inputs" / "synthetic" / "A.json")
    B = load_json_matrix(PROJECT_ROOT / "inputs" / "synthetic" / "B.json")

    T_old = load_json_matrix(matrices_dir / "T_old_ls_kxl.json").T
    W_old = T_old.T

    T_new_path = matrices_dir / "T_new.json"
    if T_new_path.exists():
        T_new = load_json_matrix(T_new_path)
    else:
        T_new = load_json_matrix(matrices_dir / "T_new_kxl.json").T
    W_new = T_new.T

    JA = load_json_matrix(matrices_dir / "J_A.json")
    JB = load_json_matrix(matrices_dir / "J_B.json")

    # Sweep Data
    sweep_data = load_json(matrices_dir / "lambda_sweep.json")
    
    # Robustness Data
    a_rot_data = load_json(matrices_dir / "A_rot_sweep.json")
    A_rots = np.array(a_rot_data["data"])  # (n_angles, m, k)
    angles = np.array(a_rot_data["meta"]["angles_rad"])

    # Labels (hardcoded per synthetic.py)
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=int)

    # Reconstruct B_rots
    B_old_rots = [rot @ W_old for rot in A_rots]
    B_new_rots = [rot @ W_new for rot in A_rots]

    # --- Generate Figures ---

    # 1. Heatmaps (numbered to match existing naming convention)
    _plot_heatmap(W_old, "Heatmap: T_old (Baseline)", out_dir, "03_heatmap_T_old")
    _plot_heatmap(W_new, "Heatmap: T_new (Equivariant)", out_dir, "04_heatmap_T_new")
    _plot_heatmap(JA, "Heatmap: J^A", out_dir, "05_heatmap_JA")
    _plot_heatmap(JB, "Heatmap: J^B", out_dir, "06_heatmap_JB")

    # 2. Trade-offs
    lambdas = [r["lambda"] for r in sweep_data["rows"]]
    fid = [r["mse_fid"] for r in sweep_data["rows"]]
    sym = [r["sym_err"] for r in sweep_data["rows"]]

    _plot_tradeoff(
        lambdas,
        fid,
        "Lambda",
        "MSE Fidelity",
        "Trade-off: MSE_fid vs Lambda",
        out_dir,
        "08_tradeoff_mse_vs_lambda",
    )
    _plot_tradeoff(
        lambdas,
        sym,
        "Lambda",
        "Symmetry Error",
        "Trade-off: Symmetry Error vs Lambda",
        out_dir,
        "09_tradeoff_sym_vs_lambda",
    )

    # 3. Robustness Scatter Plots (chaos vs order)
    B_old_stacked = np.vstack(B_old_rots)
    B_new_stacked = np.vstack(B_new_rots)
    all_embeddings = np.vstack([B_old_stacked, B_new_stacked])
    labels_rep = np.tile(labels, len(angles))
    n_old = B_old_stacked.shape[0]

    print("Computing embeddings...")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_embeddings)
    old_pca = pca.transform(B_old_stacked)
    new_pca = pca.transform(B_new_stacked)
    _plot_embedding_comparison(
        old_pca, new_pca, labels_rep, "PCA", out_dir, "10a_robustness_pca"
    )

    # MDS
    mds = MDS(n_components=2, random_state=42, normalized_stress=False)
    all_2d_mds = mds.fit_transform(all_embeddings)
    old_mds = all_2d_mds[:n_old]
    new_mds = all_2d_mds[n_old:]
    _plot_embedding_comparison(
        old_mds, new_mds, labels_rep, "MDS", out_dir, "10b_robustness_mds"
    )

    # t-SNE
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, all_embeddings.shape[0] // 4)
    )
    all_2d_tsne = tsne.fit_transform(all_embeddings)
    old_tsne = all_2d_tsne[:n_old]
    new_tsne = all_2d_tsne[n_old:]
    _plot_embedding_comparison(
        old_tsne, new_tsne, labels_rep, "t-SNE", out_dir, "10c_robustness_tsne"
    )

    # UMAP
    try:
        import umap

        umap_reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, all_embeddings.shape[0] // 4),
        )
        all_2d_umap = umap_reducer.fit_transform(all_embeddings)
        old_umap = all_2d_umap[:n_old]
        new_umap = all_2d_umap[n_old:]
        _plot_embedding_comparison(
            old_umap, new_umap, labels_rep, "UMAP", out_dir, "10d_robustness_umap"
        )
    except ImportError:
        print("UMAP not installed, skipping UMAP plot.")

    print(f"Done. Figures written to {out_dir}/figures/")


if __name__ == "__main__":
    out_dir_path = Path("outputs/synthetic")
    if len(sys.argv) > 1:
        out_dir_path = Path(sys.argv[1])

    if not out_dir_path.exists():
        print(f"Error: {out_dir_path} does not exist. Run synthetic experiments first.")
        sys.exit(1)

    run_synthetic_viz(out_dir_path)

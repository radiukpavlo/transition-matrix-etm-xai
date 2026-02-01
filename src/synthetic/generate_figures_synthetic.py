#!/usr/bin/env python3
"""Generate publication-ready figures for Synthetic experiments.

Combines functionality from original generate_figures.py and generate_figures_extended.py.
Generates 19 total figures across heatmaps, scatter plots, tradeoffs, and robustness analysis.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, List, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.linear_model import LinearRegression

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary utils
from src.etm.utils import load_json_matrix
from src.synthetic.core import mds_2d, rotate_2d, _labels_for_15
from src.synthetic.viz_utils import (
    configure_style, save_figure,
    CLASS_COLORS, LIGHT_COLORS, CLASS_MARKERS,
    MAJOR_GRID_STYLE, TITLE_FONT_SIZE,
    MARKER_SIZE_MEDIUM, MARKER_SIZE_LARGE, MARKER_SIZE_SMALL, LINE_MARKER_SIZE
)

# --- STYLE CONFIGURATION ---
configure_style()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Plotting Functions from generate_figures.py ---

def _plot_heatmap(M: np.ndarray, title: str, out_dir: Path, stem: str, xlabel: str = "Features", ylabel: str = "Components") -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5)) 
    im = ax.imshow(M, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(x, y, marker="o", linewidth=3, markersize=LINE_MARKER_SIZE)
    ax.grid(True, **MAJOR_GRID_STYLE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_figure(fig, out_dir, stem)


def _plot_scatter(X2d: np.ndarray, title: str, out_dir: Path, stem: str) -> None:
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in np.unique(labels):
        idx = labels == c
        ax.scatter(X2d[idx, 0], X2d[idx, 1], label=f"Class {c}", s=MARKER_SIZE_MEDIUM, color=CLASS_COLORS[c], marker=CLASS_MARKERS[c])
    ax.set_title(title)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.legend(borderaxespad=0)
    ax.grid(True, **MAJOR_GRID_STYLE)
    save_figure(fig, out_dir, stem)


def _plot_singular_values(svals: np.ndarray, title: str, out_dir: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(np.arange(1, len(svals) + 1), svals, marker="o", linewidth=2, color=CLASS_COLORS[2]) # Blue
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("σ")
    ax.grid(True, **MAJOR_GRID_STYLE)
    save_figure(fig, out_dir, stem)


def _plot_embedding_comparison(
    old_2d: np.ndarray,
    new_2d: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    out_dir: Path,
    stem: str,
) -> None:
    """Plot Old vs New embeddings side-by-side."""
    fig, axs = plt.subplots(1, 2, figsize=(22, 10))
    
    # Left: B*_old
    for c in np.unique(labels):
        idx = labels == c
        axs[0].scatter(
            old_2d[idx, 0],
            old_2d[idx, 1],
            s=MARKER_SIZE_MEDIUM,
            alpha=0.7,
            color=LIGHT_COLORS[c % len(LIGHT_COLORS)],
            edgecolor=CLASS_COLORS[c % len(CLASS_COLORS)],
            linewidth=0.8,
            label=f"Class {c}",
            marker=CLASS_MARKERS[c % len(CLASS_MARKERS)]
        )
    axs[0].set_title(f"{method_name}: $B^*_{{old}}$")
    axs[0].set_xlabel(f"{method_name}-1")
    axs[0].set_ylabel(f"{method_name}-2")
    axs[0].grid(True, **MAJOR_GRID_STYLE)
    
    # Right: B*_new
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
    axs[1].set_title(f"{method_name}: $B^*_{{new}}$")
    axs[1].set_xlabel(f"{method_name}-1")
    axs[1].set_ylabel(f"{method_name}-2")
    axs[1].grid(True, **MAJOR_GRID_STYLE)

    # Global Legend (Unified)
    handles, labels_txt = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels_txt, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    
    plt.subplots_adjust(bottom=0.20, wspace=0.2)
    save_figure(fig, out_dir, stem)


# --- Plotting Functions from generate_figures_extended.py ---

def plot_error_vs_angle(metrics: Dict[str, Any], out_dir: Path) -> None:
    angles_deg = np.array(metrics["angles_deg"])
    results_old = np.array(metrics["mse_old"])
    results_new = np.array(metrics["mse_new"])
    
    # Vivid Colors
    COLOR_OLD = "#FF1493" # DeepPink
    COLOR_NEW = "#1E90FF" # DodgerBlue

    # Create subplot with ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Main Plot (MSE)
    ax1.plot(angles_deg, results_old, marker='o', color=COLOR_OLD, label="Old Method ($T_{old}$)", linewidth=4, markersize=LINE_MARKER_SIZE)
    ax1.plot(angles_deg, results_new, marker='o', color=COLOR_NEW, label="New Method ($T_{new}$)", linewidth=4, markersize=LINE_MARKER_SIZE)
    ax1.set_ylabel("MSE (Fidelity)")
    ax1.set_title(f"Robustness to Rotation (Stress Test)\nRange: [-90, 90]")
    ax1.legend(loc='upper center', ncol=2, frameon=False) 
    ax1.grid(True, **MAJOR_GRID_STYLE)
    
    # Ratio Plot
    ratio = results_old / results_new
    ax2.plot(angles_deg, ratio, marker='s', color='#800080', linewidth=3, markersize=LINE_MARKER_SIZE, label="Ratio ($MSE_{old} / MSE_{new}$)")
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=2)
    ax2.set_xlabel("Rotation Angle (degrees)")
    ax2.set_ylabel("Error Ratio\n(Higher is Better)")
    ax2.grid(True, **MAJOR_GRID_STYLE)
    ax2.legend(loc='upper right', frameon=False)
    ax2.set_xlim(-90, 90)

    plt.tight_layout()
    save_figure(fig, out_dir, "12_error_vs_angle")


def plot_displacement_vectors(demo_data: Dict[str, Any], out_dir: Path) -> None:
    angle_deg = demo_data["angle_deg"]
    X_target = np.array(demo_data["B_target"])
    X_pred_old = np.array(demo_data["B_pred_old"])
    X_pred_new = np.array(demo_data["B_pred_new"])

    pca = PCA(n_components=2)
    pca.fit(X_target)
    
    xy_target = pca.transform(X_target)
    xy_old = pca.transform(X_pred_old)
    xy_new = pca.transform(X_pred_new)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    
    # Shared limits
    all_xy = np.vstack([xy_target, xy_old, xy_new])
    x_min, x_max = all_xy[:,0].min(), all_xy[:,0].max()
    y_min, y_max = all_xy[:,1].min(), all_xy[:,1].max()
    pad = (x_max - x_min) * 0.1
    xlim = (x_min - pad, x_max + pad)
    ylim = (y_min - pad, y_max + pad)

    labels = np.array([0]*5 + [1]*5 + [2]*5)

    def plot_arrows(ax, start, end, title):
        for c in np.unique(labels):
            idx = labels == c
            
            # Ideal (Target) - Static
            ax.scatter(
                start[idx, 0], start[idx, 1],
                color=CLASS_COLORS[c],
                marker=CLASS_MARKERS[c],
                s=MARKER_SIZE_LARGE,
                alpha=1.0,
                edgecolor='black',
                linewidth=1.2,
                label=f"Ideal (Class {c})",
                zorder=5
            )
            
            # Predicted - Rotated (Predictions from rotation)
            ax.scatter(
                end[idx, 0], end[idx, 1],
                color=LIGHT_COLORS[c],
                marker=CLASS_MARKERS[c],
                s=200,
                alpha=0.9,
                edgecolor=CLASS_COLORS[c],
                linewidth=0.8,
                label=f"Predicted (Class {c})",
                zorder=4
            )
        
        # Arrows
        for i in range(len(start)):
            ax.arrow(start[i,0], start[i,1], 
                     end[i,0] - start[i,0], end[i,1] - start[i,1],
                     head_width=pad*0.1, length_includes_head=True, 
                     color='black', alpha=0.6, width=pad*0.005, zorder=2)
        
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.grid(True, **MAJOR_GRID_STYLE)

    plot_arrows(axes[0], xy_target, xy_old, f"Old Method (Angle={angle_deg:.0f}°)")
    plot_arrows(axes[1], xy_target, xy_new, f"New Method (Angle={angle_deg:.0f}°)")

    # Global Legend
    handles, lbls = axes[0].get_legend_handles_labels()
    sorted_pairs = sorted(zip(lbls, handles), key=lambda x: x[0].split()[-1] + x[0].split()[0])
    
    fig.legend(
        [h for l, h in sorted_pairs], 
        [l for l, h in sorted_pairs], 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=3, 
        frameon=False,
    )
    
    plt.subplots_adjust(bottom=0.20, wspace=0.2)
    save_figure(fig, out_dir, "11_displacement_vectors")


def plot_comparison_extended(
    emb: np.ndarray, 
    method_name: str, 
    stem: str, 
    indices: dict,
    labels: np.ndarray,
    labels_rot: np.ndarray,
    out_dir: Path
) -> None:
    """Plot Rotated vs Static embeddings for extended robustness figures."""
    # Extract parts
    e_old_static = emb[indices["old_static"]]
    e_new_static = emb[indices["new_static"]]
    e_old_rot = emb[indices["old_rot"]]
    e_new_rot = emb[indices["new_rot"]]
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 8)) 
    
    # Helper to plot one side
    def _plot_side(ax, e_static, e_rot, title):
        # 1. Plot Rotated (Background)
        for c in np.unique(labels):
            color = LIGHT_COLORS[c % len(LIGHT_COLORS)]
            marker = CLASS_MARKERS[c % len(CLASS_MARKERS)]
            
            idx_rot = labels_rot == c
            ax.scatter(
                e_rot[idx_rot, 0], e_rot[idx_rot, 1],
                s=MARKER_SIZE_SMALL, 
                alpha=0.6,
                color=color,
                edgecolor=CLASS_COLORS[c % len(CLASS_COLORS)],
                linewidth=0.5,
                label=f"Rotated (Class {c})",
                marker=marker
            )
        
        # 2. Plot Static (Foreground)
        for c in np.unique(labels):
            color = CLASS_COLORS[c % len(CLASS_COLORS)]
            marker = CLASS_MARKERS[c % len(CLASS_MARKERS)]
            
            idx_stat = labels == c
            ax.scatter(
                e_static[idx_stat, 0], e_static[idx_stat, 1],
                s=MARKER_SIZE_LARGE,
                alpha=1.0,
                color=color,
                edgecolor='black',  
                linewidth=1.5,
                label=f"Static (Class {c})",
                marker=marker,
                zorder=10           
            )
        
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel(f"{method_name}-1")
        ax.set_ylabel(f"{method_name}-2")
        ax.grid(True, **MAJOR_GRID_STYLE)
        

    # Left: Old Comparison
    _plot_side(axs[0], e_old_static, e_old_rot, f"{method_name}: $B^*_{{old}}$ (Rotated vs Static)")
    
    # Right: New Comparison
    _plot_side(axs[1], e_new_static, e_new_rot, f"{method_name}: $B^*_{{new}}$ (Rotated vs Static)")
    
    # Global Legend
    handles, lbls = axs[0].get_legend_handles_labels()
    sorted_pairs = sorted(zip(lbls, handles), key=lambda x: x[0].split()[-1] + x[0].split()[0])
    
    fig.legend(
        [h for l, h in sorted_pairs], 
        [l for l, h in sorted_pairs], 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05), 
        ncol=3, 
        frameon=False,
    )
    
    plt.subplots_adjust(bottom=0.20, wspace=0.2)
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

    A_2d = load_json_matrix(matrices_dir / "A_2d.json")
    B_2d = load_json_matrix(matrices_dir / "B_2d.json")

    # Sweep Data
    sweep_data = load_json(matrices_dir / "lambda_sweep.json")
    
    # Robustness Data (Original small sweep)
    a_rot_data = load_json(matrices_dir / "A_rot_sweep.json")
    A_rots = np.array(a_rot_data["data"])  # (n_angles, m, k)
    angles = np.array(a_rot_data["meta"]["angles_rad"])

    # Labels (hardcoded per synthetic.py)
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=int)

    # Reconstruct B_rots for small sweep
    B_old_rots = [rot @ W_old for rot in A_rots]
    B_new_rots = [rot @ W_new for rot in A_rots]

    # --- Generate Original Figures (01-10) ---

    # 1. Heatmaps (numbered to match existing naming convention)
    _plot_heatmap(W_old, "Heatmap: T_old (Baseline)", out_dir, "03_heatmap_T_old", xlabel="Features (Input Dim)", ylabel="Latent Components")
    _plot_heatmap(W_new, "Heatmap: T_new (Equivariant)", out_dir, "04_heatmap_T_new", xlabel="Features (Input Dim)", ylabel="Latent Components")
    _plot_heatmap(JA, "Heatmap: J^A", out_dir, "05_heatmap_JA", xlabel="Dimension 2", ylabel="Dimension 1")
    _plot_heatmap(JB, "Heatmap: J^B", out_dir, "06_heatmap_JB", xlabel="Dimension 2", ylabel="Dimension 1")

    # 1b. Scatter Plots (MDS)
    _plot_scatter(A_2d, "Synthetic: MDS(A)", out_dir, "01_mds_A")
    _plot_scatter(B_2d, "Synthetic: MDS(B)", out_dir, "02_mds_B")

    # 1c. Singular Values
    svals = np.array([r["meta"]["singular_values"] for r in sweep_data["rows"] if r["lambda"] == sweep_data["default_lambda"]][0])
    _plot_singular_values(svals, f"Singular values of M (λ={sweep_data['default_lambda']})", out_dir, "07_singular_values_M")

    # 2. Trade-offs
    lambdas = [r["lambda"] for r in sweep_data["rows"]]
    fid = [r["mse_fid"] for r in sweep_data["rows"]]
    sym = [r["sym_err"] for r in sweep_data["rows"]]

    _plot_tradeoff(
        lambdas, fid, "Lambda", "MSE Fidelity",
        "Trade-off: MSE_fid vs Lambda", out_dir, "08_tradeoff_mse_vs_lambda"
    )
    _plot_tradeoff(
        lambdas, sym, "Lambda", "Symmetry Error",
        "Trade-off: Symmetry Error vs Lambda", out_dir, "09_tradeoff_sym_vs_lambda"
    )

    # 3. Robustness Scatter Plots [Figure 10]
    B_old_stacked = np.vstack(B_old_rots)
    B_new_stacked = np.vstack(B_new_rots)
    all_embeddings = np.vstack([B_old_stacked, B_new_stacked])
    labels_rep = np.tile(labels, len(angles))
    n_old = B_old_stacked.shape[0]

    print("Computing embeddings for Figure 10...")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_embeddings)
    old_pca = pca.transform(B_old_stacked)
    new_pca = pca.transform(B_new_stacked)
    _plot_embedding_comparison(old_pca, new_pca, labels_rep, "PCA", out_dir, "10a_robustness_pca")

    # MDS
    mds = MDS(n_components=2, random_state=42, normalized_stress=False, n_init=4)
    all_2d_mds = mds.fit_transform(all_embeddings)
    old_mds = all_2d_mds[:n_old]
    new_mds = all_2d_mds[n_old:]
    _plot_embedding_comparison(old_mds, new_mds, labels_rep, "MDS", out_dir, "10b_robustness_mds")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_embeddings.shape[0] // 4))
    all_2d_tsne = tsne.fit_transform(all_embeddings)
    old_tsne = all_2d_tsne[:n_old]
    new_tsne = all_2d_tsne[n_old:]
    _plot_embedding_comparison(old_tsne, new_tsne, labels_rep, "t-SNE", out_dir, "10c_robustness_tsne")

    # UMAP
    try:
        import umap
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, all_embeddings.shape[0] // 4), n_jobs=1)
        all_2d_umap = umap_reducer.fit_transform(all_embeddings)
        old_umap = all_2d_umap[:n_old]
        new_umap = all_2d_umap[n_old:]
        _plot_embedding_comparison(old_umap, new_umap, labels_rep, "UMAP", out_dir, "10d_robustness_umap")
    except ImportError:
        print("UMAP not installed, skipping UMAP plot.")


    # --- Generate Extended Figures (11-13) ---

    # Extended Metrics & Demo
    metrics_path = matrices_dir / "robustness_metrics_extended.json"
    demo_path = matrices_dir / "displacement_test_data.json"

    if metrics_path.exists():
        print("Plotting Error vs Angle (Figure 12)...")
        metrics = load_json(metrics_path)
        plot_error_vs_angle(metrics, out_dir)

    if demo_path.exists():
        print("Plotting Displacement Vectors (Figure 11)...")
        demo_data = load_json(demo_path)
        plot_displacement_vectors(demo_data, out_dir)

    # Figure 13 Generation (Re-running rotation logic)
    print("Generating Extended Robustness Figures (Figure 13)...")
    
    # Estimate Decoders specific for this viz
    random_state = 42
    A_2d_est = mds_2d(A, random_state=random_state, normalized_stress=False)
    decoderA = LinearRegression().fit(A_2d_est, A)
    
    # Generate Rotations
    angles_deg_ext = np.arange(-120, 120 + 1e-9, 15)
    angles_rad_ext = np.radians(angles_deg_ext)
    
    B_old_rots_ext = []
    B_new_rots_ext = []
    
    for angle in angles_rad_ext:
        A2 = rotate_2d(A_2d_est, angle)
        # Decode A_rot
        A_rot = decoderA.predict(A2)
        # Predict B
        B_old_rots_ext.append(A_rot @ W_old)
        B_new_rots_ext.append(A_rot @ W_new)
        
    B_old_rots_stacked_ext = np.vstack(B_old_rots_ext)
    B_new_rots_stacked_ext = np.vstack(B_new_rots_ext)
    
    # Static Baselines
    B_old_static_ext = A @ W_old
    B_new_static_ext = A @ W_new
    
    # Stack all for common embedding
    all_data_ext = np.vstack([
        B_old_static_ext, 
        B_new_static_ext, 
        B_old_rots_stacked_ext, 
        B_new_rots_stacked_ext
    ])
    
    n = B.shape[0]
    n_rot = B_old_rots_stacked_ext.shape[0]
    
    indices = {
        "old_static": slice(0, n),
        "new_static": slice(n, 2*n),
        "old_rot": slice(2*n, 2*n + n_rot),
        "new_rot": slice(2*n + n_rot, 2*n + 2*n_rot)
    }
    
    labels_ext = _labels_for_15()
    labels_rot_ext = np.tile(labels_ext, len(angles_deg_ext))
    
    print(f"Total samples for extended reduction: {all_data_ext.shape[0]}")

    # A. PCA
    print("Running extended PCA...")
    pca_ext = PCA(n_components=2, random_state=random_state)
    all_pca_ext = pca_ext.fit_transform(all_data_ext)
    plot_comparison_extended(all_pca_ext, "PCA", "13a_robustness_pca", indices, labels_ext, labels_rot_ext, out_dir)
    
    # B. MDS
    print("Running extended MDS...")
    mds_ext = MDS(n_components=2, random_state=random_state, normalized_stress=False, n_init=4)
    all_mds_ext = mds_ext.fit_transform(all_data_ext)
    plot_comparison_extended(all_mds_ext, "MDS", "13b_robustness_mds", indices, labels_ext, labels_rot_ext, out_dir)
    
    # C. t-SNE
    print("Running extended t-SNE...")
    tsne_ext = TSNE(n_components=2, random_state=random_state, perplexity=30)
    all_tsne_ext = tsne_ext.fit_transform(all_data_ext)
    plot_comparison_extended(all_tsne_ext, "t-SNE", "13c_robustness_tsne", indices, labels_ext, labels_rot_ext, out_dir)
    
    # D. UMAP
    try:
        import umap
        print("Running extended UMAP...")
        reducer_ext = umap.UMAP(n_components=2, random_state=random_state, n_jobs=1)
        all_umap_ext = reducer_ext.fit_transform(all_data_ext)
        plot_comparison_extended(all_umap_ext, "UMAP", "13d_robustness_umap", indices, labels_ext, labels_rot_ext, out_dir)
    except ImportError:
        print("UMAP not installed, skipping extended UMAP.")

    print(f"Done. Figures written to {out_dir}/figures/")


if __name__ == "__main__":
    out_dir_path = Path("outputs/synthetic")
    if len(sys.argv) > 1:
        out_dir_path = Path(sys.argv[1])

    if not out_dir_path.exists():
        print(f"Error: {out_dir_path} does not exist. Run synthetic experiments first.")
        # We don't exit here strictly to allow running in isolation if output dir exists manually
        # but typically this folder structure is needed.
        sys.exit(1)

    run_synthetic_viz(out_dir_path)

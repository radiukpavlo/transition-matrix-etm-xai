#!/usr/bin/env python3
"""Generate publication-ready figures for MNIST experiments.

Follows strict visual style guidelines and relies on pre-computed data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mnist import mnist_viz_utils as viz

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_robustness_curve(
    angles: np.ndarray,
    vals_old: List[float],
    vals_new: List[float],
    ylabel: str,
    title: str,
    out_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(angles, vals_old, marker=viz.CLASS_MARKERS[0], label="$T_{old}$", color=viz.COLOR_CYCLE[0], linewidth=2.5, markersize=8)
    ax.plot(angles, vals_new, marker=viz.CLASS_MARKERS[1], label="$T_{new}$", color=viz.COLOR_CYCLE[1], linewidth=2.5, markersize=8)
    
    ax.grid(True, **viz.MAJOR_GRID_STYLE)
    ax.set_xlabel("Rotation angle (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    viz.enforce_bold_text(ax)
    viz.save_figure(fig, out_dir, stem)

def plot_symmetry_bar(
    sym_old: float,
    sym_new: float,
    out_dir: Path,
    stem: str,
    subtitle: str = ""
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ["$T_{old}$", "$T_{new}$"]
    values = [sym_old, sym_new]
    colors = [viz.COLOR_CYCLE[0], viz.COLOR_CYCLE[1]]
    
    bars = ax.bar(labels, values, color=colors, edgecolor=viz.DARK_EDGE_COLOR, width=0.6)
    ax.bar_label(bars, fmt="%.2e", padding=3, fontweight="bold")
    
    ax.set_ylabel(r"$||T J_A - J_B T||_F$")
    ax.set_title(f"Symmetry Error\n{subtitle}")
    ax.grid(axis='y', **viz.MAJOR_GRID_STYLE)
    
    viz.enforce_bold_text(ax)
    viz.save_figure(fig, out_dir, stem)

def plot_embeddings(
    old_2d: np.ndarray,
    new_2d: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    out_dir: Path,
    stem: str,
    subtitle: str = ""
) -> None:
    """Plot Old vs New embeddings side-by-side."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    
    # Common settings
    pt_size = 15
    alpha = 0.7
    
    # Left: Old
    sc0 = axs[0].scatter(
        old_2d[:, 0], old_2d[:, 1], c=labels, cmap="tab10", s=pt_size, alpha=alpha, edgecolor='none'
    )
    axs[0].set_title(f"{method_name}: $B^*_{{old}}$ (Chaos)")
    axs[0].set_xlabel("Dim 1")
    axs[0].set_ylabel("Dim 2")
    axs[0].grid(True, **viz.MAJOR_GRID_STYLE)
    
    # Right: New
    sc1 = axs[1].scatter(
        new_2d[:, 0], new_2d[:, 1], c=labels, cmap="tab10", s=pt_size, alpha=alpha, edgecolor='none'
    )
    axs[1].set_title(f"{method_name}: $B^*_{{new}}$ (Order)")
    axs[1].set_xlabel("Dim 1")
    axs[1].set_ylabel("Dim 2")
    axs[1].grid(True, **viz.MAJOR_GRID_STYLE)
    
    # Colorbar (sharedish)
    cbar = fig.colorbar(sc1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Digit Class")
    cbar.set_ticks(range(10))
    
    fig.suptitle(f"{method_name} Projection - {subtitle}", fontsize=viz.TITLE_FONT_SIZE, fontweight="bold")
    
    for ax in axs:
        viz.enforce_bold_text(ax)
    
    # Bold colorbar
    cbar.ax.yaxis.label.set_fontweight("bold")
    for l in cbar.ax.get_yticklabels():
        l.set_fontweight("bold")

    viz.save_figure(fig, out_dir, stem)

def plot_chaos_figure(
    samples_path: Path,
    out_dir: Path,
    stem: str,
    subtitle: str = ""
) -> None:
    """
    Demonstrate reconstructed images of each rotated digit for B_old and B_new.
    Rows: Real Rotated, Recon Old, Recon New.
    Cols: Digits 0-9.
    """
    if not samples_path.exists():
        print(f"Skipping Chaos Figure: {samples_path} not found.")
        return
        
    data = np.load(samples_path)
    orig = data["orig"]      # (10, 28, 28)
    func_old = data["recon_old"] # (10, 28, 28)
    func_new = data["recon_new"] # (10, 28, 28)
    labels = data["labels"]  # (10,)
    
    # Ensure sorted by label 0-9
    idxs = np.argsort(labels)
    orig = orig[idxs]
    func_old = func_old[idxs]
    func_new = func_new[idxs]
    labels = labels[idxs]
    
    n = 10
    fig, axs = plt.subplots(3, n, figsize=(n * 1.5, 5))
    
    # Titles for rows
    row_titles = ["Input (Rotated)", "Recon ($T_{old}$)", "Recon ($T_{new}$)"]
    
    for i in range(n):
        # Row 0: Original
        axs[0, i].imshow(orig[i], cmap="gray", vmin=0, vmax=1)
        axs[0, i].axis("off")
        axs[0, i].set_title(str(labels[i]), fontweight="bold", fontsize=viz.BASE_FONT_SIZE)
        
        # Row 1: Old
        axs[1, i].imshow(func_old[i], cmap="gray", vmin=0, vmax=1)
        axs[1, i].axis("off")
        
        # Row 2: New
        axs[2, i].imshow(func_new[i], cmap="gray", vmin=0, vmax=1)
        axs[2, i].axis("off")
    
    # Set row labels
    for r, txt in enumerate(row_titles):
        # Add text to the left of the first column
        # Using figure coordinates or axes coordinates of the first plot
        axs[r, 0].text(-0.2, 0.5, txt, transform=axs[r, 0].transAxes, 
                       rotation=90, va='center', ha='right', fontweight='bold', fontsize=viz.BASE_FONT_SIZE)

    fig.suptitle(f"Rotated Digit Reconstruction Analysis\n{subtitle}", fontsize=viz.TITLE_FONT_SIZE, fontweight="bold")
    plt.tight_layout()
    viz.save_figure(fig, out_dir, stem)

def generate_subset_figures(subset: str, matrices_dir: Path, out_dir: Path) -> None:
    print(f"--- Generating figures for subset: {subset} ---")
    
    # 1. Metrics JSON
    metrics_path = matrices_dir / f"mnist_metrics_{subset}.json"
    if not metrics_path.exists():
        print(f"Metrics not found: {metrics_path}")
        return
        
    metrics = load_json(metrics_path)
    rob = metrics["robustness"]
    angles = np.array(rob["angles_deg"])
    
    # Robustness Curves
    plot_robustness_curve(
        angles, rob["mean_ssim_old"], rob["mean_ssim_new"], 
        "Mean SSIM", f"Robustness: SSIM ({subset})", out_dir, f"08_robustness_ssim_{subset}"
    )
    plot_robustness_curve(
        angles, rob["mean_psnr_old"], rob["mean_psnr_new"], 
        "Mean PSNR (dB)", f"Robustness: PSNR ({subset})", out_dir, f"09_robustness_psnr_{subset}"
    )
    
    # Symmetry Bar
    sym_old = metrics["symmetry_error_fro"]["old"]
    sym_new = metrics["symmetry_error_fro"]["new"]
    plot_symmetry_bar(sym_old, sym_new, out_dir, f"07b_symmetry_bar_{subset}", subtitle=f"({subset})")
    
    # 2. Chaos Figure
    chaos_path = matrices_dir / f"mnist_chaos_samples_{subset}.npz"
    plot_chaos_figure(chaos_path, out_dir, f"10_chaos_figure_{subset}", subtitle=f"Subset: {subset}")
    
    # 3. Scatter Plots
    emb_path = matrices_dir / f"mnist_embeddings_{subset}.npz"
    if emb_path.exists():
        data = np.load(emb_path)
        old_stack = data["B_star_old"]
        new_stack = data["B_star_new"]
        labels = data["labels"]
        # angles = data["angles"] # Unused for now
        
        # Re-compute projections here to ensure style consistency?
        # Or did eval.py compute projections? Eval.py computed embeddings *stacked* but not 2D projections.
        # Eval.py computed PCA/MDS and saved *figures*.
        # Wait, eval.py SAVED figures? 
        # I removed plotting from eval.py. So I must compute projections here from the stacked embeddings.
        
        all_emb = np.vstack([old_stack, new_stack])
        n_old = old_stack.shape[0]
        
        print(f"Computing projections for {subset} (N={all_emb.shape[0]})...")
        
        from sklearn.decomposition import PCA
        from sklearn.manifold import MDS, TSNE
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        all_pca = pca.fit_transform(all_emb)
        plot_embeddings(all_pca[:n_old], all_pca[n_old:], labels, "PCA", out_dir, f"09a_scatter_pca_{subset}", subset)
        
        # MDS (might be slow)
        # Check size. If too large, maybe skip or subsample further?
        # eval.py subsampled to 128 per angle -> 128*6 = 768 points. MDS is fine.
        mds = MDS(n_components=2, random_state=42, normalized_stress='auto', n_init=1, max_iter=300)
        all_mds = mds.fit_transform(all_emb)
        plot_embeddings(all_mds[:n_old], all_mds[n_old:], labels, "MDS", out_dir, f"09b_scatter_mds_{subset}", subset)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_emb.shape[0]//4))
        all_tsne = tsne.fit_transform(all_emb)
        plot_embeddings(all_tsne[:n_old], all_tsne[n_old:], labels, "t-SNE", out_dir, f"09c_scatter_tsne_{subset}", subset)
        
        # UMAP
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            all_umap = reducer.fit_transform(all_emb)
            plot_embeddings(all_umap[:n_old], all_umap[n_old:], labels, "UMAP", out_dir, f"09d_scatter_umap_{subset}", subset)
        except ImportError:
            print("UMAP not installed, skipping.")

def run_mnist_viz(out_dir: Path) -> None:
    viz.configure_style()
    matrices_dir = out_dir / "matrices"
    if not matrices_dir.exists():
        print(f"Error: Matrices directory not found at {matrices_dir}")
        return

    # Process both subsets if available
    for subset in ["train", "test"]:
        generate_subset_figures(subset, matrices_dir, out_dir)

if __name__ == "__main__":
    out_dir_path = Path("outputs/mnist")
    if len(sys.argv) > 1:
        out_dir_path = Path(sys.argv[1])
        
    run_mnist_viz(out_dir_path)

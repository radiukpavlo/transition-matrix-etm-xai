#!/usr/bin/env python3
"""Generate publication-ready figures for MNIST experiments."""

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

# --- STYLE CONFIGURATION (Matching Example) ---
BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 2
TITLE_FONT_SIZE = BASE_FONT_SIZE + 2
LEGEND_FONT_SIZE = max(BASE_FONT_SIZE - 2, 10)

COLOR_CYCLE = mpl.colormaps["tab10"].colors
MAJOR_GRID_STYLE = {"color": "#c7ccd6", "linewidth": 0.9, "alpha": 0.7}
DARK_EDGE_COLOR = "#1f2937"
DARK_TEXT_COLOR = "#0f172a"

mpl.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "font.weight": "bold",
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.titleweight": "bold",
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.framealpha": 0.92,
        "axes.edgecolor": DARK_EDGE_COLOR,
        "axes.labelcolor": DARK_TEXT_COLOR,
        "axes.titlecolor": DARK_TEXT_COLOR,
        "axes.linewidth": 1.1,
        "grid.color": MAJOR_GRID_STYLE["color"],
        "grid.linewidth": MAJOR_GRID_STYLE["linewidth"],
        "grid.alpha": MAJOR_GRID_STYLE["alpha"],
        "savefig.dpi": 300,
    }
)
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_CYCLE)


def _enforce_bold_text(ax: mpl.axes.Axes) -> None:
    """Ensure all text elements in the axes are bold."""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for text in ax.texts:
        text.set_fontweight("bold")
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontweight("bold")


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save figure in PDF, SVG, and PNG formats."""
    fig.tight_layout()
    for ax in fig.axes:
        _enforce_bold_text(ax)
    
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    for ext in ("pdf", "svg", "png"):
        fig.savefig(figures_dir / f"{stem}.{ext}", format=ext, bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_robustness_curve(
    angles: np.ndarray,
    vals_old: List[float],
    vals_new: List[float],
    ylabel: str,
    title: str,
    out_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(angles, vals_old, marker="o", label="T_old", linewidth=2)
    ax.plot(angles, vals_new, marker="o", label="T_new", linewidth=2)
    ax.grid(True, **MAJOR_GRID_STYLE)
    ax.set_xlabel("Rotation angle (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save_figure(fig, out_dir, stem)


def _plot_symmetry_bar(
    sym_old: float,
    sym_new: float,
    out_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["T_old", "T_new"]
    values = [sym_old, sym_new]
    ax.bar(labels, values, color=[COLOR_CYCLE[0], COLOR_CYCLE[1]], edgecolor=DARK_EDGE_COLOR)
    ax.set_ylabel("||T J_A - J_B T||_F")
    ax.set_title("Symmetry error")
    _save_figure(fig, out_dir, stem)


def _plot_mnist_embedding(
    old_2d: np.ndarray,
    new_2d: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    out_dir: Path,
    stem: str,
) -> None:
    """Plot Old vs New embeddings side-by-side (MNIST chaos vs order)."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: B*_old - Chaos (scattered points, lost cluster structure)
    scatter0 = axs[0].scatter(
        old_2d[:, 0],
        old_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=12,
        alpha=0.6,
        edgecolor="none",
    )
    cbar0 = plt.colorbar(scatter0, ax=axs[0])
    cbar0.set_label("Digit")
    cbar0.ax.yaxis.label.set_fontweight("bold")
    for l in cbar0.ax.get_yticklabels():
        l.set_fontweight("bold")
    axs[0].set_title(f"{method_name}: $B^*_{{old}}$ (Chaos)")
    axs[0].set_xlabel(f"{method_name}-1")
    axs[0].set_ylabel(f"{method_name}-2")
    axs[0].grid(True, **MAJOR_GRID_STYLE)
    
    # Right: B*_new - Order (preserved cluster structure)
    scatter1 = axs[1].scatter(
        new_2d[:, 0],
        new_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=12,
        alpha=0.6,
        edgecolor="none",
    )
    cbar1 = plt.colorbar(scatter1, ax=axs[1])
    cbar1.set_label("Digit")
    cbar1.ax.yaxis.label.set_fontweight("bold")
    for l in cbar1.ax.get_yticklabels():
        l.set_fontweight("bold")
    axs[1].set_title(f"{method_name}: $B^*_{{new}}$ (Order)")
    axs[1].set_xlabel(f"{method_name}-1")
    axs[1].set_ylabel(f"{method_name}-2")
    axs[1].grid(True, **MAJOR_GRID_STYLE)

    _save_figure(fig, out_dir, stem)


def run_mnist_viz(out_dir: Path) -> None:
    print(f"Generating MNIST figures from {out_dir}")
    matrices_dir = out_dir / "matrices"

    # 1. Load Metrics
    metrics_path = matrices_dir / "mnist_metrics.json"
    if not metrics_path.exists():
        print(f"Metrics not found at {metrics_path}. Run Stage 2 first.")
        return

    metrics = load_json(metrics_path)

    # --- Robustness Curves (renamed to match existing convention) ---
    rob = metrics["robustness"]
    angles = np.array(rob["angles_deg"])
    ssim_old = rob["mean_ssim_old"]
    ssim_new = rob["mean_ssim_new"]
    psnr_old = rob["mean_psnr_old"]
    psnr_new = rob["mean_psnr_new"]

    _plot_robustness_curve(
        angles, ssim_old, ssim_new, "Mean SSIM", "Robustness: SSIM vs rotation", out_dir, "08_robustness_ssim_vs_angle"
    )
    _plot_robustness_curve(
        angles, psnr_old, psnr_new, "Mean PSNR (dB)", "Robustness: PSNR vs rotation", out_dir, "09_robustness_psnr_vs_angle"
    )

    # --- Symmetry Error Bar ---
    sym_old = metrics["symmetry_error_fro"]["old"]
    sym_new = metrics["symmetry_error_fro"]["new"]
    
    _plot_symmetry_bar(sym_old, sym_new, out_dir, "07b_symmetry_error_bar")

    # --- Robustness Scatter Plots ---
    embeddings_path = matrices_dir / "mnist_robustness_embeddings.npz"
    if not embeddings_path.exists():
        print(f"Embeddings not found at {embeddings_path}. Run Stage 2 first.")
        return

    data = np.load(embeddings_path)
    B_star_old = data["B_star_old"]
    B_star_new = data["B_star_new"]
    labels = data["labels"]

    all_embeddings = np.vstack([B_star_old, B_star_new])
    n_old = B_star_old.shape[0]

    print("Computing MNIST embeddings...")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_embeddings)
    old_pca = pca.transform(B_star_old)
    new_pca = pca.transform(B_star_new)
    _plot_mnist_embedding(old_pca, new_pca, labels, "PCA", out_dir, "09a_mnist_scatter_pca")

    # MDS
    mds = MDS(
        n_components=2,
        random_state=42,
        normalized_stress="auto",
        max_iter=300,
        n_init=1,
    )
    all_2d_mds = mds.fit_transform(all_embeddings)
    old_mds = all_2d_mds[:n_old]
    new_mds = all_2d_mds[n_old:]
    _plot_mnist_embedding(old_mds, new_mds, labels, "MDS", out_dir, "09b_mnist_scatter_mds")

    # t-SNE
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, all_embeddings.shape[0] // 4)
    )
    all_2d_tsne = tsne.fit_transform(all_embeddings)
    old_tsne = all_2d_tsne[:n_old]
    new_tsne = all_2d_tsne[n_old:]
    _plot_mnist_embedding(old_tsne, new_tsne, labels, "t-SNE", out_dir, "09c_mnist_scatter_tsne")

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
        _plot_mnist_embedding(old_umap, new_umap, labels, "UMAP", out_dir, "09d_mnist_scatter_umap")
    except ImportError:
        print("UMAP not installed.")

    print(f"Done. Figures written to {out_dir}/figures/")


if __name__ == "__main__":
    out_dir_path = Path("outputs/mnist")
    if len(sys.argv) > 1:
        out_dir_path = Path(sys.argv[1])

    if not out_dir_path.exists():
        print(f"Error: {out_dir_path} does not exist. Run MNIST experiments first.")
        sys.exit(1)

    run_mnist_viz(out_dir_path)

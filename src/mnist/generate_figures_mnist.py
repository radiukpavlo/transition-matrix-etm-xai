#!/usr/bin/env python3
"""Generate publication-ready figures for MNIST experiments.

Follows strict visual style guidelines and relies on pre-computed data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

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
    axs[0].set_title(f"{method_name}: $B^*_{{old}}$")
    axs[0].set_xlabel("Dim 1")
    axs[0].set_ylabel("Dim 2")
    axs[0].grid(True, **viz.MAJOR_GRID_STYLE)
    
    # Right: New
    sc1 = axs[1].scatter(
        new_2d[:, 0], new_2d[:, 1], c=labels, cmap="tab10", s=pt_size, alpha=alpha, edgecolor='none'
    )
    axs[1].set_title(f"{method_name}: $B^*_{{new}}$")
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

# --- New Functions for On-the-Fly Generation (Figures 03-06) ---

from src.mnist.model import MNISTCNN, CNNConfig
from src.mnist.data import get_raw_dataloaders
from src.etm.utils import load_json_matrix

# Configuration for on-the-fly evaluation
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM_MEAN = 0.1307
NORM_STD = 0.3081

def load_data_and_model(out_dir: Path) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader, Dict[str, np.ndarray]]:
    print(f"Loading resources (device={DEVICE})...")
    
    # 1. Model
    model_path = out_dir / "models" / "mnist_cnn_k490.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    ckpt = torch.load(model_path, map_location=DEVICE)
    # Reconstruct config from checkpoint if possible, or use default
    cfg_dict = ckpt.get("cfg", {})
    model_cfg = CNNConfig(**cfg_dict)
    model = MNISTCNN(model_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # 2. Data (Test Set, Raw for visualization)
    _, test_loader = get_raw_dataloaders(PROJECT_ROOT, BATCH_SIZE, num_workers=0, device=torch.device(DEVICE))
    
    # 3. Transition Matrices
    matrices_dir = out_dir / "matrices"
    T_old = load_json_matrix(matrices_dir / "T_old_kxl.json") # This is W = T^T (k, l)
    T_new = load_json_matrix(matrices_dir / "T_new_kxl.json") # This is W = T^T (k, l)
    
    matrices = {"T_old": T_old, "T_new": T_new}
    
    return model, test_loader, matrices

def reconstruct_batch(model, W_kxl: np.ndarray, x: torch.Tensor) -> np.ndarray:
    """Reconstruct batch x (N, 1, 28, 28) using transition W (k, l)."""
    # Normalize for feature extraction
    x_norm = (x.to(DEVICE) - NORM_MEAN) / NORM_STD
    with torch.no_grad():
        A = model.penultimate(x_norm).cpu().numpy().astype(np.float64) # (N, k)
    
    B_hat = A @ W_kxl # (N, l)
    recon = np.clip(B_hat.reshape(-1, 28, 28), 0.0, 1.0)
    return recon

def compute_metrics(model, matrices: Dict[str, np.ndarray], loader) -> Dict[str, Dict[str, list]]:
    print("Computing metrics on test set...")
    metrics = {
        "ssim": {"T_old": [], "T_new": []},
        "psnr": {"T_old": [], "T_new": []}
    }
    
    for x, _ in tqdm(loader, desc="Evaluating"):
        x_np = x.numpy().reshape(-1, 28, 28)
        
        # Reconstruct Old
        rec_old = reconstruct_batch(model, matrices["T_old"], x)
        # Reconstruct New
        rec_new = reconstruct_batch(model, matrices["T_new"], x)
        
        for i in range(x.shape[0]):
            orig = x_np[i]
            r_o = rec_old[i]
            r_n = rec_new[i]
            
            # SSIM
            metrics["ssim"]["T_old"].append(ssim(orig, r_o, data_range=1.0))
            metrics["ssim"]["T_new"].append(ssim(orig, r_n, data_range=1.0))
            
            # PSNR
            metrics["psnr"]["T_old"].append(psnr(orig, r_o, data_range=1.0))
            metrics["psnr"]["T_new"].append(psnr(orig, r_n, data_range=1.0))
            
    return metrics

def plot_reconstructions(
    model, 
    W_kxl: np.ndarray, 
    loader, 
    title: str, 
    out_dir: Path, 
    filename: str
) -> None:
    # Get a fixed batch
    x_batch, y_batch = next(iter(loader))
    
    # Take first 8 samples for a clean 2x4 grid or 1x8
    n_samples = 8
    x_sample = x_batch[:n_samples]
    
    recon = reconstruct_batch(model, W_kxl, x_sample)
    orig = x_sample.numpy().reshape(-1, 28, 28)
    
    fig, axs = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3.5))
    
    for i in range(n_samples):
        # Top: Original
        axs[0, i].imshow(orig[i], cmap="gray", vmin=0, vmax=1)
        axs[0, i].axis("off")
        if i == 0:
            axs[0, i].text(-0.2, 0.5, "Original", transform=axs[0, i].transAxes, 
                           rotation=90, va='center', ha='right', fontweight='bold', fontsize=viz.BASE_FONT_SIZE)

        # Bottom: Reconstructed
        axs[1, i].imshow(recon[i], cmap="gray", vmin=0, vmax=1)
        axs[1, i].axis("off")
        if i == 0:
            axs[1, i].text(-0.2, 0.5, "Reconstructed", transform=axs[1, i].transAxes, 
                           rotation=90, va='center', ha='right', fontweight='bold', fontsize=viz.BASE_FONT_SIZE)
            
    fig.suptitle(title, fontsize=viz.TITLE_FONT_SIZE, fontweight="bold")
    plt.tight_layout()
    viz.save_figure(fig, out_dir, filename)

def plot_metric_histogram(
    vals_old: list, 
    vals_new: list, 
    metric_name: str, 
    out_dir: Path, 
    filename: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bins = 50
    alpha = 0.6
    
    ax.hist(vals_old, bins=bins, alpha=alpha, label="$T_{old}$", color=viz.COLOR_CYCLE[0], density=True)
    ax.hist(vals_new, bins=bins, alpha=alpha, label="$T_{new}$", color=viz.COLOR_CYCLE[1], density=True)
    
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Density")
    ax.set_title(f"{metric_name} Distribution (Test Set)")
    ax.legend()
    ax.grid(True, **viz.MAJOR_GRID_STYLE)
    
    viz.enforce_bold_text(ax)
    viz.save_figure(fig, out_dir, filename)

def run_mnist_viz(out_dir: Path) -> None:
    viz.configure_style()
    matrices_dir = out_dir / "matrices"
    if not matrices_dir.exists():
        print(f"Error: Matrices directory not found at {matrices_dir}")
        return

    # --- Part 1: On-the-fly generation for Figures 03-06 ---
    try:
        model, test_loader, matrices = load_data_and_model(out_dir)
        
        # 03. Reconstructions T_old
        plot_reconstructions(
            model, matrices["T_old"], test_loader, 
            "Reconstructions ($T_{old}$)", out_dir, "03_reconstructions_T_old"
        )
        
        # 04. Reconstructions T_new
        plot_reconstructions(
            model, matrices["T_new"], test_loader, 
            "Reconstructions ($T_{new}$)", out_dir, "04_reconstructions_T_new"
        )
        
        # Calculate metrics (SSIM/PSNR distributions)
        metrics = compute_metrics(model, matrices, test_loader)
        
        # 05. SSIM
        plot_metric_histogram(
            metrics["ssim"]["T_old"], metrics["ssim"]["T_new"], 
            "SSIM", out_dir, "05_ssim_comparison"
        )
        
        # 06. PSNR
        plot_metric_histogram(
            metrics["psnr"]["T_old"], metrics["psnr"]["T_new"], 
            "PSNR (dB)", out_dir, "06_psnr_comparison"
        )
    except Exception as e:
        print(f"Warning: Could not generate on-the-fly figures (03-06): {e}")
        import traceback
        traceback.print_exc()

    # --- Part 2: Pre-computed subset figures (Figures 07-10) ---
    # Process both subsets if available
    for subset in ["train", "test"]:
        generate_subset_figures(subset, matrices_dir, out_dir)

if __name__ == "__main__":
    out_dir_path = Path("outputs/mnist")
    if len(sys.argv) > 1:
        out_dir_path = Path(sys.argv[1])
        
    run_mnist_viz(out_dir_path)

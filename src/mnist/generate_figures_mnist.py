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

def plot_complex_robustness_curve(
    angles: np.ndarray,
    vals_old: List[float],
    vals_new: List[float],
    ylabel: str,
    title: str,
    out_dir: Path,
    stem: str,
) -> None:
    """Plot robustness curve with ratio subplot."""
    # Filter for -90 to 90
    mask = (angles >= -90) & (angles <= 90)
    a = angles[mask]
    v_old = np.array(vals_old)[mask]
    v_new = np.array(vals_new)[mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Ax1: Main metrics
    ax1.plot(a, v_old, marker=viz.CLASS_MARKERS[0], label="$T_{old}$", color=viz.COLOR_CYCLE[0], linewidth=3, markersize=10)
    ax1.plot(a, v_new, marker=viz.CLASS_MARKERS[1], label="$T_{new}$", color=viz.COLOR_CYCLE[1], linewidth=3, markersize=10)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend(loc="upper center", ncol=2, frameon=False)
    ax1.grid(True, **viz.MAJOR_GRID_STYLE)
    
    # Ax2: Advantage Ratio (New / Old for Score Metrics)
    # If standard is Error (lower better), we use Old/New.
    # Here SSIM/PSNR are scores (higher better), so New/Old > 1 is good.
    ratio = v_new / v_old
    ratio_label = "Advantage ($T_{new} / T_{old}$)"
    
    ax2.plot(a, ratio, marker='s', color='#800080', linewidth=3, markersize=8, label=ratio_label)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=2)
    ax2.set_xlabel("Rotation angle (deg)")
    ax2.set_ylabel("Advantage Ratio")
    ax2.legend(loc="upper center", frameon=False)
    ax2.grid(True, **viz.MAJOR_GRID_STYLE)
    ax2.set_xlim(-90, 90)
    
    # viz.enforce_bold_text(fig.axes) # Disabled
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
    # ax.bar_label(bars, fmt="%.2e", padding=3, fontweight="bold")
    ax.bar_label(bars, fmt="%.2e", padding=3) # Regular weight
    
    ax.set_ylabel(r"$||T J_A - J_B T||_F$")
    ax.set_title(f"Symmetry Error\n{subtitle}")
    ax.grid(axis='y', **viz.MAJOR_GRID_STYLE)
    
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
    axs[0].set_xlabel(f"{method_name}-1")
    axs[0].set_ylabel(f"{method_name}-2")
    axs[0].grid(True, **viz.MAJOR_GRID_STYLE)
    
    # Right: New
    sc1 = axs[1].scatter(
        new_2d[:, 0], new_2d[:, 1], c=labels, cmap="tab10", s=pt_size, alpha=alpha, edgecolor='none'
    )
    axs[1].set_title(f"{method_name}: $B^*_{{new}}$")
    axs[1].set_xlabel(f"{method_name}-1")
    axs[1].set_ylabel(f"{method_name}-2")
    axs[1].grid(True, **viz.MAJOR_GRID_STYLE)
    
    # Colorbar (sharedish)
    cbar = fig.colorbar(sc1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Digit Class")
    cbar.set_ticks(range(10))
    
    fig.suptitle(f"{method_name} Projection - {subtitle}", fontsize=viz.TITLE_FONT_SIZE)
    viz.save_figure(fig, out_dir, stem)

def plot_extended_scatter(
    old_static_2d: np.ndarray,
    old_rot_2d: np.ndarray,
    new_static_2d: np.ndarray,
    new_rot_2d: np.ndarray,
    labels: np.ndarray, # single set of labels, repeated for rot
    method_name: str,
    out_dir: Path,
    stem: str,
) -> None:
    """
    Generate Extended Robustness Figures (11a-d).
    Left: Old (Rotated Light vs Static Vivid)
    Right: New (Rotated Light vs Static Vivid)
    """
    fig, axs = plt.subplots(1, 2, figsize=(22, 10))
    
    n_static = len(labels)
    n_rot_old = old_rot_2d.shape[0]
    # For faint background, replicate labels
    labels_rot = np.tile(labels, n_rot_old // n_static) if n_rot_old > n_static else labels
    
    def _plot_side(ax, static_2d, rot_2d, title):
        # 1. Rotated (Background)
        ax.scatter(
            rot_2d[:, 0], rot_2d[:, 1],
            c="lightgray", # Or use tabulated colors but very faint
            s=40, alpha=0.3, 
            edgecolor='none',
            label="Rotated",
            zorder=1
        )
        
        # 2. Static (Foreground)
        ax.scatter(
            static_2d[:, 0], static_2d[:, 1],
            c=labels, cmap="tab10",
            s=120, alpha=1.0,
            edgecolor='black', linewidth=1.5,
            label="Static",
            zorder=2
        )
        
        ax.set_title(title)
        ax.set_xlabel(f"{method_name}-1")
        ax.set_ylabel(f"{method_name}-2")
        ax.grid(True, **viz.MAJOR_GRID_STYLE)
        
    # Left: Old
    _plot_side(axs[0], old_static_2d, old_rot_2d, f"{method_name}: $B^*_{{old}}$ (Rotated vs Static)")
    
    # Right: New
    _plot_side(axs[1], new_static_2d, new_rot_2d, f"{method_name}: $B^*_{{new}}$ (Rotated vs Static)")
    
    # Add Colorbar for class identification
    # Since we used "tab10" for static, we can add a colorbar
    # Create a dummy scalar mappable
    sm = plt.cm.ScalarMappable(cmap="tab10", norm=plt.Normalize(vmin=0, vmax=9))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Digit Class")
    cbar.set_ticks(range(10))
    
    plt.subplots_adjust(bottom=0.15)
    viz.save_figure(fig, out_dir, stem)

def plot_chaos_figure(
    samples_path: Path,
    out_dir: Path,
    stem: str,
    subtitle: str = ""
) -> None:
    if not samples_path.exists():
        print(f"Skipping Chaos Figure: {samples_path} not found.")
        return
        
    data = np.load(samples_path)
    orig = data["orig"]
    func_old = data["recon_old"]
    func_new = data["recon_new"]
    labels = data["labels"]
    
    idxs = np.argsort(labels)
    orig = orig[idxs]
    func_old = func_old[idxs]
    func_new = func_new[idxs]
    labels = labels[idxs]
    
    n = 10
    fig, axs = plt.subplots(3, n, figsize=(n * 1.5, 5))
    row_titles = ["Input (Rotated)", "Recon ($T_{old}$)", "Recon ($T_{new}$)"]
    
    for i in range(n):
        axs[0, i].imshow(orig[i], cmap="gray", vmin=0, vmax=1)
        axs[0, i].axis("off")
        axs[0, i].set_title(str(labels[i]), fontsize=viz.BASE_FONT_SIZE)
        
        axs[1, i].imshow(func_old[i], cmap="gray", vmin=0, vmax=1)
        axs[1, i].axis("off")
        
        axs[2, i].imshow(func_new[i], cmap="gray", vmin=0, vmax=1)
        axs[2, i].axis("off")
    
    for r, txt in enumerate(row_titles):
        axs[r, 0].text(-0.2, 0.5, txt, transform=axs[r, 0].transAxes, 
                       rotation=90, va='center', ha='right', fontsize=viz.BASE_FONT_SIZE)

    fig.suptitle(f"Rotated Digit Reconstruction Analysis\n{subtitle}", fontsize=viz.TITLE_FONT_SIZE)
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
    
    # Robustness Curves (Complex)
    plot_complex_robustness_curve(
        angles, rob["mean_ssim_old"], rob["mean_ssim_new"], 
        "Mean SSIM", f"Robustness: SSIM vs Angle ({subset})", out_dir, f"08_robustness_ssim_vs_angle_{subset}"
    )
    plot_complex_robustness_curve(
        angles, rob["mean_psnr_old"], rob["mean_psnr_new"], 
        "Mean PSNR (dB)", f"Robustness: PSNR vs Angle ({subset})", out_dir, f"09_robustness_psnr_vs_angle_{subset}"
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
        
        all_emb = np.vstack([old_stack, new_stack])
        n_old = old_stack.shape[0]
        
        # Calculate Static vs Rotated
        try:
            static_angle_idx = np.where(angles == 0)[0][0]
            n_samples = n_old // len(angles)
            start = static_angle_idx * n_samples
            end = (static_angle_idx + 1) * n_samples
            
            # Static embeddings indices
            # old_static = old_stack[start:end]
            
            print(f"Computing projections for {subset} (N={all_emb.shape[0]})...")
            
            from sklearn.decomposition import PCA
            from sklearn.manifold import MDS, TSNE
            import warnings
            warnings.filterwarnings("ignore")
            
            # Helper to run reduction and plot both types
            def run_and_plot(reducer, name, stem_simple, stem_complex):
                emb_all = reducer.fit_transform(all_emb)
                
                # Split Old/New
                emb_old_all = emb_all[:n_old]
                emb_new_all = emb_all[n_old:]
                
                # Split Static/Rotated for Complex Plot
                emb_old_stat = emb_old_all[start:end]
                emb_new_stat = emb_new_all[start:end]
                
                # 1. Simple Plot (09)
                plot_embeddings(emb_old_all, emb_new_all, labels, name, out_dir, stem_simple, subset)
                
                # 2. Complex Plot (11)
                plot_extended_scatter(
                    emb_old_stat, emb_old_all,
                    emb_new_stat, emb_new_all,
                    labels[:n_samples], # Assume labels are repeated per angle, so take first chunk
                    name, out_dir, stem_complex
                )

            # A. PCA
            pca = PCA(n_components=2, random_state=42)
            run_and_plot(pca, "PCA", f"09a_scatter_pca_{subset}", f"11a_robustness_pca_{subset}")
            
            # B. MDS
            if all_emb.shape[0] <= 3000: # Threshold for slowness
                mds = MDS(n_components=2, random_state=42, normalized_stress='auto', n_init=1)
                run_and_plot(mds, "MDS", f"09b_scatter_mds_{subset}", f"11b_robustness_mds_{subset}")
            else:
                print("Skipping MDS (N too large)")
                
            # C. t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            run_and_plot(tsne, "t-SNE", f"09c_scatter_tsne_{subset}", f"11c_robustness_tsne_{subset}")
            
            # D. UMAP
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                # Note: User requested 131_robustness_umap, assuming typo for 11d or specific request.
                # Plan said 11d. Let's use 11d but also 131 if needed. I will stick to plan (11d) unless directed.
                # User prompted: "11a... and 131_robustness_umap". Okay, I will use 131 as requested.
                complex_stem = f"131_robustness_umap_{subset}"
                run_and_plot(reducer, "UMAP", f"09d_scatter_umap_{subset}", complex_stem)
            except ImportError:
                print("UMAP skipped")
                
        except Exception as e:
            print(f"Error computing extended figures: {e}")
            import traceback
            traceback.print_exc()

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
    cfg_dict = ckpt.get("cfg", {})
    model_cfg = CNNConfig(**cfg_dict)
    model = MNISTCNN(model_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # 2. Data
    _, test_loader = get_raw_dataloaders(PROJECT_ROOT, BATCH_SIZE, num_workers=0, device=torch.device(DEVICE))
    
    # 3. Transition Matrices
    matrices_dir = out_dir / "matrices"
    T_old = load_json_matrix(matrices_dir / "T_old_kxl.json") 
    T_new = load_json_matrix(matrices_dir / "T_new_kxl.json") 
    
    matrices = {"T_old": T_old, "T_new": T_new}
    return model, test_loader, matrices

def reconstruct_batch(model, W_kxl: np.ndarray, x: torch.Tensor) -> np.ndarray:
    x_norm = (x.to(DEVICE) - NORM_MEAN) / NORM_STD
    with torch.no_grad():
        A = model.penultimate(x_norm).cpu().numpy().astype(np.float64) 
    
    B_hat = A @ W_kxl 
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
        
        rec_old = reconstruct_batch(model, matrices["T_old"], x)
        rec_new = reconstruct_batch(model, matrices["T_new"], x)
        
        for i in range(x.shape[0]):
            orig = x_np[i]
            r_o = rec_old[i]
            r_n = rec_new[i]
            
            metrics["ssim"]["T_old"].append(ssim(orig, r_o, data_range=1.0))
            metrics["ssim"]["T_new"].append(ssim(orig, r_n, data_range=1.0))
            
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
    x_batch, y_batch = next(iter(loader))
    n_samples = 8
    x_sample = x_batch[:n_samples]
    
    recon = reconstruct_batch(model, W_kxl, x_sample)
    orig = x_sample.numpy().reshape(-1, 28, 28)
    
    fig, axs = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3.5))
    
    for i in range(n_samples):
        axs[0, i].imshow(orig[i], cmap="gray", vmin=0, vmax=1)
        axs[0, i].axis("off")
        if i == 0:
            axs[0, i].text(-0.2, 0.5, "Original", transform=axs[0, i].transAxes, 
                           rotation=90, va='center', ha='right', fontsize=viz.BASE_FONT_SIZE)

        axs[1, i].imshow(recon[i], cmap="gray", vmin=0, vmax=1)
        axs[1, i].axis("off")
        if i == 0:
            axs[1, i].text(-0.2, 0.5, "Reconstructed", transform=axs[1, i].transAxes, 
                           rotation=90, va='center', ha='right', fontsize=viz.BASE_FONT_SIZE)
            
    fig.suptitle(title, fontsize=viz.TITLE_FONT_SIZE)
    # Remove bold text
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
        
        plot_reconstructions(
            model, matrices["T_old"], test_loader, 
            "Reconstructions ($T_{old}$)", out_dir, "03_reconstructions_T_old"
        )
        
        plot_reconstructions(
            model, matrices["T_new"], test_loader, 
            "Reconstructions ($T_{new}$)", out_dir, "04_reconstructions_T_new"
        )
        
        metrics = compute_metrics(model, matrices, test_loader)
        
        plot_metric_histogram(
            metrics["ssim"]["T_old"], metrics["ssim"]["T_new"], 
            "SSIM", out_dir, "05_ssim_comparison"
        )
        
        plot_metric_histogram(
            metrics["psnr"]["T_old"], metrics["psnr"]["T_new"], 
            "PSNR (dB)", out_dir, "06_psnr_comparison"
        )
    except Exception as e:
        print(f"Warning: Could not generate on-the-fly figures (03-06): {e}")
        import traceback
        traceback.print_exc()

    # --- Part 2: Pre-computed subset figures (Figures 07-11/13) ---
    for subset in ["train", "test"]:
        generate_subset_figures(subset, matrices_dir, out_dir)

if __name__ == "__main__":
    out_dir_path = Path("outputs/mnist")
    if len(sys.argv) > 1:
        out_dir_path = Path(sys.argv[1])
        
    run_mnist_viz(out_dir_path)

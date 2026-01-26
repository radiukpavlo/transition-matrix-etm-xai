#!/usr/bin/env python3
"""Generate publication-ready figures for MNIST experiments (Figures 03-06).

Generates:
3. 03_reconstructions_T_old ([Original | Reconstructed])
4. 04_reconstructions_T_new ([Original | Reconstructed])
5. 05_ssim_comparison (Histogram/Boxplot)
6. 06_psnr_comparison (Histogram/Boxplot)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

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
from src.mnist.model import MNISTCNN, CNNConfig
from src.mnist.data import get_raw_dataloaders
from src.etm.utils import load_json_matrix

# Configuration
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
    
    # Process a subset or full set? Let's do full test set for histograms
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
    
    # specific digits to show? Let's just take the first 10 distinct digits if possible, or just first 10
    # Let's simple take first 8 samples for a clean 2x4 grid or 1x8
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

def run_viz(out_dir: Path):
    viz.configure_style()
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
    
    # Calculate metrics
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

if __name__ == "__main__":
    out_dir_path = Path("outputs/mnist")
    if len(sys.argv) > 1:
        out_dir_path = Path(sys.argv[1])
        
    run_viz(out_dir_path)

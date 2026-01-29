#!/usr/bin/env python3
"""Generate publication-ready figures for MNIST experiments.

Follows strict visual style guidelines and relies on pre-computed data.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import visualization utilities
from src.mnist import mnist_viz_utils as viz
# Import rotation utility
from src.mnist.rotate import rotate_batch

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
    
    # Ax2: Advantage Ratio
    ratio = v_new / v_old
    ratio_label = "Advantage ($T_{new} / T_{old}$)"
    
    ax2.plot(a, ratio, marker='s', color='#800080', linewidth=3, markersize=8, label=ratio_label)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=2)
    ax2.set_xlabel("Rotation angle (deg)")
    ax2.set_ylabel("Advantage Ratio")
    ax2.legend(loc="upper center", frameon=False)
    ax2.grid(True, **viz.MAJOR_GRID_STYLE)
    ax2.set_xlim(-90, 90)
    
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
            s=60, alpha=0.3, # Increased size slightly for visibility
            edgecolor='none',
            label="Rotated",
            zorder=1
        )
        
        # 2. Static (Foreground)
        ax.scatter(
            static_2d[:, 0], static_2d[:, 1],
            c=labels, cmap="tab10",
            s=150, alpha=1.0,
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
    
    # Add Colorbar for class identification "outside the figure plot"
    # To match Figure 09 style:
    sm = plt.cm.ScalarMappable(cmap="tab10", norm=plt.Normalize(vmin=0, vmax=9))
    sm.set_array([])
    # fraction=0.02, pad=0.04 puts it nicely on the right
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Digit Class")
    cbar.set_ticks(range(10))
    
    plt.subplots_adjust(bottom=0.15)
    viz.save_figure(fig, out_dir, stem)

def plot_symmetry_bar_train_test(train_metrics: dict, test_metrics: dict, out_dir: Path, stem: str) -> None:
    sym_old_train = float(train_metrics["symmetry_error_fro"]["old"])
    sym_new_train = float(train_metrics["symmetry_error_fro"]["new"])
    sym_old_test = float(test_metrics["symmetry_error_fro"]["old"])
    sym_new_test = float(test_metrics["symmetry_error_fro"]["new"])

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
    groups = ["Train", "Test"]
    x = np.arange(len(groups))
    width = 0.35
    ax.bar(x - width / 2, [sym_old_train, sym_old_test], width, label="Baseline $T_{old}$", color=viz.COLOR_CYCLE[0])
    ax.bar(x + width / 2, [sym_new_train, sym_new_test], width, label="ETM $T_{new}$", color=viz.COLOR_CYCLE[1])
    ax.set_ylabel(r"$\|T J^A - J^B T\|_F$")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Symmetry Defect (log scale)")
    ax.grid(axis='y', **viz.MAJOR_GRID_STYLE)
    
    viz.save_figure(fig, out_dir, stem)


def plot_lambda_sweep_symerr(lambda_sweep: dict, out_dir: Path, stem: str, highlight_lambda: float = 0.5) -> None:
    rows = lambda_sweep["rows"]
    lambdas = [float(r["lambda"]) for r in rows]
    sym_err = [float(r["symmetry_error"]) for r in rows]
    x_vals = [1e-4 if l == 0.0 else l for l in lambdas]  # visualize 0 on log axis

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
    ax.plot(x_vals, sym_err, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ (log scale)")
    ax.set_ylabel(r"$\|T J^A - J^B T\|_F$")
    ax.set_title(r"MNIST: Symmetry Defect vs. $\lambda$")
    ax.grid(True, **viz.MAJOR_GRID_STYLE)

    for xv, l, se in zip(x_vals, lambdas, sym_err):
        if abs(l - highlight_lambda) < 1e-12:
            ax.scatter([xv], [se], s=40, zorder=5)
            ax.annotate(rf"$\lambda={highlight_lambda}$", (xv, se), textcoords="offset points",
                        xytext=(6, 6), fontsize=7)

    viz.save_figure(fig, out_dir, stem)


def plot_generator_singular_values(gen_sv: dict, out_dir: Path, stem: str) -> None:
    svA = np.array(gen_sv["GA_singular_values"], dtype=float)
    svB = np.array(gen_sv["GB_singular_values"], dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
    ax.semilogy(svA, label=r"$G_A$ fit")
    ax.semilogy(svB, label=r"$G_B$ fit")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value (log)")
    ax.set_title("Generator Estimation: Singular Value Spectra")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, **viz.MAJOR_GRID_STYLE)
    
    viz.save_figure(fig, out_dir, stem)


def plot_extended_robustness_curves(metrics_test: dict, out_dir: Path) -> None:
    """Generate extended robustness curves (Test set): 14a, 14b, 14c, 14d."""
    angles = np.array(metrics_test["robustness"]["angles_deg"], dtype=float)
    ssim_old = np.array(metrics_test["robustness"]["mean_ssim_old"], dtype=float)
    ssim_new = np.array(metrics_test["robustness"]["mean_ssim_new"], dtype=float)
    psnr_old = np.array(metrics_test["robustness"]["mean_psnr_old"], dtype=float)
    psnr_new = np.array(metrics_test["robustness"]["mean_psnr_new"], dtype=float)

    # 14a: SSIM
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
    ax.plot(angles, ssim_old, label="Baseline SSIM")
    ax.plot(angles, ssim_new, label="ETM SSIM")
    ax.set_xlabel("Rotation angle (deg)")
    ax.set_ylabel("Mean SSIM")
    ax.set_title("MNIST Robustness (Test): SSIM vs Rotation, ±90°")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, **viz.MAJOR_GRID_STYLE)
    viz.save_figure(fig, out_dir, "14a_robustness_ssim_vs_angle_test_90deg")

    # 14b: PSNR
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
    ax.plot(angles, psnr_old, label="Baseline PSNR")
    ax.plot(angles, psnr_new, label="ETM PSNR")
    ax.set_xlabel("Rotation angle (deg)")
    ax.set_ylabel("Mean PSNR")
    ax.set_title("MNIST Robustness (Test): PSNR vs Rotation, ±90°")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, **viz.MAJOR_GRID_STYLE)
    viz.save_figure(fig, out_dir, "14b_robustness_psnr_vs_angle_test_90deg")

    # 14c: Delta SSIM
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
    ax.axhline(0.0, linewidth=0.8, color='black')
    ax.plot(angles, ssim_new - ssim_old, marker="o")
    ax.set_xlabel("Rotation angle (deg)")
    ax.set_ylabel(r"$\Delta$SSIM (ETM - Baseline)")
    ax.set_title("MNIST Robustness Gain: SSIM Difference vs Rotation")
    ax.grid(True, **viz.MAJOR_GRID_STYLE)
    viz.save_figure(fig, out_dir, "14c_delta_ssim_vs_angle_test_90deg")

    # 14d: Delta PSNR
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
    ax.axhline(0.0, linewidth=0.8, color='black')
    ax.plot(angles, psnr_new - psnr_old, marker="o")
    ax.set_xlabel("Rotation angle (deg)")
    ax.set_ylabel(r"$\Delta$PSNR (ETM - Baseline)")
    ax.set_title("MNIST Robustness Gain: PSNR Difference vs Rotation")
    ax.grid(True, **viz.MAJOR_GRID_STYLE)
    viz.save_figure(fig, out_dir, "14d_delta_psnr_vs_angle_test_90deg")


def plot_mega_panel(out_dir: Path, stem: str) -> None:
    # Requires availability of specific source figures in PNG format
    # Source paths (we rely on the 'png' folder we just populated)
    # Layout: 2 rows x 3 cols
    
    png_dir = out_dir / "figures" / "png"
    
    # Files expected to exist:
    # (a) 05_ssim_comparison.png
    # (b) 06_psnr_comparison.png
    # (c) 12a_symmetry_bar_train_test.png
    # (d) 08_robustness_ssim_vs_angle_test.png (Assuming this is what was intended, or 14a?)
    #     Note: User original code referenced "08_robustness_ssim_vs_angle_test.png" 
    #     which comes from generate_subset_figures("test").
    # (e) 09_robustness_psnr_vs_angle_test.png
    # (f) 12b_lambda_sweep_symerr.png
    
    panel_sources = [
        "05_ssim_comparison.png",
        "06_psnr_comparison.png",
        "12a_symmetry_bar_train_test.png",
        "08_robustness_ssim_vs_angle_test.png",
        "09_robustness_psnr_vs_angle_test.png",
        "12b_lambda_sweep_symerr.png"
    ]
    
    panel_paths = [png_dir / fname for fname in panel_sources]
    
    # Check existence
    for p in panel_paths:
        if not p.exists():
            print(f"Warning: Missing panel source for mega figure: {p}")
            return

    panel_w, panel_h = 900, 600
    cols, rows = 3, 2

    imgs = [Image.open(p).convert("RGB").resize((panel_w, panel_h)) for p in panel_paths]
    mega = Image.new("RGB", (panel_w * cols, panel_h * rows), color=(255, 255, 255))

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    try:
        # Try a standard font, else default
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        try:
             font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
        except IOError:
             font = ImageFont.load_default()

    draw = ImageDraw.Draw(mega)
    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        mega.paste(im, (c * panel_w, r * panel_h))
        # small white background for label
        draw.rectangle([c * panel_w + 10, r * panel_h + 10, c * panel_w + 110, r * panel_h + 60],
                       fill=(255, 255, 255))
        draw.text((c * panel_w + 20, r * panel_h + 15), labels[idx], fill=(0, 0, 0), font=font)

    # Save manually to the 3 folders using logic similar to viz.save_figure but for PIL image
    # Note: viz.save_figure works on matplotlib figures. Here we have a PIL image.
    # We will replicate the saving logic.
    formats = ["png", "pdf"] # SVG not directly supported by PIL save easily without conversion, leaving as PNG/PDF
    # Actually user requested SVG for everything. PIL doesn't save SVG. 
    # But since this is a raster composition, wrapping it in SVG is just embedding the raster.
    # For now, we will save as PNG and PDF.
    
    base_figures_dir = out_dir / "figures"
    for fmt in ["png", "pdf"]:
        fmt_dir = base_figures_dir / fmt
        fmt_dir.mkdir(parents=True, exist_ok=True)
        #For PDF, resolution arg is different usually, but save handles it
        mega.save(fmt_dir / f"{stem}.{fmt}") 
        
    print(f"Generated Mega Panel: {stem}")
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM_MEAN = 0.1307
NORM_STD = 0.3081

# Models and Data Loading (Needed for Figures 03-06 and 10)
from src.mnist.model import MNISTCNN, CNNConfig
from src.mnist.data import get_raw_dataloaders
from src.etm.utils import load_json_matrix

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
    """Reconstruct batch x (N, 1, 28, 28) using transition W (k, l)."""
    # Normalize for feature extraction
    x_norm = (x.to(DEVICE) - NORM_MEAN) / NORM_STD
    with torch.no_grad():
        A = model.penultimate(x_norm).cpu().numpy().astype(np.float64) # (N, k)
    
    B_hat = A @ W_kxl # (N, l)
    recon = np.clip(B_hat.reshape(-1, 28, 28), 0.0, 1.0)
    return recon

def generate_chaos_variants(
    model, 
    test_loader, 
    matrices: Dict[str, np.ndarray], 
    out_dir: Path, 
    subset: str
) -> None:
    """
    Generate Figures 10a (0deg), 10b (+-45deg), 10c (+-90deg) on the fly.
    Features:
    - Large visible digits.
    - 4 Rows: Input, Old, New, Difference (Improvement).
    - Distinct Axes.
    """
    # 1. Select distinct samples (one per digit 0-9)
    # We iterate loader until we find one of each.
    samples_by_digit = {}
    for x, y in test_loader:
        for i in range(len(y)):
            lbl = int(y[i])
            if lbl not in samples_by_digit:
                samples_by_digit[lbl] = x[i:i+1] # Keep as (1, 1, 28, 28)
            if len(samples_by_digit) == 10:
                break
        if len(samples_by_digit) == 10:
            break
            
    # Sort 0-9
    digits = [samples_by_digit[i] for i in range(10)]
    input_batch = torch.cat(digits, dim=0).to(DEVICE) # (10, 1, 28, 28)
    
    # 2. Define Variants
    variants = [
        ("10a", 0.0),
        ("10b", 45.0),
        ("10c", 90.0)
    ]
    
    # Fixed seed for angle signs
    rng = np.random.default_rng(42)
    
    for stem_prefix, angle_mag in variants:
        stem = f"{stem_prefix}_chaos_figure_{subset}"
        print(f"Generating {stem} (Angle magnitude: {angle_mag})...")
        
        # Prepare Rotated Inputs
        if angle_mag == 0:
            angles = np.zeros(10)
        else:
            # Random sign per digit
            signs = rng.choice([-1, 1], size=10)
            angles = signs * angle_mag
            
        theta = torch.tensor(angles * math.pi / 180.0, device=DEVICE, dtype=torch.float32)
        rotated_inputs = rotate_batch(input_batch, theta).detach() # (10, 1, 28, 28)
        
        # Reconstruct
        recon_old = reconstruct_batch(model, matrices["T_old"], rotated_inputs.cpu())
        recon_new = reconstruct_batch(model, matrices["T_new"], rotated_inputs.cpu())
        orig_np = rotated_inputs.cpu().numpy().reshape(-1, 28, 28)
        
        # Compute Visual Advantage (Difference Row)
        # Improvement = Error_Old - Error_New
        # Error = |Input - Recon|
        # Positive (Green) means New has less error (Better).
        err_old = np.abs(orig_np - recon_old)
        err_new = np.abs(orig_np - recon_new)
        improvement = err_old - err_new
        
        # Plotting
        n = 10
        fig, axs = plt.subplots(4, n, figsize=(n * 2.0, 9)) # Taller/Wider for clarity
        
        row_titles = ["Input", "Recon ($T_{old}$)", "Recon ($T_{new}$)", "Advantage\n($T_{new}$ vs $T_{old}$)"]
        
        # Global color scale for difference map
        # centered at 0. Green for positive (better), Red for negative (worse)
        div_norm = mcolors.TwoSlopeNorm(vmin=improvement.min(), vcenter=0., vmax=improvement.max())
        cmap_diff = plt.cm.RdYlGn # Red-Yellow-Green
        
        for i in range(n):
            # Row 0: Input
            axs[0, i].imshow(orig_np[i], cmap="gray", vmin=0, vmax=1)
            axs[0, i].axis("off")
            title_text = f"{i}\n({angles[i]:.0f}°)" if angle_mag > 0 else f"{i}"
            axs[0, i].set_title(title_text, fontsize=viz.BASE_FONT_SIZE + 2)
            
            # Row 1: Old
            axs[1, i].imshow(recon_old[i], cmap="gray", vmin=0, vmax=1)
            axs[1, i].axis("off")
            
            # Row 2: New
            axs[2, i].imshow(recon_new[i], cmap="gray", vmin=0, vmax=1)
            axs[2, i].axis("off")
            
            # Row 3: Difference (Advantage)
            im_diff = axs[3, i].imshow(improvement[i], cmap=cmap_diff, norm=div_norm)
            axs[3, i].axis("off")
            
        # Row Labels (Far Left)
        for r, txt in enumerate(row_titles):
            # Place text reasonably to the left
            axs[r, 0].text(-0.25, 0.5, txt, transform=axs[r, 0].transAxes, 
                           rotation=90, va='center', ha='right', fontsize=viz.BASE_FONT_SIZE)
        
        # Add colorbar for the difference row
        # Position it to the right of the bottom row
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.15]) # [left, bottom, width, height]
        cbar = fig.colorbar(im_diff, cax=cbar_ax, orientation="vertical")
        cbar.set_label("Improvement", fontsize=viz.LEGEND_FONT_SIZE)
        cbar.set_ticks([improvement.min(), 0, improvement.max()])
        cbar.ax.set_yticklabels(["Worse", "0", "Better"])
        
        fig.suptitle(f"Reconstruction & Robustness Analysis ({subset})\nAngle Magnitude: {angle_mag}°", fontsize=viz.TITLE_FONT_SIZE)
        
        # Separate axes clearly (handled by subplots, but we ensure spacing)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        viz.save_figure(fig, out_dir, stem)


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

    # --- Part 1: On-the-fly generation ---
    # We need model interactions for Figures 03-06 (Recons/Hist) AND 10 (Chaos Variants)
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
        
        # 10. Chaos Variants (10a, 10b, 10c) for "test" subset (represented by loader)
        generate_chaos_variants(model, test_loader, matrices, out_dir, "test")
        
    except Exception as e:
        print(f"Warning: Could not generate on-the-fly figures (03-06, 10): {e}")
        import traceback
        traceback.print_exc()

    # --- Part 2: Pre-computed subset figures (Figures 07-11/13 etc) ---
    for subset in ["train", "test"]:
        generate_subset_figures(subset, matrices_dir, out_dir)
        
    # --- Part 3: Mega Panel & Extra Figures (from legacy generate_mnist_mega...) ---
    try:
        print("Generating consolidated mega figures...")
        train_metrics = load_json(matrices_dir / "mnist_metrics_train.json")
        test_metrics = load_json(matrices_dir / "mnist_metrics_test.json")
        lambda_sweep = load_json(matrices_dir / "lambda_sweep.json")
        gen_sv = load_json(matrices_dir / "generator_singular_values.json")
        
        # 12a. Symmetry Bar (Train vs Test)
        plot_symmetry_bar_train_test(
            train_metrics, test_metrics, 
            out_dir, "12a_symmetry_bar_train_test"
        )
        
        # 12b. Lambda Sweep
        plot_lambda_sweep_symerr(
            lambda_sweep, out_dir, "12b_lambda_sweep_symerr"
        )
        
        # 13. Generator SV
        plot_generator_singular_values(
            gen_sv, out_dir, "13_generator_singular_values"
        )
        
        # 14. Extended Robustness (Test 90deg)
        plot_extended_robustness_curves(test_metrics, out_dir)
        
        # 12. Mega Panel (Must be last to ensure inputs exist)
        plot_mega_panel(out_dir, "12_mega_mnist_panel")
        
    except FileNotFoundError as e:
        print(f"Skipping Mega Figures due to missing data: {e}")
    except Exception as e:
        print(f"Error generating mega figures: {e}")
        import traceback
        traceback.print_exc()

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
            print(f"Computing projections for {subset} (N={all_emb.shape[0]})...")
            
            from sklearn.decomposition import PCA
            from sklearn.manifold import MDS, TSNE
            import warnings
            warnings.filterwarnings("ignore")
            
            # Helper to run reduction and plot both types
            def run_and_plot(reducer, name, stem_simple, stem_complex):
                # Ensure reproducibility of reducer
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
                    labels[:n_samples], 
                    name, out_dir, stem_complex
                )

            # A. PCA
            pca = PCA(n_components=2, random_state=42)
            run_and_plot(pca, "PCA", f"09a_scatter_pca_{subset}", f"11a_robustness_pca_{subset}")
            
            # B. MDS
            if all_emb.shape[0] <= 3000:
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
                complex_stem = f"11d_robustness_umap_{subset}" # Using standard naming scheme
                run_and_plot(reducer, "UMAP", f"09d_scatter_umap_{subset}", complex_stem)
            except ImportError:
                print("UMAP skipped")
                
        except Exception as e:
            print(f"Error computing extended figures: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    out_dir_path = Path("outputs/mnist")
    if len(sys.argv) > 1:
        out_dir_path = Path(sys.argv[1])
        
    run_mnist_viz(out_dir_path)

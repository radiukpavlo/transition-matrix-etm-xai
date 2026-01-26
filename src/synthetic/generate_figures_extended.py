#!/usr/bin/env python3
"""
Extended Synthetic Experiments Visualization.

Generates Figures 13a-13d: Robustness comparing Static vs Rotated embeddings
using dimensionality reduction (PCA, MDS, t-SNE, UMAP).

Angle range: [-120, +120] (degrees)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, List, Tuple, Dict
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.linear_model import LinearRegression

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def plot_error_vs_angle(metrics: Dict[str, Any], out_dir: Path) -> None:
    angles_deg = np.array(metrics["angles_deg"])
    results_old = np.array(metrics["mse_old"])
    results_new = np.array(metrics["mse_new"])
    start_deg = metrics["start_deg"]
    end_deg = metrics["end_deg"]

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
    print(f"Saved 12_error_vs_angle")


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
    # Get handles from one ax
    handles, lbls = axes[0].get_legend_handles_labels()
    # Sort: Class 0 Pred, Class 0 Ideal, Class 1 Pred, ... 
    # Ideal (Target) vs Predicted (Rotated). Predicted is "Rotated" concept here (LIGHT colors).
    # "Ideal" starts with I, "Predicted" starts with P. 
    # We want Class grouping.
    sorted_pairs = sorted(zip(lbls, handles), key=lambda x: x[0].split()[-1] + x[0].split()[0])
    
    fig.legend(
        [h for l, h in sorted_pairs], 
        [l for l, h in sorted_pairs], 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=3, # 3 classes * 2 types = 6 items
        frameon=False,
    )
    
    plt.subplots_adjust(bottom=0.20, wspace=0.2)

    save_figure(fig, out_dir, "11_displacement_vectors")
    print(f"Saved 11_displacement_vectors")



def generate_extended_figures():
    out_dir = PROJECT_ROOT / "outputs" / "synthetic"
    matrices_dir = out_dir / "matrices"
    
    # 1. Load Data
    print("Loading matrices...")
    A = load_json_matrix(PROJECT_ROOT / "inputs" / "synthetic" / "A.json")
    B = load_json_matrix(PROJECT_ROOT / "inputs" / "synthetic" / "B.json")
    
    W_old = load_json_matrix(matrices_dir / "T_old_ls_kxl.json") # k x l
    
    if (matrices_dir / "T_new.json").exists():
        T_new = load_json_matrix(matrices_dir / "T_new.json") # l x k
        W_new = T_new.T
    else:
        W_new = load_json_matrix(matrices_dir / "T_new_kxl.json")

    # 2. Re-estimate Decoders
    print("Estimating generators/decoders...")
    random_state = 42
    # Estimate A_2d, B_2d
    A_2d = mds_2d(A, random_state=random_state, normalized_stress=False)
    B_2d = mds_2d(B, random_state=random_state, normalized_stress=False)
    
    decoderA = LinearRegression().fit(A_2d, A)
    # decoderB not strictly needed for visualization if we use A_rot -> T -> B_pred
    
    # 3. Generate Rotations
    print("Generating rotations (-120 to +120)...")
    angles_deg = np.arange(-120, 120 + 1e-9, 15)
    angles_rad = np.radians(angles_deg)
    
    B_old_rots = []
    B_new_rots = []
    
    for angle in angles_rad:
        A2 = rotate_2d(A_2d, angle)
        # Decode A_rot
        A_rot = decoderA.predict(A2)
        
        # Predict B
        B_old_rots.append(A_rot @ W_old)
        B_new_rots.append(A_rot @ W_new)
        
    B_old_rots_stacked = np.vstack(B_old_rots)
    B_new_rots_stacked = np.vstack(B_new_rots)
    
    # Static Baselines (No rotation)
    B_old_static = A @ W_old
    B_new_static = A @ W_new
    
    # 4. Dimensionality Reduction
    # Stack everything to find a common embedding space
    # Order: [B_old_static, B_new_static, B_old_rots, B_new_rots]
    all_data = np.vstack([
        B_old_static, 
        B_new_static, 
        B_old_rots_stacked, 
        B_new_rots_stacked
    ])
    
    # Indices to slice back later
    n = B.shape[0] # 15
    n_rot = B_old_rots_stacked.shape[0] # 15 * num_angles
    
    idx_old_static = slice(0, n)
    idx_new_static = slice(n, 2*n)
    idx_old_rot = slice(2*n, 2*n + n_rot)
    idx_new_rot = slice(2*n + n_rot, 2*n + 2*n_rot)
    
    labels = _labels_for_15()
    labels_rot = np.tile(labels, len(angles_deg))
    
    print(f"Total samples for reduction: {all_data.shape[0]}")
    
    # --- STYLE CONSTANTS ---
    # Imported from viz_utils
    
    # --- PLOTTING HELPER ---
    def plot_comparison(emb: np.ndarray, method_name: str, stem: str):
        # Extract parts
        e_old_static = emb[idx_old_static]
        e_new_static = emb[idx_new_static]
        e_old_rot = emb[idx_old_rot]
        e_new_rot = emb[idx_new_rot]
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 8)) # Increased from (14, 6)
        
        # Helper to plot one side
        def _plot_side(ax, e_static, e_rot, title):
            # 1. Plot Rotated (Background)
            for c in np.unique(labels):
                color = LIGHT_COLORS[c % len(LIGHT_COLORS)]
                marker = CLASS_MARKERS[c % len(CLASS_MARKERS)]
                
                idx_rot = labels_rot == c
                ax.scatter(
                    e_rot[idx_rot, 0], e_rot[idx_rot, 1],
                    s=MARKER_SIZE_SMALL, # Was 40, now 80
                    alpha=0.6,
                    color=color,
                    edgecolor=CLASS_COLORS[c % len(CLASS_COLORS)], # Slight edge for contrast? Or strict light?
                    # User said: "light red ... represent rotated dots"
                    # User didn't say no edge. I'll use thin edge of main color to help visibility vs white, 
                    # OR just the light color. "Light Yellow" on white is invisible without edge.
                    # I will add a thin edge matching the vivid color.
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
                    s=MARKER_SIZE_LARGE, # Was 120, now 240
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
        # We need handles for: Class 0, 1, 2 (Static) AND Class 0, 1, 2 (Rotated)
        # They should be in axs[0] (or axs[1])
        handles, lbls = axs[0].get_legend_handles_labels()
        
        # Sort by Class Num then Type (Rotated/Static)
        # key=lambda x: x[0].split()[-1] + x[0].split()[0]
        # "Rotated (Class 0)" -> "0)Rotated"
        # "Static (Class 0)" -> "0)Static"
        # R < S, so Rotated comes first.
        sorted_pairs = sorted(zip(lbls, handles), key=lambda x: x[0].split()[-1] + x[0].split()[0])
        
        fig.legend(
            [h for l, h in sorted_pairs], 
            [l for l, h in sorted_pairs], 
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.05), 
            ncol=3, # 3 classes * 2 variants = 6 items. 3 cols = 2 rows.
            frameon=False,
        )
        
        plt.subplots_adjust(bottom=0.20, wspace=0.2)
        
        save_figure(fig, out_dir, stem)
        print(f"Saved {stem}")

    # A. PCA
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=random_state)
    all_pca = pca.fit_transform(all_data)
    plot_comparison(all_pca, "PCA", "13a_robustness_pca")
    
    # B. MDS
    print("Running MDS...")
    # MDS is slow on large N, but N here is roughly (15 + 15 + 15*17 + 15*17) ~ 540 samples. Doable.
    mds = MDS(n_components=2, random_state=random_state, normalized_stress=False)
    all_mds = mds.fit_transform(all_data)
    plot_comparison(all_mds, "MDS", "13b_robustness_mds")
    
    # C. t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    all_tsne = tsne.fit_transform(all_data)
    plot_comparison(all_tsne, "t-SNE", "13c_robustness_tsne")
    
    # D. UMAP
    try:
        import umap
        print("Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        all_umap = reducer.fit_transform(all_data)
        plot_comparison(all_umap, "UMAP", "13d_robustness_umap")
    except ImportError:
        print("UMAP not installed, skipping.")
    
    # 5. Plot Extended Figures (11, 12)
    # Load data saved by run_extended.py
    metrics_path = matrices_dir / "robustness_metrics_extended.json"
    demo_path = matrices_dir / "displacement_test_data.json"

    if metrics_path.exists():
        print("Plotting Error vs Angle...")
        metrics = load_json(metrics_path)
        plot_error_vs_angle(metrics, out_dir)
    else:
        print(f"Warning: {metrics_path} not found. Skipping Figure 12.")

    if demo_path.exists():
        print("Plotting Displacement Vectors...")
        demo_data = load_json(demo_path)
        plot_displacement_vectors(demo_data, out_dir)
    else:
        print(f"Warning: {demo_path} not found. Skipping Figure 11.")

if __name__ == "__main__":
    generate_extended_figures()

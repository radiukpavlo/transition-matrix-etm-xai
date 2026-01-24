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
from typing import Any, List, Tuple

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
from src.etm.synthetic import mds_2d, rotate_2d, _labels_for_15

# --- STYLE CONFIGURATION ---
BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 2
TITLE_FONT_SIZE = BASE_FONT_SIZE + 2
LEGEND_FONT_SIZE = max(BASE_FONT_SIZE - 2, 10)
DARK_EDGE_COLOR = "#1f2937"
DARK_TEXT_COLOR = "#0f172a"
MAJOR_GRID_STYLE = {"color": "#c7ccd6", "linewidth": 0.9, "alpha": 0.7}

mpl.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "font.weight": "bold",
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.titleweight": "bold",
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.framealpha": 0.92,
        "axes.edgecolor": DARK_EDGE_COLOR,
        "axes.labelcolor": DARK_TEXT_COLOR,
        "axes.titlecolor": DARK_TEXT_COLOR,
        "savefig.dpi": 300,
    }
)

def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save figure in PDF, SVG, and PNG formats."""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    for ext in ("pdf", "svg", "png"):
        fig.savefig(figures_dir / f"{stem}.{ext}", format=ext, bbox_inches="tight")
    plt.close(fig)

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
    
    # Vivid Color Palette for Classes (0, 1, 2)
    # 0: Vivid Red, 1: Vivid Lime/Green, 2: Vivid Blue
    CLASS_COLORS = ["#FF1493", "#00C957", "#1E90FF"]  # DeepPink, EmeraldGreen, DodgerBlue
    
    # --- PLOTTING HELPER ---
    def plot_comparison(emb: np.ndarray, method_name: str, stem: str):
        # Extract parts
        e_old_static = emb[idx_old_static]
        e_new_static = emb[idx_new_static]
        e_old_rot = emb[idx_old_rot]
        e_new_rot = emb[idx_new_rot]
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Helper to plot one side
        def _plot_side(ax, e_static, e_rot, title):
            for c in np.unique(labels):
                color = CLASS_COLORS[c % len(CLASS_COLORS)]
                
                # Rotated points (Background/Context) - Larger and clearer
                idx_rot = labels_rot == c
                ax.scatter(
                    e_rot[idx_rot, 0], e_rot[idx_rot, 1],
                    s=40,              # Increased from 10
                    alpha=0.4,         # More visible
                    color=color,
                    label=f"Rotated (Class {c})",
                    marker='x',
                    linewidth=1.2
                )
                
                # Static points (Foreground/Focus) - Very large and distinct
                idx_stat = labels == c
                ax.scatter(
                    e_static[idx_stat, 0], e_static[idx_stat, 1],
                    s=180,              # Increased from 80
                    alpha=1.0,
                    color=color,
                    edgecolor='black',  # High contrast edge
                    linewidth=2.0,      # Thicker edge
                    label=f"Static (Class {c})",
                    marker='o',
                    zorder=10           # Ensure on top
                )
            
            ax.set_title(title, fontsize=TITLE_FONT_SIZE)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.grid(True, **MAJOR_GRID_STYLE)
            
            # Custom Legend to avoid duplication and clutter
            # We only show one entry per class (Static) to keep it clean, 
            # or separate Rotated/Static if needed. Ideally just Class separation colors.
            # But the user might want to distinguish. Let's keep full legend but de-duplicate.
            # Actually, grouping by class is better.
            handles, lbls = ax.get_legend_handles_labels()
            by_label = dict(zip(lbls, handles))
            # Sort by label to keep Rotated/Static groups or Class groups
            sorted_keys = sorted(by_label.keys()) 
            ax.legend([by_label[k] for k in sorted_keys], sorted_keys, loc='best')

        # Left: Old Comparison
        _plot_side(axs[0], e_old_static, e_old_rot, f"{method_name}: $B^*_{{old}}$ (Rotated vs Static)")
        
        # Right: New Comparison
        _plot_side(axs[1], e_new_static, e_new_rot, f"{method_name}: $B^*_{{new}}$ (Rotated vs Static)")
        
        _save_figure(fig, out_dir, stem)
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

if __name__ == "__main__":
    generate_extended_figures()

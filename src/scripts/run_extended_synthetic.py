#!/usr/bin/env python3
"""
Extended Synthetic Experiments (Stress Test & New Viz).

1.  Loads existing synthetic matrices.
2.  Performs stress test with larger rotation angles (e.g. -120 to +120).
3.  Generates new visualizations:
    -   Displacement Vectors (Ideal vs Predicted).
    -   Error vs Angle plots.
"""

import math
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.linear_model import LinearRegression

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etm.utils import load_json_matrix
from src.etm.synthetic import mds_2d, rotate_2d, mse_fid
from src.etm.viz_utils import (
    configure_style, save_figure,
    CLASS_COLORS, LIGHT_COLORS, CLASS_MARKERS,
    MAJOR_GRID_STYLE, TITLE_FONT_SIZE
)

# --- CONFIG ---
# We'll read from configs/synthetic.yaml just for the angle range and step
CONFIG_PATH = PROJECT_ROOT / "configs" / "synthetic.yaml"

def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# --- PLOTTING STYLE ---
configure_style()


def run_extended_experiments():
    cfg = _load_config()
    out_dir = PROJECT_ROOT / "outputs" / "synthetic"
    matrices_dir = out_dir / "matrices"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading matrices...")
    A = load_json_matrix(PROJECT_ROOT / "inputs" / "synthetic" / "A.json")
    B = load_json_matrix(PROJECT_ROOT / "inputs" / "synthetic" / "B.json")
    
    # Load learned matrices
    # T_old_ls_kxl.json -> W_old (k x l)
    W_old = load_json_matrix(matrices_dir / "T_old_ls_kxl.json")

    # T_new.json (l x k) -> W_new (k x l)
    # Check if T_new.json exists, otherwise use T_new_kxl.json
    if (matrices_dir / "T_new.json").exists():
        T_new = load_json_matrix(matrices_dir / "T_new.json")
        W_new = T_new.T
    else:
        # Fallback if T_new.json not found but T_new_kxl exists
         W_new = load_json_matrix(matrices_dir / "T_new_kxl.json")

    # Re-estimate generators/decoders to get A_2d, B_2d and decoders
    # We could load them if saved, but re-computing is fast and safer unless seeded perfectly.
    # To ensure consistency, let's use the same random state from config.
    # Note: src.etm.synthetic.estimate_generator_via_bridge does the whole thing.
    # We just need A_2d and the decoders.
    
    print("Re-estimating decoders for simulation...")
    random_state = cfg.get("mds_random_state", 42)
    normalized_stress = cfg.get("mds_normalized_stress", False)

    # A -> MDS -> A_2d -(dec)-> A_hat
    A_2d = mds_2d(A, random_state=random_state, normalized_stress=normalized_stress)
    decoderA = LinearRegression().fit(A_2d, A)

    # B -> MDS -> B_2d -(dec)-> B_hat
    B_2d = mds_2d(B, random_state=random_state, normalized_stress=normalized_stress)
    decoderB = LinearRegression().fit(B_2d, B)

    # --- 1. EXTENDED ROTATION LOOP ---
    range_deg = cfg.get("robustness_range_degrees", [-60, 60]) # Fallback if not set
    step_deg = cfg.get("robustness_step_degrees", 5)
    
    start_deg, end_deg = range_deg[0], range_deg[1]
    # np.arange excludes endpoint, so add a tiny bit
    angles_deg = np.arange(start_deg, end_deg + 1e-9, step_deg)
    angles_rad = np.radians(angles_deg)

    print(f"Running stress test for angles: {start_deg} to {end_deg} (step {step_deg})")

    results_old = [] # (angle, mse)
    results_new = [] # (angle, mse)

    # To visualize "Displacement Vectors", we need a specific (large) angle
    # Let's pick the max angle in the positive direction for the demo plot.
    demo_angle_deg = end_deg 
    demo_idx = -1 # Index of the demo angle
    
    # Store demo data: targets and predictions for Old/New
    demo_data = {} 

    for i, angle in enumerate(angles_rad):
        # 1. Rotate latent space
        A2 = rotate_2d(A_2d, angle)
        B2 = rotate_2d(B_2d, angle)
        
        # 2. Decode to ambient space ("Ground Truth" for this rotation)
        A_rot = decoderA.predict(A2)
        B_target = decoderB.predict(B2)
        
        # 3. Predict using Transition Matrix
        B_pred_old = A_rot @ W_old
        B_pred_new = A_rot @ W_new
        
        # 4. Compute MSE
        mse_old = mse_fid(B_target, B_pred_old)
        mse_new = mse_fid(B_target, B_pred_new)
        
        results_old.append(mse_old)
        results_new.append(mse_new)

        # Store data for visualization if it's the demo angle
        if i == len(angles_rad) - 1: # Just taking the last one for now or match specific
             # Wait, finding exactly demo_angle_deg might be safer by value
             pass
        
        if abs(angles_deg[i] - demo_angle_deg) < 1e-5:
            demo_data = {
                "angle_deg": angles_deg[i],
                "B_target": B_target,
                "B_pred_old": B_pred_old,
                "B_pred_new": B_pred_new
            }

    # Vivid Colors
    # Old/New line plots
    COLOR_OLD = "#FF1493" # DeepPink
    COLOR_NEW = "#1E90FF" # DodgerBlue
    
    # --- 2. VIZ: ERROR vs ANGLE ---
    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, results_old, marker='o', color=COLOR_OLD, label="Old Method ($T_{old}$)", linewidth=2.5)
    plt.plot(angles_deg, results_new, marker='o', color=COLOR_NEW, label="New Method ($T_{new}$)", linewidth=2.5)
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("MSE (Fidelity)")
    plt.title(f"Robustness to Rotation (Stress Test)\nRange: [{start_deg}, {end_deg}]")
    plt.legend()
    plt.grid(True, **MAJOR_GRID_STYLE)
    save_figure(plt.gcf(), out_dir, "12_error_vs_angle")
    print(f"Saved 12_error_vs_angle")


    # --- 3. VIZ: DISPLACEMENT VECTORS ---
    # Need to project B_target, B_pred_old, B_pred_new into 2D to plot arrows.
    # We should use a COMMON projection.
    # Let's use MDS on the union of all standard samples (B) + demo targets + demo predictions to define the space,
    # OR just train MDS on B and project others? MDS doesn't really "project" new points easily without Out-of-Sample extension.
    # PCA is easier for projection. Let's use PCA fitted on the original B (or B_target).
    
    if not demo_data:
        print("Warning: Demo angle data not found via exact match. Using last angle.")
        demo_data = {
             "angle_deg": angles_deg[-1],
             # Re-compute for last just in case loop logic missed it
             "B_target": decoderB.predict(rotate_2d(B_2d, angles_rad[-1])),
             "B_pred_old": decoderA.predict(rotate_2d(A_2d, angles_rad[-1])) @ W_old,
             "B_pred_new": decoderA.predict(rotate_2d(A_2d, angles_rad[-1])) @ W_new
        }

    from sklearn.decomposition import PCA
    
    # Collect all points to fit PCA or just fit on B_target?
    # Fitting on B_target (the "truth" at this angle) makes sense to see deviations from it.
    X_target = demo_data["B_target"]
    X_pred_old = demo_data["B_pred_old"]
    X_pred_new = demo_data["B_pred_new"]

    pca = PCA(n_components=2)
    # Fit on Target to establish the "Truth" plane
    pca.fit(X_target)
    
    xy_target = pca.transform(X_target)
    xy_old = pca.transform(X_pred_old)
    xy_new = pca.transform(X_pred_new)
    
    # Plotting
    # Two subplots: Old vs New
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Shared limits
    all_xy = np.vstack([xy_target, xy_old, xy_new])
    x_min, x_max = all_xy[:,0].min(), all_xy[:,0].max()
    y_min, y_max = all_xy[:,1].min(), all_xy[:,1].max()
    pad = (x_max - x_min) * 0.1
    xlim = (x_min - pad, x_max + pad)
    ylim = (y_min - pad, y_max + pad)

    # Infer labels for demo data
    # Assuming demo data preserves order of input B (15 samples: 5x0, 5x1, 5x2)
    labels = np.array([0]*5 + [1]*5 + [2]*5)

    def plot_arrows(ax, start, end, title):
        # start = Ideal (Static/Target style), end = Predicted (Rotated/Light style)
        
        # Plot points per class
        for c in np.unique(labels):
            idx = labels == c
            
            # Ideal (Target) points -> Class Color + Class Marker (Solid)
            # Use zorder=5 to stay above arrows
            ax.scatter(
                start[idx, 0], start[idx, 1],
                color=CLASS_COLORS[c],
                marker=CLASS_MARKERS[c],
                s=120,
                alpha=1.0,
                edgecolor='black',
                linewidth=1.2,
                label=f"Ideal (Class {c})",
                zorder=5
            )
            
            # Predicted points -> Light Class Color + Class Marker (Solid but light)
            # Use zorder=4 to stay above arrows (usually) or below ideal?
            ax.scatter(
                end[idx, 0], end[idx, 1],
                color=LIGHT_COLORS[c],
                marker=CLASS_MARKERS[c],
                s=100,
                alpha=0.9,
                edgecolor=CLASS_COLORS[c], # Thin edge of main color
                linewidth=0.8,
                label=f"Predicted (Class {c})",
                zorder=4
            )
        
        # Arrows - Black
        for i in range(len(start)):
            ax.arrow(start[i,0], start[i,1], 
                     end[i,0] - start[i,0], end[i,1] - start[i,1],
                     head_width=pad*0.1, length_includes_head=True, 
                     color='black', alpha=0.6, width=pad*0.005, zorder=2)
        
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Smart Legend
        handles, lbls = ax.get_legend_handles_labels()
        # Sort by Class then Type (Ideal/Pred)
        # Type is first word. Class is last word.
        # "Ideal (Class 0)", "Predicted (Class 0)"
        sorted_pairs = sorted(zip(lbls, handles), key=lambda x: x[0].split()[-1] + x[0].split()[0])
        ax.legend([h for l, h in sorted_pairs], [l for l, h in sorted_pairs], loc='best', ncol=2, fontsize='small')

        ax.grid(True, **MAJOR_GRID_STYLE)

    plot_arrows(axes[0], xy_target, xy_old, f"Old Method (Angle={demo_data['angle_deg']:.0f}°)")
    plot_arrows(axes[1], xy_target, xy_new, f"New Method (Angle={demo_data['angle_deg']:.0f}°)")

    save_figure(fig, out_dir, "11_displacement_vectors")
    print(f"Saved 11_displacement_vectors")


if __name__ == "__main__":
    run_extended_experiments()

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

import numpy as np
import yaml
from sklearn.linear_model import LinearRegression

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etm.utils import load_json_matrix
from src.synthetic.core import mds_2d, rotate_2d, mse_fid
from src.synthetic.core import mds_2d, rotate_2d, mse_fid


# --- CONFIG ---
# We'll read from configs/synthetic.yaml just for the angle range and step
CONFIG_PATH = PROJECT_ROOT / "configs" / "synthetic.yaml"

def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)




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
                "angle_deg": float(angles_deg[i]),
                "B_target": B_target.tolist(), # Convert to list for JSON
                "B_pred_old": B_pred_old.tolist(),
                "B_pred_new": B_pred_new.tolist()
            }

    # Save Metrics for Figure 12
    robustness_metrics = {
        "angles_deg": angles_deg.tolist(),
        "mse_old": results_old,
        "mse_new": results_new,
        "start_deg": start_deg,
        "end_deg": end_deg
    }
    from src.etm.utils import save_json
    save_json(matrices_dir / "robustness_metrics_extended.json", robustness_metrics)

    # Save Demo Data for Figure 11
    if not demo_data:
        print("Warning: Demo angle data not found via exact match. Using last angle.")
        # Re-compute for last just in case loop logic missed it
        demo_data = {
             "angle_deg": float(angles_deg[-1]),
             "B_target": decoderB.predict(rotate_2d(B_2d, angles_rad[-1])).tolist(),
             "B_pred_old": (decoderA.predict(rotate_2d(A_2d, angles_rad[-1])) @ W_old).tolist(),
             "B_pred_new": (decoderA.predict(rotate_2d(A_2d, angles_rad[-1])) @ W_new).tolist()
        }
    
    save_json(matrices_dir / "displacement_test_data.json", demo_data)
    print("Saved extended metrics and demo data to matrices/")


if __name__ == "__main__":
    run_extended_experiments()

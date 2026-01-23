"""
Generate figures for Synthetic experiments from saved outputs.
Usage: python src/scripts/generate_figures_synthetic.py [output_dir]
Default output_dir: outputs/synthetic/
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary utils
from src.etm.utils import load_json_matrix

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def run_synthetic_viz(out_dir: Path):
    print(f"Generating synthetic figures from {out_dir}")
    matrices_dir = out_dir / "matrices"
    figures_dir = out_dir / "figures_recreated"
    figures_dir.mkdir(exist_ok=True, parents=True)

    # Load Matrices
    inputs = PROJECT_ROOT / "inputs" / "synthetic" # Or outputs if they are copied there?
    # run_synthetic loads from inputs, but saves to out_dir/matrices
    # We should trust out_dir/matrices for results
    
    A = load_json_matrix(matrices_dir / "J_A.json") # Wait, J_A is different.
    # The pipeline copies base matrices? No. But we need A/B/T.
    # T_old etc are saved in matrices_dir.
    # But A and B (original) might not be saved in matrices_dir by default? 
    # run_synthetic loads A/B from repo/inputs.
    # Let's load from repo/inputs.
    
    A = load_json_matrix(PROJECT_ROOT / "inputs" / "synthetic" / "A.json")
    B = load_json_matrix(PROJECT_ROOT / "inputs" / "synthetic" / "B.json")
    
    # Load Results
    T_old = load_json_matrix(matrices_dir / "T_old_ls_kxl.json").T # W_old was saved as T_old_ls
    W_old = T_old.T
    
    T_new_path = matrices_dir / "T_new.json"
    if T_new_path.exists():
        T_new = load_json_matrix(T_new_path)
    else:
        # Fallback if saved as T_new_kxl
        T_new = load_json_matrix(matrices_dir / "T_new_kxl.json").T
    W_new = T_new.T
    
    JA = load_json_matrix(matrices_dir / "J_A.json")
    JB = load_json_matrix(matrices_dir / "J_B.json")
    
    # Load Sweep Data
    sweep_data = load_json(matrices_dir / "lambda_sweep.json")
    summary = load_json(matrices_dir / "summary.json")
    
    # Load A_rot_sweep for robustness
    a_rot_data = load_json(matrices_dir / "A_rot_sweep.json")
    A_rots = np.array(a_rot_data["data"]) # shape (n_angles, m, k)
    angles = np.array(a_rot_data["meta"]["angles_rad"])

    # Labels (hardcoded as per synthetic.py)
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=int)

    # Reconstruct B_rots
    B_old_rots = [rot @ W_old for rot in A_rots]
    B_new_rots = [rot @ W_new for rot in A_rots]

    # --- PLOTTING ---

    # 1. Heatmaps
    def heatmap(M: np.ndarray, title: str, path: Path):
        plt.figure(figsize=(6, 5))
        plt.imshow(M, aspect="auto")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    heatmap(W_old, "Heatmap: T_old (Baseline)", figures_dir / "heatmap_T_old.png")
    heatmap(W_new, "Heatmap: T_new (Equivariant)", figures_dir / "heatmap_T_new.png")
    heatmap(JA, "Heatmap: J^A", figures_dir / "heatmap_JA.png")
    heatmap(JB, "Heatmap: J^B", figures_dir / "heatmap_JB.png")

    # 2. Trade-offs
    lambdas = [r["lambda"] for r in sweep_data["rows"]]
    fid = [r["mse_fid"] for r in sweep_data["rows"]]
    sym = [r["sym_err"] for r in sweep_data["rows"]]

    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, fid, marker="o")
    plt.title("Trade-off: MSE_fid vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("MSE Fidelity")
    plt.tight_layout()
    plt.savefig(figures_dir / "tradeoff_mse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, sym, marker="o")
    plt.title("Trade-off: Symmetry Error vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Symmetry Error")
    plt.tight_layout()
    plt.savefig(figures_dir / "tradeoff_sym.png", dpi=160)
    plt.close()

    # 3. Robustness Scatter
    B_old_stacked = np.vstack(B_old_rots)
    B_new_stacked = np.vstack(B_new_rots)
    all_embeddings = np.vstack([B_old_stacked, B_new_stacked])
    labels_rep = np.tile(labels, len(angles))
    n_old = B_old_stacked.shape[0]

    def plot_embedding_comparison(old_2d, new_2d, labels_rep, method_name: str, out_path: Path):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for c in np.unique(labels_rep):
            idx = labels_rep == c
            plt.scatter(old_2d[idx, 0], old_2d[idx, 1], s=12, alpha=0.7, label=f"Class {c}")
        plt.title(f"{method_name}: $B^*_{{old}}$ (Chaos)")
        plt.xlabel(f"{method_name}-1")
        plt.ylabel(f"{method_name}-2")
        plt.legend()

        plt.subplot(1, 2, 2)
        for c in np.unique(labels_rep):
            idx = labels_rep == c
            plt.scatter(new_2d[idx, 0], new_2d[idx, 1], s=12, alpha=0.7, label=f"Class {c}")
        plt.title(f"{method_name}: $B^*_{{new}}$ (Order)")
        plt.xlabel(f"{method_name}-1")
        plt.ylabel(f"{method_name}-2")
        plt.legend()

        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()

    print("Computing embeddings...")
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_embeddings)
    old_pca = pca.transform(B_old_stacked)
    new_pca = pca.transform(B_new_stacked)
    plot_embedding_comparison(old_pca, new_pca, labels_rep, "PCA", figures_dir / "robustness_pca.png")

    # MDS
    mds = MDS(n_components=2, random_state=42, normalized_stress=False)
    # Using a subset if needed, but for synthetic it's small (15 * num_angles ~ 100 points)
    all_2d_mds = mds.fit_transform(all_embeddings)
    old_mds = all_2d_mds[:n_old]
    new_mds = all_2d_mds[n_old:]
    plot_embedding_comparison(old_mds, new_mds, labels_rep, "MDS", figures_dir / "robustness_mds.png")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_embeddings.shape[0] // 4))
    all_2d_tsne = tsne.fit_transform(all_embeddings)
    old_tsne = all_2d_tsne[:n_old]
    new_tsne = all_2d_tsne[n_old:]
    plot_embedding_comparison(old_tsne, new_tsne, labels_rep, "t-SNE", figures_dir / "robustness_tsne.png")

    # UMAP
    try:
        import umap
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, all_embeddings.shape[0] // 4))
        all_2d_umap = umap_reducer.fit_transform(all_embeddings)
        old_umap = all_2d_umap[:n_old]
        new_umap = all_2d_umap[n_old:]
        plot_embedding_comparison(old_umap, new_umap, labels_rep, "UMAP", figures_dir / "robustness_umap.png")
    except ImportError:
        print("UMAP not installed, skipping UMAP plot.")

    print("Done synthetic visualization.")

if __name__ == "__main__":
    out_dir = Path("outputs/synthetic")
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])
    
    if not out_dir.exists():
        print(f"Error: {out_dir} does not exist. Run synthetic experiments first.")
        sys.exit(1)
        
    run_synthetic_viz(out_dir)

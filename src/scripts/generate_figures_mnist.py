"""
Generate figures for MNIST experiments from saved outputs.
Usage: python src/scripts/generate_figures_mnist.py [output_dir]
Default output_dir: outputs/mnist/
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

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def run_mnist_viz(out_dir: Path):
    print(f"Generating MNIST figures from {out_dir}")
    matrices_dir = out_dir / "matrices"
    figures_dir = out_dir / "figures_recreated"
    figures_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load Metrics
    metrics_path = matrices_dir / "mnist_metrics.json"
    if not metrics_path.exists():
        print(f"Metrics not found at {metrics_path}. Run Stage 2 first.")
        return

    metrics = load_json(metrics_path)

    # --- Robustness Curves ---
    rob = metrics["robustness"]
    angles = np.array(rob["angles_deg"])
    ssim_old = rob["mean_ssim_old"]
    ssim_new = rob["mean_ssim_new"]
    psnr_old = rob["mean_psnr_old"]
    psnr_new = rob["mean_psnr_new"]

    plt.figure(figsize=(6, 4))
    plt.plot(angles, ssim_old, marker="o", label="T_old")
    plt.plot(angles, ssim_new, marker="o", label="T_new")
    plt.xlabel("Rotation angle (deg)")
    plt.ylabel("Mean SSIM")
    plt.title("Robustness: SSIM vs rotation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "robustness_ssim.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(angles, psnr_old, marker="o", label="T_old")
    plt.plot(angles, psnr_new, marker="o", label="T_new")
    plt.xlabel("Rotation angle (deg)")
    plt.ylabel("Mean PSNR (dB)")
    plt.title("Robustness: PSNR vs rotation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "robustness_psnr.png", dpi=160)
    plt.close()

    # --- Symmetry Error Bar ---
    sym_old = metrics["symmetry_error_fro"]["old"]
    sym_new = metrics["symmetry_error_fro"]["new"]

    plt.figure(figsize=(4, 4))
    plt.bar(["T_old", "T_new"], [sym_old, sym_new])
    plt.ylabel("||T J_A - J_B T||_F")
    plt.title("Symmetry error (single λ)")
    plt.tight_layout()
    plt.savefig(figures_dir / "symmetry_error_bar.png", dpi=160)
    plt.close()

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

    def plot_mnist_embedding(old_2d, new_2d, labels, method_name: str, out_path: Path):
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(old_2d[:, 0], old_2d[:, 1], c=labels, cmap='tab10', s=8, alpha=0.6)
        plt.colorbar(scatter, label='Digit')
        plt.title(f"{method_name}: $B^*_{{old}}$ (MNIST)")
        plt.xlabel(f"{method_name}-1")
        plt.ylabel(f"{method_name}-2")

        plt.subplot(1, 2, 2)
        scatter = plt.scatter(new_2d[:, 0], new_2d[:, 1], c=labels, cmap='tab10', s=8, alpha=0.6)
        plt.colorbar(scatter, label='Digit')
        plt.title(f"{method_name}: $B^*_{{new}}$ (MNIST)")
        plt.xlabel(f"{method_name}-1")
        plt.ylabel(f"{method_name}-2")

        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()

    print("Computing MNIST embeddings...")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_embeddings)
    old_pca = pca.transform(B_star_old)
    new_pca = pca.transform(B_star_new)
    plot_mnist_embedding(old_pca, new_pca, labels, "PCA", figures_dir / "mnist_scatter_pca.png")

    # MDS
    # Use subset for MDS if too large (it's N*angles = 128*6 = 768 points, totally fine)
    mds = MDS(n_components=2, random_state=42, normalized_stress='auto', max_iter=300, n_init=1)
    all_2d_mds = mds.fit_transform(all_embeddings)
    old_mds = all_2d_mds[:n_old]
    new_mds = all_2d_mds[n_old:]
    plot_mnist_embedding(old_mds, new_mds, labels, "MDS", figures_dir / "mnist_scatter_mds.png")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_embeddings.shape[0] // 4))
    all_2d_tsne = tsne.fit_transform(all_embeddings)
    old_tsne = all_2d_tsne[:n_old]
    new_tsne = all_2d_tsne[n_old:]
    plot_mnist_embedding(old_tsne, new_tsne, labels, "t-SNE", figures_dir / "mnist_scatter_tsne.png")

    # UMAP
    try:
        import umap
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, all_embeddings.shape[0] // 4))
        all_2d_umap = umap_reducer.fit_transform(all_embeddings)
        old_umap = all_2d_umap[:n_old]
        new_umap = all_2d_umap[n_old:]
        plot_mnist_embedding(old_umap, new_umap, labels, "UMAP", figures_dir / "mnist_scatter_umap.png")
    except ImportError:
        print("UMAP not installed.")

    print("Done MNIST visualization.")

if __name__ == "__main__":
    out_dir = Path("outputs/mnist")
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])
    
    if not out_dir.exists():
        print(f"Error: {out_dir} does not exist.")
        sys.exit(1)
        
    run_mnist_viz(out_dir)

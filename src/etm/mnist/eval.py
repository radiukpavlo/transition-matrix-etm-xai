"""Evaluation and visualization for MNIST experiments.

Metrics:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- symmetry error ||T J^A - J^B T||_F
- robustness curves over rotated test inputs

Generates the MNIST-required figures (10+), when called from the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from .rotate import rotate_batch
from ..utils import ensure_dir, save_json


@dataclass
class EvalConfig:
    device: str = "cpu"
    n_eval_samples: int = 2000
    n_viz_samples: int = 16
    rotation_deg_max: float = 30.0
    rotation_deg_step: float = 5.0


def symmetry_error(T_lxk: np.ndarray, J_A: np.ndarray, J_B: np.ndarray, squared: bool = False) -> float:
    D = T_lxk @ J_A - J_B @ T_lxk
    val = float(np.linalg.norm(D, ord="fro"))
    return val * val if squared else val


def _normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std


@torch.no_grad()
def _features(model, images_norm: torch.Tensor) -> torch.Tensor:
    return model.penultimate(images_norm)


@torch.no_grad()
def reconstruct_images(
    model,
    W_kxl: np.ndarray,
    images01: torch.Tensor,
    mean: float,
    std: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    x01 = images01.to(device)
    x_norm = _normalize(x01, mean, std)
    A = _features(model, x_norm).cpu().numpy().astype(np.float64)
    B_hat = A @ W_kxl
    orig = x01.cpu().numpy().astype(np.float64).reshape(-1, 28, 28)
    recon = np.clip(B_hat.reshape(-1, 28, 28), 0.0, 1.0)
    return orig, recon


def image_metrics(orig: np.ndarray, recon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = orig.shape[0]
    ssim_vals = np.zeros(n, dtype=np.float64)
    psnr_vals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ssim_vals[i] = ssim(orig[i], recon[i], data_range=1.0)
        psnr_vals[i] = psnr(orig[i], recon[i], data_range=1.0)
    return ssim_vals, psnr_vals


def plot_recon_grid(orig: np.ndarray, recon: np.ndarray, title: str, path: Path) -> None:
    import math as _m

    n = orig.shape[0]
    cols = int(_m.sqrt(n))
    rows = int(_m.ceil(n / cols))
    plt.figure(figsize=(2 * cols, 2 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.concatenate([orig[i], recon[i]], axis=1), cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_histogram(a: np.ndarray, b: np.ndarray, la: str, lb: str, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(a, bins=30, alpha=0.6, label=la)
    plt.hist(b, bins=30, alpha=0.6, label=lb)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def evaluate_and_plot(
    out_root: Path,
    model,
    test_loader_raw,
    W_old: np.ndarray,
    W_new: np.ndarray,
    J_A: np.ndarray,
    J_B: np.ndarray,
    cfg: EvalConfig,
    normalize_mean: float,
    normalize_std: float,
    logger,
) -> Dict[str, object]:
    ensure_dir(out_root / "figures")
    ensure_dir(out_root / "matrices")

    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    images_list: List[torch.Tensor] = []
    seen = 0
    for x, _y in test_loader_raw:
        if seen >= cfg.n_eval_samples:
            break
        take = min(x.shape[0], cfg.n_eval_samples - seen)
        images_list.append(x[:take])
        seen += take

    images01 = torch.cat(images_list, dim=0)

    orig, recon_old = reconstruct_images(model, W_old, images01, normalize_mean, normalize_std, device)
    _orig, recon_new = reconstruct_images(model, W_new, images01, normalize_mean, normalize_std, device)

    ssim_old, psnr_old = image_metrics(orig, recon_old)
    ssim_new, psnr_new = image_metrics(orig, recon_new)

    n_viz = min(cfg.n_viz_samples, orig.shape[0])
    plot_recon_grid(orig[:n_viz], recon_old[:n_viz], "Original | Reconstructed (T_old)", out_root / "figures" / "03_recon_grid_old.png")
    plot_recon_grid(orig[:n_viz], recon_new[:n_viz], "Original | Reconstructed (T_new)", out_root / "figures" / "04_recon_grid_new.png")

    plot_histogram(ssim_old, ssim_new, "T_old", "T_new", "SSIM distribution (test subset)", out_root / "figures" / "05_ssim_hist.png")
    plot_histogram(psnr_old, psnr_new, "T_old", "T_new", "PSNR distribution (test subset)", out_root / "figures" / "06_psnr_hist.png")

    sym_old = symmetry_error(W_old.T, J_A, J_B, squared=False)
    sym_new = symmetry_error(W_new.T, J_A, J_B, squared=False)

    # Bar (supplementary) - the required curve is produced in pipeline.py
    plt.figure(figsize=(4, 4))
    plt.bar(["T_old", "T_new"], [sym_old, sym_new])
    plt.ylabel("||T J_A - J_B T||_F")
    plt.title("Symmetry error (single λ)")
    plt.tight_layout()
    plt.savefig(out_root / "figures" / "07b_symmetry_error_bar.png", dpi=160)
    plt.close()

    angles = np.arange(-cfg.rotation_deg_max, cfg.rotation_deg_max + 1e-9, cfg.rotation_deg_step)
    mean_ssim_old, mean_ssim_new, mean_psnr_old, mean_psnr_new = [], [], [], []

    images_rob = images01[: min(256, images01.shape[0])].to(device)

    for deg in angles:
        theta = torch.tensor(float(deg) * math.pi / 180.0, device=device)
        x_rot = rotate_batch(images_rob, theta).detach().cpu()

        o, ro = reconstruct_images(model, W_old, x_rot, normalize_mean, normalize_std, device)
        _o2, rn = reconstruct_images(model, W_new, x_rot, normalize_mean, normalize_std, device)

        s_o, p_o = image_metrics(o, ro)
        s_n, p_n = image_metrics(o, rn)

        mean_ssim_old.append(float(np.mean(s_o)))
        mean_ssim_new.append(float(np.mean(s_n)))
        mean_psnr_old.append(float(np.mean(p_o)))
        mean_psnr_new.append(float(np.mean(p_n)))

    plt.figure(figsize=(6, 4))
    plt.plot(angles, mean_ssim_old, marker="o", label="T_old")
    plt.plot(angles, mean_ssim_new, marker="o", label="T_new")
    plt.xlabel("Rotation angle (deg)")
    plt.ylabel("Mean SSIM")
    plt.title("Robustness: SSIM vs rotation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "figures" / "08_robustness_ssim_vs_angle.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(angles, mean_psnr_old, marker="o", label="T_old")
    plt.plot(angles, mean_psnr_new, marker="o", label="T_new")
    plt.xlabel("Rotation angle (deg)")
    plt.ylabel("Mean PSNR (dB)")
    plt.title("Robustness: PSNR vs rotation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "figures" / "09_robustness_psnr_vs_angle.png", dpi=160)
    plt.close()

    # Qualitative rotated grid at 15°
    theta = torch.tensor(15.0 * math.pi / 180.0, device=device)
    x15 = rotate_batch(images_rob[: min(8, images_rob.shape[0])], theta).detach().cpu()
    o15, ro15 = reconstruct_images(model, W_old, x15, normalize_mean, normalize_std, device)
    _o15, rn15 = reconstruct_images(model, W_new, x15, normalize_mean, normalize_std, device)

    plt.figure(figsize=(10, 6))
    for i in range(o15.shape[0]):
        plt.subplot(3, o15.shape[0], i + 1)
        plt.imshow(o15[i], cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        if i == 0:
            plt.title("Rotated input")

        plt.subplot(3, o15.shape[0], o15.shape[0] + i + 1)
        plt.imshow(ro15[i], cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        if i == 0:
            plt.title("Recon T_old")

        plt.subplot(3, o15.shape[0], 2 * o15.shape[0] + i + 1)
        plt.imshow(rn15[i], cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        if i == 0:
            plt.title("Recon T_new")

    plt.tight_layout()
    plt.savefig(out_root / "figures" / "10_qualitative_rotated_grid.png", dpi=160)
    plt.close()

    # --- ROBUSTNESS SCATTER PLOTS (PCA, MDS, t-SNE, UMAP) ---
    # Visualize predicted embeddings B* colored by digit label to show cluster structure
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, TSNE
    
    scatter_angles = np.array([-25.0, -15.0, -5.0, 5.0, 15.0, 25.0])  # degrees
    n_scatter = min(128, images01.shape[0])
    
    # Collect labels for scatter subset from loader
    labels_list: List[int] = []
    images_scatter_list: List[torch.Tensor] = []
    scatter_seen = 0
    for x, y in test_loader_raw:
        if scatter_seen >= n_scatter:
            break
        take = min(x.shape[0], n_scatter - scatter_seen)
        images_scatter_list.append(x[:take])
        labels_list.extend(y[:take].tolist())
        scatter_seen += take
    
    images_scatter = torch.cat(images_scatter_list, dim=0).to(device)
    base_labels = np.array(labels_list)
    
    recon_old_all = []
    recon_new_all = []
    all_labels = []
    
    for deg in scatter_angles:
        theta = torch.tensor(float(deg) * math.pi / 180.0, device=device)
        x_rot = rotate_batch(images_scatter, theta).detach().cpu()
        
        _orig_s, recon_old_s = reconstruct_images(model, W_old, x_rot, normalize_mean, normalize_std, device)
        _orig_s2, recon_new_s = reconstruct_images(model, W_new, x_rot, normalize_mean, normalize_std, device)
        
        recon_old_all.append(recon_old_s.reshape(n_scatter, -1))
        recon_new_all.append(recon_new_s.reshape(n_scatter, -1))
        all_labels.extend(base_labels.tolist())
    
    recon_old_stacked = np.vstack(recon_old_all)  # (N*angles, 784)
    recon_new_stacked = np.vstack(recon_new_all)  # (N*angles, 784)
    all_embeddings = np.vstack([recon_old_stacked, recon_new_stacked])
    all_labels = np.array(all_labels)
    
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
    
    n_old = recon_old_stacked.shape[0]
    
    # 1) PCA
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_embeddings)
    old_pca = pca.transform(recon_old_stacked)
    new_pca = pca.transform(recon_new_stacked)
    plot_mnist_embedding(old_pca, new_pca, all_labels, "PCA", out_root / "figures" / "09a_mnist_scatter_pca.png")
    
    # 2) MDS
    mds = MDS(n_components=2, random_state=42, normalized_stress='auto', max_iter=300, n_init=1)
    all_2d_mds = mds.fit_transform(all_embeddings)
    old_mds = all_2d_mds[:n_old]
    new_mds = all_2d_mds[n_old:]
    plot_mnist_embedding(old_mds, new_mds, all_labels, "MDS", out_root / "figures" / "09b_mnist_scatter_mds.png")
    
    # 3) t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_embeddings.shape[0] // 4))
    all_2d_tsne = tsne.fit_transform(all_embeddings)
    old_tsne = all_2d_tsne[:n_old]
    new_tsne = all_2d_tsne[n_old:]
    plot_mnist_embedding(old_tsne, new_tsne, all_labels, "t-SNE", out_root / "figures" / "09c_mnist_scatter_tsne.png")
    
    # 4) UMAP
    import umap
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, all_embeddings.shape[0] // 4))
    all_2d_umap = umap_reducer.fit_transform(all_embeddings)
    old_umap = all_2d_umap[:n_old]
    new_umap = all_2d_umap[n_old:]
    plot_mnist_embedding(old_umap, new_umap, all_labels, "UMAP", out_root / "figures" / "09d_mnist_scatter_umap.png")

    metrics = {
        "n_eval_samples": int(orig.shape[0]),
        "ssim": {
            "old_mean": float(np.mean(ssim_old)),
            "old_std": float(np.std(ssim_old)),
            "new_mean": float(np.mean(ssim_new)),
            "new_std": float(np.std(ssim_new)),
        },
        "psnr": {
            "old_mean": float(np.mean(psnr_old)),
            "old_std": float(np.std(psnr_old)),
            "new_mean": float(np.mean(psnr_new)),
            "new_std": float(np.std(psnr_new)),
        },
        "symmetry_error_fro": {"old": sym_old, "new": sym_new},
        "robustness": {
            "angles_deg": angles.tolist(),
            "mean_ssim_old": mean_ssim_old,
            "mean_ssim_new": mean_ssim_new,
            "mean_psnr_old": mean_psnr_old,
            "mean_psnr_new": mean_psnr_new,
        },
    }

    save_json(out_root / "matrices" / "mnist_metrics.json", metrics)

    logger.info(
        "MNIST eval subset: "
        f"SSIM old={metrics['ssim']['old_mean']:.4f}, new={metrics['ssim']['new_mean']:.4f}; "
        f"PSNR old={metrics['psnr']['old_mean']:.2f}, new={metrics['psnr']['new_mean']:.2f}; "
        f"sym old={sym_old:.3e}, new={sym_new:.3e}"
    )

    return metrics

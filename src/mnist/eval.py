"""Evaluation and data extraction for MNIST experiments.

Metrics:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- symmetry error ||T J^A - J^B T||_F
- robustness curves over rotated test inputs

Saves all data required for figure generation to disk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from .rotate import rotate_batch
from etm.utils import ensure_dir, save_json


@dataclass
class EvalConfig:
    device: str = "cpu"
    n_eval_samples: int = 5000
    n_viz_samples: int = 30
    rotation_deg_max: float = 90.0
    rotation_deg_step: float = 10.0


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


def evaluate_and_save(
    out_root: Path,
    model,
    loader,
    W_old: np.ndarray,
    W_new: np.ndarray,
    J_A: np.ndarray,
    J_B: np.ndarray,
    cfg: EvalConfig,
    normalize_mean: float,
    normalize_std: float,
    logger,
    subset_name: str,
) -> Dict[str, object]:
    ensure_dir(out_root / "matrices")

    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    # 1. Collect evaluation samples
    images_list: List[torch.Tensor] = []
    labels_list: List[int] = []
    seen = 0
    
    # Store reference to full loader iteration for digit selection
    all_images_for_digits = []
    all_labels_for_digits = []

    for x, y in loader:
        # Collect for metrics
        if seen < cfg.n_eval_samples:
            take = min(x.shape[0], cfg.n_eval_samples - seen)
            images_list.append(x[:take])
            labels_list.extend(y[:take].tolist())
            seen += take
        
        # Collect for digit selection (keep all until we have enough)
        if len(all_images_for_digits) * loader.batch_size < 5000: # Limit memory
            all_images_for_digits.append(x)
            all_labels_for_digits.append(y)

    images01 = torch.cat(images_list, dim=0)
    labels = np.array(labels_list)

    # 2. Main Metrics (SSIM/PSNR on unrotated)
    orig, recon_old = reconstruct_images(model, W_old, images01, normalize_mean, normalize_std, device)
    _orig, recon_new = reconstruct_images(model, W_new, images01, normalize_mean, normalize_std, device)

    ssim_old, psnr_old = image_metrics(orig, recon_old)
    ssim_new, psnr_new = image_metrics(orig, recon_new)

    sym_old = symmetry_error(W_old.T, J_A, J_B, squared=False)
    sym_new = symmetry_error(W_new.T, J_A, J_B, squared=False)

    # 3. Robustness Curves
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

    # 4. Embeddings for Scatter Plot (PCA/MDS etc)
    # Use a subset of labeled data
    scatter_angles = np.array([-90.0, -45.0, -5.0, 5.0, 45.0, 90.0])
    n_scatter = min(128, images01.shape[0])
    
    images_scatter = images01[:n_scatter].to(device)
    labels_scatter = labels[:n_scatter]
    
    recon_old_all = []
    recon_new_all = []
    all_labels = []
    
    for deg in scatter_angles:
        theta = torch.tensor(float(deg) * math.pi / 180.0, device=device)
        x_rot = rotate_batch(images_scatter, theta).detach().cpu()
        
        _, recon_old_s = reconstruct_images(model, W_old, x_rot, normalize_mean, normalize_std, device)
        _, recon_new_s = reconstruct_images(model, W_new, x_rot, normalize_mean, normalize_std, device)
        
        recon_old_all.append(recon_old_s.reshape(n_scatter, -1))
        recon_new_all.append(recon_new_s.reshape(n_scatter, -1))
        all_labels.extend(labels_scatter.tolist())
    
    recon_old_stacked = np.vstack(recon_old_all)
    recon_new_stacked = np.vstack(recon_new_all)
    all_labels = np.array(all_labels)

    # 5. Extract specific rotated digit samples for "The Chaos Figure"
    # Select one random instance of each digit 0-9
    # Rotate by random angle in [-90, 90]
    
    all_imgs = torch.cat(all_images_for_digits, dim=0)
    all_lbls = torch.cat(all_labels_for_digits, dim=0).cpu().numpy()
    
    chaos_samples_orig = []
    chaos_samples_old = []
    chaos_samples_new = []
    chaos_labels = []
    chaos_angles = []

    np.random.seed(42)  # For reproducibility of sample selection
    
    for digit in range(10):
        # Find indices for this digit
        indices = np.where(all_lbls == digit)[0]
        if len(indices) == 0:
            continue
            
        # Pick one random sample
        idx = np.random.choice(indices)
        img = all_imgs[idx:idx+1].to(device)
        
        # Pick random angle in [-90, 90]
        angle_deg = np.random.uniform(-90, 90)
        theta = torch.tensor(float(angle_deg) * math.pi / 180.0, device=device)
        
        # Rotate
        img_rot = rotate_batch(img, theta).detach().cpu()
        
        # Reconstruct
        orig_rot, rec_old = reconstruct_images(model, W_old, img_rot, normalize_mean, normalize_std, device)
        _, rec_new = reconstruct_images(model, W_new, img_rot, normalize_mean, normalize_std, device)
        
        chaos_samples_orig.append(orig_rot[0])
        chaos_samples_old.append(rec_old[0])
        chaos_samples_new.append(rec_new[0])
        chaos_labels.append(digit)
        chaos_angles.append(angle_deg)

    # Convert to arrays
    chaos_samples_orig = np.array(chaos_samples_orig)
    chaos_samples_old = np.array(chaos_samples_old)
    chaos_samples_new = np.array(chaos_samples_new)

    # 6. Save Everything
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

    save_json(out_root / "matrices" / f"mnist_metrics_{subset_name}.json", metrics)

    np.savez(
        out_root / "matrices" / f"mnist_embeddings_{subset_name}.npz",
        B_star_old=recon_old_stacked,
        B_star_new=recon_new_stacked,
        labels=all_labels,
        angles=scatter_angles
    )
    
    np.savez(
        out_root / "matrices" / f"mnist_chaos_samples_{subset_name}.npz",
        orig=chaos_samples_orig,
        recon_old=chaos_samples_old,
        recon_new=chaos_samples_new,
        labels=np.array(chaos_labels),
        angles=np.array(chaos_angles)
    )

    logger.info(
        f"MNIST eval subset ({subset_name}): "
        f"SSIM old={metrics['ssim']['old_mean']:.4f}, new={metrics['ssim']['new_mean']:.4f}; "
        f"PSNR old={metrics['psnr']['old_mean']:.2f}, new={metrics['psnr']['new_mean']:.2f}; "
        f"sym old={sym_old:.3e}, new={sym_new:.3e}"
    )

    return metrics

"""Infinitesimal generator estimation for MNIST (SO(2) rotation).

Critical manuscript requirement: compute exact derivatives with respect to θ using PyTorch autograd.

We compute derivatives via torch.autograd.functional.jvp (forward-mode AD).

Then estimate generators via least squares:
  A J_A^T ≈ dA/dθ|_{0}
  B J_B^T ≈ dB/dθ|_{0}

We solve using reduced normal equations with SVD-based pseudoinverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .rotate import rotate_batch, jvp_theta
from ..utils import ensure_dir, save_json, save_json_matrix


@dataclass
class GeneratorConfig:
    n_samples: int = 5000
    batch_size: int = 128
    tau: float = 1e-8
    device: str = "cpu"
    normalize_mean: float = 0.1307
    normalize_std: float = 0.3081


def _svd_pinv(M: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_inv = np.array([1.0 / si if si > tau else 0.0 for si in s], dtype=np.float64)
    return (Vt.T * s_inv) @ U.T, s


def _normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std


def compute_A_and_dA(model, images01: torch.Tensor, device: torch.device, mean: float, std: float, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute A(θ=0) and dA/dθ|_{θ=0} using finite differences.
    
    We use central differences: dA/dθ ≈ (A(eps) - A(-eps)) / (2*eps)
    """
    images01 = images01.to(device)
    
    with torch.no_grad():
        # A at theta=0
        x_n = _normalize(images01, mean, std)
        A = model.penultimate(x_n)
        
        # A at theta=+eps
        theta_pos = torch.tensor(eps, device=device)
        x_rot_pos = rotate_batch(images01, theta_pos)
        x_n_pos = _normalize(x_rot_pos, mean, std)
        A_pos = model.penultimate(x_n_pos)
        
        # A at theta=-eps
        theta_neg = torch.tensor(-eps, device=device)
        x_rot_neg = rotate_batch(images01, theta_neg)
        x_n_neg = _normalize(x_rot_neg, mean, std)
        A_neg = model.penultimate(x_n_neg)
        
        # Central difference
        dA = (A_pos - A_neg) / (2.0 * eps)
    
    return A, dA


def compute_B_and_dB(images01: torch.Tensor, device: torch.device, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute B(θ=0) and dB/dθ|_{θ=0} using finite differences.
    
    We use central differences: dB/dθ ≈ (B(eps) - B(-eps)) / (2*eps)
    """
    images01 = images01.to(device)
    
    with torch.no_grad():
        # B at theta=0
        B = images01.flatten(1)
        
        # B at theta=+eps
        theta_pos = torch.tensor(eps, device=device)
        x_rot_pos = rotate_batch(images01, theta_pos)
        B_pos = x_rot_pos.flatten(1)
        
        # B at theta=-eps
        theta_neg = torch.tensor(-eps, device=device)
        x_rot_neg = rotate_batch(images01, theta_neg)
        B_neg = x_rot_neg.flatten(1)
        
        # Central difference
        dB = (B_pos - B_neg) / (2.0 * eps)
    
    return B, dB


def estimate_generators(repo_root: Path, out_root: Path, model, raw_loader, cfg: GeneratorConfig, logger) -> Dict[str, object]:
    ensure_dir(out_root / "matrices")
    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    A_list, dA_list, B_list, dB_list = [], [], [], []
    seen = 0

    for x01, _y in raw_loader:
        if seen >= cfg.n_samples:
            break
        take = min(x01.shape[0], cfg.n_samples - seen)
        x01 = x01[:take]

        A, dA = compute_A_and_dA(model, x01, device=device, mean=cfg.normalize_mean, std=cfg.normalize_std)
        B, dB = compute_B_and_dB(x01, device=device)

        A_list.append(A.cpu())
        dA_list.append(dA.cpu())
        B_list.append(B.cpu())
        dB_list.append(dB.cpu())
        seen += take

        if seen % (cfg.batch_size * 10) == 0:
            logger.info(f"Generator estimation: processed {seen}/{cfg.n_samples} samples")

    A = torch.cat(A_list, dim=0).numpy().astype(np.float64)
    dA = torch.cat(dA_list, dim=0).numpy().astype(np.float64)
    B = torch.cat(B_list, dim=0).numpy().astype(np.float64)
    dB = torch.cat(dB_list, dim=0).numpy().astype(np.float64)

    logger.info(f"Collected A {A.shape}, dA {dA.shape}, B {B.shape}, dB {dB.shape}")

    G_A = A.T @ A
    H_A = A.T @ dA
    G_B = B.T @ B
    H_B = B.T @ dB

    pinv_GA, s_GA = _svd_pinv(G_A, tau=cfg.tau)
    pinv_GB, s_GB = _svd_pinv(G_B, tau=cfg.tau)

    J_A_T = pinv_GA @ H_A
    J_B_T = pinv_GB @ H_B

    J_A = J_A_T.T
    J_B = J_B_T.T

    save_json_matrix(out_root / "matrices" / "J_A.json", J_A, name="J^A", source="MNIST (autograd JVP, reduced normal equations)", meta={"n_samples": int(seen), "tau": cfg.tau})
    save_json_matrix(out_root / "matrices" / "J_B.json", J_B, name="J^B", source="MNIST (autograd JVP, reduced normal equations)", meta={"n_samples": int(seen), "tau": cfg.tau})

    save_json(out_root / "matrices" / "generator_singular_values.json", {
        "GA_singular_values": s_GA.tolist(),
        "GB_singular_values": s_GB.tolist(),
        "tau": cfg.tau,
        "n_samples": int(seen),
    })

    return {
        "n_samples": int(seen),
        "A_shape": list(A.shape),
        "B_shape": list(B.shape),
        "tau": cfg.tau,
        "GA_min_sv": float(np.min(s_GA)),
        "GB_min_sv": float(np.min(s_GB)),
    }

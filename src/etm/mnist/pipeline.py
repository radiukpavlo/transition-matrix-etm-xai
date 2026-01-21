"""MNIST end-to-end pipeline (manuscript Section 3.5).

This pipeline is designed for a standard Python environment.
Per the user request for this sandbox, we do NOT execute MNIST computations here,
but the repository contains a complete, runnable MNIST pipeline.

Stages:
1) Load MNIST from data/mnist/ (no downloads).
2) Train CNN FM (k=490) or load existing weights.
3) Estimate generators J^A and J^B using autograd JVP at θ=0.
4) Estimate transition matrices T_old and T_new, plus a λ sweep.
5) Evaluate SSIM/PSNR, symmetry error, robustness curves, and generate figures.

All outputs are written under outputs/mnist/ and outputs/logs/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .data import MNISTDataConfig, assert_mnist_present, get_dataloaders, get_raw_dataloaders, mnist_root
from .eval import EvalConfig, evaluate_and_plot, symmetry_error
from .generators import GeneratorConfig, estimate_generators
from .model import CNNConfig, MNISTCNN
from .train import TrainConfig, train_cnn
from .transition import TransitionConfig, save_transition_matrices, solve_T_new_lsqr, solve_T_old
from ..utils import (
    configure_logger,
    ensure_dir,
    load_json_matrix,
    save_json,
    set_global_seed,
    system_info,
    utc_timestamp,
)


@dataclass
class MNISTPipelineConfig:
    seed: int = 42
    device: str = "cpu"

    data: MNISTDataConfig = field(default_factory=MNISTDataConfig)
    model: CNNConfig = field(default_factory=CNNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generators: GeneratorConfig = field(default_factory=GeneratorConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    lambda_sweep: Optional[List[float]] = None

    def __post_init__(self):
        if self.lambda_sweep is None:
            self.lambda_sweep = [0.0, 0.1, 0.5, 1.0]


def run_mnist(repo_root: Path, out_root: Path, cfg: MNISTPipelineConfig) -> Dict[str, object]:
    ensure_dir(out_root)
    ensure_dir(repo_root / "outputs" / "logs")
    ensure_dir(out_root / "runs")

    run_id = f"mnist_{utc_timestamp()}"
    log_path = repo_root / "outputs" / "logs" / f"{run_id}.log"
    logger = configure_logger(log_path)

    logger.info("Starting MNIST pipeline")
    logger.info(json.dumps({"system": system_info()}, indent=2))

    seed_info = set_global_seed(cfg.seed, deterministic_torch=True)
    logger.info(f"Seeding info: {seed_info}")

    device = torch.device(cfg.device)

    # Data
    data_root = mnist_root(repo_root)
    assert_mnist_present(data_root)

    # Training loaders (normalized)
    cfg.train.device = cfg.device
    train_loader, test_loader = get_dataloaders(repo_root, cfg.data, device=device)

    # Raw loaders (x in [0,1]) for MM pixels and autograd rotation
    train_raw_loader, test_raw_loader = get_raw_dataloaders(
        repo_root,
        batch_size=cfg.generators.batch_size,
        num_workers=cfg.data.num_workers,
        device=device,
    )

    # Model
    model = MNISTCNN(cfg.model)

    # Train or load weights
    weights_path = out_root / "models" / "mnist_cnn_k490.pt"
    ensure_dir(weights_path.parent)
    if weights_path.exists():
        logger.info(f"Loading existing model weights: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
    else:
        logger.info("Training MNIST CNN (no existing weights found)")
        train_cnn(out_root, train_loader, test_loader, cfg.model, cfg.train, logger)
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

    # Generator estimation (autograd)
    # Ensure generator normalization matches training normalization.
    cfg.generators.normalize_mean = cfg.data.normalize_mean
    cfg.generators.normalize_std = cfg.data.normalize_std
    cfg.generators.device = cfg.device
    gen_info = estimate_generators(repo_root, out_root, model, train_raw_loader, cfg.generators, logger)

    J_A = load_json_matrix(out_root / "matrices" / "J_A.json")
    J_B = load_json_matrix(out_root / "matrices" / "J_B.json")

    # Transition matrices: collect (A,B) subset from raw loader
    logger.info(f"Collecting A,B for transition estimation (n_samples={cfg.transition.n_samples})")
    k = cfg.model.k
    l = 28 * 28

    A_list: List[np.ndarray] = []
    B_list: List[np.ndarray] = []
    seen = 0
    mean = cfg.data.normalize_mean
    std = cfg.data.normalize_std

    model = model.to(device)
    model.eval()

    for x01, _y in train_raw_loader:
        if seen >= cfg.transition.n_samples:
            break
        take = min(x01.shape[0], cfg.transition.n_samples - seen)
        x01 = x01[:take]
        with torch.no_grad():
            x_norm = (x01.to(device) - mean) / std
            A = model.penultimate(x_norm).cpu().numpy().astype(np.float64)
        B = x01.cpu().numpy().astype(np.float64).reshape(take, l)
        A_list.append(A)
        B_list.append(B)
        seen += take

    A_sub = np.vstack(A_list)
    B_sub = np.vstack(B_list)
    logger.info(f"Transition data shapes: A_sub={A_sub.shape}, B_sub={B_sub.shape}")

    # Baseline and equivariant solutions
    W_old, info_old = solve_T_old(A_sub, B_sub, tau=cfg.transition.tau)

    W_new, info_new = solve_T_new_lsqr(
        A_sub,
        B_sub,
        J_A,
        J_B,
        lambda_=cfg.transition.lambda_,
        iter_lim=cfg.transition.lsqr_iter_lim,
        atol=cfg.transition.lsqr_atol,
        btol=cfg.transition.lsqr_btol,
    )

    save_transition_matrices(out_root, W_old, W_new, info_old, info_new)

    # λ sweep (required symmetry-error vs λ figure)
    sweep = []
    for lam in (cfg.lambda_sweep or []):
        if lam == cfg.transition.lambda_:
            W = W_new
            solver_info = info_new
        elif lam == 0.0:
            W = W_old
            solver_info = {"note": "λ=0 uses T_old baseline"}
        else:
            W, solver_info = solve_T_new_lsqr(
                A_sub,
                B_sub,
                J_A,
                J_B,
                lambda_=lam,
                iter_lim=max(50, cfg.transition.lsqr_iter_lim // 2),
                atol=cfg.transition.lsqr_atol,
                btol=cfg.transition.lsqr_btol,
            )
        sweep.append(
            {
                "lambda": float(lam),
                "symmetry_error": symmetry_error(W.T, J_A, J_B, squared=False),
                "symmetry_error_sq": symmetry_error(W.T, J_A, J_B, squared=True),
                "solver": solver_info,
            }
        )

    save_json(out_root / "matrices" / "lambda_sweep.json", {"rows": sweep, "note": "λ=0 row corresponds to baseline"})

    # Plot symmetry error vs λ
    import matplotlib.pyplot as plt

    lambdas = [r["lambda"] for r in sweep]
    syms = [r["symmetry_error"] for r in sweep]
    ensure_dir(out_root / "figures")
    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, syms, marker="o")
    plt.xlabel("λ")
    plt.ylabel("||T J_A - J_B T||_F")
    plt.title("Symmetry error vs λ")
    plt.tight_layout()
    plt.savefig(out_root / "figures" / "07_symmetry_error_vs_lambda.png", dpi=160)
    plt.close()

    # Evaluation & figures
    cfg.eval.device = cfg.device
    metrics = evaluate_and_plot(
        out_root,
        model,
        test_raw_loader,
        W_old,
        W_new,
        J_A,
        J_B,
        cfg.eval,
        normalize_mean=mean,
        normalize_std=std,
        logger=logger,
    )

    manifest = {
        "run_id": run_id,
        "seed": cfg.seed,
        "device": cfg.device,
        "n_transition_samples": int(seen),
        "configs": {
            "data": cfg.data.__dict__,
            "model": cfg.model.__dict__,
            "train": cfg.train.__dict__,
            "generators": cfg.generators.__dict__,
            "transition": cfg.transition.__dict__,
            "eval": cfg.eval.__dict__,
            "lambda_sweep": list(cfg.lambda_sweep or []),
        },
        "generator_info": gen_info,
        "metrics": metrics,
        "log_path": str(log_path.relative_to(repo_root)),
    }

    save_json(out_root / "runs" / f"{run_id}_manifest.json", manifest)
    logger.info(f"MNIST pipeline complete. Manifest: outputs/mnist/runs/{run_id}_manifest.json")
    return manifest

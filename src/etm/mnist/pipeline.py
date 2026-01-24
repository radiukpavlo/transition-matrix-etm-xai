"""MNIST end-to-end pipeline (manuscript Section 3.5).

This pipeline is designed for a standard Python environment.
Per the user request for this sandbox, we do NOT execute MNIST computations here,
but the repository contains a complete, runnable MNIST pipeline.

Refactored Stages:
1) Train/Extract: Train CNN (or load), estimate generators, extract A/B samples.
2) Experiments: Load A/B and generators, compute T_old/T_new, sweep lambda, evaluate.

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


def run_stage1_train_extract(repo_root: Path, out_root: Path, cfg: MNISTPipelineConfig) -> Dict[str, object]:
    ensure_dir(out_root)
    ensure_dir(repo_root / "outputs" / "logs")
    ensure_dir(out_root / "runs")
    ensure_dir(out_root / "matrices")

    run_id = f"mnist_stage1_{utc_timestamp()}"
    log_path = repo_root / "outputs" / "logs" / f"{run_id}.log"
    logger = configure_logger(log_path)

    logger.info("Starting MNIST Stage 1: Train & Extract")
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
    
    # Check if we should retrain based on user intent? 
    # For now, if weights exist, we load them unless explicit overwrite logic is added (not requested).
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
    save_json(out_root / "matrices" / "gen_info.json", gen_info)

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

    # Reuse train_raw_loader logic from original
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
    logger.info(f"Extracted transition data: A_sub={A_sub.shape}, B_sub={B_sub.shape}")
    
    np.save(out_root / "matrices" / "A_sub.npy", A_sub)
    np.save(out_root / "matrices" / "B_sub.npy", B_sub)
    logger.info(f"Saved A_sub.npy and B_sub.npy to {out_root / 'matrices'}")

    logger.info("Stage 1 complete.")
    return {"run_id": run_id, "gen_info": gen_info, "samples_extracted": seen}


def run_stage2_experiments(repo_root: Path, out_root: Path, cfg: MNISTPipelineConfig) -> Dict[str, object]:
    run_id = f"mnist_stage2_{utc_timestamp()}"
    log_path = repo_root / "outputs" / "logs" / f"{run_id}.log"
    logger = configure_logger(log_path)

    logger.info("Starting MNIST Stage 2: Experiments")
    
    seed_info = set_global_seed(cfg.seed, deterministic_torch=True)
    device = torch.device(cfg.device)

    # Check dependencies
    matrices_dir = out_root / "matrices"
    if not (matrices_dir / "J_A.json").exists():
        raise FileNotFoundError("J_A.json not found. Run Stage 1 first.")
    if not (matrices_dir / "A_sub.npy").exists():
        raise FileNotFoundError("A_sub.npy not found. Run Stage 1 first.")

    # Load Matrices
    J_A = load_json_matrix(matrices_dir / "J_A.json")
    J_B = load_json_matrix(matrices_dir / "J_B.json")
    A_sub = np.load(matrices_dir / "A_sub.npy")
    B_sub = np.load(matrices_dir / "B_sub.npy")
    
    # Load generation info if available
    gen_info = {}
    if (matrices_dir / "gen_info.json").exists():
        with open(matrices_dir / "gen_info.json", "r") as f:
            gen_info = json.load(f)

    logger.info(f"Loaded matrices: A={A_sub.shape}, B={B_sub.shape}")
    logger.info(f"Solving Transition Matrices (lam={cfg.transition.lambda_})...")

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

    # λ sweep
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

    # Evaluate
    # Need model and data for evaluation
    # Load Model
    model = MNISTCNN(cfg.model)
    weights_path = out_root / "models" / "mnist_cnn_k490.pt"
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
    else:
        logger.warning(f"Model weights not found at {weights_path}. Evaluation will be random!")

    # Load Data (Test Set Only needed for Eval generally, but check evaluate_and_plot)
    # evaluate_and_plot uses test_raw_loader
    train_raw_loader, test_raw_loader = get_raw_dataloaders(
        repo_root,
        batch_size=cfg.generators.batch_size,
        num_workers=cfg.data.num_workers,
        device=device,
    )

    # Evaluation & figures
    cfg.eval.device = cfg.device
    
    logger.info("Evaluating on TRAIN subset...")
    metrics_train = evaluate_and_plot(
        out_root,
        model,
        train_raw_loader,
        W_old,
        W_new,
        J_A,
        J_B,
        cfg.eval,
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        logger=logger,
        subset_name="train",
    )

    logger.info("Evaluating on TEST subset...")
    metrics_test = evaluate_and_plot(
        out_root,
        model,
        test_raw_loader,
        W_old,
        W_new,
        J_A,
        J_B,
        cfg.eval,
        normalize_mean=cfg.data.normalize_mean,
        normalize_std=cfg.data.normalize_std,
        logger=logger,
        subset_name="test",
    )
    
    metrics = {
        "train": metrics_train,
        "test": metrics_test,
    }
    
    manifest = {
        "run_id": run_id,
        "seed": cfg.seed,
        "device": cfg.device,
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
        "log_path": str(log_path.relative_to(repo_root)) if log_path.is_relative_to(repo_root) else str(log_path),
    }

    save_json(out_root / "runs" / f"{run_id}_manifest.json", manifest)
    logger.info(f"MNIST Stage 2 complete. Manifest: outputs/mnist/runs/{run_id}_manifest.json")
    return manifest

def run_mnist(repo_root: Path, out_root: Path, cfg: MNISTPipelineConfig) -> Dict[str, object]:
    """Legacy wrapper to run both stages sequentially."""
    run_stage1_train_extract(repo_root, out_root, cfg)
    return run_stage2_experiments(repo_root, out_root, cfg)

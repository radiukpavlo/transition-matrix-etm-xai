"""Run the MNIST experiment (baseline vs equivariant method).

This implements the experiment structure described in the baseline PDF and
reused in the manuscript:
- Train a CNN on MNIST.
- Extract penultimate-layer features (k=490).
- Define mental model features as pixels (l=784).
- Use a random subsample of m=10,000 training images to estimate T.
- Evaluate reconstructions on 1,000 test images with SSIM and PSNR.

Notes for this sandbox:
- Outbound network from the container may be disabled, so MNIST downloading may
  fail here. The script will exit with a clear message if MNIST files are missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np

# Ensure the repository root (parent of /scripts) is on the import path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.mnist_idx import download_mnist_idx, load_mnist
from src.models.cnn_mnist import MNISTCNN
from src.methods.transition_baseline import solve_baseline_transition
from src.methods.generator_estimation import estimate_generators_least_squares
from src.methods.transition_equivariant import solve_equivariant_large_cg
from src.utils.metrics import compute_image_metrics, fidelity_mse, symmetry_defect
from src.utils.plotting import save_image_grid, save_histogram
from src.utils.repro import set_global_seed, collect_env_info


def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[mnist] train epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in pbar:
            xb = _to_device(xb, device)
            yb = _to_device(yb, device)

            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())

            pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        print(f"[mnist] epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

    return {"train_loss": epoch_loss, "train_acc": epoch_acc}


def extract_features_and_pixels(
    model: MNISTCNN,
    images: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (A, B) where
    - A: (m,k) penultimate features
    - B: (m,l) pixels flattened in [0,1]
    """
    model.eval()
    X = torch.from_numpy(images).float() / 255.0
    X = X.unsqueeze(1)  # (m,1,28,28)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    feats = []
    with torch.no_grad():
        for (xb,) in tqdm(dl, desc="[mnist] extract features", leave=False):
            xb = _to_device(xb, device)
            _, f = model(xb)
            feats.append(f.detach().cpu().numpy())

    A = np.concatenate(feats, axis=0).astype(np.float64)
    B = (images.reshape(images.shape[0], -1).astype(np.float64) / 255.0)
    return A, B


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/mnist', help='MNIST IDX directory')
    parser.add_argument('--download', action='store_true', help='Attempt to download MNIST IDX files')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--m_train', type=int, default=10000)
    parser.add_argument('--m_test', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=0.01, help='Rotation step in radians for generator estimation')
    parser.add_argument('--lambda_', type=float, default=0.5, dest='lambda_')
    parser.add_argument('--gen_samples', type=int, default=2000, help='Samples used to estimate generators')
    parser.add_argument('--cg_maxiter', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)

    outdir = Path('outputs/mnist')
    outdir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if args.download:
        print('[mnist] attempting download (may fail in restricted sandboxes)')
        download_mnist_idx(data_dir)

    try:
        mnist = load_mnist(data_dir)
        X_train, y_train = mnist.x_train, mnist.y_train
        X_test, y_test = mnist.x_test, mnist.y_test
    except FileNotFoundError as e:
        print('[mnist] MNIST IDX files not found:', e)
        print('[mnist] Place the 4 MNIST gzip IDX files into', data_dir)
        print('[mnist] Or re-run with --download in an environment with internet access.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[mnist] device:', device)

    # Train CNN on the full MNIST training set (baseline PDF).
    model = MNISTCNN().to(device)

    Xtr = torch.from_numpy(X_train).float().unsqueeze(1) / 255.0
    ytr = torch.from_numpy(y_train).long()
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)

    train_stats = train_cnn(model, train_loader, device=device, epochs=args.epochs, lr=args.lr)

    # Subsample for transition matrix estimation / evaluation.
    rng = np.random.default_rng(args.seed)
    idx_train = rng.choice(len(X_train), size=args.m_train, replace=False)
    idx_test = rng.choice(len(X_test), size=args.m_test, replace=False)

    X_train_sub = X_train[idx_train]
    X_test_sub = X_test[idx_test]

    # Extract (A,B) matrices.
    A_train, B_train = extract_features_and_pixels(model, X_train_sub, device=device, batch_size=args.batch_size)
    A_test, B_test = extract_features_and_pixels(model, X_test_sub, device=device, batch_size=args.batch_size)

    # Baseline transition matrix
    baseline = solve_baseline_transition(A_train, B_train)
    T_old = baseline.T

    # Generator estimation
    # Use subset of training sample for generator estimation to reduce cost.
    gen_idx = rng.choice(args.m_train, size=min(args.gen_samples, args.m_train), replace=False)

    # Rotate images by epsilon (radians)
    from skimage.transform import rotate

    def rot_imgs(imgs: np.ndarray, angle_rad: float) -> np.ndarray:
        angle_deg = angle_rad * 180.0 / math.pi
        out = np.empty_like(imgs, dtype=np.float64)
        for i in range(imgs.shape[0]):
            out[i] = rotate(imgs[i], angle=angle_deg, resize=False, order=1, mode='constant', cval=0.0, preserve_range=True)
        return out

    X_gen = X_train_sub[gen_idx]
    X_gen_eps = rot_imgs(X_gen, args.epsilon)

    # Features
    A_gen, B_gen = extract_features_and_pixels(model, X_gen, device=device, batch_size=args.batch_size)
    A_gen_eps, B_gen_eps = extract_features_and_pixels(model, X_gen_eps.astype(np.uint8), device=device, batch_size=args.batch_size)

    gen = estimate_generators_least_squares(
        A=A_gen,
        A_eps=A_gen_eps,
        B=B_gen,
        B_eps=B_gen_eps,
        eps=args.epsilon,
        ridge=1e-6,
    )
    J_A, J_B = gen.J_A, gen.J_B

    # Equivariant transition matrix
    sol_new, cg_info = solve_equivariant_large_cg(
        A=A_train,
        B=B_train,
        J_A=J_A,
        J_B=J_B,
        lam=args.lambda_,
        ridge=1e-6,
        maxiter=args.cg_maxiter,
        rtol=1e-6,
        atol=0.0,
        x0=T_old,
    )
    T_new = sol_new.T

    # Reconstructions on test set
    B_old_hat_test = A_test @ T_old.T
    B_new_hat_test = A_test @ T_new.T

    met_old = compute_image_metrics(
        B_test.reshape(-1, 28, 28),
        np.clip(B_old_hat_test, 0, 1).reshape(-1, 28, 28),
        data_range=1.0,
    )
    met_new = compute_image_metrics(
        B_test.reshape(-1, 28, 28),
        np.clip(B_new_hat_test, 0, 1).reshape(-1, 28, 28),
        data_range=1.0,
    )

    # Save summaries
    summary = {
        'hyperparams': {
            'seed': args.seed,
            'm_train': args.m_train,
            'm_test': args.m_test,
            'epsilon': args.epsilon,
            'lambda': args.lambda_,
            'gen_samples': int(min(args.gen_samples, args.m_train)),
            'cg_maxiter': args.cg_maxiter,
        },
        'train_cnn': train_stats,
        'generator_estimation': {
            'cond_A': float(gen.cond_A),
            'cond_B': float(gen.cond_B),
        },
        'baseline': {
            'fidelity_mse_train': float(fidelity_mse(B_train, baseline.B_hat).mse),
            'sym_defect': float(symmetry_defect(T_old, J_A, J_B).frob),
            'ssim_mean_test': float(np.mean(met_old['ssim'])),
            'psnr_mean_test': float(np.mean(met_old['psnr'])),
        },
        'equivariant': {
            'fidelity_mse_train': float(fidelity_mse(B_train, A_train @ T_new.T).mse),
            'sym_defect': float(symmetry_defect(T_new, J_A, J_B).frob),
            'ssim_mean_test': float(np.mean(met_new['ssim'])),
            'psnr_mean_test': float(np.mean(met_new['psnr'])),
            'cg_info': {
                'converged': bool(cg_info.converged),
                'n_iter': int(cg_info.n_iter),
                'final_resid_norm': float(cg_info.final_residual_norm),
            },
        },
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))

    # Save T matrices
    np.save(outdir / 'T_old_lxk.npy', T_old)
    np.save(outdir / 'T_new_lxk.npy', T_new)
    np.save(outdir / 'J_A.npy', J_A)
    np.save(outdir / 'J_B.npy', J_B)

    # Plots: histograms of SSIM and PSNR
    save_histogram(met_old['ssim'], outdir / 'ssim_old.png', xlabel='SSIM', title='SSIM (old)')
    save_histogram(met_new['ssim'], outdir / 'ssim_new.png', xlabel='SSIM', title='SSIM (new)')
    save_histogram(met_old['psnr'], outdir / 'psnr_old.png', xlabel='PSNR (dB)', title='PSNR (old)')
    save_histogram(met_new['psnr'], outdir / 'psnr_new.png', xlabel='PSNR (dB)', title='PSNR (new)')

    # Example reconstructions
    # take first 25 test images
    save_image_grid(
        X_test_sub[:25].astype(np.float64) / 255.0,
        outdir / 'orig_grid.png',
        n_rows=5,
        n_cols=5,
        title='Original test images',
    )
    save_image_grid(
        np.clip(B_old_hat_test[:25], 0, 1).reshape(-1, 28, 28),
        outdir / 'recon_old_grid.png',
        n_rows=5,
        n_cols=5,
        title='Reconstruction (old)',
    )
    save_image_grid(
        np.clip(B_new_hat_test[:25], 0, 1).reshape(-1, 28, 28),
        outdir / 'recon_new_grid.png',
        n_rows=5,
        n_cols=5,
        title='Reconstruction (new)',
    )

    env = collect_env_info()
    (outdir / 'env.json').write_text(json.dumps(env, indent=2))

    print('[mnist] wrote outputs to', outdir)


if __name__ == '__main__':
    main()

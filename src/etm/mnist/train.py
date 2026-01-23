"""Training for the MNIST CNN.

Saves:
- weights: outputs/mnist/models/mnist_cnn_k490.pt
- training curves: outputs/mnist/figures/01_train_loss.png and 02_train_accuracy.png
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from .model import MNISTCNN, CNNConfig
from ..utils import ensure_dir, save_json


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    n_train_samples: Optional[int] = None


def evaluate_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def train_cnn(
    out_root: Path,
    train_loader,
    test_loader,
    cfg_model: CNNConfig,
    cfg_train: TrainConfig,
    logger,
) -> Tuple[Path, Dict[str, object]]:
    device = torch.device(cfg_train.device)
    ensure_dir(out_root / "models")
    ensure_dir(out_root / "figures")

    model = MNISTCNN(cfg_model).to(device)
    opt = Adam(model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay)

    history = {"epoch": [], "loss": [], "train_acc": [], "test_acc": [], "seconds": []}
    
    logger.info(f"Starting training for {cfg_train.epochs} epochs (device={device})")

    for epoch in range(1, cfg_train.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        samples_seen = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg_train.epochs}", leave=True):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            
            samples_seen += x.size(0)
            if cfg_train.n_train_samples and samples_seen >= cfg_train.n_train_samples:
                break

        sec = time.time() - t0
        train_acc = evaluate_accuracy(model, train_loader, device)
        test_acc = evaluate_accuracy(model, test_loader, device)
        mean_loss = float(np.mean(losses)) if losses else float("nan")

        logger.info(
            f"Epoch {epoch}: loss={mean_loss:.6f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, sec={sec:.1f}"
        )

        history["epoch"].append(epoch)
        history["loss"].append(mean_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["seconds"].append(sec)

    weights_path = out_root / "models" / "mnist_cnn_k490.pt"
    torch.save({"model_state": model.state_dict(), "cfg": cfg_model.__dict__}, weights_path)
    save_json(out_root / "models" / "training_history.json", history)

    # Save history to CSV
    import csv
    with open(out_root / "models" / "training_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "train_acc", "test_acc", "seconds"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["loss"][i],
                history["train_acc"][i],
                history["test_acc"][i],
                history["seconds"][i]
            ])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("MNIST CNN training loss")
    plt.tight_layout()
    plt.savefig(out_root / "figures" / "01_train_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_acc"], marker="o", label="Train")
    plt.plot(history["epoch"], history["test_acc"], marker="o", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST CNN accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "figures" / "02_train_accuracy.png", dpi=160)
    plt.close()

    return weights_path, history

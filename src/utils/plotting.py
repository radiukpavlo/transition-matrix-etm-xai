"""Plotting utilities for the synthetic and MNIST experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def save_scatter_side_by_side(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    labels: Sequence[int],
    outpath: Path,
    *,
    title_left: str,
    title_right: str,
) -> None:
    """Save the required 2-panel scatter plot for Section 3.4.

    Args:
        left_xy: (m, 2)
        right_xy: (m, 2)
        labels: length m
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)
    axes = axes.ravel()

    axes[0].scatter(left_xy[:, 0], left_xy[:, 1], c=labels, s=40)
    axes[0].set_title(title_left)
    axes[0].set_xlabel('dim1')
    axes[0].set_ylabel('dim2')

    axes[1].scatter(right_xy[:, 0], right_xy[:, 1], c=labels, s=40)
    axes[1].set_title(title_right)
    axes[1].set_xlabel('dim1')
    axes[1].set_ylabel('dim2')

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_tradeoff_curve(
    xs: Sequence[float],
    ys: Sequence[float],
    outpath: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def save_histogram(
    values: np.ndarray,
    outpath: Path,
    *,
    xlabel: str,
    title: str,
    bins: int = 30,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 4))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def save_image_grid(
    images: np.ndarray,
    outpath: Path,
    *,
    n_rows: int,
    n_cols: int,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Save a grid of grayscale images.

    images: (n, H, W)
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx < images.shape[0]:
            ax.imshow(images[idx], cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

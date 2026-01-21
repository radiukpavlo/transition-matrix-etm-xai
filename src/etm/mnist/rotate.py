"""Differentiable SO(2) rotation for MNIST images.

We rotate images using affine_grid + grid_sample so that PyTorch autograd can
differentiate outputs with respect to the rotation parameter θ.

The manuscript requires autograd-based derivatives (not finite differences)
for generator estimation on MNIST.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def rotation_matrix_2x3(theta: torch.Tensor) -> torch.Tensor:
    """Return a batch of 2×3 rotation matrices for affine_grid.

    theta: shape () or (N,) angles in radians.
    returns: (N,2,3)
    """
    if theta.dim() == 0:
        theta = theta[None]
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(c)
    row1 = torch.stack([c, -s, z], dim=-1)
    row2 = torch.stack([s, c, z], dim=-1)
    return torch.stack([row1, row2], dim=1)


def rotate_batch(
    images: torch.Tensor,
    theta: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Rotate a batch of images by theta radians.

    images: (N,1,H,W)
    theta: scalar tensor or (N,) tensor in radians
    """
    N = images.size(0)
    A = rotation_matrix_2x3(theta).to(dtype=images.dtype, device=images.device)
    # Expand A to match batch size if theta was scalar (A has shape (1, 2, 3))
    if A.size(0) == 1 and N > 1:
        A = A.expand(N, -1, -1)
    grid = F.affine_grid(A, size=images.size(), align_corners=align_corners)
    return F.grid_sample(images, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def jvp_theta(func, theta0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (y(theta0), dy/dθ(theta0)) for scalar θ using forward-mode AD."""
    if not hasattr(torch.autograd.functional, "jvp"):
        raise RuntimeError("torch.autograd.functional.jvp is unavailable; cannot compute exact derivative.")
    theta0 = theta0.detach().clone().requires_grad_(True)
    y, dy = torch.autograd.functional.jvp(func, (theta0,), (torch.ones_like(theta0),), create_graph=False)
    return y, dy


def degrees_to_radians(deg: float) -> float:
    return deg * math.pi / 180.0

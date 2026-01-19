"""Reproducibility utilities (seeding, environment capture).

This project is designed to be runnable end-to-end via scripts/run_all.py.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and (if available) PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism flags (may reduce performance but improves reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class EnvInfo:
    created_at: str
    python: str
    platform: str
    numpy: str
    scipy: str
    sklearn: str
    torch: str
    device: str


def collect_env_info(device: Optional[str] = None) -> Dict[str, Any]:
    """Collect versions/hardware notes for the reproducibility checklist."""
    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _ver(pkg: str) -> str:
        try:
            mod = __import__(pkg)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "not_installed"

    torch_ver = "not_installed"
    dev = "cpu"
    if torch is not None:
        torch_ver = getattr(torch, "__version__", "unknown")
        if device is not None:
            dev = device
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"

    info = EnvInfo(
        created_at=created_at,
        python=sys.version.replace("\n", " "),
        platform=f"{platform.platform()} ({platform.machine()})",
        numpy=_ver("numpy"),
        scipy=_ver("scipy"),
        sklearn=_ver("sklearn"),
        torch=torch_ver,
        device=dev,
    )
    return asdict(info)

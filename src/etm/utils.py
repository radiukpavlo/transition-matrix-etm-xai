"""Utility functions for reproducible experiments.

All runs log:
- global seeds (Python/NumPy/PyTorch)
- lightweight system + package versions
- explicit file paths for artifacts

Matrix JSON format:
{
  "name": str,
  "shape": [rows, cols],
  "dtype": str,
  "source": str,
  "data": [[...], ...],
  "meta": {...}
}

This matches the project contract and supports end-to-end reproducibility.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def repo_root() -> Path:
    """Return repository root path (assumes this file lives in src/etm)."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def utc_timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def set_global_seed(seed: int, deterministic_torch: bool = True) -> Dict[str, Any]:
    """Set Python/NumPy (and optionally PyTorch) seeds."""
    random.seed(seed)
    np.random.seed(seed)

    info: Dict[str, Any] = {
        "seed": int(seed),
        "deterministic_torch": bool(deterministic_torch),
        "torch": None,
    }

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        info["torch"] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    except Exception as e:  # pragma: no cover
        info["torch"] = {"error": str(e)}

    return info


def system_info() -> Dict[str, Any]:
    """Collect lightweight system and environment information."""
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
        "git": None,
        "packages": {},
    }

    try:
        root = repo_root()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root).decode().strip()
        info["git"] = {"commit": commit, "dirty": bool(status), "status": status}
    except Exception:
        info["git"] = None

    for mod in ["numpy", "scipy", "sklearn", "matplotlib", "torch", "torchvision", "skimage"]:
        try:
            m = __import__(mod)
            info["packages"][mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info["packages"][mod] = None

    return info


def configure_logger(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


@dataclasses.dataclass
class JsonMatrix:
    name: str
    shape: Tuple[int, int]
    dtype: str
    source: str
    data: Sequence[Sequence[float]]
    meta: Optional[Dict[str, Any]] = None


def save_json_matrix(path: Path, mat: np.ndarray, name: str, source: str, meta: Optional[Dict[str, Any]] = None) -> None:
    ensure_dir(path.parent)
    payload = {
        "name": name,
        "shape": list(mat.shape),
        "dtype": str(mat.dtype),
        "source": source,
        "data": mat.tolist(),
        "meta": meta or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json_matrix(path: Path) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    arr = np.asarray(obj["data"], dtype=np.float64)
    expected = tuple(obj["shape"])
    if arr.shape != expected:
        raise ValueError(f"Shape mismatch for {path}: expected {expected}, got {arr.shape}")
    return arr


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_repeating_decimal(token: str) -> float:
    """Parse tokens like '0.8(4)' or '-0.(4)'.

    Convention:
    - 'a.b(c)' means decimal expansion where the digits inside parentheses repeat.
      Example: 0.8(4) = 0.84444...
    - '-0.(4)' = -0.44444...

    If token contains no repeating pattern, float(token) is returned.
    """
    t = token.strip()
    if "(" not in t or ")" not in t:
        return float(t)

    sign = -1.0 if t.startswith("-") else 1.0
    t2 = t[1:] if t.startswith("-") else t

    if "." not in t2:
        raise ValueError(f"Unexpected repeating decimal format: {token}")

    int_part_str, frac = t2.split(".", 1)
    int_part = int(int_part_str) if int_part_str else 0

    nonrep, rep = frac.split("(", 1)
    rep = rep.split(")", 1)[0]

    n = len(nonrep)
    r = len(rep)

    nonrep_val = int(nonrep) if nonrep else 0
    rep_val = int(rep) if rep else 0

    denom_nonrep = 10 ** n
    frac_val = nonrep_val / denom_nonrep
    if r > 0:
        denom_rep = (10 ** r - 1)
        frac_val += rep_val / (denom_nonrep * denom_rep)

    return sign * (int_part + frac_val)

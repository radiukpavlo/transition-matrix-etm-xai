"""Extract MNIST.zip into the repository data/ folder.

Expected archive layout (as provided by the user):
  raw/train-images-idx3-ubyte
  raw/train-labels-idx1-ubyte
  raw/t10k-images-idx3-ubyte
  raw/t10k-labels-idx1-ubyte

We re-home files to torchvision-compatible layout:
  data/mnist/MNIST/raw/...

Usage:
  python scripts/extract_mnist_zip.py --zip /path/to/MNIST.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_path", required=True, help="Path to MNIST.zip")
    ap.add_argument("--repo", dest="repo", default=str(Path(__file__).resolve().parents[1]), help="Repo root")
    args = ap.parse_args()

    repo = Path(args.repo)
    zpath = Path(args.zip_path)
    out_raw = repo / "data" / "mnist" / "MNIST" / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zpath, "r") as z:
        members = [m for m in z.namelist() if m.startswith("raw/") and not m.endswith("/")]
        if not members:
            raise RuntimeError(f"No 'raw/' members found in {zpath}")
        for m in members:
            name = Path(m).name
            tgt = out_raw / name
            with z.open(m) as src, open(tgt, "wb") as dst:
                dst.write(src.read())

    print(f"Extracted {len(members)} files into {out_raw}")


if __name__ == "__main__":
    main()

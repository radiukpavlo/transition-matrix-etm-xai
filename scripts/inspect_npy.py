#!/usr/bin/env python3
"""
Utility script to inspect a NumPy (.npy) file.

Usage:
    python scripts/inspect_npy.py <path_to_npy_file>
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a .npy file")
    parser.add_argument("path", type=Path, help="Path to the .npy file")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: File not found at {args.path}")
        sys.exit(1)

    try:
        data = np.load(args.path)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"=== Inspection: {args.path.name} ===")
    print(f"Path:  {args.path.resolve()}")
    print(f"Type:  {type(data)}")

    if isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print("-" * 30)

        if data.size > 0:
            if np.issubdtype(data.dtype, np.number):
                print(f"Min:   {np.min(data)}")
                print(f"Max:   {np.max(data)}")
                print(f"Mean:  {np.mean(data)}")
                print(f"Std:   {np.std(data)}")
            else:
                print("Non-numeric data, skipping stats.")
            
            print("-" * 30)
            print("Data Snippet (first 5 items/rows):")
            # Print a reasonable slice depending on dimensionality
            if data.ndim == 1:
                print(data[:10])
            else:
                print(data[:5])
        else:
            print("Array is empty.")
    else:
        print("Loaded object is not a numpy array.")

    print("=" * 30)


if __name__ == "__main__":
    main()

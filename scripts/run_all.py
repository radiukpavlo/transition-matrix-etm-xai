"""Run all experiments and generate all output artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("\n$", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    code1 = run([python, str(repo_root / "scripts" / "run_synthetic.py")])
    if code1 != 0:
        sys.exit(code1)

    code2 = run([python, str(repo_root / "scripts" / "run_mnist.py")])
    if code2 != 0:
        sys.exit(code2)


if __name__ == "__main__":
    main()

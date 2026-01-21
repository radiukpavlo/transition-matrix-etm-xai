"""Run synthetic and/or MNIST experiments.

Examples:
  python run_all.py --synthetic
  python run_all.py --mnist
  python run_all.py --all

Config files (YAML):
  configs/synthetic.yaml
  configs/mnist.yaml

Logs:
  outputs/logs/

Note: this script imports the MNIST pipeline lazily so synthetic runs do not depend on torchvision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from etm.config import dataclass_from_dict  # noqa: E402
from etm.synthetic import SyntheticConfig, run_synthetic  # noqa: E402
from etm.utils import configure_logger, ensure_dir, repo_root, save_json, set_global_seed, utc_timestamp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="Run synthetic pipeline")
    ap.add_argument("--mnist", action="store_true", help="Run MNIST pipeline")
    ap.add_argument("--all", action="store_true", help="Run both pipelines")
    ap.add_argument("--synthetic-config", type=str, default="configs/synthetic.yaml")
    ap.add_argument("--mnist-config", type=str, default="configs/mnist.yaml")
    args = ap.parse_args()

    if not (args.synthetic or args.mnist or args.all):
        ap.error("Select at least one of --synthetic, --mnist, or --all")

    root = repo_root()
    ensure_dir(root / "outputs" / "logs")

    run_id = f"run_all_{utc_timestamp()}"
    logger = configure_logger(root / "outputs" / "logs" / f"{run_id}.log")
    logger.info(f"Repository root: {root}")

    # Load synthetic config
    syn_cfg = SyntheticConfig()
    if Path(args.synthetic_config).exists():
        syn_dict = yaml.safe_load(Path(args.synthetic_config).read_text()) or {}
        syn_cfg = dataclass_from_dict(SyntheticConfig, syn_dict)

    # Load MNIST config lazily (only if needed)
    mn_cfg = None
    if args.mnist or args.all:
        from etm.mnist.pipeline import MNISTPipelineConfig  # noqa: E402

        mn_cfg = MNISTPipelineConfig()
        if Path(args.mnist_config).exists():
            mn_dict = yaml.safe_load(Path(args.mnist_config).read_text()) or {}
            mn_cfg = dataclass_from_dict(MNISTPipelineConfig, mn_dict)

    manifest = {"run_id": run_id, "synthetic": None, "mnist": None}

    if args.synthetic or args.all:
        set_global_seed(42)
        logger.info("Running synthetic pipeline")
        syn_out = root / "outputs" / "synthetic"
        ensure_dir(syn_out)
        summary = run_synthetic(root, syn_out, syn_cfg)
        manifest["synthetic"] = summary

    if args.mnist or args.all:
        logger.info("Running MNIST pipeline")
        from etm.mnist.pipeline import run_mnist  # noqa: E402

        mn_out = root / "outputs" / "mnist"
        ensure_dir(mn_out)
        out = run_mnist(root, mn_out, mn_cfg)  # type: ignore[arg-type]
        manifest["mnist"] = out

    save_json(root / "outputs" / "logs" / f"{run_id}_manifest.json", manifest)
    logger.info("All requested pipelines complete")


if __name__ == "__main__":
    main()

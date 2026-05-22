from __future__ import annotations

import argparse
import subprocess
import sys

from utils.config import ensure_dirs, load_config
from utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Trustify strict reproduction entrypoint.")
    parser.add_argument("--config", default="configs/trustify_reproduction.yaml")
    parser.add_argument("--stage", choices=["prepare", "train_text", "train_image", "train_hybrid", "evaluate"], required=True)
    parser.add_argument("--csv", help="Prepared dataset CSV for stage=prepare")
    parser.add_argument("--predictions", help="Prediction CSV for hybrid/evaluate stages")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed_everything(cfg["experiment"]["seed"], cfg["experiment"]["deterministic"])

    if args.stage == "prepare":
        if not args.csv:
            raise SystemExit("--csv is required for prepare")
        subprocess.run([sys.executable, "-m", "scripts.prepare_dataset", "--config", args.config, "--csv", args.csv], check=True)
    elif args.stage == "train_text":
        subprocess.run([sys.executable, "-m", "training.train_text", "--config", args.config], check=True)
    elif args.stage == "train_image":
        subprocess.run([sys.executable, "-m", "training.train_image", "--config", args.config], check=True)
    elif args.stage == "train_hybrid":
        if not args.predictions:
            raise SystemExit("--predictions is required for train_hybrid")
        subprocess.run([sys.executable, "-m", "training.train_hybrid", "--config", args.config, "--predictions", args.predictions], check=True)
    elif args.stage == "evaluate":
        if not args.predictions:
            raise SystemExit("--predictions is required for evaluate")
        subprocess.run([sys.executable, "-m", "evaluation.evaluate", "--config", args.config, "--predictions", args.predictions], check=True)


if __name__ == "__main__":
    main()

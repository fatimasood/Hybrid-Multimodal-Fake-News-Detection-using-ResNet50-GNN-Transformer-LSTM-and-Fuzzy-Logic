from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate import evaluate_fuzzy_csv
from utils.config import load_config
from utils.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trustify_reproduction.yaml")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = {"hybrid_fis": evaluate_fuzzy_csv(args.config, args.predictions, "hybrid")}
    out = Path(cfg["paths"]["logs_dir"]) / "ablation_results.json"
    write_json(out, results)
    print(f"Wrote ablation results to {out}")


if __name__ == "__main__":
    main()


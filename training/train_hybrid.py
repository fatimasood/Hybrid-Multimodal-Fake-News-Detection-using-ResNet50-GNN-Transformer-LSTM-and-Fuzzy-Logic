from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models.fuzzy.fuzzy_system import build_fis
from models.fuzzy.threshold_optimization import tune_thresholds_ga, tune_thresholds_grid
from utils.config import load_config
from utils.logging import write_json


def train_hybrid_thresholds(config_path: str, prediction_csv: str) -> None:
    cfg = load_config(config_path)
    frame = pd.read_csv(prediction_csv)
    labels = frame["label"].to_numpy(dtype=float)
    confidence = frame[["text_confidence", "image_confidence"]].mean(axis=1).to_numpy(dtype=float)
    grid = tune_thresholds_grid(confidence, labels, cfg["fuzzy"]["mfce_grid_points"])
    ga_cfg = cfg["fuzzy"]["ga"]
    ga = tune_thresholds_ga(
        confidence,
        labels,
        seed=cfg["experiment"]["seed"],
        population_size=ga_cfg["population_size"],
        generations=ga_cfg["generations"],
        mutation_rate=ga_cfg["mutation_rate"],
    )
    fis = build_fis(cfg["fuzzy"])
    sample_outputs = [
        fis.infer(float(row.text_confidence), float(row.image_confidence), getattr(row, "caption_text_similarity", None))
        for row in frame.head(5).itertuples(index=False)
    ]
    out = Path(cfg["paths"]["checkpoints_dir"]) / "fuzzy_thresholds.json"
    write_json(out, {"grid": {"t_low": grid[0], "t_high": grid[1], "mfce": grid[2]}, "ga": {"t_low": ga[0], "t_high": ga[1], "mfce": ga[2]}, "sample_outputs": sample_outputs})
    print(f"Wrote fuzzy threshold optimization summary to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trustify_reproduction.yaml")
    parser.add_argument("--predictions", required=True, help="Validation CSV with label,text_confidence,image_confidence.")
    args = parser.parse_args()
    train_hybrid_thresholds(args.config, args.predictions)


if __name__ == "__main__":
    main()


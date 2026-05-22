from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.compare_paper import compare_to_paper
from evaluation.metrics import binary_metrics
from models.fuzzy.fuzzy_system import build_fis
from utils.config import load_config
from utils.logging import write_json


def evaluate_fuzzy_csv(config_path: str, csv_path: str, method: str = "hybrid") -> dict:
    cfg = load_config(config_path)
    frame = pd.read_csv(csv_path)
    required = {"label", "text_confidence", "image_confidence"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")
    fis = build_fis(cfg["fuzzy"])
    predictions = []
    traces = []
    for row in frame.itertuples(index=False):
        similarity = getattr(row, "caption_text_similarity", None)
        output = fis.infer(float(row.text_confidence), float(row.image_confidence), similarity)
        predictions.append(output["prediction"])
        traces.append(output["trace"])
    metrics = binary_metrics(frame["label"].to_numpy(), np.asarray(predictions))
    dataset = cfg["dataset"]["active_dataset"]
    comparison = compare_to_paper(dataset, method, metrics) if (dataset, method) else {}
    return {"metrics": metrics, "paper_delta": comparison, "rule_traces": traces[:10]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Trustify fuzzy predictions.")
    parser.add_argument("--config", default="configs/trustify_reproduction.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--method", default="hybrid")
    parser.add_argument("--out", default="logs/evaluation.json")
    args = parser.parse_args()
    result = evaluate_fuzzy_csv(args.config, args.predictions, args.method)
    write_json(Path(args.out), result)
    print(result["metrics"])


if __name__ == "__main__":
    main()


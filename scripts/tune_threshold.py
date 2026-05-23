from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.metrics import best_f1_threshold, binary_metrics_with_scores
from utils.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune a binary decision threshold on validation predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--score-column", required=True, choices=["text_confidence", "image_confidence"])
    parser.add_argument("--out", default="logs/threshold.json")
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    y_true = frame["label"].to_numpy(dtype=int)
    y_score = frame[args.score_column].to_numpy(dtype=float)
    threshold, validation_f1 = best_f1_threshold(y_true, y_score)
    metrics = binary_metrics_with_scores(y_true, y_score, threshold)
    payload = {
        "score_column": args.score_column,
        "threshold": threshold,
        "validation_f1": validation_f1,
        "metrics_at_threshold": metrics,
    }
    write_json(Path(args.out), payload)
    print(payload)


if __name__ == "__main__":
    main()

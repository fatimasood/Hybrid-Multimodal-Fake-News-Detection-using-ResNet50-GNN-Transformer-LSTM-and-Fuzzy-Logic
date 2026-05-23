from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.metrics import binary_metrics_with_scores
from models.fuzzy.fuzzy_system import build_fis
from utils.config import load_config
from utils.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trustify fuzzy inference over combined branch predictions.")
    parser.add_argument("--config", default="configs/trustify_reproduction.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="logs/fuzzy_test_predictions.csv")
    parser.add_argument("--metrics-out", default="logs/fuzzy_test_metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fis = build_fis(cfg["fuzzy"])
    frame = pd.read_csv(args.predictions)
    rows = []
    traces = []
    for row in frame.itertuples(index=False):
        output = fis.infer(
            text_confidence=float(row.text_confidence),
            image_confidence=float(row.image_confidence),
            caption_text_similarity=float(getattr(row, "caption_text_similarity", 1.0)),
        )
        rows.append(
            {
                "label": int(row.label),
                "text_confidence": float(row.text_confidence),
                "image_confidence": float(row.image_confidence),
                "caption_text_similarity": float(getattr(row, "caption_text_similarity", 1.0)),
                "decision_score": output["decision_score"],
                "prediction": output["prediction"],
                "fuzzy_confidence_score": output["fuzzy_confidence_score"],
            }
        )
        traces.append(output["trace"])

    out_frame = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(out_path, index=False)
    metrics = binary_metrics_with_scores(
        out_frame["label"].to_numpy(dtype=int),
        out_frame["decision_score"].to_numpy(dtype=float),
        threshold=cfg["fuzzy"]["decision_threshold"],
    )
    write_json(args.metrics_out, {"metrics": metrics, "sample_rule_traces": traces[:10]})
    print(metrics)
    print(f"Wrote fuzzy predictions to {out_path}")


if __name__ == "__main__":
    main()

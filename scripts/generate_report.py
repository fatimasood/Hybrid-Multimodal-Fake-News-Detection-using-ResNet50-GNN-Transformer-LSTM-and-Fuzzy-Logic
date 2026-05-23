from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from evaluation.metrics import binary_metrics_with_scores
from utils.logging import write_json


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["Fake", "Real"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_curves(y_true: np.ndarray, y_score: np.ndarray, roc_out: Path, pr_out: Path, title_prefix: str) -> None:
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title(f"{title_prefix} ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_out, dpi=220)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.title(f"{title_prefix} Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(pr_out, dpi=220)
    plt.close()


def plot_metric_bar(metrics_by_name: dict[str, dict], out: Path) -> None:
    names = list(metrics_by_name)
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.15
    plt.figure(figsize=(10, 5))
    for idx, metric in enumerate(metric_names):
        values = [metrics_by_name[name].get(metric, 0.0) for name in names]
        plt.bar(x + (idx - 2) * width, values, width=width, label=metric.replace("_", " ").title())
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def load_prediction(path: str, score_column: str, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    frame = pd.read_csv(path)
    y_true = frame["label"].to_numpy(dtype=int)
    y_score = frame[score_column].to_numpy(dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    return y_true, y_score, y_pred, binary_metrics_with_scores(y_true, y_score, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research metrics, plots, and markdown report.")
    parser.add_argument("--text-predictions")
    parser.add_argument("--image-predictions")
    parser.add_argument("--hybrid-predictions")
    parser.add_argument("--text-threshold", type=float, default=0.5)
    parser.add_argument("--image-threshold", type=float, default=0.5)
    parser.add_argument("--hybrid-threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", default="logs/research_report")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_name: dict[str, dict] = {}

    if args.text_predictions:
        y_true, y_score, y_pred, metrics = load_prediction(args.text_predictions, "text_confidence", args.text_threshold)
        metrics_by_name["Text LSTM"] = metrics
        plot_confusion_matrix(y_true, y_pred, out_dir / "text_confusion_matrix.png", "Text LSTM Confusion Matrix")
        plot_curves(y_true, y_score, out_dir / "text_roc.png", out_dir / "text_pr.png", "Text LSTM")

    if args.image_predictions:
        y_true, y_score, y_pred, metrics = load_prediction(args.image_predictions, "image_confidence", args.image_threshold)
        metrics_by_name["Image GNN-Transformer"] = metrics
        plot_confusion_matrix(y_true, y_pred, out_dir / "image_confusion_matrix.png", "Image Branch Confusion Matrix")
        plot_curves(y_true, y_score, out_dir / "image_roc.png", out_dir / "image_pr.png", "Image Branch")

    if args.hybrid_predictions:
        frame = pd.read_csv(args.hybrid_predictions)
        if "decision_score" in frame.columns:
            score = frame["decision_score"].to_numpy(dtype=float)
        else:
            score = frame[["text_confidence", "image_confidence"]].mean(axis=1).to_numpy(dtype=float)
        y_true = frame["label"].to_numpy(dtype=int)
        y_pred = (score >= args.hybrid_threshold).astype(int)
        metrics = binary_metrics_with_scores(y_true, score, args.hybrid_threshold)
        metrics_by_name["Fuzzy Hybrid"] = metrics
        plot_confusion_matrix(y_true, y_pred, out_dir / "hybrid_confusion_matrix.png", "Fuzzy Hybrid Confusion Matrix")
        plot_curves(y_true, score, out_dir / "hybrid_roc.png", out_dir / "hybrid_pr.png", "Fuzzy Hybrid")

    plot_metric_bar(metrics_by_name, out_dir / "metric_comparison.png")
    write_json(out_dir / "metrics_summary.json", metrics_by_name)

    markdown = ["# Research Report\n", "\n## Metrics\n"]
    for name, metrics in metrics_by_name.items():
        markdown.append(f"\n### {name}\n")
        markdown.append("```json\n")
        markdown.append(json.dumps(metrics, indent=2))
        markdown.append("\n```\n")
    markdown.append("\n## Figures\n")
    for image in sorted(out_dir.glob("*.png")):
        markdown.append(f"- {image.name}\n")
    (out_dir / "REPORT.md").write_text("".join(markdown), encoding="utf-8")
    print(f"Wrote report to {out_dir}")


if __name__ == "__main__":
    main()

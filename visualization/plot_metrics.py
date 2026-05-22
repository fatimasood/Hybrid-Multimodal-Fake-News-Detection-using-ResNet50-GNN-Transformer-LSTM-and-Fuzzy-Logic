from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric_bars(results: dict, out_path: str | Path) -> None:
    labels = list(results)
    accuracies = [results[label]["accuracy"] for label in labels]
    f1_scores = [results[label]["f1"] for label in labels]
    x = range(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar([i - 0.2 for i in x], accuracies, width=0.4, label="Accuracy")
    plt.bar([i + 0.2 for i in x], f1_scores, width=0.4, label="F1")
    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


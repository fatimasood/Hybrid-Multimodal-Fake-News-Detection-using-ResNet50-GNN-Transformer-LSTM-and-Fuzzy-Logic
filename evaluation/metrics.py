from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def summarize_runs(metrics: list[dict]) -> dict:
    keys = [key for key in metrics[0] if key != "confusion_matrix"]
    return {
        key: {
            "mean": float(np.mean([run[key] for run in metrics])),
            "std": float(np.std([run[key] for run in metrics], ddof=0)),
        }
        for key in keys
    }


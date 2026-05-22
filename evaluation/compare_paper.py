from __future__ import annotations


PAPER_RESULTS = {
    ("twitter", "text_only"): {"precision": 0.968, "recall": 0.944, "f1": 0.955, "accuracy": 0.938},
    ("twitter", "image_only"): {"precision": 0.987, "recall": 0.960, "f1": 0.973, "accuracy": 0.965},
    ("twitter", "hybrid"): {"precision": 0.984, "recall": 0.972, "f1": 0.978, "accuracy": 0.961},
    ("buzzfeed", "text_only"): {"precision": 0.907, "recall": 0.912, "f1": 0.910, "accuracy": 0.928},
    ("buzzfeed", "image_only"): {"precision": 0.977, "recall": 0.920, "f1": 0.957, "accuracy": 0.962},
    ("buzzfeed", "hybrid"): {"precision": 0.949, "recall": 0.922, "f1": 0.935, "accuracy": 0.959},
    ("politifact", "text_only"): {"precision": 0.920, "recall": 0.914, "f1": 0.916, "accuracy": 0.922},
    ("politifact", "image_only"): {"precision": 0.957, "recall": 0.920, "f1": 0.946, "accuracy": 0.924},
    ("politifact", "hybrid"): {"precision": 0.949, "recall": 0.923, "f1": 0.939, "accuracy": 0.922},
}


def compare_to_paper(dataset: str, method: str, metrics: dict) -> dict:
    target = PAPER_RESULTS[(dataset.lower(), method)]
    return {key: float(metrics[key] - target[key]) for key in target}


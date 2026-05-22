from __future__ import annotations

import numpy as np


def mean_fuzzy_classification_error(predicted_memberships: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted_memberships - labels)))


def tune_thresholds_grid(confidences: np.ndarray, labels: np.ndarray, grid_points: int = 41) -> tuple[float, float, float]:
    best = (0.25, 0.75, float("inf"))
    grid = np.linspace(0.0, 1.0, grid_points)
    for t_low in grid:
        for t_high in grid:
            if t_high <= t_low:
                continue
            memberships = np.clip((confidences - t_low) / (t_high - t_low), 0.0, 1.0)
            mfce = mean_fuzzy_classification_error(memberships, labels)
            if mfce < best[2]:
                best = (float(t_low), float(t_high), mfce)
    return best


def tune_thresholds_ga(
    confidences: np.ndarray,
    labels: np.ndarray,
    seed: int,
    population_size: int = 24,
    generations: int = 20,
    mutation_rate: float = 0.08,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    population = np.sort(rng.uniform(0.0, 1.0, size=(population_size, 2)), axis=1)
    best = (0.25, 0.75, float("inf"))
    for _ in range(generations):
        scored = []
        for t_low, t_high in population:
            if t_high <= t_low:
                t_high = min(1.0, t_low + 1e-3)
            memberships = np.clip((confidences - t_low) / (t_high - t_low), 0.0, 1.0)
            mfce = mean_fuzzy_classification_error(memberships, labels)
            scored.append((mfce, t_low, t_high))
            if mfce < best[2]:
                best = (float(t_low), float(t_high), float(mfce))
        scored.sort(key=lambda item: item[0])
        parents = np.array([[item[1], item[2]] for item in scored[: max(2, population_size // 4)]])
        children = []
        while len(children) < population_size:
            p1, p2 = parents[rng.integers(0, len(parents), size=2)]
            alpha = rng.uniform()
            child = alpha * p1 + (1.0 - alpha) * p2
            if rng.uniform() < mutation_rate:
                child += rng.normal(0.0, 0.05, size=2)
            child = np.sort(np.clip(child, 0.0, 1.0))
            children.append(child)
        population = np.asarray(children)
    return best

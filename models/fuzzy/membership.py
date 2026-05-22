from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianMF:
    mean: float
    sigma: float

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        sigma = max(float(self.sigma), 1e-8)
        return np.exp(-0.5 * ((np.asarray(x) - self.mean) / sigma) ** 2)


@dataclass
class ThresholdMF:
    t_low: float
    t_high: float

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        x_arr = np.asarray(x)
        denom = max(self.t_high - self.t_low, 1e-8)
        return np.clip((x_arr - self.t_low) / denom, 0.0, 1.0)


def default_gaussian_memberships() -> dict[str, GaussianMF]:
    return {
        "Low": GaussianMF(mean=0.0, sigma=0.20),
        "Medium": GaussianMF(mean=0.5, sigma=0.20),
        "High": GaussianMF(mean=1.0, sigma=0.20),
    }


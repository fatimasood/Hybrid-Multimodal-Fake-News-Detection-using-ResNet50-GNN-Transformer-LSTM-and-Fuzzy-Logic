from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.fuzzy.membership import GaussianMF, default_gaussian_memberships


@dataclass
class FuzzyRule:
    text: str
    image: str
    consequent: str
    weight: float = 1.0


class TrustifyFuzzyInferenceSystem:
    """Gaussian fuzzification, rule firing, centroid defuzzification, and FCS."""

    def __init__(
        self,
        rules: list[FuzzyRule],
        output_centroids: dict[str, float],
        decision_threshold: float = 0.5,
        memberships: dict[str, GaussianMF] | None = None,
    ):
        self.rules = rules
        self.output_centroids = output_centroids
        self.decision_threshold = decision_threshold
        self.memberships = memberships or default_gaussian_memberships()

    @staticmethod
    def binary_membership(real_confidence: float) -> dict[str, float]:
        fake_confidence = 1.0 - real_confidence
        return {"real": float(real_confidence), "fake": float(fake_confidence)}

    def linguistic_membership(self, confidence: float) -> dict[str, float]:
        return {name: float(mf(confidence)) for name, mf in self.memberships.items()}

    def infer(self, text_confidence: float, image_confidence: float, caption_text_similarity: float | None = None) -> dict:
        text_mu = self.binary_membership(text_confidence)
        image_mu = self.binary_membership(image_confidence)
        numerator = 0.0
        denominator = 0.0
        weighted_firing = 0.0
        weighted_denominator = 0.0
        trace = []

        for idx, rule in enumerate(self.rules, start=1):
            mu = min(text_mu[rule.text], image_mu[rule.image])
            if caption_text_similarity is not None:
                mu = min(mu, float(caption_text_similarity))
            centroid = self.output_centroids[rule.consequent]
            numerator += mu * centroid
            denominator += mu
            weighted_firing += rule.weight * mu
            weighted_denominator += rule.weight
            trace.append(
                {
                    "rule": idx,
                    "text": rule.text,
                    "image": rule.image,
                    "consequent": rule.consequent,
                    "mu": mu,
                    "weight": rule.weight,
                    "centroid": centroid,
                }
            )

        crisp = numerator / denominator if denominator > 0 else 0.0
        fcs = weighted_firing / weighted_denominator if weighted_denominator > 0 else 0.0
        prediction = int(crisp >= self.decision_threshold)
        return {
            "decision_score": float(crisp),
            "prediction": prediction,
            "fuzzy_confidence_score": float(fcs),
            "text_linguistic": self.linguistic_membership(text_confidence),
            "image_linguistic": self.linguistic_membership(image_confidence),
            "trace": trace,
        }


def build_fis(config: dict) -> TrustifyFuzzyInferenceSystem:
    rules = [FuzzyRule(**rule) for rule in config["rules"]]
    return TrustifyFuzzyInferenceSystem(
        rules=rules,
        output_centroids=config["output_centroids"],
        decision_threshold=config["decision_threshold"],
    )


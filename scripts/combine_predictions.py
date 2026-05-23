from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine text and image branch predictions for fuzzy evaluation.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="logs/hybrid_test_predictions.csv")
    args = parser.parse_args()

    text = pd.read_csv(args.text)
    image = pd.read_csv(args.image)
    if len(text) != len(image):
        raise ValueError(f"Prediction lengths differ: text={len(text)}, image={len(image)}")
    if not np.array_equal(text["label"].to_numpy(), image["label"].to_numpy()):
        raise ValueError("Text and image prediction labels differ; check split/order.")

    combined = pd.DataFrame(
        {
            "label": text["label"].astype(int),
            "text_confidence": text["text_confidence"].astype(float),
            "image_confidence": image["image_confidence"].astype(float),
            # Placeholder semantic alignment score from confidence agreement.
            # Replace with a caption-text similarity model if reproducing this submodule separately.
            "caption_text_similarity": 1.0 - np.abs(text["text_confidence"].astype(float) - image["image_confidence"].astype(float)),
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"Wrote {len(combined)} rows to {out}")


if __name__ == "__main__":
    main()

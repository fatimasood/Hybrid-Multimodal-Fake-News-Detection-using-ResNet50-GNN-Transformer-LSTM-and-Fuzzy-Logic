from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Colab runner for the Fakeddit Trustify-style experiment.")
    parser.add_argument("--metadata", default="data/raw/fakeddit/multimodal_only_samples/multimodal_train.tsv")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--config", default="configs/fakeddit_colab.yaml")
    parser.add_argument("--skip-image-download", action="store_true")
    parser.add_argument("--skip-image-training", action="store_true")
    args = parser.parse_args()

    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "scripts/adapt_fakeddit.py",
            "--input",
            args.metadata,
            "--out",
            "data/processed/fakeddit_prepared.csv",
            "--sample-size",
            str(args.sample_size),
            "--balanced",
            "--image-root",
            "data/raw/fakeddit/images",
        ]
    )
    if not args.skip_image_download:
        run(
            [
                sys.executable,
                "scripts/download_fakeddit_images.py",
                "--csv",
                "data/processed/fakeddit_prepared.csv",
                "--workers",
                "16",
                "--keep-only-downloaded",
            ]
        )
    run([sys.executable, "reproduce.py", "--config", args.config, "--stage", "prepare", "--csv", "data/processed/fakeddit_prepared.csv"])
    run([sys.executable, "reproduce.py", "--config", args.config, "--stage", "train_text"])
    run([sys.executable, "scripts/predict_text.py", "--config", args.config, "--split", "test"])

    if not args.skip_image_training:
        run([sys.executable, "reproduce.py", "--config", args.config, "--stage", "train_image"])
        run([sys.executable, "scripts/predict_image.py", "--config", args.config, "--split", "test"])
        run(
            [
                sys.executable,
                "scripts/combine_predictions.py",
                "--text",
                "logs/text_test_predictions.csv",
                "--image",
                "logs/image_test_predictions.csv",
                "--out",
                "logs/hybrid_test_predictions.csv",
            ]
        )
        run([sys.executable, "scripts/predict_fuzzy.py", "--config", args.config, "--predictions", "logs/hybrid_test_predictions.csv"])
        run(
            [
                sys.executable,
                "scripts/generate_report.py",
                "--text-predictions",
                "logs/text_test_predictions.csv",
                "--image-predictions",
                "logs/image_test_predictions.csv",
                "--hybrid-predictions",
                "logs/fuzzy_test_predictions.csv",
            ]
        )


if __name__ == "__main__":
    main()

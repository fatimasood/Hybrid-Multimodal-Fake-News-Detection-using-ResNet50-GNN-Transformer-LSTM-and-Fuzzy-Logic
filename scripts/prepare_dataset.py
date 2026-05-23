from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data.datasets import read_dataset_csv
from data.splits import make_three_way_split
from preprocessing.text import build_text_preprocessor
from preprocessing.vocabulary import build_vocabulary
from utils.config import load_config
from utils.logging import write_json
from utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paper-aligned splits and vocabulary.")
    parser.add_argument("--config", default="configs/trustify_reproduction.yaml")
    parser.add_argument("--csv", required=True, help="Prepared CSV exposing text, image_path, caption, label columns.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["experiment"]["seed"], cfg["experiment"]["deterministic"])
    dataset_cfg = cfg["dataset"]
    split_cfg = dataset_cfg["split"]
    prep_cfg = cfg["preprocessing"]["text"]

    frame = read_dataset_csv(args.csv, dataset_cfg["label_column"], dataset_cfg["label_mapping"])
    text_preprocessor = build_text_preprocessor(prep_cfg)
    tokenized = [text_preprocessor.clean(text) for text in frame[dataset_cfg["text_column"]].fillna("")]
    vocab = build_vocabulary(tokenized)

    train_df, val_df, test_df = make_three_way_split(
        frame,
        dataset_cfg["label_column"],
        split_cfg["train"],
        split_cfg["validation"],
        split_cfg["test"],
        cfg["experiment"]["seed"],
        split_cfg["stratify"],
    )

    out = Path(cfg["paths"]["processed_dir"]) / dataset_cfg["active_dataset"]
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "validation.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    write_json(out / "vocab.json", vocab.token_to_idx)
    write_json(out / "split_counts.json", {"train": len(train_df), "validation": len(val_df), "test": len(test_df)})
    print(f"Prepared splits in {out}")


if __name__ == "__main__":
    main()

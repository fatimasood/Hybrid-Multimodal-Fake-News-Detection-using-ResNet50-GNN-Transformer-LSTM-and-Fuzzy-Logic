from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect prepared dataset labels and text lengths.")
    parser.add_argument("--csv", default="data/processed/fakeddit_prepared.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    print("Rows:", len(frame))
    print("Columns:", frame.columns.tolist())
    print("Label counts:", frame["label"].value_counts().sort_index().to_dict())
    lengths = frame["text"].fillna("").astype(str).str.split().str.len()
    print("Text word length summary:")
    print(lengths.describe())
    print("Examples:")
    print(frame[["text", "label", "image_path"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

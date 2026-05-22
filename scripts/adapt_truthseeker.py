from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt the Kaggle TruthSeeker CSV to the reproduction CSV schema.")
    parser.add_argument(
        "--input",
        default="data/raw/truthseekertwitterdataset2023/Truth_Seeker_Model_Dataset.csv",
    )
    parser.add_argument("--out", default="data/processed/truthseeker_prepared.csv")
    parser.add_argument("--text-source", choices=["tweet", "statement", "statement_tweet"], default="statement_tweet")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if "BinaryNumTarget" not in frame.columns:
        raise ValueError("Expected BinaryNumTarget label column in TruthSeeker dataset.")

    if args.text_source == "tweet":
        text = frame["tweet"].fillna("")
    elif args.text_source == "statement":
        text = frame["statement"].fillna("")
    else:
        text = frame["statement"].fillna("") + " " + frame["tweet"].fillna("")

    prepared = pd.DataFrame(
        {
            "text": text,
            "image_path": "",
            "caption": "",
            "label": frame["BinaryNumTarget"].astype(int),
        }
    )
    prepared = prepared.dropna(subset=["text", "label"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out, index=False)
    print(f"Wrote {len(prepared)} rows to {out}")
    print(prepared["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

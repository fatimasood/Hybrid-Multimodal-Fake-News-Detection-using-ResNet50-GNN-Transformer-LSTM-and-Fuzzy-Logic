from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"None of these columns found: {candidates}. Available columns: {frame.columns.tolist()}")


def normalize_fakeddit_label(frame: pd.DataFrame) -> pd.Series:
    if "2_way_label" in frame.columns:
        # Fakeddit convention: 0=true, 1=fake. Project convention: 1=real, 0=fake.
        return 1 - frame["2_way_label"].astype(int)
    if "6_way_label" in frame.columns:
        # Fakeddit 6-way: 0=true, all other classes are fake-like categories.
        return (frame["6_way_label"].astype(int) == 0).astype(int)
    if "label" in frame.columns:
        values = frame["label"]
        if pd.api.types.is_numeric_dtype(values):
            return values.astype(int)
        normalized = values.astype(str).str.lower().str.strip()
        return normalized.map({"real": 1, "true": 1, "fake": 0, "false": 0}).astype(int)
    raise ValueError("No supported label column found. Expected 2_way_label, 6_way_label, or label.")


def image_path_from_id_or_url(frame: pd.DataFrame, image_root: str) -> pd.Series:
    if "image_path" in frame.columns:
        return frame["image_path"].fillna("").astype(str)

    if "id" in frame.columns:
        ids = frame["id"].fillna("").astype(str)
        return ids.apply(lambda value: f"{image_root}/{value}.jpg" if value else "")

    if "image_url" in frame.columns:
        urls = frame["image_url"].fillna("").astype(str)
        return urls.apply(lambda url: url.split("/")[-1] if url else "")

    raise ValueError("No supported image field found. Expected image_path, id, or image_url.")


def stratified_sample(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    _, sample = train_test_split(
        frame,
        test_size=n,
        random_state=seed,
        shuffle=True,
        stratify=frame["label"] if frame["label"].nunique() > 1 else None,
    )
    return sample.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt Fakeddit metadata into this project's multimodal CSV schema.")
    parser.add_argument("--input", required=True, help="Fakeddit CSV/TSV file, e.g. all_train.tsv or multimodal_train.tsv.")
    parser.add_argument("--out", default="data/processed/fakeddit_prepared.csv")
    parser.add_argument("--image-root", default="data/raw/fakeddit/images")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = read_table(Path(args.input))
    text_column = find_column(frame, ["clean_title", "title", "text", "statement"])

    if "hasImage" in frame.columns:
        frame = frame[frame["hasImage"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    prepared = pd.DataFrame(
        {
            "text": frame[text_column].fillna("").astype(str),
            "image_path": image_path_from_id_or_url(frame, args.image_root),
            "caption": frame[text_column].fillna("").astype(str),
            "label": normalize_fakeddit_label(frame),
        }
    )
    prepared = prepared[prepared["text"].str.len() > 0].dropna(subset=["label"]).reset_index(drop=True)
    prepared = stratified_sample(prepared, args.sample_size, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out, index=False)
    print(f"Wrote {len(prepared)} rows to {out}")
    print("Project label convention: 1=real, 0=fake")
    print(prepared["label"].value_counts().sort_index().to_dict())
    print("Important: image_path values must point to existing local image files before train_image.")


if __name__ == "__main__":
    main()

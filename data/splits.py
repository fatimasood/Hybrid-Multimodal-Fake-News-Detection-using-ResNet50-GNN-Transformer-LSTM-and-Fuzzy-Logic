from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def make_three_way_split(
    frame: pd.DataFrame,
    label_column: str,
    train_size: float,
    validation_size: float,
    test_size: float,
    seed: int,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_size + validation_size + test_size - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    labels = frame[label_column] if stratify else None
    train_df, remaining = train_test_split(
        frame,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    remaining_val_fraction = validation_size / (validation_size + test_size)
    remaining_labels = remaining[label_column] if stratify else None
    val_df, test_df = train_test_split(
        remaining,
        train_size=remaining_val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=remaining_labels,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


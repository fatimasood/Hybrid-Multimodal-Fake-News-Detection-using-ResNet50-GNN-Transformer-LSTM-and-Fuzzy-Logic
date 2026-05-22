from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocessing.image import load_rgb_image
from preprocessing.vocabulary import Vocabulary


class TrustifyMultimodalDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        text_column: str,
        image_column: str,
        caption_column: str,
        label_column: str,
        vocab: Vocabulary,
        text_preprocessor: Callable[[str], list[str]],
        image_transform,
        max_text_length: int,
        max_caption_length: int,
        image_root: str | Path | None = None,
        allow_missing_images: bool = False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.text_column = text_column
        self.image_column = image_column
        self.caption_column = caption_column
        self.label_column = label_column
        self.vocab = vocab
        self.text_preprocessor = text_preprocessor
        self.image_transform = image_transform
        self.max_text_length = max_text_length
        self.max_caption_length = max_caption_length
        self.image_root = Path(image_root) if image_root else None
        self.allow_missing_images = allow_missing_images

    def __len__(self) -> int:
        return len(self.frame)

    def _image_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path))
        if not path.is_absolute() and self.image_root is not None:
            path = self.image_root / path
        return path

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[idx]
        text_tokens = self.text_preprocessor(str(row.get(self.text_column, "")))
        caption_tokens = self.text_preprocessor(str(row.get(self.caption_column, "")))
        text_ids = self.vocab.encode(text_tokens, self.max_text_length)
        caption_ids = self.vocab.encode(caption_tokens, self.max_caption_length, add_bos_eos=True)
        label = int(row[self.label_column])

        raw_image_value = row.get(self.image_column, "")
        image_path = self._image_path(raw_image_value)
        if str(raw_image_value).strip() and image_path.exists() and image_path.is_file():
            image = self.image_transform(load_rgb_image(image_path))
        elif self.allow_missing_images:
            image = torch.zeros(3, 224, 224)
        else:
            raise FileNotFoundError(f"Image not found for multimodal sample: {image_path}")

        return {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "caption_ids": torch.tensor(caption_ids, dtype=torch.long),
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "raw_text": str(row.get(self.text_column, "")),
        }


def read_dataset_csv(path: str | Path, label_column: str, label_mapping: dict[str, int]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if label_column not in frame.columns:
        raise ValueError(f"Label column {label_column!r} missing from {path}")
    if not pd.api.types.is_numeric_dtype(frame[label_column]):
        normalized = frame[label_column].astype(str).str.lower().str.strip()
        frame[label_column] = normalized.map(label_mapping)
    frame = frame.dropna(subset=[label_column]).copy()
    frame[label_column] = frame[label_column].astype(int)
    return frame

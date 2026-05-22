from __future__ import annotations

from pathlib import Path

from PIL import Image
from torchvision import transforms


def build_image_transform(config: dict):
    size = config.get("resize", [224, 224])
    return transforms.Compose(
        [
            transforms.Resize(tuple(size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["normalize_mean"], std=config["normalize_std"]),
        ]
    )


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


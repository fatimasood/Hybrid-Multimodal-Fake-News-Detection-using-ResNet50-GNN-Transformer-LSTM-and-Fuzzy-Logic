from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ("processed_dir", "checkpoints_dir", "logs_dir"):
        value = config.get("paths", {}).get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)


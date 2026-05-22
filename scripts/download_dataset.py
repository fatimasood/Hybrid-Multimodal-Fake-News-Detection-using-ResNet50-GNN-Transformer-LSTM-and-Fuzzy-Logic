from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the exact Kaggle dataset named in the paper.")
    parser.add_argument("--dataset", default="sudishbasnet/truthseekertwitterdataset2023")
    parser.add_argument("--out", default="data/raw/truthseekertwitterdataset2023")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(output_dir), "--unzip"],
        check=True,
    )
    print(f"Downloaded {args.dataset} to {output_dir}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image
from tqdm import tqdm


def download_one(row: dict, timeout: int = 20) -> tuple[bool, str]:
    url = str(row.get("image_url", "")).strip()
    out_path = Path(str(row.get("image_path", "")).strip())
    if not url or not out_path:
        return False, "missing_url_or_path"
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "exists"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.save(out_path, format="JPEG", quality=90)
        return True, "downloaded"
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return False, type(exc).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Fakeddit images listed in a prepared CSV.")
    parser.add_argument("--csv", default="data/processed/fakeddit_prepared.csv")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-only-downloaded", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    if "image_url" not in frame.columns or "image_path" not in frame.columns:
        raise ValueError("CSV must contain image_url and image_path columns. Re-run scripts/adapt_fakeddit.py.")

    records = frame.to_dict("records")
    ok_indices: set[int] = set()
    failures: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, row): idx for idx, row in enumerate(records)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            ok, status = future.result()
            if ok:
                ok_indices.add(idx)
            else:
                failures[status] = failures.get(status, 0) + 1

    print({"requested": len(frame), "downloaded_or_existing": len(ok_indices), "failures": failures})
    if args.keep_only_downloaded:
        filtered = frame.iloc[sorted(ok_indices)].reset_index(drop=True)
        filtered.to_csv(args.csv, index=False)
        print(f"Updated {args.csv} with {len(filtered)} rows that have local images.")


if __name__ == "__main__":
    main()

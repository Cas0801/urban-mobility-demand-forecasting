from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def download_month(year: int, month: int, output_dir: Path) -> Path:
    if not 2009 <= year <= 2100 or not 1 <= month <= 12:
        raise ValueError("Invalid year or month")
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    destination = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing {destination}")
        return destination
    partial = destination.with_suffix(".parquet.part")
    request = Request(f"{BASE_URL}/{filename}", headers={"User-Agent": "urban-mobility-research/1.0"})
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            partial.replace(destination)
            break
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            print(f"Retrying {filename} after attempt {attempt}")
            time.sleep(2 ** attempt)
    print(f"Downloaded {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--months", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    for month in args.months:
        download_month(args.year, month, args.output)


if __name__ == "__main__":
    main()

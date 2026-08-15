from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


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
    urlretrieve(f"{BASE_URL}/{filename}", destination)
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

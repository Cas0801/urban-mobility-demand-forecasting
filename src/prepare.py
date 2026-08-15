from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


REQUIRED = ["tpep_pickup_datetime", "PULocationID"]


def read_trips(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        match = re.search(r"(\d{4})-(\d{2})", path.name)
        if not match:
            raise ValueError(f"Cannot infer source month from {path.name}")
        year, month = map(int, match.groups())
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthBegin(1)
        frame = pd.read_parquet(path, columns=REQUIRED)
        frame["tpep_pickup_datetime"] = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
        frames.append(frame[frame.tpep_pickup_datetime.between(start, end, inclusive="left")])
    trips = pd.concat(frames, ignore_index=True)
    trips["PULocationID"] = pd.to_numeric(trips["PULocationID"], errors="coerce")
    trips = trips.dropna(subset=REQUIRED)
    trips = trips[trips.PULocationID.between(1, 265)]
    return trips


def aggregate_hourly(trips: pd.DataFrame) -> pd.DataFrame:
    clean = trips.assign(timestamp=trips.tpep_pickup_datetime.dt.floor("h"))
    demand = clean.groupby(["timestamp", "PULocationID"], observed=True).size().rename("demand").reset_index()
    demand = demand.rename(columns={"PULocationID": "zone_id"})
    zones = sorted(demand.zone_id.unique())
    hours = pd.date_range(demand.timestamp.min(), demand.timestamp.max(), freq="h")
    full_index = pd.MultiIndex.from_product([hours, zones], names=["timestamp", "zone_id"])
    result = demand.set_index(["timestamp", "zone_id"]).reindex(full_index, fill_value=0).reset_index()
    result["zone_id"] = result.zone_id.astype("int16")
    result["demand"] = result.demand.astype("int32")
    return result.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hourly_zone_demand.parquet"))
    args = parser.parse_args()
    paths = sorted(args.input.glob("yellow_tripdata_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No trip files found under {args.input}")
    hourly = aggregate_hourly(read_trips(paths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(args.output, index=False)
    print(f"Saved {len(hourly):,} zone hour rows to {args.output}")


if __name__ == "__main__":
    main()

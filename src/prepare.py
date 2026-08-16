from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


REQUIRED = ["tpep_pickup_datetime", "PULocationID"]


def read_month(path: Path) -> pd.DataFrame:
    match = re.search(r"(\d{4})-(\d{2})", path.name)
    if not match:
        raise ValueError(f"Cannot infer source month from {path.name}")
    year, month = map(int, match.groups())
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    trips = pd.read_parquet(path, columns=REQUIRED)
    trips["tpep_pickup_datetime"] = pd.to_datetime(trips["tpep_pickup_datetime"], errors="coerce")
    trips = trips[trips.tpep_pickup_datetime.between(start, end, inclusive="left")]
    trips["PULocationID"] = pd.to_numeric(trips["PULocationID"], errors="coerce")
    trips = trips.dropna(subset=REQUIRED)
    return trips[trips.PULocationID.between(1, 265)]


def read_trips(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([read_month(path) for path in paths], ignore_index=True)


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


def aggregate_paths(paths: list[Path]) -> pd.DataFrame:
    monthly_counts = []
    for path in paths:
        trips = read_month(path)
        counts = (
            trips.assign(timestamp=trips.tpep_pickup_datetime.dt.floor("h"))
            .groupby(["timestamp", "PULocationID"], observed=True)
            .size()
            .rename("demand")
            .reset_index()
            .rename(columns={"PULocationID": "zone_id"})
        )
        monthly_counts.append(counts)
        print(f"Aggregated {len(trips):,} valid trips from {path.name}")

    demand = pd.concat(monthly_counts, ignore_index=True)
    demand = demand.groupby(["timestamp", "zone_id"], as_index=False, observed=True).demand.sum()
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
    hourly = aggregate_paths(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(args.output, index=False)
    print(f"Saved {len(hourly):,} zone hour rows to {args.output}")


if __name__ == "__main__":
    main()

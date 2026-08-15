import pandas as pd
import pytest

from src.features import build_features, make_supervised
from src.prepare import aggregate_hourly
from src.train import temporal_split
from src.monitor import create_drift_report


def test_hourly_aggregation_includes_zero_demand():
    trips = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2024-01-01 00:10", "2024-01-01 02:20"]),
        "PULocationID": [1, 1],
    })
    result = aggregate_hourly(trips)
    assert result.demand.tolist() == [1, 0, 1]


def test_hourly_aggregation_keeps_zone_grid():
    trips = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2024-01-01 00:10", "2024-01-01 01:20"]),
        "PULocationID": [1, 2],
    })
    result = aggregate_hourly(trips)
    assert len(result) == 4
    assert result.demand.sum() == 2


def test_lag_features_use_only_past_values():
    hours = pd.date_range("2024-01-01", periods=200, freq="h")
    hourly = pd.DataFrame({"timestamp": hours, "zone_id": 1, "demand": range(200)})
    featured = build_features(hourly)
    first = featured.iloc[0]
    assert first.lag_1 == first.demand - 1
    assert first.lag_24 == first.demand - 24
    assert first.lag_168 == first.demand - 168


def test_temporal_split_is_strictly_ordered():
    data = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=24 * 30, freq="h")})
    train, validation, test = temporal_split(data)
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()


def test_direct_target_and_baseline_are_horizon_correct():
    hours = pd.date_range("2024-01-01", periods=400, freq="h")
    hourly = pd.DataFrame({"timestamp": hours, "zone_id": 1, "demand": range(400)})
    supervised = make_supervised(build_features(hourly), horizon=24)
    row = supervised.iloc[0]
    assert row.target == row.demand + 24
    assert row.baseline == row.target - 168
    assert row.target_timestamp == row.timestamp + pd.Timedelta(hours=24)


def test_drift_report_flags_shifted_distribution():
    profile = {
        name: {"edges": [float("-inf"), 5, float("inf")], "proportions": [0.5, 0.5], "missing_rate": 0.0}
        for name in ["lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168"]
    }
    shifted = pd.DataFrame({name: [100] * 20 for name in profile})
    report = create_drift_report(shifted, profile)
    assert report["status"] == "alert"
    assert all(item["psi"] > 0.25 for item in report["features"].values())

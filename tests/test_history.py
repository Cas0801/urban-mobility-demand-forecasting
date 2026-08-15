import pandas as pd
import pytest

from src.history import historical_features


def sample_history(hours=200):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=hours, freq="h"),
        "zone_id": 1,
        "demand": range(hours),
    })


def test_historical_features_are_computed_before_origin():
    data = sample_history()
    origin = pd.Timestamp("2024-01-09 00:00:00")

    features = historical_features(data, zone_id=1, forecast_origin=origin)

    assert features["lag_1"] == 191
    assert features["lag_24"] == 168
    assert features["lag_168"] == 24
    assert features["rolling_mean_24"] == pytest.approx(sum(range(168, 192)) / 24)
    assert features["rolling_mean_168"] == pytest.approx(sum(range(24, 192)) / 168)


def test_historical_features_require_complete_week():
    with pytest.raises(ValueError, match="168 complete hours"):
        historical_features(sample_history(100), zone_id=1, forecast_origin=pd.Timestamp("2024-01-05"))

import pytest
from pydantic import ValidationError

from src.api import ForecastRequest, predict_request


def valid_request(**updates):
    values = dict(
        horizon_hours=1, zone_id=161, target_hour=18, target_day_of_week=4,
        target_is_weekend=0, target_month=3, lag_1=120, lag_24=110,
        lag_168=105, rolling_mean_24=100, rolling_mean_168=95,
    )
    values.update(updates)
    return ForecastRequest(**values)


def test_request_rejects_invalid_hour():
    with pytest.raises(ValidationError):
        valid_request(target_hour=24)


def test_prediction_returns_interval_for_one_hour():
    response = predict_request(valid_request())
    assert response.prediction >= 0
    assert 0 <= response.p10 <= response.p90


def test_prediction_rejects_unsupported_horizon():
    with pytest.raises(ValueError):
        predict_request(valid_request(horizon_hours=12))

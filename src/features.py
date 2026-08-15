from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "zone_id", "target_hour", "target_day_of_week", "target_is_weekend", "target_month",
    "lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168",
]


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    data = hourly.sort_values(["zone_id", "timestamp"]).copy()
    data["hour"] = data.timestamp.dt.hour.astype("int8")
    data["day_of_week"] = data.timestamp.dt.dayofweek.astype("int8")
    data["is_weekend"] = data.day_of_week.ge(5).astype("int8")
    data["month"] = data.timestamp.dt.month.astype("int8")
    groups = data.groupby("zone_id", observed=True)["demand"]
    data["lag_1"] = groups.shift(1)
    data["lag_24"] = groups.shift(24)
    data["lag_168"] = groups.shift(168)
    shifted = groups.shift(1)
    data["rolling_mean_24"] = shifted.groupby(data.zone_id).rolling(24).mean().reset_index(level=0, drop=True)
    data["rolling_mean_168"] = shifted.groupby(data.zone_id).rolling(168).mean().reset_index(level=0, drop=True)
    history_columns = ["lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168"]
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=history_columns).reset_index(drop=True)


def make_supervised(featured: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create a direct forecast target using only features available at the origin."""
    if not 1 <= horizon < 168:
        raise ValueError("Horizon must be between 1 and 167 hours")
    data = featured.copy()
    groups = data.groupby("zone_id", observed=True)["demand"]
    data["target"] = groups.shift(-horizon)
    data["baseline"] = groups.shift(168 - horizon)
    data["target_timestamp"] = data.timestamp + pd.Timedelta(hours=horizon)
    data["target_hour"] = data.target_timestamp.dt.hour.astype("int8")
    data["target_day_of_week"] = data.target_timestamp.dt.dayofweek.astype("int8")
    data["target_is_weekend"] = data.target_day_of_week.ge(5).astype("int8")
    data["target_month"] = data.target_timestamp.dt.month.astype("int8")
    return data.dropna(subset=FEATURE_COLUMNS + ["target", "baseline"]).reset_index(drop=True)

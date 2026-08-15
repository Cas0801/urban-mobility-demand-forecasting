from __future__ import annotations

from datetime import datetime

import pandas as pd


HISTORY_COLUMNS = ("lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168")


def historical_features(hourly: pd.DataFrame, zone_id: int, forecast_origin: datetime) -> dict[str, float]:
    zone = hourly.loc[hourly["zone_id"].eq(zone_id), ["timestamp", "demand"]].copy()
    if zone.empty:
        raise ValueError(f"Zone {zone_id} is not available in the history snapshot")

    zone["timestamp"] = pd.to_datetime(zone["timestamp"])
    demand = zone.drop_duplicates("timestamp").set_index("timestamp")["demand"].sort_index()
    origin = pd.Timestamp(forecast_origin).floor("h")
    history = demand.loc[demand.index < origin]
    if len(history) < 168:
        raise ValueError("At least 168 complete hours of history are required before the forecast origin")

    required = {
        "lag_1": origin - pd.Timedelta(hours=1),
        "lag_24": origin - pd.Timedelta(hours=24),
        "lag_168": origin - pd.Timedelta(hours=168),
    }
    if any(timestamp not in demand.index for timestamp in required.values()):
        raise ValueError("The selected forecast origin has incomplete historical coverage")

    return {
        "lag_1": float(demand.loc[required["lag_1"]]),
        "lag_24": float(demand.loc[required["lag_24"]]),
        "lag_168": float(demand.loc[required["lag_168"]]),
        "rolling_mean_24": float(history.tail(24).mean()),
        "rolling_mean_168": float(history.tail(168).mean()),
    }

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import FEATURE_COLUMNS


MODEL_DIR = Path("artifacts")
SUPPORTED_HORIZONS = (1, 6, 24)


class ForecastRequest(BaseModel):
    horizon_hours: int = Field(description="Direct forecast horizon", examples=[1])
    zone_id: int = Field(ge=1, le=265)
    target_hour: int = Field(ge=0, le=23)
    target_day_of_week: int = Field(ge=0, le=6)
    target_is_weekend: int = Field(ge=0, le=1)
    target_month: int = Field(ge=1, le=12)
    lag_1: float = Field(ge=0)
    lag_24: float = Field(ge=0)
    lag_168: float = Field(ge=0)
    rolling_mean_24: float = Field(ge=0)
    rolling_mean_168: float = Field(ge=0)


class ForecastResponse(BaseModel):
    horizon_hours: int
    prediction: float
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None


@lru_cache(maxsize=8)
def load_booster(filename: str) -> lgb.Booster:
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return lgb.Booster(model_file=str(path))


def predict_request(request: ForecastRequest) -> ForecastResponse:
    if request.horizon_hours not in SUPPORTED_HORIZONS:
        raise ValueError(f"Supported horizons are {SUPPORTED_HORIZONS}")
    row = request.model_dump(exclude={"horizon_hours"})
    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    prediction = max(0.0, float(load_booster(f"lightgbm_h{request.horizon_hours}.txt").predict(frame)[0]))
    response = ForecastResponse(horizon_hours=request.horizon_hours, prediction=prediction)
    if request.horizon_hours == 1:
        response.p10 = max(0.0, float(load_booster("lightgbm_h1_q10.txt").predict(frame)[0]))
        response.p50 = max(0.0, float(load_booster("lightgbm_h1_q50.txt").predict(frame)[0]))
        response.p90 = max(0.0, float(load_booster("lightgbm_h1_q90.txt").predict(frame)[0]))
    return response


app = FastAPI(title="Urban Mobility Demand Forecast API", version="1.0.0")


@app.get("/health")
def health():
    available = {horizon: (MODEL_DIR / f"lightgbm_h{horizon}.txt").exists() for horizon in SUPPORTED_HORIZONS}
    return {"status": "healthy" if all(available.values()) else "degraded", "models": available}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    try:
        return predict_request(request)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

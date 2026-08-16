from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluate import metrics
from src.features import FEATURE_COLUMNS, build_features, make_supervised
from src.train import HORIZONS, fit_model, new_model


def choose_origins(
    data: pd.DataFrame,
    folds: int = 3,
    minimum_train_days: int = 120,
    validation_days: int = 14,
    test_days: int = 28,
) -> list[pd.Timestamp]:
    earliest = data.target_timestamp.min().normalize()
    latest = data.target_timestamp.max().normalize()
    first_origin = earliest + pd.Timedelta(days=minimum_train_days + validation_days)
    last_origin = latest - pd.Timedelta(days=test_days - 1)
    if first_origin > last_origin:
        raise ValueError("Not enough history for the requested rolling backtest")
    values = pd.date_range(first_origin, last_origin, periods=folds)
    return [pd.Timestamp(value).floor("D") for value in values]


def rolling_origin_splits(
    data: pd.DataFrame,
    origins: list[pd.Timestamp],
    validation_days: int = 14,
    test_days: int = 28,
):
    for origin in origins:
        origin = pd.Timestamp(origin)
        validation_start = origin - pd.Timedelta(days=validation_days)
        test_end = origin + pd.Timedelta(days=test_days)
        train = data[data.target_timestamp < validation_start]
        validation = data[(data.target_timestamp >= validation_start) & (data.target_timestamp < origin)]
        test = data[(data.target_timestamp >= origin) & (data.target_timestamp < test_end)]
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError(f"Empty rolling split at {origin}")
        yield origin, train, validation, test


def summarize(results: pd.DataFrame) -> dict:
    test = results[results.model.eq("lightgbm")].copy()
    summary = {}
    for horizon, group in test.groupby("horizon_hours"):
        summary[str(horizon)] = {
            "folds": int(len(group)),
            "rmse_mean": float(group.rmse.mean()),
            "rmse_std": float(group.rmse.std(ddof=0)),
            "improvement_mean": float(group.rmse_improvement.mean()),
            "improved_folds": int((group.rmse_improvement > 0).sum()),
        }
    return summary


def run_backtest(
    input_path: Path,
    artifacts: Path,
    horizons=HORIZONS,
    folds: int = 3,
) -> pd.DataFrame:
    featured = build_features(pd.read_parquet(input_path))
    rows = []
    for horizon in horizons:
        supervised = make_supervised(featured, horizon)
        origins = choose_origins(supervised, folds=folds)
        for fold, (origin, train, validation, test) in enumerate(rolling_origin_splits(supervised, origins), start=1):
            model = fit_model(new_model(n_estimators=400), train, validation)
            prediction = model.predict(test[FEATURE_COLUMNS])
            peak_threshold = train.target.quantile(0.9)
            baseline_metrics = metrics(test.target, test.baseline, peak_threshold)
            model_metrics = metrics(test.target, prediction, peak_threshold)
            improvement = 1 - model_metrics["rmse"] / baseline_metrics["rmse"]
            common = {
                "horizon_hours": horizon,
                "fold": fold,
                "origin": origin,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
            }
            rows.append({**common, "model": "same_hour_last_week", "rmse_improvement": 0.0, **baseline_metrics})
            rows.append({**common, "model": "lightgbm", "rmse_improvement": improvement, **model_metrics})
            print(f"Finished horizon {horizon}, fold {fold}, origin {origin.date()}, improvement {improvement:.2%}")

    results = pd.DataFrame(rows)
    artifacts.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts / "backtest_metrics.csv", index=False)
    (artifacts / "backtest_summary.json").write_text(json.dumps(summarize(results), indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/hourly_zone_demand.parquet"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    results = run_backtest(args.input, args.artifacts, tuple(args.horizons), args.folds)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate import metrics
from src.features import FEATURE_COLUMNS, build_features, make_supervised


HORIZONS = (1, 6, 24)
CATEGORICAL = ["zone_id", "target_hour", "target_day_of_week", "target_month"]
MONITORED_FEATURES = ["lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168"]


def temporal_split(data: pd.DataFrame, validation_days: int = 14, test_days: int = 14):
    time_column = "target_timestamp" if "target_timestamp" in data else "timestamp"
    end = data[time_column].max()
    test_start = end - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    validation_start = test_start - pd.Timedelta(days=validation_days)
    return (
        data[data[time_column] < validation_start],
        data[(data[time_column] >= validation_start) & (data[time_column] < test_start)],
        data[data[time_column] >= test_start],
    )


def new_model(objective: str = "poisson", alpha: float | None = None):
    parameters = dict(
        objective=objective, n_estimators=700, learning_rate=0.04,
        num_leaves=63, min_child_samples=50, subsample=0.9,
        colsample_bytree=0.9, random_state=42, n_jobs=4,
        verbosity=-1, force_row_wise=True,
    )
    if alpha is not None:
        parameters["alpha"] = alpha
    return lgb.LGBMRegressor(**parameters)


def fit_model(model, train: pd.DataFrame, validation: pd.DataFrame):
    model.fit(
        train[FEATURE_COLUMNS], train.target,
        eval_set=[(validation[FEATURE_COLUMNS], validation.target)],
        callbacks=[lgb.early_stopping(60, verbose=False)],
        categorical_feature=CATEGORICAL,
    )
    return model


def make_reference_profile(data: pd.DataFrame, bins: int = 10) -> dict:
    profile = {}
    for column in MONITORED_FEATURES:
        values = data[column].dropna()
        edges = sorted(set(float(value) for value in values.quantile([index / bins for index in range(bins + 1)])))
        if len(edges) < 2:
            edges = [float(values.min()) - 1, float(values.max()) + 1]
        edges[0], edges[-1] = float("-inf"), float("inf")
        counts = pd.cut(values, bins=edges, include_lowest=True).value_counts(sort=False, normalize=True)
        profile[column] = {"edges": edges, "proportions": counts.tolist(), "missing_rate": float(data[column].isna().mean())}
    return profile


def run(input_path: Path, artifacts: Path, horizons=HORIZONS) -> pd.DataFrame:
    featured = build_features(pd.read_parquet(input_path))
    artifacts.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    sample_frames = []
    probabilistic_summary = {}
    metadata = {"features": FEATURE_COLUMNS, "horizons": list(horizons), "runs": {}}

    for horizon in horizons:
        supervised = make_supervised(featured, horizon)
        train, validation, test = temporal_split(supervised)
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError(f"Not enough history for horizon {horizon}")
        model = fit_model(new_model(), train, validation)
        peak_threshold = train.target.quantile(0.9)

        for split_name, split in (("validation", validation), ("test", test)):
            prediction = model.predict(split[FEATURE_COLUMNS])
            metric_rows.append({"horizon_hours": horizon, "model": "same_hour_last_week", "split": split_name,
                                **metrics(split.target, split.baseline, peak_threshold)})
            metric_rows.append({"horizon_hours": horizon, "model": "lightgbm", "split": split_name,
                                **metrics(split.target, prediction, peak_threshold)})
            if split_name == "test":
                sample = split[["target_timestamp", "zone_id", "target", "baseline"]].copy()
                sample["prediction"] = prediction
                sample["horizon_hours"] = horizon
                sample_frames.append(sample.sample(min(5000, len(sample)), random_state=42))

        model.booster_.save_model(artifacts / f"lightgbm_h{horizon}.txt")

        if horizon == 1:
            quantile_predictions = {}
            for quantile in (0.1, 0.5, 0.9):
                quantile_model = fit_model(new_model("quantile", quantile), train, validation)
                quantile_predictions[quantile] = np.clip(quantile_model.predict(test[FEATURE_COLUMNS]), 0, None)
                quantile_model.booster_.save_model(artifacts / f"lightgbm_h1_q{int(quantile * 100)}.txt")
            lower, median, upper = (quantile_predictions[value] for value in (0.1, 0.5, 0.9))
            probabilistic_summary = {
                "nominal_coverage": 0.8,
                "empirical_coverage": float(((test.target >= lower) & (test.target <= upper)).mean()),
                "mean_interval_width": float((upper - lower).mean()),
                "median_metrics": metrics(test.target, median, peak_threshold),
            }
            busiest_zones = train.groupby("zone_id").target.mean().nlargest(20).index
            interval_sample = test.loc[test.zone_id.isin(busiest_zones), ["target_timestamp", "zone_id", "target"]].copy()
            interval_sample["p10"] = lower[test.zone_id.isin(busiest_zones)]
            interval_sample["p50"] = median[test.zone_id.isin(busiest_zones)]
            interval_sample["p90"] = upper[test.zone_id.isin(busiest_zones)]
            interval_sample.to_csv(artifacts / "interval_sample.csv", index=False)
        metadata["runs"][str(horizon)] = {
            "train_end": str(train.target_timestamp.max()),
            "validation_end": str(validation.target_timestamp.max()),
            "test_end": str(test.target_timestamp.max()),
            "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        }
        if horizon == 1:
            metadata["reference_profile"] = make_reference_profile(train)

    results = pd.DataFrame(metric_rows)
    results.to_csv(artifacts / "metrics.csv", index=False)
    pd.concat(sample_frames, ignore_index=True).to_csv(artifacts / "prediction_sample.csv", index=False)
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (artifacts / "probabilistic_metrics.json").write_text(json.dumps(probabilistic_summary, indent=2))
    print(results.to_string(index=False))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/hourly_zone_demand.parquet"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    args = parser.parse_args()
    run(args.input, args.artifacts, tuple(args.horizons))


if __name__ == "__main__":
    main()

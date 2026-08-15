from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_features
from src.train import MONITORED_FEATURES


def population_stability_index(values: pd.Series, reference: dict, epsilon: float = 1e-6) -> float:
    clean = values.dropna()
    actual = pd.cut(clean, bins=reference["edges"], include_lowest=True).value_counts(sort=False, normalize=True).to_numpy()
    expected = np.asarray(reference["proportions"])
    actual = np.clip(actual, epsilon, None)
    expected = np.clip(expected, epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def create_drift_report(data: pd.DataFrame, profile: dict, warning_threshold: float = 0.1, alert_threshold: float = 0.25) -> dict:
    features = {}
    overall = "ok"
    for column in MONITORED_FEATURES:
        psi = population_stability_index(data[column], profile[column])
        missing_rate = float(data[column].isna().mean())
        status = "alert" if psi >= alert_threshold else "warning" if psi >= warning_threshold else "ok"
        if status == "alert":
            overall = "alert"
        elif status == "warning" and overall == "ok":
            overall = "warning"
        features[column] = {"psi": psi, "missing_rate": missing_rate, "status": status}
    return {"status": overall, "rows": len(data), "features": features}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/hourly_zone_demand.parquet"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift_report.json"))
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    featured = build_features(pd.read_parquet(args.input))
    recent = featured[featured.timestamp >= featured.timestamp.max() - pd.Timedelta(days=args.days)]
    profile = json.loads(args.metadata.read_text())["reference_profile"]
    report = create_drift_report(recent, profile)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

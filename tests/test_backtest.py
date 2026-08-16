import pandas as pd

from src.backtest import choose_origins, rolling_origin_splits, summarize


def test_rolling_splits_are_ordered_and_disjoint():
    timestamps = pd.date_range("2024-01-01", periods=24 * 240, freq="h")
    data = pd.DataFrame({"target_timestamp": timestamps})
    origins = choose_origins(data, folds=3, minimum_train_days=90)

    splits = list(rolling_origin_splits(data, origins))

    assert len(splits) == 3
    for origin, train, validation, test in splits:
        assert train.target_timestamp.max() < validation.target_timestamp.min()
        assert validation.target_timestamp.max() < test.target_timestamp.min()
        assert test.target_timestamp.min() == origin


def test_backtest_summary_reports_stability():
    results = pd.DataFrame({
        "horizon_hours": [1, 1, 1],
        "model": ["lightgbm"] * 3,
        "rmse": [10.0, 12.0, 11.0],
        "rmse_improvement": [0.10, -0.02, 0.05],
    })

    summary = summarize(results)["1"]

    assert summary["folds"] == 3
    assert summary["rmse_mean"] == 11.0
    assert summary["improved_folds"] == 2

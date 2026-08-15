import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(actual, predicted, peak_threshold=None) -> dict:
    actual = np.asarray(actual)
    predicted = np.clip(np.asarray(predicted), 0, None)
    result = {
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "wmape": float(np.abs(actual - predicted).sum() / max(actual.sum(), 1)),
    }
    if peak_threshold is not None:
        mask = actual >= peak_threshold
        result["peak_rmse"] = float(mean_squared_error(actual[mask], predicted[mask]) ** 0.5) if mask.any() else None
    return result

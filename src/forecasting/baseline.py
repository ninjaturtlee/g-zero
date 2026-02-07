import numpy as np


def seasonal_naive_forecast(y: np.ndarray, season: int = 7) -> np.ndarray:
    """
    Seasonal naive forecast: y_hat[t] = y[t-season].
    For t < season, repeat y[0].
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    pred = np.zeros(n, dtype=float)
    for t in range(n):
        pred[t] = y[t - season] if (t - season) >= 0 else y[0]
    return pred
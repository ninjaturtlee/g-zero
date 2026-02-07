from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass
class XGBForecastModel:
    model: XGBRegressor

    @staticmethod
    def build(params: Dict[str, Any], seed: int = 42) -> "XGBForecastModel":
        m = XGBRegressor(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            objective=str(params.get("objective", "reg:squarederror")),
            random_state=int(seed),
            n_jobs=int(params.get("n_jobs", -1)),
        )
        return XGBForecastModel(model=m)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def make_time_features(series: pd.Series) -> pd.DataFrame:
    """
    Calendar + seasonality features for daily series.
    """
    idx = pd.to_datetime(series.index)
    df = pd.DataFrame(index=idx)

    # calendar
    df["dow"] = idx.dayofweek
    df["dom"] = idx.day
    df["month"] = idx.month

    # strong weekly seasonality encoded smoothly
    dow = idx.dayofweek.to_numpy()
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    # simple trend feature
    df["t"] = np.arange(len(df), dtype=float)

    return df


def add_lags(df: pd.DataFrame, y: pd.Series, lags=(1, 7), roll_window: int = 7) -> pd.DataFrame:
    out = df.copy()
    for l in lags:
        out[f"lag_{l}"] = y.shift(l)
    out[f"rollmean_{roll_window}"] = y.shift(1).rolling(roll_window).mean()
    return out


def train_val_split_time(
    X: pd.DataFrame,
    y: np.ndarray,
    val_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    n = len(X)
    cut = int(n * (1 - val_ratio))
    X_tr = X.iloc[:cut].copy()
    X_va = X.iloc[cut:].copy()
    y_tr = y[:cut]
    y_va = y[cut:]
    return X_tr, y_tr, X_va, y_va
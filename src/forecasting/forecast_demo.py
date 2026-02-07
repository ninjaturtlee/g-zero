import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.io import load_yaml
from src.utils.metrics import mae, rmse
from src.forecasting.baseline import seasonal_naive_forecast
from src.forecasting.xgboost_model import (
    XGBForecastModel,
    make_time_features,
    add_lags,
    train_val_split_time,
)


def generate_synthetic_series(days: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days, freq="D")

    t = np.arange(days)
    base = 8000.0
    weekly = 2000.0 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 800.0, size=days)
    y = base + weekly + noise

    # occasional demand spikes (e.g., holidays)
    spike_days = rng.choice(days, size=max(1, days // 30), replace=False)
    y[spike_days] += rng.uniform(5000, 12000, size=len(spike_days))

    y = np.clip(y, 0, None)
    return pd.Series(y, index=idx, name="demand")


def main() -> None:
    sim_cfg = load_yaml(Path("configs/sim.yaml"))
    train_cfg = load_yaml(Path("configs/train.yaml"))

    days = int(sim_cfg["horizon_days"])
    seed = int(sim_cfg["seed"])

    series = generate_synthetic_series(days, seed=seed)

    # Baseline: seasonal naive (weekly)
    baseline_pred_full = seasonal_naive_forecast(series.values, season=7)

    # XGBoost features
    base_feats = make_time_features(series)
    feats = add_lags(
        base_feats,
        series,
        lags=tuple(train_cfg["lags"]),
        roll_window=int(train_cfg["roll_window"]),
    )

    # Align (drop NaNs from lags/rolling)
    df = feats.copy()
    df["y"] = series.values
    df = df.dropna()
    y = df["y"].values
    X = df.drop(columns=["y"])

    X_tr, y_tr, X_va, y_va = train_val_split_time(
        X, y, val_ratio=float(train_cfg["test_ratio"])
    )

    model = XGBForecastModel.build(train_cfg["xgboost"], seed=seed)
    model.fit(X_tr, y_tr)
    xgb_pred = model.predict(X_va)

    # Baseline on same validation window (align length)
    baseline_df = pd.Series(baseline_pred_full, index=series.index).loc[df.index]
    cut = int(len(df) * (1 - float(train_cfg["test_ratio"])))
    baseline_va = baseline_df.iloc[cut:].values

    # --- Overall metrics ---
    base_mae = mae(y_va, baseline_va)
    base_rmse = rmse(y_va, baseline_va)
    xgb_mae = mae(y_va, xgb_pred)
    xgb_rmse = rmse(y_va, xgb_pred)

    improvement = (base_mae - xgb_mae) / base_mae * 100

    print("\n=== Forecasting Demo (Synthetic) ===")
    print(f"Train rows: {len(X_tr)} | Val rows: {len(X_va)}")

    print("\nBaseline (seasonal naive, weekly):")
    print(f"  MAE : {base_mae:.2f}")
    print(f"  RMSE: {base_rmse:.2f}")

    print("\nXGBoost:")
    print(f"  MAE : {xgb_mae:.2f}")
    print(f"  RMSE: {xgb_rmse:.2f}")

    print(f"\nMAE improvement vs baseline: {improvement:.1f}%")

    # --- Tail-risk evaluation (top 10% demand days in validation) ---
    tail_pct = 90
    threshold = np.percentile(y_va, tail_pct)
    mask = y_va >= threshold

    base_tail_mae = mae(y_va[mask], baseline_va[mask])
    xgb_tail_mae = mae(y_va[mask], xgb_pred[mask])
    tail_improvement = (base_tail_mae - xgb_tail_mae) / base_tail_mae * 100

    print(f"\nTail-risk evaluation (top {100 - tail_pct}% demand days in validation):")
    print(f"  Threshold (>= p{tail_pct}): {threshold:.2f}")
    print(f"  Baseline MAE: {base_tail_mae:.2f}")
    print(f"  XGBoost MAE : {xgb_tail_mae:.2f}")
    print(f"  Improvement on spike days: {tail_improvement:.1f}%\n")


if __name__ == "__main__":
    main()

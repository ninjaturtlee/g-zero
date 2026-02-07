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

from src.forecasting.bank_real_generator import BankRealGeneratorConfig, generate_bank_real_series


def underforecast_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Penalize only under-forecasting (stockout-risk proxy)."""
    under = np.maximum(y_true - y_pred, 0.0)
    return float(np.mean(under))


def moving_average_forecast(y: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Simple causal baseline: predicts next value as mean of last `window` values.
    Pads first `window` with the first available average.
    """
    y = np.asarray(y, dtype=float)
    preds = np.zeros_like(y)
    for i in range(len(y)):
        start = max(0, i - window)
        hist = y[start:i]
        if len(hist) == 0:
            preds[i] = y[0]
        else:
            preds[i] = float(np.mean(hist))
    return preds


def main() -> None:
    sim_cfg = load_yaml(Path("configs/sim.yaml"))
    train_cfg = load_yaml(Path("configs/train.yaml"))

    days = int(sim_cfg["horizon_days"])
    seed = int(sim_cfg["seed"])

    # --- Bank-real synthetic demand ---
    series = generate_bank_real_series(BankRealGeneratorConfig(days=days, seed=seed))

    # --- Baselines ---
    seasonal_pred_full = seasonal_naive_forecast(series.values, season=7)
    ma7_pred_full = moving_average_forecast(series.values, window=7)

    # --- Features for ML model ---
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

    # Leakage sanity: index must be strictly increasing
    assert df.index.is_monotonic_increasing, "Time index is not sorted (leakage risk)."

    y = df["y"].values
    X = df.drop(columns=["y"])

    # Time split (no leakage)
    X_tr, y_tr, X_va, y_va = train_val_split_time(X, y, val_ratio=float(train_cfg["test_ratio"]))

    # Train model (point forecast)
    model = XGBForecastModel.build(train_cfg["xgboost"], seed=seed)
    model.fit(X_tr, y_tr)
    xgb_pred = model.predict(X_va)

    # Align baselines to same validation window
    seasonal_df = pd.Series(seasonal_pred_full, index=series.index).loc[df.index]
    ma7_df = pd.Series(ma7_pred_full, index=series.index).loc[df.index]
    cut = int(len(df) * (1 - float(train_cfg["test_ratio"])))
    seasonal_va = seasonal_df.iloc[cut:].values
    ma7_va = ma7_df.iloc[cut:].values

    # Sanity: lengths match
    assert len(y_va) == len(seasonal_va) == len(ma7_va) == len(xgb_pred), "Length mismatch."

    # --- Core metrics ---
    def report_block(name: str, yhat: np.ndarray) -> dict:
        return {
            "mae": mae(y_va, yhat),
            "rmse": rmse(y_va, yhat),
            "under_mae": underforecast_mae(y_va, yhat),
        }

    seasonal_m = report_block("seasonal", seasonal_va)
    ma7_m = report_block("ma7", ma7_va)
    xgb_m = report_block("xgb", xgb_pred)

    # Improvements vs seasonal naive
    impr_mae = (seasonal_m["mae"] - xgb_m["mae"]) / seasonal_m["mae"] * 100
    impr_under = (seasonal_m["under_mae"] - xgb_m["under_mae"]) / seasonal_m["under_mae"] * 100 if seasonal_m["under_mae"] > 0 else 0.0

    # --- Tail-risk evaluation ---
    tail_pct = 80
    threshold = float(np.percentile(y_va, tail_pct))
    mask = y_va >= threshold
    n_tail = int(mask.sum())

    def tail_report(yhat: np.ndarray) -> tuple[float, float]:
        return mae(y_va[mask], yhat[mask]), underforecast_mae(y_va[mask], yhat[mask])

    seasonal_tail_mae, seasonal_tail_under = tail_report(seasonal_va)
    ma7_tail_mae, ma7_tail_under = tail_report(ma7_va)
    xgb_tail_mae, xgb_tail_under = tail_report(xgb_pred)

    tail_impr = (seasonal_tail_mae - xgb_tail_mae) / seasonal_tail_mae * 100 if seasonal_tail_mae > 0 else 0.0
    tail_under_impr = (seasonal_tail_under - xgb_tail_under) / seasonal_tail_under * 100 if seasonal_tail_under > 0 else 0.0

    # --- Worst misses (under-forecast days) ---
    # under_err = actual - pred (positive = under-forecast)
    under_err_seasonal = y_va - seasonal_va
    under_err_xgb = y_va - xgb_pred

    worst_seasonal = np.argsort(-under_err_seasonal)[:5]  # biggest under-forecast
    worst_xgb = np.argsort(-under_err_xgb)[:5]

    print("\n=== Forecasting Demo (Bank-Real Synthetic) ===")
    print(f"Train rows: {len(X_tr)} | Val rows: {len(X_va)}")
    print(f"Tail set: top {100 - tail_pct}% (n={n_tail} days) | threshold >= p{tail_pct}: {threshold:.2f}")

    print("\nBaselines + Model:")
    print("Seasonal naive (weekly):")
    print(f"  MAE : {seasonal_m['mae']:.2f} | RMSE: {seasonal_m['rmse']:.2f} | Under-MAE: {seasonal_m['under_mae']:.2f}")
    print("Moving average (7d):")
    print(f"  MAE : {ma7_m['mae']:.2f} | RMSE: {ma7_m['rmse']:.2f} | Under-MAE: {ma7_m['under_mae']:.2f}")
    print("XGBoost:")
    print(f"  MAE : {xgb_m['mae']:.2f} | RMSE: {xgb_m['rmse']:.2f} | Under-MAE: {xgb_m['under_mae']:.2f}")

    print(f"\nImprovement vs seasonal naive:")
    print(f"  MAE improvement: {impr_mae:.1f}%")
    print(f"  Under-forecast reduction: {impr_under:.1f}%")

    print(f"\nTail-risk (top {100 - tail_pct}% demand days):")
    print(f"  Seasonal naive tail MAE: {seasonal_tail_mae:.2f} | tail under-MAE: {seasonal_tail_under:.2f}")
    print(f"  MA(7)         tail MAE: {ma7_tail_mae:.2f} | tail under-MAE: {ma7_tail_under:.2f}")
    print(f"  XGBoost       tail MAE: {xgb_tail_mae:.2f} | tail under-MAE: {xgb_tail_under:.2f}")
    print(f"  Tail MAE improvement vs seasonal: {tail_impr:.1f}%")
    print(f"  Tail under-forecast reduction vs seasonal: {tail_under_impr:.1f}%")

    print("\nWorst 5 UNDER-forecast days (Seasonal naive):")
    for i in worst_seasonal:
        print(
            f"  i={i:03d} | actual={y_va[i]:.2f} | seasonal={seasonal_va[i]:.2f} | xgb={xgb_pred[i]:.2f} | "
            f"(actual-seasonal)={under_err_seasonal[i]:.2f}"
        )

    print("\nWorst 5 UNDER-forecast days (XGBoost):")
    for i in worst_xgb:
        print(
            f"  i={i:03d} | actual={y_va[i]:.2f} | seasonal={seasonal_va[i]:.2f} | xgb={xgb_pred[i]:.2f} | "
            f"(actual-xgb)={under_err_xgb[i]:.2f}"
        )

    print("")


if __name__ == "__main__":
    main()
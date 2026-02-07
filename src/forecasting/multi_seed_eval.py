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
    under = np.maximum(y_true - y_pred, 0.0)
    return float(np.mean(under))


def run_once(seed: int) -> dict:
    sim_cfg = load_yaml(Path("configs/sim.yaml"))
    train_cfg = load_yaml(Path("configs/train.yaml"))

    days = int(sim_cfg["horizon_days"])

    # Bank-real synthetic
    series = generate_bank_real_series(BankRealGeneratorConfig(days=days, seed=seed))

    # Baseline seasonal
    seasonal_pred_full = seasonal_naive_forecast(series.values, season=7)

    # Features
    base_feats = make_time_features(series)
    feats = add_lags(
        base_feats,
        series,
        lags=tuple(train_cfg["lags"]),
        roll_window=int(train_cfg["roll_window"]),
    )

    df = feats.copy()
    df["y"] = series.values
    df = df.dropna()

    y = df["y"].values
    X = df.drop(columns=["y"])

    X_tr, y_tr, X_va, y_va = train_val_split_time(X, y, val_ratio=float(train_cfg["test_ratio"]))

    model = XGBForecastModel.build(train_cfg["xgboost"], seed=seed)
    model.fit(X_tr, y_tr)
    xgb_pred = model.predict(X_va)

    seasonal_df = pd.Series(seasonal_pred_full, index=series.index).loc[df.index]
    cut = int(len(df) * (1 - float(train_cfg["test_ratio"])))
    seasonal_va = seasonal_df.iloc[cut:].values

    # Metrics
    base_mae = mae(y_va, seasonal_va)
    xgb_mae = mae(y_va, xgb_pred)
    base_under = underforecast_mae(y_va, seasonal_va)
    xgb_under = underforecast_mae(y_va, xgb_pred)

    mae_impr = (base_mae - xgb_mae) / base_mae * 100 if base_mae > 0 else 0.0
    under_impr = (base_under - xgb_under) / base_under * 100 if base_under > 0 else 0.0

    # Tail
    tail_pct = 80
    thr = float(np.percentile(y_va, tail_pct))
    mask = y_va >= thr
    n_tail = int(mask.sum())

    base_tail_mae = mae(y_va[mask], seasonal_va[mask])
    xgb_tail_mae = mae(y_va[mask], xgb_pred[mask])
    base_tail_under = underforecast_mae(y_va[mask], seasonal_va[mask])
    xgb_tail_under = underforecast_mae(y_va[mask], xgb_pred[mask])

    tail_mae_impr = (base_tail_mae - xgb_tail_mae) / base_tail_mae * 100 if base_tail_mae > 0 else 0.0
    tail_under_impr = (base_tail_under - xgb_tail_under) / base_tail_under * 100 if base_tail_under > 0 else 0.0

    return {
        "seed": seed,
        "mae_impr_pct": mae_impr,
        "under_impr_pct": under_impr,
        "tail_mae_impr_pct": tail_mae_impr,
        "tail_under_impr_pct": tail_under_impr,
        "n_tail": n_tail,
        "val_rows": int(len(y_va)),
        "tail_threshold": thr,
        "baseline_mae": float(base_mae),
        "xgb_mae": float(xgb_mae),
    }


def main() -> None:
    seeds = list(range(40, 60))  # 20 runs
    rows = [run_once(s) for s in seeds]
    df = pd.DataFrame(rows)

    print("\n=== Multi-seed Evaluation (Bank-Real Synthetic) ===")
    print(f"Runs: {len(df)} | Seeds: {seeds[0]}..{seeds[-1]}")
    print("\nKey metrics (mean ± std):")
    for col in ["mae_impr_pct", "under_impr_pct", "tail_mae_impr_pct", "tail_under_impr_pct"]:
        print(f"- {col}: {df[col].mean():.2f} ± {df[col].std():.2f}")

    print("\nTail sample size (n_tail):")
    print(f"- mean: {df['n_tail'].mean():.2f} | min: {df['n_tail'].min()} | max: {df['n_tail'].max()}")

    print("\nTop 5 tail improvements (by tail_under_impr_pct):")
    top = df.sort_values("tail_under_impr_pct", ascending=False).head(5)
    print(top[["seed", "tail_under_impr_pct", "tail_mae_impr_pct", "under_impr_pct", "mae_impr_pct", "n_tail"]].to_string(index=False))

    print("\nBottom 5 tail improvements (sanity):")
    bot = df.sort_values("tail_under_impr_pct", ascending=True).head(5)
    print(bot[["seed", "tail_under_impr_pct", "tail_mae_impr_pct", "under_impr_pct", "mae_impr_pct", "n_tail"]].to_string(index=False))

    # Optional: write CSV
    out_path = Path("artifacts/forecast_eval")
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "multi_seed_eval.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}\n")


if __name__ == "__main__":
    main()
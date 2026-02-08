from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.io import load_yaml
from src.forecasting.bank_real_generator import BankRealGeneratorConfig, generate_bank_real_series
from src.forecasting.baseline import seasonal_naive_forecast
from src.forecasting.xgboost_model import (
    XGBForecastModel,
    make_time_features,
    add_lags,
    train_val_split_time,
)
from src.simulator.policies import reorder_point_policy
from src.simulator.atm_inventory_sim import simulate_atm_inventory_ops


def underforecast_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.maximum(y_true - y_pred, 0.0)))


def get_plan_window(sim_cfg: dict, total_days: int) -> int:
    horizon = int(sim_cfg.get("plan_horizon_days", sim_cfg.get("planning_horizon_days", 60)))
    if horizon >= total_days - 30:
        horizon = max(30, int(0.2 * total_days))
    return horizon


def get_simulator_params(sim_cfg: dict) -> dict:
    atm = sim_cfg.get("atm", {})
    carbon = sim_cfg.get("carbon", {})

    return {
        "initial_cash": float(atm.get("initial_cash", 25000)),
        "max_capacity": float(atm.get("max_capacity", 50000)),
        "restock_fixed_cost": float(atm.get("restock_fixed_cost", 150.0)),
        "holding_cost_rate_daily": float(atm.get("holding_cost_rate_daily", 0.0001)),
        "co2_per_truck_km": float(carbon.get("co2_per_truck_km", 0.85)),
        "avg_trip_distance_km": float(carbon.get("avg_trip_distance_km", carbon.get("avg_trip_distance", 15.0))),
    }


def compute_totals(out) -> dict:
    """
    HARD-MAPPED to your SimulationResult fields (confirmed):
    ['cash_levels','cashouts','co2_kg','op_cost','replenishments','served_demand','summary','unmet_demand']
    """
    repl = np.asarray(out.replenishments, dtype=float)
    op_cost = np.asarray(out.op_cost, dtype=float)
    co2 = np.asarray(out.co2_kg, dtype=float)
    cashouts = np.asarray(out.cashouts, dtype=float)
    unmet = np.asarray(out.unmet_demand, dtype=float)

    repl_trips = int(np.sum(repl > 0))
    total_cost = float(np.sum(op_cost))
    total_co2 = float(np.sum(co2))
    unmet_total = float(np.sum(unmet))
    cashout_rate = float(np.mean(cashouts > 0))

    return {
        "repl_trips": repl_trips,
        "total_cost": total_cost,
        "total_co2_kg": total_co2,
        "unmet_demand": unmet_total,
        "cashout_rate": cashout_rate,
    }


def forecast_baseline(y_all: np.ndarray, horizon: int) -> np.ndarray:
    pred_full = seasonal_naive_forecast(y_all, season=7)
    return pred_full[-horizon:]


def forecast_xgb(series: pd.Series, horizon: int, sim_cfg: dict, train_cfg: dict) -> np.ndarray:
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

    X = df.drop(columns=["y"])
    y = df["y"].values.astype(float)

    X_tr, y_tr, X_va, y_va = train_val_split_time(X, y, val_ratio=0.20)

    model = XGBForecastModel.build(train_cfg["xgboost"], seed=int(sim_cfg.get("seed", 42)))
    model.fit(X_tr, y_tr)
    pred_va = model.predict(X_va)

    # Use the most recent horizon points for decision planning
    return pred_va[-horizon:]


def simulate_ops(demand: np.ndarray, reorder_point: float, sim_params: dict):
    policy = reorder_point_policy(reorder_point)

    return simulate_atm_inventory_ops(
        demand=demand,
        initial_cash=sim_params["initial_cash"],
        max_capacity=sim_params["max_capacity"],
        policy=policy,
        restock_fixed_cost=sim_params["restock_fixed_cost"],
        holding_cost_rate_daily=sim_params["holding_cost_rate_daily"],
        co2_per_truck_km=sim_params["co2_per_truck_km"],
        avg_trip_distance_km=sim_params["avg_trip_distance_km"],
    )


def run_method(method: str, q: int, series: pd.Series, sim_cfg: dict, train_cfg: dict, sim_params: dict) -> dict:
    y_all = series.values.astype(float)
    horizon = get_plan_window(sim_cfg, total_days=len(y_all))
    y_true = y_all[-horizon:]

    if method == "baseline":
        y_pred = forecast_baseline(y_all, horizon=horizon)

    elif method == "xgb":
        y_pred = forecast_xgb(series, horizon=horizon, sim_cfg=sim_cfg, train_cfg=train_cfg)

        # safety fallback: if under-forecast risk worse than baseline, fallback
        base_pred = forecast_baseline(y_all, horizon=horizon)
        if underforecast_mae(y_true, y_pred) > underforecast_mae(y_true, base_pred):
            y_pred = base_pred
            method = "xgb_fallback_to_baseline"
    else:
        raise ValueError("Unknown method")

    rp = float(np.percentile(y_pred, q))

    out = simulate_ops(demand=y_true, reorder_point=rp, sim_params=sim_params)
    totals = compute_totals(out)

    return {
        "method": f"{method}_p{q}",
        "reorder_point": round(rp, 2),
        "cashout_rate": round(totals["cashout_rate"], 4),
        "repl_trips": int(totals["repl_trips"]),
        "unmet_demand": round(totals["unmet_demand"], 2),
        "total_cost": round(totals["total_cost"], 2),
        "total_co2_kg": round(totals["total_co2_kg"], 2),
    }


def main():
    sim_cfg = load_yaml(Path("configs/sim.yaml"))
    train_cfg = load_yaml(Path("configs/train.yaml"))

    days = int(sim_cfg.get("horizon_days", 365))
    seed = int(sim_cfg.get("seed", 42))

    series = generate_bank_real_series(BankRealGeneratorConfig(days=days, seed=seed))
    sim_params = get_simulator_params(sim_cfg)

    rows = []
    for method in ["baseline", "xgb"]:
        for q in [50, 80, 90]:
            rows.append(run_method(method, q, series, sim_cfg, train_cfg, sim_params))

    df = pd.DataFrame(rows)

    # Sort: baseline then xgb, and by quantile
    df["method_base"] = df["method"].str.replace(r"_p\d+$", "", regex=True)
    df["q"] = df["method"].str.extract(r"_p(\d+)$").astype(int)
    df = df.sort_values(["method_base", "q"]).drop(columns=["method_base", "q"])

    print("\n=== Benchmark Table (Forecast -> Decision Outcomes) ===")
    print(df.to_string(index=False))
    print("")

    out_dir = Path("artifacts/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "benchmark_table.csv", index=False)
    print(f"Saved: {out_dir / 'benchmark_table.csv'}\n")


if __name__ == "__main__":
    main()
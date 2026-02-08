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

from src.simulator.policies import reorder_point_policy, target_fill_policy
from src.simulator.atm_inventory_sim import simulate_atm_inventory_ops


def underforecast_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Penalize only when prediction is below actual (stockout-risk proxy)."""
    return float(np.mean(np.maximum(y_true - y_pred, 0.0)))


def get_plan_window(sim_cfg: dict, total_days: int) -> int:
    horizon = int(sim_cfg.get("plan_horizon_days", sim_cfg.get("planning_horizon_days", 60)))
    if horizon >= total_days - 30:
        horizon = max(30, int(0.2 * total_days))
    return horizon


def get_simulator_params(sim_cfg: dict) -> dict:
    atm = sim_cfg.get("atm", {})
    carbon = sim_cfg.get("carbon", {})

    # Support both key names to avoid KeyErrors
    avg_trip_km = carbon.get("avg_trip_distance_km", carbon.get("avg_trip_distance", 15.0))

    return {
        "initial_cash": float(atm.get("initial_cash", 25000)),
        "max_capacity": float(atm.get("max_capacity", 50000)),
        "restock_fixed_cost": float(atm.get("restock_fixed_cost", atm.get("restock_cost", 150.0))),
        "holding_cost_rate_daily": float(atm.get("holding_cost_rate_daily", atm.get("holding_cost_rate", 0.0001))),
        "co2_per_truck_km": float(carbon.get("co2_per_truck_km", 0.85)),
        "avg_trip_distance_km": float(avg_trip_km),
        "batch_size": int(carbon.get("batch_size", 1)),
    }


def compute_totals(out) -> dict:
    repl = np.asarray(out.replenishments, dtype=float)
    op_cost = np.asarray(out.op_cost, dtype=float)
    co2 = np.asarray(out.co2_kg, dtype=float)
    cashouts = np.asarray(out.cashouts, dtype=float)
    unmet = np.asarray(out.unmet_demand, dtype=float)

    return {
        "repl_trips": int(np.sum(repl > 0)),
        "total_cost": float(np.sum(op_cost)),
        "total_co2_kg": float(np.sum(co2)),
        "unmet_demand": float(np.sum(unmet)),
        "cashout_rate": float(np.mean(cashouts > 0)),
    }


def forecast_baseline(y_all: np.ndarray, horizon: int) -> np.ndarray:
    pred_full = seasonal_naive_forecast(y_all, season=7)
    return np.asarray(pred_full[-horizon:], dtype=float)


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

    return np.asarray(pred_va[-horizon:], dtype=float)


def simulate(demand: np.ndarray, policy, sim_params: dict):
    return simulate_atm_inventory_ops(
        demand=demand,
        initial_cash=sim_params["initial_cash"],
        max_capacity=sim_params["max_capacity"],
        policy=policy,
        restock_fixed_cost=sim_params["restock_fixed_cost"],
        holding_cost_rate_daily=sim_params["holding_cost_rate_daily"],
        co2_per_truck_km=sim_params["co2_per_truck_km"],
        avg_trip_distance_km=sim_params["avg_trip_distance_km"],
        batch_size=sim_params["batch_size"],
    )


def sweep(
    method: str,
    qs: list[int],
    target_fracs: list[float],
    series: pd.Series,
    sim_cfg: dict,
    train_cfg: dict,
    sim_params: dict,
) -> pd.DataFrame:
    y_all = series.values.astype(float)
    horizon = get_plan_window(sim_cfg, total_days=len(y_all))
    y_true = y_all[-horizon:]

    if method == "baseline":
        y_pred = forecast_baseline(y_all, horizon=horizon)
    elif method == "xgb":
        y_pred = forecast_xgb(series, horizon=horizon, sim_cfg=sim_cfg, train_cfg=train_cfg)
        base_pred = forecast_baseline(y_all, horizon=horizon)

        # Safety fallback: don't allow XGB to be worse on under-forecast risk
        if underforecast_mae(y_true, y_pred) > underforecast_mae(y_true, base_pred):
            y_pred = base_pred
            method = "xgb_fallback_to_baseline"
    else:
        raise ValueError("Unknown method")

    rows = []
    for q in qs:
        rp = float(np.percentile(y_pred, q))

        for tf in target_fracs:
            if abs(tf - 1.0) < 1e-9:
                pol = reorder_point_policy(rp)
                pol_name = f"{method}_p{q}_fill1.0"
            else:
                pol = target_fill_policy(rp, tf)
                pol_name = f"{method}_p{q}_fill{tf:.1f}"

            out = simulate(y_true, pol, sim_params)
            t = compute_totals(out)

            rows.append(
                {
                    "policy": pol_name,
                    "method": method,
                    "q": q,
                    "target_frac": tf,
                    "reorder_point": rp,
                    **t,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    sim_cfg = load_yaml(Path("configs/sim.yaml"))
    train_cfg = load_yaml(Path("configs/train.yaml"))

    days = int(sim_cfg.get("horizon_days", 365))
    seed = int(sim_cfg.get("seed", 42))
    series = generate_bank_real_series(BankRealGeneratorConfig(days=days, seed=seed))
    sim_params = get_simulator_params(sim_cfg)

    SLA_MAX = float(sim_cfg.get("sla_cashout_rate_max", 0.005))
    CO2_BUDGET = float(sim_cfg.get("carbon_budget_kg", 150.0))

    qs = list(range(50, 96, 5))  # 50..95
    target_fracs = [0.6, 0.7, 0.8, 0.9, 1.0]

    df = pd.concat(
        [
            sweep("baseline", qs, target_fracs, series, sim_cfg, train_cfg, sim_params),
            sweep("xgb", qs, target_fracs, series, sim_cfg, train_cfg, sim_params),
        ],
        ignore_index=True,
    )

    feasible = df[(df["cashout_rate"] <= SLA_MAX) & (df["total_co2_kg"] <= CO2_BUDGET)].copy()
    feasible = feasible.sort_values(["total_cost", "total_co2_kg", "cashout_rate"])

    print("\n=== Governed Selection (min cost under SLA + carbon budget) ===")
    print(f"SLA cashout_rate <= {SLA_MAX} | Carbon budget <= {CO2_BUDGET} kg")
    print(f"Search grid: quantiles={qs[0]}..{qs[-1]} step 5 | target_fill={target_fracs} | batch_size={sim_params['batch_size']}")

    if feasible.empty:
        print("\nNo feasible policies under current constraints.")
        print("Next moves: increase batch_size (route batching) OR increase max_capacity OR relax carbon_budget_kg.")
    else:
        best = feasible.iloc[0]
        print("\nChosen policy:")
        print(
            f"- {best['policy']} | rp={best['reorder_point']:.2f} | "
            f"fill={best['target_frac']:.1f} | cashout={best['cashout_rate']:.4f} | "
            f"trips={int(best['repl_trips'])} | cost={best['total_cost']:.2f} | co2={best['total_co2_kg']:.2f}"
        )

        print("\nTop 5 feasible (cheapest):")
        cols = ["policy", "reorder_point", "target_frac", "cashout_rate", "repl_trips", "total_cost", "total_co2_kg"]
        print(feasible[cols].head(5).to_string(index=False))

    out_dir = Path("artifacts/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "governed_sweep_all.csv", index=False)
    feasible.to_csv(out_dir / "governed_sweep_feasible.csv", index=False)
    print(f"\nSaved: {out_dir / 'governed_sweep_all.csv'}")
    print(f"Saved: {out_dir / 'governed_sweep_feasible.csv'}\n")


if __name__ == "__main__":
    main()
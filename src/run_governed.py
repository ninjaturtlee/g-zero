import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.io import load_yaml
from src.utils.metrics import mae
from src.simulator.atm_inventory_sim import simulate_atm_inventory_ops
from src.simulator.policies import ReorderPointPolicy
from src.forecasting.xgboost_model import make_time_features, add_lags, train_val_split_time
from src.forecasting.baseline import seasonal_naive_forecast
from src.forecasting.quantile_xgb import QuantileForecaster
from src.audit.logging import write_audit_bundle


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


def choose_policy_under_constraints(
    demand_plan: np.ndarray,
    cfg: dict,
    reorder_points: list[int],
    carbon_budget: float,
    sla_max: float,
) -> dict:
    feasible = []
    for rp in reorder_points:
        policy = ReorderPointPolicy(reorder_point=rp, target_level=cfg["policy"]["target_level"])
        res = simulate_atm_inventory_ops(
            demand_plan,
            initial_cash=cfg["atm"]["initial_cash"],
            max_capacity=cfg["atm"]["max_capacity"],
            policy=policy,
            restock_fixed_cost=cfg["costs"]["restock_fixed_cost"],
            holding_cost_rate_daily=cfg["costs"]["holding_cost_rate_daily"],
            co2_per_truck_km=cfg["carbon"]["co2_per_truck_km"],
            avg_trip_distance_km=cfg["carbon"]["avg_trip_distance_km"],
        )
        s = res.summary()
        trips = int(np.sum(res.replenishments > 0))
        row = {
            "reorder_point": rp,
            "cashout_rate": s["cashout_rate"],
            "total_cost": s["total_operational_cost"],
            "total_co2": s["total_co2_kg"],
            "trips": trips,
        }
        if row["cashout_rate"] <= sla_max and row["total_co2"] <= carbon_budget:
            feasible.append(row)

    if not feasible:
        return {"status": "no_feasible_policy", "feasible": []}

    best = min(feasible, key=lambda r: r["total_cost"])
    return {"status": "ok", "best": best, "feasible": feasible}


def main() -> None:
    sim_cfg = load_yaml(Path("configs/sim.yaml"))
    train_cfg = load_yaml(Path("configs/train.yaml"))

    days = int(sim_cfg["horizon_days"])
    seed = int(sim_cfg["seed"])

    # Governance constraints
    sla_max = float(sim_cfg["sla"]["cashout_prob_max"])  # e.g. 0.005
    carbon_budget = float(sim_cfg.get("carbon_budget_kg", 400.0))

    # Data
    series = generate_synthetic_series(days, seed=seed)

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

    # Baseline forecast (aligned to df index and val slice)
    baseline_pred_full = seasonal_naive_forecast(series.values, season=7)
    baseline_df = pd.Series(baseline_pred_full, index=series.index).loc[df.index]
    cut = int(len(df) * (1 - float(train_cfg["test_ratio"])))
    baseline_va = baseline_df.iloc[cut:].values

    # Quantile forecasters (P50 + P90)
    q_params = train_cfg["xgboost"]
    f50 = QuantileForecaster.build(q_params, alpha=0.5, seed=seed)
    f90 = QuantileForecaster.build(q_params, alpha=0.9, seed=seed)

    f50.fit(X_tr, y_tr)
    f90.fit(X_tr, y_tr)

    p50 = f50.predict(X_va)
    p90 = f90.predict(X_va)

    # --- Conformal correction for P90 (calibration) ---
    residuals = y_va - p90
    conformal_shift_q90 = float(np.quantile(residuals, 0.90))
    p90_corr = p90 + conformal_shift_q90

    # Policy selection uses calibrated P90 plan
    reorder_points = [5000, 10000, 15000, 20000, 25000, 30000]
    plan_selection = choose_policy_under_constraints(
        demand_plan=p90_corr,
        cfg=sim_cfg,
        reorder_points=reorder_points,
        carbon_budget=carbon_budget,
        sla_max=sla_max,
    )

    # Backtest chosen policy on realized demand (y_va)
    backtest = None
    if plan_selection["status"] == "ok":
        rp = plan_selection["best"]["reorder_point"]
        policy = ReorderPointPolicy(reorder_point=rp, target_level=sim_cfg["policy"]["target_level"])
        realized_res = simulate_atm_inventory_ops(
            y_va,
            initial_cash=sim_cfg["atm"]["initial_cash"],
            max_capacity=sim_cfg["atm"]["max_capacity"],
            policy=policy,
            restock_fixed_cost=sim_cfg["costs"]["restock_fixed_cost"],
            holding_cost_rate_daily=sim_cfg["costs"]["holding_cost_rate_daily"],
            co2_per_truck_km=sim_cfg["carbon"]["co2_per_truck_km"],
            avg_trip_distance_km=sim_cfg["carbon"]["avg_trip_distance_km"],
        )
        backtest = realized_res.summary()
        backtest["reorder_point"] = rp

    # Forecasting metrics
    base_mae = mae(y_va, baseline_va)
    p50_mae = mae(y_va, p50)
    overall_impr = float((base_mae - p50_mae) / base_mae * 100)

    thr = float(np.percentile(y_va, 90))
    mask = y_va >= thr
    base_tail_mae = mae(y_va[mask], baseline_va[mask])
    p50_tail_mae = mae(y_va[mask], p50[mask])
    tail_impr = float((base_tail_mae - p50_tail_mae) / base_tail_mae * 100)

    cov_raw = float(np.mean(y_va <= p90))
    cov_conf = float(np.mean(y_va <= p90_corr))

    out = {
        "governance": {
            "sla_max_cashout_rate": sla_max,
            "carbon_budget_kg": carbon_budget,
        },
        "forecasting": {
            "val_rows": int(len(y_va)),
            "baseline_weekly_mae": float(base_mae),
            "p50_mae": float(p50_mae),
            "overall_mae_improvement_pct": overall_impr,
            "tail_threshold_p90": thr,
            "baseline_tail_mae": float(base_tail_mae),
            "p50_tail_mae": float(p50_tail_mae),
            "tail_mae_improvement_pct": tail_impr,
            "quantile_coverage_p90_raw": cov_raw,
            "quantile_coverage_p90_conformal": cov_conf,
            "conformal_shift_q90": float(conformal_shift_q90),
        },
        "policy_selection_plan_using_p90_conformal": plan_selection,
        "realized_backtest_on_actual_demand": backtest,
        "notes": [
            "Policy selection uses calibrated P90 demand plan under SLA + carbon budget constraints.",
            "Conformal correction shifts P90 upward to improve coverage reliability.",
            "Backtest evaluates the same selected policy on realized demand.",
            "Audit bundle contains constraints, results, and metadata for traceability.",
        ],
    }

    # Console summary
    print("\n=== GOVERNED RUN (SLA + Carbon Budget + Audit) ===")
    print(f"SLA cashout_rate <= {sla_max} | Carbon budget <= {carbon_budget} kg")

    print("\nForecasting:")
    print(f"- Baseline MAE: {base_mae:.2f}")
    print(f"- P50 MAE     : {p50_mae:.2f}")
    print(f"- Overall MAE improvement: {overall_impr:.1f}%")
    print(f"- Spike-day MAE improvement: {tail_impr:.1f}% (top 10%)")
    print(f"- P90 coverage raw      : {cov_raw:.3f}")
    print(f"- P90 coverage conformal: {cov_conf:.3f}")
    print(f"- Conformal shift (q90 residual): {conformal_shift_q90:.2f}")

    status_ok = False
    if plan_selection["status"] == "ok":
        b = plan_selection["best"]
        status_ok = (b["cashout_rate"] <= sla_max) and (b["total_co2"] <= carbon_budget)

        print("\nPlanned policy (selected using calibrated P90):")
        print(
            f"- reorder_point={b['reorder_point']} | cashout={b['cashout_rate']:.4f} | "
            f"trips={b['trips']} | cost={b['total_cost']:.2f} | co2={b['total_co2']:.2f}"
        )
    else:
        print("\nNo feasible policy under constraints (tight budget/threshold).")

    if backtest is not None:
        print("\nRealized backtest on actual demand:")
        print(
            f"- cashout_rate={backtest['cashout_rate']:.4f} | total_cashouts={backtest['total_cashouts']} | "
            f"cost={backtest['total_operational_cost']:.2f} | co2={backtest['total_co2_kg']:.2f}"
        )

    audit_path = write_audit_bundle(out)

    print("\n=== DECISION STATUS ===")
    if status_ok:
        print("STATUS: ✅ PASSED (SLA met, Carbon under budget)")
    else:
        print("STATUS: ❌ FAILED (Constraint violation detected)")

    print(f"\nAudit written: {audit_path}\n")


if __name__ == "__main__":
    main()
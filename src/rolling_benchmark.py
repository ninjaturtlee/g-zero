# src/rolling_benchmark.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.io import load_yaml
from src.forecasting.bank_real_generator import BankRealGeneratorConfig, generate_bank_real_series
from src.forecasting.xgboost_model import make_time_features, add_lags, XGBForecastModel


# -----------------------------
# Config + helpers
# -----------------------------

@dataclass(frozen=True)
class FleetParams:
    n_atms: int = 10
    days: int = 365
    seed: int = 42

    # Inventory / costs
    initial_cash: float = 25000.0
    max_capacity: float = 50000.0
    holding_cost_rate_daily: float = 0.0001
    restock_fixed_cost: float = 150.0

    # Carbon + routing
    co2_per_truck_km: float = 0.85
    avg_trip_distance_km: float = 15.0
    carbon_budget_kg: float = 80.0
    route_penalty_per_extra_stop_cost: float = 40.0

    # Policy + forecasting
    K: int = 7               # lookback window for baseline
    q: int = 80              # percentile for reorder risk threshold
    retrain_every_days: int = 7  # XGB retrain frequency

    # SLA
    sla_cashout_rate_max: float = 0.005


def _safe_percentile(x: np.ndarray, q: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    return float(np.percentile(x, q))


def _load_params(sim_cfg: dict) -> FleetParams:
    atm = sim_cfg.get("atm", {})
    carbon = sim_cfg.get("carbon", {})
    costs = sim_cfg.get("costs", {})
    sla = sim_cfg.get("sla", {})

    carbon_budget = float(sim_cfg.get("carbon_budget_kg", carbon.get("carbon_budget_kg", 80.0)))
    sla_max = float(sim_cfg.get("sla_cashout_rate_max", sla.get("cashout_prob_max", 0.005)))

    # IMPORTANT: allow either sim_cfg["route_penalty"] or carbon key
    route_penalty = float(sim_cfg.get("route_penalty", carbon.get("route_penalty_per_extra_stop_cost", 40.0)))

    # IMPORTANT: prefer carbon.avg_trip_distance_km but allow avg_trip_distance
    avg_trip_km = float(carbon.get("avg_trip_distance_km", carbon.get("avg_trip_distance", 15.0)))

    return FleetParams(
        n_atms=int(sim_cfg.get("fleet_n_atms", 10)),
        days=int(sim_cfg.get("horizon_days", 365)),
        seed=int(sim_cfg.get("seed", 42)),

        initial_cash=float(atm.get("initial_cash", 25000.0)),
        max_capacity=float(atm.get("max_capacity", 50000.0)),
        holding_cost_rate_daily=float(atm.get("holding_cost_rate_daily", costs.get("holding_cost_rate_daily", 0.0001))),
        restock_fixed_cost=float(atm.get("restock_fixed_cost", costs.get("restock_fixed_cost", 150.0))),

        co2_per_truck_km=float(carbon.get("co2_per_truck_km", 0.85)),
        avg_trip_distance_km=avg_trip_km,
        carbon_budget_kg=carbon_budget,
        route_penalty_per_extra_stop_cost=route_penalty,

        K=int(sim_cfg.get("rolling_K", 7)),
        q=int(sim_cfg.get("rolling_q", 80)),
        retrain_every_days=int(sim_cfg.get("xgb_retrain_every_days", 7)),

        sla_cashout_rate_max=sla_max,
    )


def _generate_fleet_demands(params: FleetParams) -> pd.DataFrame:
    """
    Generate bank-real demand series per ATM.
    Columns: atm_000, atm_001, ...
    """
    cols = []
    for i in range(params.n_atms):
        s = generate_bank_real_series(
            BankRealGeneratorConfig(days=params.days, seed=params.seed + 1000 * (i + 1))
        )
        cols.append(s.rename(f"atm_{i:03d}"))
    return pd.concat(cols, axis=1)


def _build_features(series: pd.Series, train_cfg: dict) -> Tuple[pd.DataFrame, np.ndarray]:
    base = make_time_features(series)
    feats = add_lags(
        base,
        series,
        lags=tuple(train_cfg["lags"]),
        roll_window=int(train_cfg["roll_window"]),
    )
    df = feats.copy()
    df["y"] = series.values
    df = df.dropna()
    y = df["y"].values.astype(float)
    X = df.drop(columns=["y"])
    return X, y


def _xgb_predict_next_and_resid_q(
    series: pd.Series,
    train_cfg: dict,
    *,
    seed: int,
    q: int,
    t: int,
    cached_model: Optional[object],
    cached_until_idx: int,
    retrain_every_days: int,
) -> Tuple[float, float, object, int]:
    """
    Predict demand for day t using history up to t-1 (NO LEAKAGE).
    Return:
      - point_pred for day t
      - resid_q (qth percentile of residuals on training window)
      - model + cache_until
    """
    # retrain schedule
    need_retrain = (cached_model is None) or (t > cached_until_idx)

    model = cached_model
    if need_retrain:
        train_series = series.iloc[:t]  # up to t-1
        X_tr, y_tr = _build_features(train_series, train_cfg)

        # too little history -> fallback
        if len(y_tr) < 60:
            last = train_series.values[-max(7, len(train_series)) :]
            point_pred = float(np.mean(last))
            resid_q = 0.0
            return point_pred, resid_q, cached_model, cached_until_idx

        model = XGBForecastModel.build(train_cfg["xgboost"], seed=seed)
        model.fit(X_tr, y_tr)

        cached_until_idx = t + max(1, retrain_every_days) - 1
        cached_model = model

    # Build feature row for day t (still only using info up to t-1)
    # We create a "next_series" that includes day t so feature builder can produce the row,
    # BUT y at t is not used by the model (we only take the last X row).
    next_series = series.iloc[: t + 1]
    X_all, _ = _build_features(next_series, train_cfg)

    x_next = X_all.iloc[[-1]]
    point_pred = float(model.predict(x_next)[0])

    # residual quantile on train window
    train_series = series.iloc[:t]
    X_tr, y_tr = _build_features(train_series, train_cfg)
    if len(y_tr) < 60:
        resid_q = 0.0
    else:
        yhat = model.predict(X_tr)
        resid = (y_tr - yhat).astype(float)
        resid_q = float(np.percentile(resid, q))

    return point_pred, resid_q, cached_model, cached_until_idx


# -----------------------------
# Fleet closed-loop simulator
# -----------------------------

def rolling_fleet_run(
    method: str,
    demands: pd.DataFrame,
    params: FleetParams,
    train_cfg: dict,
    *,
    batch_size: int,
) -> Dict[str, float]:
    """
    Multi-ATM closed-loop run with REAL batching + hard carbon cap.

    DAY ORDER (THIS IS THE FIX):
      1) compute rp using history up to t-1
      2) decide refills using start-of-day cash
      3) execute allowed refills (subject to remaining carbon)
      4) apply withdrawals for day t
      5) record cashouts/unmet
      6) add holding cost (end-of-day)

    Note: rp is a *risk threshold* for refill trigger.
    """

    n_days = demands.shape[0]
    n_atms = demands.shape[1]

    # State: start-of-day cash
    cash = np.full(n_atms, params.initial_cash, dtype=float)

    # Accounting
    total_cost = 0.0
    total_co2 = 0.0
    total_cashout_atm_days = 0
    total_unmet = 0.0
    total_trips = 0

    # Carbon per trip (DO NOT DIVIDE BY BATCH SIZE HERE — batching reduces trips)
    co2_per_trip = float(params.co2_per_truck_km * params.avg_trip_distance_km)

    # For XGB: per-ATM model cache
    model_cache: List[Optional[object]] = [None] * n_atms
    cache_until: List[int] = [-1] * n_atms

    series_list = [demands.iloc[:, i] for i in range(n_atms)]

    for t in range(n_days):
        # 1) compute reorder thresholds rp[i] using history up to t-1
        rp = np.zeros(n_atms, dtype=float)

        for i in range(n_atms):
            s = series_list[i]

            # warm-up: be conservative using max seen so far
            if t < max(params.K, 7):
                rp[i] = float(np.max(s.values[: t + 1]))
                continue

            if method == "baseline":
                window = s.values[t - params.K : t]  # up to t-1
                rp[i] = _safe_percentile(window, params.q)

            elif method == "xgb":
                point_pred, resid_q, m, until = _xgb_predict_next_and_resid_q(
                    series=s,
                    train_cfg=train_cfg,
                    seed=params.seed + 17 * (i + 1),
                    q=params.q,
                    t=t,
                    cached_model=model_cache[i],
                    cached_until_idx=cache_until[i],
                    retrain_every_days=params.retrain_every_days,
                )
                model_cache[i] = m
                cache_until[i] = until
                rp[i] = max(0.0, float(point_pred + resid_q))

            else:
                raise ValueError(f"Unknown method: {method}")

        # 2) decide which ATMs want refill based on START-OF-DAY cash
        # IMPORTANT: policy spec is cash <= reorder_point
        want_refill = cash <= rp
        candidates = np.where(want_refill)[0].tolist()

        # 3) execute allowed refills subject to carbon hard cap
        if candidates:
            # rank by urgency: larger (rp - cash) is more urgent
            urgency = [(i, float(rp[i] - cash[i])) for i in candidates]
            urgency.sort(key=lambda x: x[1], reverse=True)

            # how many trips remain under budget?
            remaining_budget = float(params.carbon_budget_kg - total_co2)
            trips_left = int(np.floor(remaining_budget / co2_per_trip + 1e-12))

            # max stops we can serve with remaining trips
            max_stops = trips_left * int(batch_size)

            if max_stops > 0:
                chosen = [i for i, _ in urgency[:max_stops]]
                n_chosen = len(chosen)

                trips_today = int(np.ceil(n_chosen / int(batch_size)))
                if trips_today > 0:
                    # refill chosen ATMs to max capacity
                    cash[chosen] = float(params.max_capacity)

                    # fixed cost per trip
                    total_cost += float(trips_today) * float(params.restock_fixed_cost)

                    # routing penalty per extra stop inside each trip
                    remaining = n_chosen
                    extra_stops = 0
                    for _ in range(trips_today):
                        stops = min(int(batch_size), remaining)
                        extra_stops += max(0, stops - 1)
                        remaining -= stops

                    total_cost += float(extra_stops) * float(params.route_penalty_per_extra_stop_cost)

                    # carbon cost
                    total_co2 += float(trips_today) * float(co2_per_trip)
                    total_trips += int(trips_today)

        # 4) apply withdrawals for day t
        demand_today = demands.iloc[t, :].values.astype(float)

        served = np.minimum(cash, demand_today)
        unmet = demand_today - served
        cash = cash - served  # end-of-day cash

        # 5) record SLA / unmet
        cashouts_today = int(np.sum(unmet > 0))
        total_cashout_atm_days += cashouts_today
        total_unmet += float(np.sum(unmet))

        # 6) holding cost (end-of-day)
        total_cost += float(np.sum(cash) * float(params.holding_cost_rate_daily))

    # SLA: fraction of ATM-days with cashout
    denom = float(n_days * n_atms)
    cashout_rate = float(total_cashout_atm_days / denom)

    carbon_ok = total_co2 <= float(params.carbon_budget_kg) + 1e-9
    sla_ok = cashout_rate <= float(params.sla_cashout_rate_max) + 1e-12

    if carbon_ok and sla_ok:
        status = "ok"
    elif (not carbon_ok) and (not sla_ok):
        status = "carbon_and_sla_violation"
    elif not carbon_ok:
        status = "carbon_violation"
    else:
        status = "sla_violation"

    return {
        "method": method,
        "status": status,
        "total_cost": float(total_cost),
        "total_co2_kg": float(total_co2),
        "cashout_rate": float(cashout_rate),
        "repl_trips": int(total_trips),
        "unmet_demand": float(total_unmet),
        "carbon_budget_kg": float(params.carbon_budget_kg),
        "batch_size": int(batch_size),
        "q": int(params.q),
        "K": int(params.K),
        "route_penalty": float(params.route_penalty_per_extra_stop_cost),
        "n_atms": int(params.n_atms),
        "days": int(params.days),
    }


def main() -> None:
    sim_cfg = load_yaml(Path("configs/sim.yaml"))
    train_cfg = load_yaml(Path("configs/train.yaml"))

    params = _load_params(sim_cfg)
    demands = _generate_fleet_demands(params)

    rows: List[Dict[str, float]] = []
    for batch_size in [1, 2, 3]:
        for method in ["baseline", "xgb"]:
            rows.append(
                rolling_fleet_run(
                    method=method,
                    demands=demands,
                    params=params,
                    train_cfg=train_cfg,
                    batch_size=batch_size,
                )
            )

    df = pd.DataFrame(rows).sort_values(["batch_size", "method"]).reset_index(drop=True)
    df = df[
        [
            "method",
            "status",
            "total_cost",
            "total_co2_kg",
            "cashout_rate",
            "repl_trips",
            "unmet_demand",
            "carbon_budget_kg",
            "batch_size",
            "q",
            "K",
            "route_penalty",
            "n_atms",
            "days",
        ]
    ]

    print("\n=== Rolling (Closed-loop) Fleet Benchmark (Carbon-Governed) ===")
    print(df.to_string(index=False))

    outdir = Path("artifacts/benchmarks")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "rolling_benchmark_fleet.csv"
    df.to_csv(outpath, index=False)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
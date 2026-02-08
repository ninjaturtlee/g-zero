from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationResult:
    cash_levels: np.ndarray
    replenishments: np.ndarray
    served_demand: np.ndarray
    unmet_demand: np.ndarray
    cashouts: np.ndarray
    op_cost: np.ndarray
    co2_kg: np.ndarray

    @property
    def summary(self) -> dict:
        demand_total = float(np.sum(self.served_demand + self.unmet_demand))
        unmet_total = float(np.sum(self.unmet_demand))
        cashout_rate = float(np.mean(self.cashouts > 0))
        return {
            "days": int(len(self.cash_levels)),
            "cashout_rate": cashout_rate,
            "total_unmet_demand": unmet_total,
            "total_demand": demand_total,
            "total_operational_cost": float(np.sum(self.op_cost)),
            "total_co2_kg": float(np.sum(self.co2_kg)),
            "repl_trips": int(np.sum(self.replenishments > 0)),
        }


def simulate_atm_inventory_ops(
    *,
    demand: np.ndarray,
    initial_cash: float,
    max_capacity: float,
    policy,
    restock_fixed_cost: float,
    holding_cost_rate_daily: float,
    co2_per_truck_km: float,
    avg_trip_distance_km: float,
    batch_size: int = 1,
    route_penalty_per_extra_stop_cost: float = 0.0,
) -> SimulationResult:
    """
    Single-ATM simulator.

    NOTE:
    - Here we keep your original "trip_co2 divided by batch_size" behavior since
      you used this to approximate shared route CO2 for a single ATM.
    - Fleet benchmark handles batching explicitly, so it should NOT use this.
    """

    demand = np.asarray(demand, dtype=float)
    n = int(len(demand))

    cash = float(initial_cash)

    cash_levels = np.zeros(n, dtype=float)
    replenishments = np.zeros(n, dtype=float)
    cashouts = np.zeros(n, dtype=float)
    served_demand = np.zeros(n, dtype=float)
    unmet_demand = np.zeros(n, dtype=float)
    co2_kg = np.zeros(n, dtype=float)
    op_cost = np.zeros(n, dtype=float)

    # per-trip CO2 approximation
    trip_co2 = (float(co2_per_truck_km) * float(avg_trip_distance_km)) / max(1, int(batch_size))

    for t in range(n):
        # refill decision
        add = float(policy.decide_replenish_amount(cash, float(max_capacity)))
        did_refill = add > 0.0

        if did_refill:
            cash = min(float(max_capacity), cash + add)
            replenishments[t] = add
            co2_kg[t] = trip_co2
            op_cost[t] += float(restock_fixed_cost) + float(route_penalty_per_extra_stop_cost) * max(0, int(batch_size) - 1)

        # realize demand
        d = float(demand[t])
        served = min(cash, d)
        unmet = max(0.0, d - cash)
        cash -= served

        cash_levels[t] = cash
        served_demand[t] = served
        unmet_demand[t] = unmet
        cashouts[t] = 1.0 if unmet > 0 else 0.0

        # holding cost
        op_cost[t] += cash * float(holding_cost_rate_daily)

    return SimulationResult(
        cash_levels=cash_levels,
        replenishments=replenishments,
        served_demand=served_demand,
        unmet_demand=unmet_demand,
        cashouts=cashouts,
        op_cost=op_cost,
        co2_kg=co2_kg,
    )
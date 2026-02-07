from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from src.simulator.policies import ReorderPointPolicy


@dataclass
class SimulationResult:
    cash_levels: np.ndarray
    cashouts: np.ndarray
    replenishments: np.ndarray
    served_demand: np.ndarray
    unmet_demand: np.ndarray
    op_cost: np.ndarray
    co2_kg: np.ndarray

    def summary(self) -> Dict[str, Any]:
        return {
            "days": int(len(self.cash_levels)),
            "cashout_rate": float(np.mean(self.cashouts)),
            "total_cashouts": int(np.sum(self.cashouts)),
            "avg_cash_level": float(np.mean(self.cash_levels)),
            "total_replenished": float(np.sum(self.replenishments)),
            "total_unmet_demand": float(np.sum(self.unmet_demand)),
            "total_operational_cost": float(np.sum(self.op_cost)),
            "total_co2_kg": float(np.sum(self.co2_kg)),
        }


def simulate_atm_inventory_ops(
    demand: np.ndarray,
    *,
    initial_cash: float,
    max_capacity: float,
    policy: ReorderPointPolicy,
    restock_fixed_cost: float,
    holding_cost_rate_daily: float,
    co2_per_truck_km: float,
    avg_trip_distance_km: float,
) -> SimulationResult:
    demand = np.asarray(demand, dtype=float)
    n = demand.shape[0]

    cash_levels = np.zeros(n)
    cashouts = np.zeros(n, dtype=int)
    replenishments = np.zeros(n)
    served_demand = np.zeros(n)
    unmet_demand = np.zeros(n)
    op_cost = np.zeros(n)
    co2_kg = np.zeros(n)

    cash = float(initial_cash)
    if cash > max_capacity:
        cash = max_capacity

    co2_per_trip = co2_per_truck_km * avg_trip_distance_km

    for t in range(n):
        d = demand[t]
        served = min(cash, d)
        served_demand[t] = served
        cash -= served

        if d > served:
            cashouts[t] = 1
            unmet_demand[t] = d - served
            cash = 0.0

        add = policy.decide_replenish_amount(cash, max_capacity)
        if add > 0:
            cash = min(max_capacity, cash + add)
            replenishments[t] = add
            op_cost[t] += restock_fixed_cost
            co2_kg[t] += co2_per_trip

        op_cost[t] += cash * holding_cost_rate_daily

        if cash < -1e-9:
            raise AssertionError("Cash went negative")

        cash_levels[t] = cash

    return SimulationResult(
        cash_levels=cash_levels,
        cashouts=cashouts,
        replenishments=replenishments,
        served_demand=served_demand,
        unmet_demand=unmet_demand,
        op_cost=op_cost,
        co2_kg=co2_kg,
    )
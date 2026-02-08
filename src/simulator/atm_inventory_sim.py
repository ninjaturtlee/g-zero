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
    demand: np.ndarray,
    *,
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
    ATM inventory simulation with:
      - Trip-based CO2 that supports route batching:
          effective_co2_per_restock = co2_per_trip / batch_size
      - Route complexity penalty:
          restock_cost_effective = restock_fixed_cost + (batch_size-1)*route_penalty
        (so batching is no longer a free lunch)
    """
    demand = np.asarray(demand, dtype=float)
    n = int(len(demand))

    cash = float(initial_cash)
    max_capacity = float(max_capacity)

    cash_levels = np.zeros(n, dtype=float)
    replenishments = np.zeros(n, dtype=float)
    served = np.zeros(n, dtype=float)
    unmet = np.zeros(n, dtype=float)
    cashouts = np.zeros(n, dtype=int)
    op_cost = np.zeros(n, dtype=float)
    co2 = np.zeros(n, dtype=float)

    co2_per_trip = float(co2_per_truck_km) * float(avg_trip_distance_km)

    batch_size = max(1, int(batch_size))
    effective_co2_per_restock = co2_per_trip / batch_size

    # batching penalty: each extra stop in a batched route costs money (time/security/handling)
    extra_stops = max(0, batch_size - 1)
    effective_restock_cost = float(restock_fixed_cost) + float(route_penalty_per_extra_stop_cost) * float(extra_stops)

    for t in range(n):
        # 1) Decide replenish amount
        add = float(policy.decide_replenish_amount(cash, max_capacity))
        add = max(0.0, min(add, max_capacity - cash))

        if add > 0.0:
            cash += add
            replenishments[t] = add
            op_cost[t] += effective_restock_cost
            co2[t] += effective_co2_per_restock

        # 2) Serve demand
        d = float(demand[t])
        if cash >= d:
            cash -= d
            served[t] = d
        else:
            served[t] = cash
            unmet[t] = d - cash
            cash = 0.0
            cashouts[t] = 1

        # 3) Holding cost
        op_cost[t] += cash * float(holding_cost_rate_daily)

        # 4) Log cash level
        cash_levels[t] = cash

    return SimulationResult(
        cash_levels=cash_levels,
        replenishments=replenishments,
        served_demand=served,
        unmet_demand=unmet,
        cashouts=cashouts,
        op_cost=op_cost,
        co2_kg=co2,
    )
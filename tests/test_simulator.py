import numpy as np
from src.simulator.atm_inventory_sim import simulate_atm_inventory_ops
from src.simulator.policies import ReorderPointPolicy


def test_cash_never_negative():
    demand = np.array([1000, 2000, 3000])
    policy = ReorderPointPolicy(2000, 45000)
    res = simulate_atm_inventory_ops(
        demand,
        initial_cash=5000,
        max_capacity=50000,
        policy=policy,
        restock_fixed_cost=150.0,
        holding_cost_rate_daily=0.0001,
        co2_per_truck_km=0.85,
        avg_trip_distance_km=15.0,
    )
    assert (res.cash_levels >= -1e-9).all()


def test_cashout_and_unmet_demand():
    demand = np.array([6000])
    policy = ReorderPointPolicy(-1, 0)
    res = simulate_atm_inventory_ops(
        demand,
        initial_cash=5000,
        max_capacity=50000,
        policy=policy,
        restock_fixed_cost=150.0,
        holding_cost_rate_daily=0.0,
        co2_per_truck_km=0.85,
        avg_trip_distance_km=15.0,
    )
    assert res.cashouts[0] == 1
    assert res.unmet_demand[0] == 1000
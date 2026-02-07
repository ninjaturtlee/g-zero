import numpy as np
from pathlib import Path

from src.simulator.atm_inventory_sim import simulate_atm_inventory_ops
from src.simulator.policies import ReorderPointPolicy
from src.utils.io import load_yaml


def generate_synthetic_demand(days: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    base = 8000.0
    weekly = 2000.0 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 800.0, size=days)
    demand = base + weekly + noise

    spike_days = rng.choice(days, size=max(1, days // 30), replace=False)
    demand[spike_days] += rng.uniform(5000, 12000, size=len(spike_days))

    return np.clip(demand, 0, None)


def main() -> None:
    cfg = load_yaml(Path("configs/sim.yaml"))
    demand = generate_synthetic_demand(cfg["horizon_days"], seed=cfg["seed"])

    # Sweep reorder points (low → high)
    reorder_points = [5000, 10000, 15000, 20000, 25000, 30000]

    print("\n=== Policy Sweep (Reorder Point) ===")
    print("reorder_point | cashout_rate | repl_trips | total_cost | total_co2_kg")
    print("-" * 72)

    for rp in reorder_points:
        policy = ReorderPointPolicy(reorder_point=rp, target_level=cfg["policy"]["target_level"])

        res = simulate_atm_inventory_ops(
            demand,
            initial_cash=cfg["atm"]["initial_cash"],
            max_capacity=cfg["atm"]["max_capacity"],
            policy=policy,
            restock_fixed_cost=cfg["costs"]["restock_fixed_cost"],
            holding_cost_rate_daily=cfg["costs"]["holding_cost_rate_daily"],
            co2_per_truck_km=cfg["carbon"]["co2_per_truck_km"],
            avg_trip_distance_km=cfg["carbon"]["avg_trip_distance_km"],
        )

        summary = res.summary()
        repl_trips = int(np.sum(res.replenishments > 0))

        print(
            f"{rp:12d} | "
            f"{summary['cashout_rate']:11.4f} | "
            f"{repl_trips:9d} | "
            f"{summary['total_operational_cost']:10.2f} | "
            f"{summary['total_co2_kg']:11.2f}"
        )


if __name__ == "__main__":
    main()
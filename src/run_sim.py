import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import numpy as np
from pathlib import Path

from src.simulator.atm_inventory_sim import simulate_atm_inventory_ops
from src.simulator.policies import ReorderPointPolicy
from src.utils.io import load_yaml


def generate_synthetic_demand(
    days: int,
    base: float = 8000.0,
    weekly_amp: float = 2000.0,
    noise_std: float = 800.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate realistic daily ATM demand:
    - weekly seasonality
    - random noise
    - occasional spikes
    """
    rng = np.random.default_rng(seed)
    t = np.arange(days)

    weekly = weekly_amp * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, noise_std, size=days)
    demand = base + weekly + noise

    # occasional demand spikes (e.g. holidays)
    spike_days = rng.choice(days, size=max(1, days // 30), replace=False)
    demand[spike_days] += rng.uniform(5000, 12000, size=len(spike_days))

    return np.clip(demand, 0, None)


def main() -> None:
    cfg = load_yaml(Path("configs/sim.yaml"))

    demand = generate_synthetic_demand(
        days=cfg["horizon_days"],
        seed=cfg["seed"],
    )

    policy = ReorderPointPolicy(
        reorder_point=cfg["policy"]["reorder_point"],
        target_level=cfg["policy"]["target_level"],
    )

    result = simulate_atm_inventory_ops(
        demand,
        initial_cash=cfg["atm"]["initial_cash"],
        max_capacity=cfg["atm"]["max_capacity"],
        policy=policy,
        restock_fixed_cost=cfg["costs"]["restock_fixed_cost"],
        holding_cost_rate_daily=cfg["costs"]["holding_cost_rate_daily"],
        co2_per_truck_km=cfg["carbon"]["co2_per_truck_km"],
        avg_trip_distance_km=cfg["carbon"]["avg_trip_distance_km"],
    )

    print("\n=== Simulation Summary ===")
    for k, v in result.summary().items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
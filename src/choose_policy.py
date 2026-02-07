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

    # occasional demand spikes (e.g., holidays)
    spike_days = rng.choice(days, size=max(1, days // 30), replace=False)
    demand[spike_days] += rng.uniform(5000, 12000, size=len(spike_days))

    return np.clip(demand, 0, None)


def main() -> None:
    cfg = load_yaml(Path("configs/sim.yaml"))
    demand = generate_synthetic_demand(cfg["horizon_days"], seed=cfg["seed"])

    # Constraints
    sla_max = cfg["sla"]["cashout_prob_max"]  # e.g. 0.005
    carbon_budget = 400.0                     # demo cap (kg CO2). Tune as needed.

    reorder_points = [5000, 10000, 15000, 20000, 25000, 30000]

    feasible = []
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

    print("\n=== Constraint-based Policy Selection ===")
    print(f"SLA constraint: cashout_rate <= {sla_max}")
    print(f"Carbon budget: total_co2 <= {carbon_budget} kg\n")

    if not feasible:
        print("No policy satisfies the constraints. Increase carbon budget or widen policy set.")
        return

    # Choose min cost among feasible policies
    best = min(feasible, key=lambda r: r["total_cost"])

    print("Feasible policies:")
    for r in feasible:
        print(
            f"- rp={r['reorder_point']:5d} | cashout={r['cashout_rate']:.4f} | "
            f"trips={r['trips']:2d} | cost={r['total_cost']:.2f} | co2={r['total_co2']:.2f}"
        )

    print("\nChosen policy (min cost among feasible):")
    print(
        f"rp={best['reorder_point']} | cashout={best['cashout_rate']:.4f} | "
        f"trips={best['trips']} | cost={best['total_cost']:.2f} | co2={best['total_co2']:.2f}"
    )


if __name__ == "__main__":
    main()
from pathlib import Path
import numpy as np

from src.utils.io import load_yaml
from src.forecasting.bank_real_generator import BankRealGeneratorConfig, generate_bank_real_series
from src.simulator.policies import reorder_point_policy
from src.simulator.atm_inventory_sim import simulate_atm_inventory_ops

cfg = load_yaml(Path("configs/sim.yaml"))

series = generate_bank_real_series(
    BankRealGeneratorConfig(
        days=int(cfg.get("horizon_days", 365)),
        seed=int(cfg.get("seed", 42)),
    )
)

demand = series.values[-60:].astype(float)
policy = reorder_point_policy(15000)

out = simulate_atm_inventory_ops(
    demand=demand,
    initial_cash=float(cfg["atm"].get("initial_cash", 25000)),
    max_capacity=float(cfg["atm"].get("max_capacity", 50000)),
    policy=policy,
    restock_fixed_cost=float(cfg["atm"].get("restock_fixed_cost", cfg["atm"].get("restock_cost", 150.0))),
    holding_cost_rate_daily=float(cfg["atm"].get("holding_cost_rate_daily", cfg["atm"].get("holding_cost_rate", 0.0001))),
    co2_per_truck_km=float(cfg["carbon"].get("co2_per_truck_km", 0.85)),
    avg_trip_distance_km=float(cfg["carbon"].get("avg_trip_distance_km", cfg["carbon"].get("avg_trip_distance", 15.0))),
)

print("TYPE:", type(out))
attrs = [a for a in dir(out) if not a.startswith("_")]
print("\nATTRIBUTES:")
print(attrs)

print("\n--- ARRAY-LIKE FIELDS ---")
for a in attrs:
    try:
        v = getattr(out, a)
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            print(a, "len=", len(v), "first=", v[0])
    except Exception:
        pass

print("\n--- SCALAR FIELDS ---")
for a in attrs:
    try:
        v = getattr(out, a)
        if isinstance(v, (int, float, np.floating, np.integer)):
            print(a, v)
    except Exception:
        pass

print("\nRESTOCK_FIXED_COST_USED:",
      cfg["atm"].get("restock_fixed_cost", cfg["atm"].get("restock_cost", None)))

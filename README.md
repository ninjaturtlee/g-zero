# g-zero

Risk-aware, carbon-budgeted operations planning for ATM cash replenishment
(forecasting + simulation + optimization-ready).

## What this is
A compact, testable core that treats sustainability as an operational constraint,
not a marketing metric.

The project simulates ATM cash inventory under uncertain demand while tracking:
- service failures (cash-outs)
- operational cost
- CO₂ emissions from replenishment trips

It is designed to support risk-aware and carbon-budgeted optimization.

## Demo (runnable)
Run a full synthetic simulation with cost + CO₂ accounting:

```bash
python -m src.run_sim
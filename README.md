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

## Constraint-based selection (optimization-ready)
Select the lowest-cost policy subject to SLA and a carbon budget:

```bash
python -m src.choose_policy

## Forecasting results (evidence)

We benchmark a classical seasonal baseline against a machine-learning model
to ensure that improvements are real and not cosmetic.

**Setup**
- Baseline: seasonal naive (weekly)
- ML model: XGBoost with lagged features, calendar features, and trend
- Evaluation: hold-out time split (no leakage)
- Data: synthetic daily demand with seasonality and demand spikes

**Results (365-day synthetic pilot)**

- Overall MAE improvement: **14.1%**
- Overall RMSE reduction: significant
- Tail-risk performance (top 10% demand days):
  - Baseline MAE: 2819.98
  - XGBoost MAE: 2306.79
  - **Improvement on spike days: 18.2%**

**Interpretation**
Seasonal naive performs strongly on average days, but the ML model
significantly reduces error during high-demand spikes.
These tail events drive cash-outs, emergency replenishments,
and unnecessary CO₂ emissions in practice.

Reducing forecast error specifically on spike days improves
downstream operational decisions under SLA and carbon constraints.

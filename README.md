```text
=== GOVERNED RUN (SLA + Carbon Budget + Audit) ===
SLA cashout_rate <= 0.005 | Carbon budget <= 400.0 kg

Forecasting:
- Baseline MAE: 1282.11
- P50 MAE     : 964.52
- Overall MAE improvement: 24.8%
- Spike-day MAE improvement: 18.4% (top 10%)
- P90 coverage raw      : 0.746
- P90 coverage conformal: 0.901
- Conformal shift (q90 residual): 810.31

Planned policy (selected using calibrated P90):
- reorder_point=15000 | cashout=0.0000 | trips=20 | cost=3232.00 | co2=255.00

Realized backtest on actual demand:
- cashout_rate=0.0000 | total_cashouts=0 | cost=2774.78 | co2=216.75

=== DECISION STATUS ===
STATUS: ✅ PASSED (SLA met, Carbon under budget)

# g-zero

Carbon-budgeted, SLA-guaranteed ATM operations with calibrated uncertainty
and audit-grade decision logs.

## Problem
ATM cash replenishment is traditionally optimized for cost and service level,
while sustainability is reported after the fact.
This creates hidden trade-offs, reactive decisions, and ungoverned emissions.

## Key idea (what’s new)
Treat carbon as a **hard operational constraint**, not an optimization preference,
and plan under **calibrated demand uncertainty**.
Every decision produces a machine-verifiable audit artifact.

## How it works (pipeline)
1) Forecast demand with uncertainty (P50 / calibrated P90)
2) Select replenishment policy under:
   - SLA constraint
   - Carbon budget
3) Backtest on realized demand
4) Emit audit bundle (inputs, models, constraints, outcome)

## Evidence
- Forecasting: +24.8% MAE improvement vs seasonal naive
- Tail risk: +18.4% MAE improvement on top-10% demand days
- Governance: SLA met with 255kg CO₂ (under 400kg budget)
- Calibration: P90 coverage corrected from 0.746 → 0.901

## Why this matters
This enables banks to:
- enforce carbon budgets at decision time
- guarantee service levels under uncertainty
- produce audit-ready justification for every operational choice

## Run
```bash
python -m src.run_governed


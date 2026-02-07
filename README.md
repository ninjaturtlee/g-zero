# g-zero: Carbon-Guaranteed ATM Operations

[![Status](https://img.shields.io/badge/Audit-PASSED-success?style=for-the-badge)]()
[![Carbon](https://img.shields.io/badge/Carbon_Budget-MET-green?style=for-the-badge)]()
[![SLA](https://img.shields.io/badge/SLA-Guaranteed-blue?style=for-the-badge)]()

**Carbon-budgeted, SLA-guaranteed ATM operations with calibrated uncertainty and audit-grade decision logs.**

---

## ⚡ Key Result: The "Governed Run"
*Evidence from a realized backtest on production-like demand.*

```text
=== GOVERNED RUN (SLA + Carbon Budget + Audit) ===
SLA cashout_rate <= 0.005 | Carbon budget <= 400.0 kg

Forecasting:
- Baseline MAE          : 1282.11
- P50 MAE               : 964.52
- Overall Improvement   : 24.8%
- Spike-day Improvement : 18.4% (Top 10% demand days)

Safety (Conformal Prediction):
- P90 coverage raw      : 0.746 (Unsafe)
- P90 coverage conformal: 0.901 (Corrected Safety)
- Conformal shift       : +810.31 units

Planned Policy:
- Order Point: 15,000 | Trips: 20 | Cost: $3232.00 | CO2: 255kg

Realized Backtest:
- Cashout Rate: 0.00% | Trips: 18 | Cost: $2774.78 | CO2: 216kg

=== DECISION STATUS ===
STATUS: ✅ PASSED (SLA met, Carbon under budget)
Audit written: artifacts/audit/audit_20260207_221602.json


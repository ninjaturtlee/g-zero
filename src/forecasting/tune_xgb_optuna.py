from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

from src.utils.io import load_yaml
from src.utils.metrics import mae
from src.forecasting.xgboost_model import make_time_features, add_lags
from src.forecasting.bank_real_generator import BankRealGeneratorConfig, generate_bank_real_series
from src.forecasting.rolling_cv import RollingCVConfig, rolling_time_splits


def underforecast_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Penalize only under-forecasting (stockout-risk proxy)."""
    return float(np.mean(np.maximum(y_true - y_pred, 0.0)))


def overforecast_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Penalize only over-forecasting (waste/overstock proxy)."""
    return float(np.mean(np.maximum(y_pred - y_true, 0.0)))


def tail_mask(y: np.ndarray, tail_pct: int = 80) -> np.ndarray:
    thr = np.percentile(y, tail_pct)
    return y >= thr


def build_features(series: pd.Series, train_cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    base_feats = make_time_features(series)
    feats = add_lags(
        base_feats,
        series,
        lags=tuple(train_cfg["lags"]),
        roll_window=int(train_cfg["roll_window"]),
    )
    df = feats.copy()
    df["y"] = series.values
    df = df.dropna()
    y = df["y"].values.astype(float)
    X = df.drop(columns=["y"])
    # sanity: ensure time ordering
    assert df.index.is_monotonic_increasing, "Index not increasing (possible leakage)."
    return X, y


def objective_factory(datasets, cv_cfg: RollingCVConfig, tail_pct: int):
    """
    Robust Optuna objective:
    - evaluates each trial across multiple seeds (datasets)
    - uses rolling CV per dataset
    - optimizes tail under-forecast risk, but penalizes over-forecasting and MAE
    - adds a worst-case penalty to prevent catastrophic regimes
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 6.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 6.0),
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",
        }

        seed_scores = []

        for X, y in datasets:
            splits = rolling_time_splits(len(y), cv_cfg)
            fold_scores = []

            for tr_idx, va_idx in splits:
                X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
                X_va, y_va = X.iloc[va_idx], y[va_idx]

                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_va)

                m_tail = tail_mask(y_va, tail_pct=tail_pct)

                tail_under = underforecast_mae(y_va[m_tail], pred[m_tail])
                overall_under = underforecast_mae(y_va, pred)
                overall_over = overforecast_mae(y_va, pred)
                overall_mae = mae(y_va, pred)

                # Risk-first objective (lower is better):
                # - prioritize tail under-forecast (stockout risk)
                # - keep overall under-forecast low
                # - punish excessive over-forecast (waste)
                # - keep MAE modest to avoid degenerate behavior
                score = (
                    0.50 * tail_under
                    + 0.25 * overall_under
                    + 0.15 * overall_over
                    + 0.10 * overall_mae
                )
                fold_scores.append(score)

            seed_scores.append(float(np.mean(fold_scores)))

        seed_scores = np.array(seed_scores, dtype=float)

        mean_score = float(np.mean(seed_scores))
        worst_score = float(np.max(seed_scores))  # worst regime
        robust_score = mean_score + 0.25 * worst_score

        # Optional: report for debugging in Optuna UI
        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("worst_score", worst_score)

        return robust_score

    return objective


def main() -> None:
    sim_cfg = load_yaml(Path("configs/sim.yaml"))
    train_cfg = load_yaml(Path("configs/train.yaml"))

    days = int(sim_cfg["horizon_days"])
    seed = int(sim_cfg["seed"])

    # Rolling CV config
    cv_block = train_cfg.get("rolling_cv", {})
    cv_cfg = RollingCVConfig(
        n_splits=int(cv_block.get("n_splits", 3)),
        val_size=int(cv_block.get("val_size", 60)),
        min_train_size=int(cv_block.get("min_train_size", 180)),
    )

    tail_pct = int(train_cfg.get("tail_pct", 80))

    # Robust seeds (multiple bank-real regimes)
    robust_seeds = train_cfg.get("robust_seeds", [40, 41, 42, 43, 44])
    robust_seeds = [int(s) for s in robust_seeds]

    # Build datasets for each seed
    datasets = []
    for s in robust_seeds:
        series = generate_bank_real_series(BankRealGeneratorConfig(days=days, seed=s))
        X, y = build_features(series, train_cfg)
        datasets.append((X, y))

    # Optuna config
    opt_block = train_cfg.get("optuna", {})
    n_trials = int(opt_block.get("n_trials", 80))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    objective = objective_factory(datasets, cv_cfg=cv_cfg, tail_pct=tail_pct)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = float(study.best_value)

    out_dir = Path("artifacts/optuna")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": n_trials,
        "rolling_cv": cv_cfg.__dict__,
        "tail_pct": tail_pct,
        "robust_seeds": robust_seeds,
        "note": "Objective = mean(seed_scores) + 0.25*max(seed_scores); each seed uses rolling CV; score weights tail_under/under/over/mae.",
    }
    (out_dir / "best_xgb_params_robust.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Optuna Tuning (ROBUST multi-seed + rolling CV) ===")
    print(f"Trials: {n_trials}")
    print(f"Tail pct: {tail_pct}")
    print(f"Robust seeds: {robust_seeds}")
    print(f"Objective (lower is better): {best_value:.4f}")
    print("\nBest params:")
    for k, v in best_params.items():
        print(f"- {k}: {v}")
    print(f"\nSaved: {out_dir / 'best_xgb_params_robust.json'}")
    print("\nNext: copy these params into configs/train.yaml under xgboost.\n")


if __name__ == "__main__":
    main()
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

# Fallback (always available) for quantiles if XGBoost quantile objective isn't supported in runtime
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class QuantileForecaster:
    model: Any
    kind: str  # "xgb" or "gbr"

    @staticmethod
    def build_xgb_quantile(params: Dict[str, Any], alpha: float, seed: int = 42) -> "QuantileForecaster":
        """
        XGBoost quantile model (preferred).
        Requires XGBoost supporting objective='reg:quantileerror' and quantile_alpha.
        """
        m = XGBRegressor(
            n_estimators=int(params.get("n_estimators", 800)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            objective="reg:quantileerror",
            quantile_alpha=float(alpha),
            random_state=int(seed),
            n_jobs=int(params.get("n_jobs", -1)),
        )
        return QuantileForecaster(model=m, kind="xgb")

    @staticmethod
    def build_fallback_quantile(alpha: float, seed: int = 42) -> "QuantileForecaster":
        """
        GradientBoostingRegressor quantile fallback (robust).
        """
        m = GradientBoostingRegressor(loss="quantile", alpha=float(alpha), random_state=int(seed))
        return QuantileForecaster(model=m, kind="gbr")

    @staticmethod
    def build(params: Dict[str, Any], alpha: float, seed: int = 42) -> "QuantileForecaster":
        """
        Try XGBoost quantile objective; if it fails, fallback to sklearn quantile regressor.
        """
        try:
            return QuantileForecaster.build_xgb_quantile(params, alpha=alpha, seed=seed)
        except TypeError:
            # In case XGBRegressor doesn't accept quantile_alpha or objective
            return QuantileForecaster.build_fallback_quantile(alpha=alpha, seed=seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)
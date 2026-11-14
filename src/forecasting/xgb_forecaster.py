"""XGBoost-based demand forecaster.

Provides a gradient-boosted tree model for demand prediction using XGBoost.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from src.common.logger import get_logger

logger = get_logger(__name__)


class XGBDemandForecaster:
    """XGBoost demand forecaster for tabular feature-based prediction."""

    def __init__(self, params: dict | None = None) -> None:
        """Initialize the XGBoost forecaster.

        Args:
            params: XGBoost hyperparameters. If ``None``, sensible defaults
                from the project configuration are used.
        """
        self.params: dict = params or {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        self._model: xgb.XGBRegressor | None = None
        logger.info("XGBDemandForecaster initialised with params=%s", self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBDemandForecaster":
        """Fit the XGBoost model on training data.

        Args:
            X: Feature matrix.
            y: Target demand values.

        Returns:
            Self for method chaining.
        """
        raise NotImplementedError("Will be implemented in Module 1")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate demand predictions for new feature data.

        Args:
            X: Feature matrix for prediction.

        Returns:
            Array of predicted demand values.
        """
        raise NotImplementedError("Will be implemented in Module 1")

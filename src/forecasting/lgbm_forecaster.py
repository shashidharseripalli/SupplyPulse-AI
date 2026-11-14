"""LightGBM-based demand forecaster.

Provides a gradient-boosted tree model for demand prediction using LightGBM.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.common.logger import get_logger

logger = get_logger(__name__)


class LGBMDemandForecaster:
    """LightGBM demand forecaster for tabular feature-based prediction."""

    def __init__(self, params: dict | None = None) -> None:
        """Initialize the LightGBM forecaster.

        Args:
            params: LightGBM hyperparameters. If ``None``, sensible defaults
                from the project configuration are used.
        """
        self.params: dict = params or {
            "objective": "regression",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        }
        self._model: lgb.LGBMRegressor | None = None
        logger.info("LGBMDemandForecaster initialised with params=%s", self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMDemandForecaster":
        """Fit the LightGBM model on training data.

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

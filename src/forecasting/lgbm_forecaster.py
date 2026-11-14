"""LightGBM-based demand forecaster.

Provides a fast gradient-boosted tree model for tabular time-series demand forecasting
using LightGBM, optimized for memory efficiency and CPU parallel execution.
"""

from typing import Sequence
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.common.config_loader import load_config
from src.common.exceptions import ModelError
from src.common.logger import get_logger

logger = get_logger(__name__)


class LGBMDemandForecaster:
    """LightGBM demand forecaster for tabular feature-based regression.

    Optimized for rapid training on multi-core CPUs.
    Defaults to configuration parameters specified in `config/settings.yaml`.

    Attributes:
        params: Hyperparameter dictionary for the underlying `LGBMRegressor`.
        model: Trained `lgb.LGBMRegressor` instance or `None` if unfitted.
        feature_names_: List of input feature names recorded during training.
    """

    def __init__(self, params: dict | None = None) -> None:
        """Initialize the LightGBM forecaster.

        Args:
            params: Custom hyperparameter dictionary. If `None`, defaults are
                loaded from `config/settings.yaml`.
        """
        config = load_config()
        cfg_params = config.get("forecasting", {}).get("lightgbm", {})

        default_params: dict = {
            "n_estimators": cfg_params.get("n_estimators", 200),
            "max_depth": cfg_params.get("max_depth", -1),
            "num_leaves": cfg_params.get("num_leaves", 31),
            "learning_rate": cfg_params.get("learning_rate", 0.08),
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "n_jobs": -1,  # Multi-threaded CPU execution
            "objective": "regression",
            "verbose": -1,
        }

        if params:
            default_params.update(params)

        self.params: dict = default_params
        self.model: lgb.LGBMRegressor | None = None
        self.feature_names_: list[str] = []
        logger.debug("LGBMDemandForecaster initialized with params=%s", self.params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        eval_set: Sequence[tuple[pd.DataFrame, pd.Series | np.ndarray]] | None = None,
    ) -> "LGBMDemandForecaster":
        """Fit the LightGBM regressor on training features and target.

        Args:
            X: Input tabular feature matrix (lags, rolling stats, calendar signals).
            y: Target demand series.
            eval_set: Optional validation sets for loss tracking and early stopping.

        Returns:
            Self instance for method chaining.

        Raises:
            ModelError: If model training fails or inputs are invalid.
        """
        try:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            self.feature_names_ = list(X.columns)
            self.model = lgb.LGBMRegressor(**self.params)

            fit_kwargs = {}
            if eval_set:
                fit_kwargs["eval_set"] = eval_set

            self.model.fit(X, y, **fit_kwargs)
            logger.info(
                "LGBMDemandForecaster trained successfully on shape X=%s, y=%d.",
                X.shape,
                len(y),
            )
            return self
        except Exception as e:
            logger.error("Failed to fit LGBMDemandForecaster: %s", e)
            raise ModelError(f"LightGBM model training failed: {e}") from e

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate demand predictions for new feature data.

        Post-processes raw predictions by clipping at 0.0 to guarantee the
        domain invariant of non-negative demand quantities.

        Args:
            X: Input feature DataFrame matching training columns.

        Returns:
            1D numpy array of non-negative predicted demand values.

        Raises:
            ModelError: If the forecaster has not been fitted prior to predict.
        """
        if self.model is None:
            raise ModelError("LGBMDemandForecaster is not fitted. Call fit() first.")

        try:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            raw_preds = self.model.predict(X)
            # Physical inventory demand cannot be negative
            return np.maximum(0.0, raw_preds)
        except Exception as e:
            logger.error("Failed during LGBMDemandForecaster inference: %s", e)
            raise ModelError(f"LightGBM inference failed: {e}") from e

    def get_feature_importances(self) -> pd.Series:
        """Retrieve feature importances from the trained tree ensemble.

        Returns:
            Series of feature importances indexed by column names, sorted descending.

        Raises:
            ModelError: If the model has not been fitted.
        """
        if self.model is None:
            raise ModelError("Cannot get feature importances from an unfitted model.")

        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names_,
            name="importance",
        ).sort_values(ascending=False)
        return importances

    @property
    def is_fitted(self) -> bool:
        """Check whether the underlying LightGBM model has been fitted."""
        return self.model is not None

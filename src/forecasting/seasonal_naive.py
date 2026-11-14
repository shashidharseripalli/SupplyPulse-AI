"""Seasonal Naive baseline forecaster.

Repeats the last observed seasonal cycle as the forecast.
Provides a robust benchmark for comparing complex machine learning models.
"""

from typing import Union
import numpy as np
import pandas as pd

from src.common.exceptions import ModelError
from src.common.logger import get_logger

logger = get_logger(__name__)


class SeasonalNaiveForecaster:
    """Seasonal Naive forecaster that repeats the last observed seasonal cycle.

    Given a seasonal period $m$ (e.g., 7 for weekly, 30 for monthly), the model
    identifies the most recent $m$ observations from the historical series and
    projects them forward periodically over the required horizon $H$.

    Attributes:
        seasonal_period: Integer length of one seasonal cycle.
        _last_season: 1D array of the last `seasonal_period` values recorded during fit.
    """

    def __init__(self, seasonal_period: int = 7) -> None:
        """Initialize the Seasonal Naive forecaster.

        Args:
            seasonal_period: Length of one seasonal cycle (default: 7 for weekly seasonality).

        Raises:
            ValueError: If seasonal_period is less than 1.
        """
        if seasonal_period < 1:
            raise ValueError(
                f"seasonal_period must be >= 1, got {seasonal_period}"
            )
        self.seasonal_period: int = seasonal_period
        self._last_season: np.ndarray | None = None
        logger.debug(
            "SeasonalNaiveForecaster initialized with seasonal_period=%d",
            self.seasonal_period,
        )

    def fit(
        self, y: Union[pd.Series, np.ndarray, list[float]]
    ) -> "SeasonalNaiveForecaster":
        """Fit the model by capturing the last seasonal cycle from the input series.

        Args:
            y: Historical demand time series.

        Returns:
            Self instance for method chaining.

        Raises:
            ModelError: If series has fewer observations than `seasonal_period`.
        """
        if isinstance(y, pd.Series):
            y_arr = y.dropna().to_numpy(dtype=float)
        else:
            y_arr = np.asarray(y, dtype=float)

        if len(y_arr) < self.seasonal_period:
            raise ModelError(
                f"Input series length ({len(y_arr)}) must be at least as large as "
                f"the seasonal_period ({self.seasonal_period})."
            )

        self._last_season = y_arr[-self.seasonal_period :].copy()
        logger.debug(
            "SeasonalNaiveForecaster fitted on %d points. Captured cycle: %s",
            len(y_arr),
            self._last_season,
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Generate demand forecast by repeating the captured seasonal cycle.

        Args:
            horizon: Positive integer number of future steps to forecast.

        Returns:
            1D numpy array of length `horizon` with non-negative predicted values.

        Raises:
            ModelError: If the model has not been fitted prior to predict.
            ValueError: If horizon is less than 1.
        """
        if self._last_season is None:
            raise ModelError(
                "SeasonalNaiveForecaster must be fitted before predict() is called."
            )
        if horizon < 1:
            raise ValueError(f"Forecast horizon must be >= 1, got {horizon}")

        repeats = int(np.ceil(horizon / self.seasonal_period))
        tiled = np.tile(self._last_season, repeats)[:horizon]

        # Enforce domain invariant: demand cannot be negative
        predictions = np.maximum(0.0, tiled)
        logger.debug(
            "Generated %d forecast steps using Seasonal Naive.", horizon
        )
        return predictions

    @property
    def is_fitted(self) -> bool:
        """Check whether the model has captured the seasonal cycle."""
        return self._last_season is not None

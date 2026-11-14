"""Seasonal Naive baseline forecaster.

Repeats the last observed seasonal cycle as the forecast.
"""

import numpy as np
import pandas as pd

from src.common.logger import get_logger

logger = get_logger(__name__)


class SeasonalNaiveForecaster:
    """Seasonal Naive forecaster that repeats the last seasonal cycle."""

    def __init__(self, seasonal_period: int = 7) -> None:
        """Initialize with seasonal period.

        Args:
            seasonal_period: Length of one seasonal cycle (default: 7 for weekly).
        """
        self.seasonal_period = seasonal_period
        self._last_season: np.ndarray | None = None

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        """Fit the model by storing the last seasonal cycle.

        Args:
            y: Time series values.

        Returns:
            Self for method chaining.
        """
        raise NotImplementedError("Will be implemented in Module 1")

    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecast by repeating seasonal pattern.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            Array of forecasted values.
        """
        raise NotImplementedError("Will be implemented in Module 1")

"""Tests for forecasting module."""
import numpy as np
import pytest
from src.forecasting.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    compute_all_metrics,
)
from src.forecasting.seasonal_naive import SeasonalNaiveForecaster


class TestEvaluationMetrics:
    """Tests for forecast evaluation metrics."""

    def test_evaluation_mae(self) -> None:
        """MAE should compute correctly for known values."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.5)

    def test_evaluation_rmse(self) -> None:
        """RMSE should compute correctly for known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(0.0)

    def test_evaluation_mape(self) -> None:
        """MAPE should compute correctly for known values."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 330.0])
        expected = np.mean([10/100, 10/200, 30/300]) * 100
        assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(expected)

    def test_compute_all_metrics_keys(self) -> None:
        """compute_all_metrics should return dict with mae, rmse, mape keys."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        metrics = compute_all_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {"mae", "rmse", "mape"}


class TestSeasonalNaive:
    """Tests for Seasonal Naive forecaster."""

    def test_seasonal_naive_not_implemented(self, sample_time_series) -> None:
        """SeasonalNaiveForecaster.fit should raise NotImplementedError."""
        forecaster = SeasonalNaiveForecaster(seasonal_period=7)
        with pytest.raises(NotImplementedError):
            forecaster.fit(sample_time_series["value"])

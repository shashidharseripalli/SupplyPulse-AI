"""Comprehensive test suite for Module 1: Demand Forecasting.

Tests:
    - Seasonal Naive baseline logic, cycle repetition, and constraints
    - XGBoost forecaster training, inference, and feature importances
    - LightGBM forecaster training, inference, and feature importances
    - Demand feature engineering pipeline (lags, rolling stats, calendar harmonics)
    - Statistical evaluation metrics (MAE, RMSE, MAPE, WAPE, SMAPE)
"""

import numpy as np
import pandas as pd
import pytest

from src.common.exceptions import ModelError
from src.forecasting.backtest import rolling_origin_backtest
from src.forecasting.evaluation import (
    compute_all_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
)
from src.forecasting.lgbm_forecaster import LGBMDemandForecaster
from src.forecasting.seasonal_naive import SeasonalNaiveForecaster
from src.forecasting.xgb_forecaster import XGBDemandForecaster
from src.preprocessing.features import (
    DemandFeaturePipeline,
    create_calendar_features,
    create_lag_features,
    create_rolling_features,
)


class TestEvaluationMetrics:
    """Tests for statistical forecast evaluation metrics."""

    def test_evaluation_mae(self) -> None:
        """MAE should compute accurate scale-dependent error."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 31.0])
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(5.0 / 3.0)

    def test_evaluation_rmse(self) -> None:
        """RMSE should penalize larger errors quadratically."""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # sqrt((9 + 16)/2) = sqrt(12.5) = 3.5355...
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(12.5))

    def test_evaluation_mape(self) -> None:
        """MAPE should compute correct percentage score for non-zero actuals."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(10.0)

    def test_evaluation_mape_zero_division_raises(self) -> None:
        """MAPE must raise ValueError when actuals contain zeros."""
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 110.0])
        with pytest.raises(ValueError, match="zero values"):
            mean_absolute_percentage_error(y_true, y_pred)

    def test_evaluation_wape(self) -> None:
        """WAPE should compute sum-weighted percentage error."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # sum(|diff|) = 10 + 20 = 30. sum(actual) = 300. WAPE = 30/300 * 100 = 10%
        assert weighted_absolute_percentage_error(y_true, y_pred) == pytest.approx(10.0)

    def test_evaluation_smape(self) -> None:
        """SMAPE should be symmetric and bounded between 0% and 200%."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([10.0, 20.0])
        assert symmetric_mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(0.0)

    def test_compute_all_metrics_keys(self) -> None:
        """compute_all_metrics should include all 5 metrics."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 21.0, 31.0])
        metrics = compute_all_metrics(y_true, y_pred)
        assert {"mae", "rmse", "mape", "wape", "smape"}.issubset(metrics.keys())


class TestSeasonalNaiveForecaster:
    """Tests for Seasonal Naive baseline logic."""

    def test_fit_and_predict_exact_cycle(self) -> None:
        """Model should repeat the exact last seasonal cycle."""
        # 14 days of weekly data (2 cycles: 1..7, 1..7)
        history = [10, 20, 30, 40, 50, 60, 70, 15, 25, 35, 45, 55, 65, 75]
        forecaster = SeasonalNaiveForecaster(seasonal_period=7)
        forecaster.fit(history)

        preds = forecaster.predict(horizon=7)
        expected = np.array([15, 25, 35, 45, 55, 65, 75], dtype=float)
        np.testing.assert_array_equal(preds, expected)

    def test_predict_extended_horizon(self) -> None:
        """Predicting across multiple periods should tile the cycle repeatedly."""
        history = [1, 2, 3]
        forecaster = SeasonalNaiveForecaster(seasonal_period=3)
        forecaster.fit(history)

        preds = forecaster.predict(horizon=7)
        expected = np.array([1, 2, 3, 1, 2, 3, 1], dtype=float)
        np.testing.assert_array_equal(preds, expected)

    def test_insufficient_history_raises(self) -> None:
        """Series shorter than seasonal_period must raise ModelError."""
        forecaster = SeasonalNaiveForecaster(seasonal_period=7)
        with pytest.raises(ModelError):
            forecaster.fit([10, 20, 30])

    def test_predict_before_fit_raises(self) -> None:
        """Calling predict() before fit() must raise ModelError."""
        forecaster = SeasonalNaiveForecaster(seasonal_period=7)
        with pytest.raises(ModelError):
            forecaster.predict(horizon=7)


class TestTreeForecasters:
    """Tests for XGBoost and LightGBM forecasters."""

    @pytest.fixture
    def synthetic_tabular_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Generate small synthetic feature matrix and demand target."""
        np.random.seed(42)
        n = 60
        X = pd.DataFrame({
            "lag_1": np.random.uniform(50, 150, n),
            "lag_7": np.random.uniform(50, 150, n),
            "rolling_mean_7": np.random.uniform(50, 150, n),
            "day_of_week": np.tile(np.arange(7), n // 7 + 1)[:n],
        })
        y = pd.Series(
            0.6 * X["lag_1"] + 0.3 * X["lag_7"] + np.random.normal(0, 5, n),
            name="demand",
        )
        return X, y

    def test_xgb_forecaster_train_and_predict(
        self, synthetic_tabular_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """XGBoost should train, predict non-negative values, and output feature importances."""
        X, y = synthetic_tabular_data
        forecaster = XGBDemandForecaster(params={"n_estimators": 20, "max_depth": 3})
        forecaster.fit(X, y)

        assert forecaster.is_fitted
        preds = forecaster.predict(X.iloc[:10])
        assert len(preds) == 10
        assert np.all(preds >= 0.0)

        importances = forecaster.get_feature_importances()
        assert len(importances) == X.shape[1]
        assert "lag_1" in importances.index

    def test_lgbm_forecaster_train_and_predict(
        self, synthetic_tabular_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """LightGBM should train, predict non-negative values, and output feature importances."""
        X, y = synthetic_tabular_data
        forecaster = LGBMDemandForecaster(params={"n_estimators": 20, "max_depth": 3})
        forecaster.fit(X, y)

        assert forecaster.is_fitted
        preds = forecaster.predict(X.iloc[:10])
        assert len(preds) == 10
        assert np.all(preds >= 0.0)

        importances = forecaster.get_feature_importances()
        assert len(importances) == X.shape[1]


class TestFeaturePipeline:
    """Tests for demand time series feature transformers."""

    @pytest.fixture
    def sample_daily_series(self) -> pd.DataFrame:
        """Create 45 days of daily demand records."""
        dates = pd.date_range("2024-01-01", periods=45, freq="D")
        np.random.seed(42)
        demand = 100 + 20 * np.sin(np.arange(45) * 2 * np.pi / 7) + np.random.normal(0, 2, 45)
        return pd.DataFrame({"date": dates, "demand": demand})

    def test_calendar_features(self, sample_daily_series: pd.DataFrame) -> None:
        """Calendar features must include day_of_week, cyclical harmonics, and weekend flags."""
        res = create_calendar_features(sample_daily_series, date_col="date")
        expected_cols = {
            "day_of_week", "day_of_month", "month", "quarter",
            "is_weekend", "day_of_year", "dow_sin", "dow_cos",
        }
        assert expected_cols.issubset(res.columns)

    def test_lag_features_shift_correctly(self, sample_daily_series: pd.DataFrame) -> None:
        """Lag-1 should equal the previous day's demand value."""
        res = create_lag_features(sample_daily_series, target_col="demand", lags=[1, 7])
        assert "lag_1" in res.columns
        assert "lag_7" in res.columns
        # lag_1 at row index 1 should equal demand at index 0
        assert res.loc[1, "lag_1"] == pytest.approx(sample_daily_series.loc[0, "demand"])

    def test_rolling_features_shift_leakage_prevention(
        self, sample_daily_series: pd.DataFrame
    ) -> None:
        """Rolling stats must use shifted target so the current day's demand is never leaked."""
        res = create_rolling_features(sample_daily_series, target_col="demand", windows=[7])
        assert "rolling_mean_7" in res.columns
        # Row 0 rolling mean must be NaN because target is shifted by 1
        assert pd.isna(res.loc[0, "rolling_mean_7"])

    def test_demand_feature_pipeline_fit_transform(
        self, sample_daily_series: pd.DataFrame
    ) -> None:
        """Full pipeline must produce non-empty, dropna-filtered feature matrix."""
        pipeline = DemandFeaturePipeline(
            date_col="date",
            target_col="demand",
            lags=[1, 7],
            windows=[7],
            drop_na=True,
        )
        pipeline.fit(sample_daily_series)
        assert len(pipeline.feature_names_) > 0

        transformed = pipeline.transform(sample_daily_series)
        assert len(transformed) < len(sample_daily_series)  # initial rows dropped
        assert not transformed.isna().any().any()

    def test_rolling_origin_backtest(
        self, sample_daily_series: pd.DataFrame
    ) -> None:
        """Rolling origin backtest should compute valid metrics across multiple folds."""
        pipeline = DemandFeaturePipeline(
            date_col="date",
            target_col="demand",
            lags=[1, 7],
            windows=[7],
            drop_na=True,
        )
        pipeline.fit(sample_daily_series)
        df_feat = pipeline.transform(sample_daily_series)

        results = rolling_origin_backtest(
            model_factory=lambda: LGBMDemandForecaster(
                params={"n_estimators": 10, "max_depth": 2, "verbose": -1}
            ),
            df=df_feat,
            feature_cols=pipeline.feature_names_,
            target_col="demand",
            n_splits=3,
        )
        assert "mean_metrics" in results
        assert "mae" in results["mean_metrics"]
        assert results["n_splits"] == 3
        assert len(results["fold_metrics"]) == 3


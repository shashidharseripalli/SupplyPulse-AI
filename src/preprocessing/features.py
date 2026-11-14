"""Time series feature engineering for demand forecasting.

Provides functions and transformers to generate leakage-proof lag features,
rolling statistics, cyclical calendar encodings, and temporal signals.
"""

from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.common.exceptions import DataError
from src.common.logger import get_logger

logger = get_logger(__name__)


def create_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
    include_cyclical: bool = True,
) -> pd.DataFrame:
    """Generate calendar features from a timestamp column.

    Extracts:
        - day_of_week (0 = Monday, 6 = Sunday)
        - day_of_month (1 to 31)
        - month (1 to 12)
        - quarter (1 to 4)
        - is_weekend (binary 0 or 1)
        - day_of_year (1 to 366)
        - Cyclical sine/cosine encodings for day_of_week and month (optional).

    Args:
        df: Input DataFrame containing the date column.
        date_col: Name of the timestamp column.
        include_cyclical: Whether to include continuous sine/cosine harmonics.

    Returns:
        New DataFrame with appended calendar feature columns.
    """
    if date_col not in df.columns:
        raise DataError(f"Date column '{date_col}' not found in DataFrame.")

    res = df.copy()
    dt_series = pd.to_datetime(res[date_col])

    res["day_of_week"] = dt_series.dt.dayofweek
    res["day_of_month"] = dt_series.dt.day
    res["month"] = dt_series.dt.month
    res["quarter"] = dt_series.dt.quarter
    res["is_weekend"] = (res["day_of_week"] >= 5).astype(int)
    res["day_of_year"] = dt_series.dt.dayofyear

    if include_cyclical:
        # Day of week has period 7
        res["dow_sin"] = np.sin(2 * np.pi * res["day_of_week"] / 7.0)
        res["dow_cos"] = np.cos(2 * np.pi * res["day_of_week"] / 7.0)
        # Month has period 12
        res["month_sin"] = np.sin(2 * np.pi * (res["month"] - 1) / 12.0)
        res["month_cos"] = np.cos(2 * np.pi * (res["month"] - 1) / 12.0)

    return res


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "demand",
    lags: Sequence[int] = (1, 2, 7, 14, 21, 28, 30),
) -> pd.DataFrame:
    """Generate historical lagged values of the target variable.

    Args:
        df: Input DataFrame sorted in chronological order.
        target_col: Name of the target series column.
        lags: Sequence of positive integer offsets to lag.

    Returns:
        DataFrame with lag_{k} columns added.
    """
    if target_col not in df.columns:
        raise DataError(f"Target column '{target_col}' not found in DataFrame.")

    res = df.copy()
    for lag in lags:
        if lag < 1:
            raise ValueError(f"Lag offset must be >= 1, got {lag}")
        res[f"lag_{lag}"] = res[target_col].shift(lag)

    return res


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "demand",
    windows: Sequence[int] = (7, 14, 30),
    min_periods: int = 1,
) -> pd.DataFrame:
    """Generate rolling window statistics over past observations.

    Crucially, shifts the series by 1 period before computing rolling stats
    to guarantee zero lookahead leakage of the contemporaneous target.

    Computes:
        - rolling_mean_{w}
        - rolling_std_{w}
        - rolling_min_{w}
        - rolling_max_{w}

    Args:
        df: Input DataFrame in chronological order.
        target_col: Name of the target series column.
        windows: Sequence of integer window lengths in days/steps.
        min_periods: Minimum number of observations required in window.

    Returns:
        DataFrame with rolling summary columns added.
    """
    if target_col not in df.columns:
        raise DataError(f"Target column '{target_col}' not found in DataFrame.")

    res = df.copy()
    # Shift target by 1 to prevent contemporary data leakage
    shifted_target = res[target_col].shift(1)

    for w in windows:
        if w < 2:
            raise ValueError(f"Rolling window size must be >= 2, got {w}")
        rolling = shifted_target.rolling(window=w, min_periods=min_periods)
        res[f"rolling_mean_{w}"] = rolling.mean()
        res[f"rolling_std_{w}"] = rolling.std().fillna(0.0)
        res[f"rolling_min_{w}"] = rolling.min()
        res[f"rolling_max_{w}"] = rolling.max()

    return res


class DemandFeaturePipeline(BaseEstimator, TransformerMixin):
    """End-to-end scikit-learn transformer for demand time series feature matrices.

    Chains calendar, lag, and rolling calculations into a unified, reproducible
    transformation.
    """

    def __init__(
        self,
        date_col: str = "date",
        target_col: str = "demand",
        lags: Sequence[int] = (1, 2, 7, 14, 21, 28, 30),
        windows: Sequence[int] = (7, 14, 30),
        drop_na: bool = True,
    ) -> None:
        """Initialize the feature engineering pipeline.

        Args:
            date_col: Timestamp column name.
            target_col: Demand target column name.
            lags: Lags to generate.
            windows: Rolling window widths to calculate.
            drop_na: Whether to drop initial rows with NaNs resulting from shifts.
        """
        self.date_col = date_col
        self.target_col = target_col
        self.lags = list(lags)
        self.windows = list(windows)
        self.drop_na = drop_na
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DemandFeaturePipeline":
        """Validate input and configure feature names.

        Args:
            X: Input time series DataFrame.
            y: Ignored (scikit-learn interface compatibility).

        Returns:
            Self instance.
        """
        transformed = self._build_features(X)
        exclude_cols = {self.date_col, self.target_col}
        self.feature_names_ = [c for c in transformed.columns if c not in exclude_cols]
        logger.debug(
            "DemandFeaturePipeline fitted. Generated %d features.",
            len(self.feature_names_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame into a rich feature set.

        Args:
            X: Input time series DataFrame.

        Returns:
            Engineered DataFrame with all lag, rolling, and temporal columns.
        """
        transformed = self._build_features(X)
        if self.drop_na:
            transformed = transformed.dropna().reset_index(drop=True)
        return transformed

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal helper executing the transformation steps sequentially."""
        res = create_calendar_features(df, date_col=self.date_col)
        res = create_lag_features(res, target_col=self.target_col, lags=self.lags)
        res = create_rolling_features(res, target_col=self.target_col, windows=self.windows)
        return res

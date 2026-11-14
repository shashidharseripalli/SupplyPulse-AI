"""Scikit-learn compatible transformers for feature engineering.

Provides reusable transformers that integrate with scikit-learn
:class:`~sklearn.pipeline.Pipeline` objects and follow the standard
``fit`` / ``transform`` API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.common.logger import get_logger

logger = get_logger(__name__)


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract calendar features from a datetime column.

    Produces the following new columns (prefixed by the source column name):

    * ``<col>_day_of_week`` – Monday = 0 … Sunday = 6
    * ``<col>_month`` – 1–12
    * ``<col>_quarter`` – 1–4
    * ``<col>_is_weekend`` – 1 if Saturday/Sunday, else 0

    Args:
        date_column: Name of the datetime column to extract features from.

    Example::

        extractor = DateFeatureExtractor(date_column="order_date")
        df_transformed = extractor.fit_transform(df)
    """

    def __init__(self, date_column: str = "date") -> None:
        self.date_column = date_column

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: Any = None,
    ) -> "DateFeatureExtractor":
        """Fit is a no-op; exists for pipeline compatibility.

        Args:
            X: Input DataFrame (unused beyond validation).
            y: Ignored.

        Returns:
            self
        """
        if self.date_column not in X.columns:
            raise ValueError(
                f"Column '{self.date_column}' not found in DataFrame. "
                f"Available columns: {list(X.columns)}"
            )
        logger.debug(
            "DateFeatureExtractor fitted on column '%s'.", self.date_column
        )
        return self

    def transform(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: Any = None,
    ) -> pd.DataFrame:
        """Create calendar features from the date column.

        Args:
            X: Input DataFrame containing *date_column*.
            y: Ignored.

        Returns:
            A copy of *X* with the four new feature columns appended.
        """
        if self.date_column not in X.columns:
            raise ValueError(
                f"Column '{self.date_column}' not found in DataFrame. "
                f"Available columns: {list(X.columns)}"
            )

        X_out = X.copy()  # noqa: N806
        dt_series = pd.to_datetime(X_out[self.date_column])

        col = self.date_column
        X_out[f"{col}_day_of_week"] = dt_series.dt.dayofweek
        X_out[f"{col}_month"] = dt_series.dt.month
        X_out[f"{col}_quarter"] = dt_series.dt.quarter
        X_out[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)

        logger.info(
            "Extracted date features from '%s': 4 new columns added.", col
        )
        return X_out


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers using the IQR (Interquartile Range) method.

    For each numeric column, values below ``Q1 - multiplier * IQR`` are
    clipped to the lower bound and values above ``Q3 + multiplier * IQR``
    are clipped to the upper bound.

    Args:
        multiplier: Scaling factor for the IQR when computing bounds.
            Defaults to ``1.5`` (standard box-plot whisker distance).

    Example::

        clipper = OutlierClipper(multiplier=1.5)
        df_clipped = clipper.fit_transform(df)
    """

    def __init__(self, multiplier: float = 1.5) -> None:
        self.multiplier = multiplier
        self.lower_bounds_: dict[str, float] = {}
        self.upper_bounds_: dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: Any = None,
    ) -> "OutlierClipper":
        """Compute the IQR-based clipping bounds for each numeric column.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            q1: float = float(X[col].quantile(0.25))
            q3: float = float(X[col].quantile(0.75))
            iqr: float = q3 - q1
            self.lower_bounds_[col] = q1 - self.multiplier * iqr
            self.upper_bounds_[col] = q3 + self.multiplier * iqr

        logger.debug(
            "OutlierClipper fitted on %d numeric columns with multiplier=%.2f.",
            len(numeric_cols),
            self.multiplier,
        )
        return self

    def transform(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: Any = None,
    ) -> pd.DataFrame:
        """Clip values to the precomputed IQR bounds.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            A copy of *X* with outlier values clipped.
        """
        X_out = X.copy()  # noqa: N806

        for col in self.lower_bounds_:
            if col not in X_out.columns:
                logger.warning(
                    "Column '%s' present during fit but missing during "
                    "transform — skipping.",
                    col,
                )
                continue
            lower = self.lower_bounds_[col]
            upper = self.upper_bounds_[col]
            X_out[col] = X_out[col].clip(lower=lower, upper=upper)

        logger.info(
            "Clipped outliers in %d columns.", len(self.lower_bounds_)
        )
        return X_out

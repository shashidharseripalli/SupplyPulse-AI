"""Backtesting engine for time series demand forecasting models.

Implements expanding-window and rolling-origin backtesting strategies
guaranteeing strict temporal causality (no forward lookahead bias).
"""

from typing import Any, Callable, Sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.common.exceptions import ModelError
from src.common.logger import get_logger
from src.forecasting.evaluation import compute_all_metrics

logger = get_logger(__name__)


def rolling_origin_backtest(
    model_factory: Callable[[], Any],
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "demand",
    n_splits: int = 5,
    max_train_size: int | None = None,
) -> dict[str, Any]:
    """Execute rolling-origin time-series cross-validation.

    Trains the model across multiple expanding/rolling folds, computes
    validation metrics for each out-of-sample window, and aggregates the results.

    Args:
        model_factory: Zero-argument callable returning a fresh forecaster instance
            (e.g., `lambda: XGBDemandForecaster()`).
        df: Input DataFrame ordered chronologically with features and target.
        feature_cols: Names of predictor columns in `df`.
        target_col: Name of the dependent variable column.
        n_splits: Number of cross-validation evaluation folds.
        max_train_size: Optional maximum training window length (for rolling vs expanding).

    Returns:
        Dictionary containing:
            - "fold_metrics": List of metric dictionaries per fold
            - "mean_metrics": Dictionary of average metrics across folds
            - "std_metrics": Standard deviation of metrics across folds
    """
    if len(df) < n_splits * 2:
        raise ModelError(
            f"Dataset length ({len(df)}) is too short for {n_splits} splits."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)
    fold_metrics: list[dict[str, float]] = []

    X = df[list(feature_cols)]
    y = df[target_col]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        metrics = compute_all_metrics(y_val.to_numpy(), preds)
        fold_metrics.append(metrics)
        logger.debug(
            "Fold %d: MAE=%.2f, RMSE=%.2f, WAPE=%.2f%%",
            fold + 1,
            metrics["mae"],
            metrics["rmse"],
            metrics["wape"],
        )

    # Compute mean and standard deviation across folds
    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean(numeric_only=True).to_dict()
    std_metrics = metrics_df.std(numeric_only=True).to_dict()

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "n_splits": n_splits,
    }

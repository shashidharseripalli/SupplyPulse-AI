"""Forecast evaluation metrics.

Provides standard regression metrics for comparing forecast accuracy.
"""

import numpy as np

from src.common.logger import get_logger

logger = get_logger(__name__)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAE score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        RMSE score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error.

    Args:
        y_true: Actual values (must be non-zero).
        y_pred: Predicted values.

    Returns:
        MAPE score as a percentage.

    Raises:
        ValueError: If y_true contains zero values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values; MAPE is undefined.")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all forecast metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary with mae, rmse, and mape keys.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }

"""Forecast evaluation metrics and performance assessment.

Provides standard statistical metrics for evaluating and comparing demand forecast accuracy:
    - MAE: Mean Absolute Error (scale-dependent error in physical units)
    - RMSE: Root Mean Squared Error (penalizes large volatility errors)
    - MAPE: Mean Absolute Percentage Error (relative percentage error)
    - WAPE: Weighted Absolute Percentage Error (industry benchmark robust against intermittent zeros)
    - SMAPE: Symmetric Mean Absolute Percentage Error (bounded relative metric)
"""

from typing import Sequence
import numpy as np
import pandas as pd

from src.common.logger import get_logger

logger = get_logger(__name__)


def mean_absolute_error(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    r"""Compute Mean Absolute Error (MAE).

    $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$

    Args:
        y_true: Ground truth observed demand values.
        y_pred: Forecasted demand values.

    Returns:
        Float MAE score in original demand units.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_t - y_p)))


def root_mean_squared_error(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    r"""Compute Root Mean Squared Error (RMSE).

    $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

    Args:
        y_true: Ground truth observed demand values.
        y_pred: Forecasted demand values.

    Returns:
        Float RMSE score in original demand units.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mean_absolute_percentage_error(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    r"""Compute Mean Absolute Percentage Error (MAPE).

    $$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

    Args:
        y_true: Ground truth observed demand values (must be non-zero).
        y_pred: Forecasted demand values.

    Returns:
        Float percentage score (e.g., 8.5 for 8.5%).

    Raises:
        ValueError: If `y_true` contains zero values.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if np.any(y_t == 0):
        raise ValueError("y_true contains zero values; MAPE is undefined.")
    return float(np.mean(np.abs((y_t - y_p) / y_t)) * 100.0)


def weighted_absolute_percentage_error(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    r"""Compute Weighted Absolute Percentage Error (WAPE / MAD/Mean ratio).

    $$\text{WAPE} = \frac{\sum_{i=1}^n |y_i - \hat{y}_i|}{\sum_{i=1}^n y_i} \times 100\%$$

    Standard supply-chain metric that aggregates errors volume-wise, avoiding
    asymptotic division explosions when individual days experience zero demand.

    Args:
        y_true: Ground truth demand.
        y_pred: Forecasted demand.

    Returns:
        Float WAPE percentage.

    Raises:
        ValueError: If total observed demand is zero.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    total_actual = float(np.sum(y_t))
    if total_actual == 0.0:
        raise ValueError("Total actual demand is zero; WAPE is undefined.")
    return float((np.sum(np.abs(y_t - y_p)) / total_actual) * 100.0)


def symmetric_mean_absolute_percentage_error(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    r"""Compute Symmetric Mean Absolute Percentage Error (SMAPE).

    $$\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^n \frac{2 \cdot |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$

    Args:
        y_true: Ground truth demand.
        y_pred: Forecasted demand.

    Returns:
        Float SMAPE bounded percentage between 0% and 200%.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_t) + np.abs(y_p)) / 2.0
    # Avoid zero division when both actual and predicted are exactly 0
    valid_mask = denominator > 0
    if not np.any(valid_mask):
        return 0.0

    smape = np.mean(np.abs(y_t[valid_mask] - y_p[valid_mask]) / denominator[valid_mask]) * 100.0
    return float(smape)


def compute_all_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Calculate all standard demand forecasting metrics at once.

    Args:
        y_true: Ground truth actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary containing mae, rmse, mape, wape, and smape metrics.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    metrics = {
        "mae": mean_absolute_error(y_t, y_p),
        "rmse": root_mean_squared_error(y_t, y_p),
        "wape": weighted_absolute_percentage_error(y_t, y_p) if np.sum(y_t) > 0 else 0.0,
        "smape": symmetric_mean_absolute_percentage_error(y_t, y_p),
    }

    # Only include strict MAPE if there are no zero values
    try:
        metrics["mape"] = mean_absolute_percentage_error(y_t, y_p)
    except ValueError:
        metrics["mape"] = float("nan")

    return metrics

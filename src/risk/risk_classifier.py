"""Supply risk classifier.

Classifies suppliers or supply-chain nodes into risk categories using
tabular features derived from historical performance and external signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logger import get_logger

logger = get_logger(__name__)


class SupplyRiskClassifier:
    """Gradient-boosted classifier for multi-level supply risk prediction."""

    def __init__(self, params: dict | None = None) -> None:
        """Initialize the supply risk classifier.

        Args:
            params: Model hyperparameters. If ``None``, sensible defaults are
                used.
        """
        self.params: dict = params or {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
        }
        self._model = None
        logger.info("SupplyRiskClassifier initialised with params=%s", self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupplyRiskClassifier":
        """Fit the risk classifier on labelled training data.

        Args:
            X: Feature matrix with supplier / node attributes.
            y: Risk-level labels.

        Returns:
            Self for method chaining.
        """
        raise NotImplementedError("Will be implemented in Module 2")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk class labels.

        Args:
            X: Feature matrix for prediction.

        Returns:
            Array of predicted risk class labels.
        """
        raise NotImplementedError("Will be implemented in Module 2")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk class probabilities.

        Args:
            X: Feature matrix for prediction.

        Returns:
            Array of shape ``(n_samples, n_classes)`` with class probabilities.
        """
        raise NotImplementedError("Will be implemented in Module 2")

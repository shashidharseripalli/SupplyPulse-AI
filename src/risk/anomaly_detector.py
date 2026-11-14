"""Supply-chain anomaly detector.

Uses unsupervised methods to flag anomalous patterns in supplier behaviour,
lead times, and other operational metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logger import get_logger

logger = get_logger(__name__)


class SupplyAnomalyDetector:
    """Isolation-Forest-style anomaly detector for supply-chain data."""

    def __init__(self, contamination: float = 0.05) -> None:
        """Initialize the anomaly detector.

        Args:
            contamination: Expected proportion of anomalies in the data,
                used to set the decision threshold (default: 0.05).
        """
        self.contamination = contamination
        self._model = None
        logger.info(
            "SupplyAnomalyDetector initialised with contamination=%.3f",
            self.contamination,
        )

    def fit(self, X: pd.DataFrame) -> "SupplyAnomalyDetector":
        """Fit the anomaly detector on historical data.

        Args:
            X: Feature matrix of normal operational data.

        Returns:
            Self for method chaining.
        """
        raise NotImplementedError("Will be implemented in Module 2")

    def detect(self, X: pd.DataFrame) -> np.ndarray:
        """Detect anomalies in new data.

        Args:
            X: Feature matrix to evaluate.

        Returns:
            Boolean array where ``True`` indicates an anomaly.
        """
        raise NotImplementedError("Will be implemented in Module 2")

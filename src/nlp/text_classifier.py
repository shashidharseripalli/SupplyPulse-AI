"""Supply-chain disruption text classifier.

Uses a transformer-based model to classify news and text snippets as
indicating a supply-chain disruption or not.
"""

from __future__ import annotations

from src.common.logger import get_logger

logger = get_logger(__name__)


class DisruptionClassifier:
    """Transformer-based binary classifier for supply-chain disruption text."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        """Initialize the disruption classifier.

        Args:
            model_name: Hugging Face model identifier to use as the backbone.
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        logger.info(
            "DisruptionClassifier initialised with model_name=%s", self.model_name
        )

    def train(self, texts: list[str], labels: list[int]) -> None:
        """Fine-tune the classifier on labelled disruption text.

        Args:
            texts: List of raw text documents.
            labels: Binary labels (1 = disruption, 0 = non-disruption).
        """
        raise NotImplementedError("Will be implemented in Module 3")

    def predict(self, texts: list[str]) -> list[int]:
        """Predict disruption labels for a batch of texts.

        Args:
            texts: List of raw text documents.

        Returns:
            List of predicted binary labels.
        """
        raise NotImplementedError("Will be implemented in Module 3")

    def predict_proba(self, texts: list[str]) -> list[float]:
        """Return disruption probability scores for a batch of texts.

        Args:
            texts: List of raw text documents.

        Returns:
            List of disruption probability scores in [0, 1].
        """
        raise NotImplementedError("Will be implemented in Module 3")

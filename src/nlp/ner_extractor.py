"""Named Entity Recognition extractor for supply-chain disruptions.

Extracts entities such as locations, organisations, and disruption types
from unstructured text using a spaCy NER pipeline.
"""

from __future__ import annotations

from src.common.logger import get_logger

logger = get_logger(__name__)


class DisruptionNERExtractor:
    """spaCy-based NER extractor for supply-chain disruption entities."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Initialize the NER extractor.

        Args:
            model_name: spaCy model identifier to load.
        """
        self.model_name = model_name
        self._nlp = None
        logger.info(
            "DisruptionNERExtractor initialised with model_name=%s", self.model_name
        )

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        """Extract named entities from a text document.

        Args:
            text: Raw text to process.

        Returns:
            List of dicts, each with keys ``"text"``, ``"label"``, ``"start"``,
            and ``"end"`` describing a recognised entity span.
        """
        raise NotImplementedError("Will be implemented in Module 3")

"""Tests for configuration loader."""
import os
import pytest
from src.common.config_loader import load_config


class TestConfigLoader:
    """Tests for load_config function."""

    def test_load_config_returns_dict(self) -> None:
        """Config loader should return a dictionary."""
        config = load_config("config/settings.yaml")
        assert isinstance(config, dict)

    def test_load_config_missing_file_raises(self) -> None:
        """Config loader should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent/config.yaml")

    def test_config_has_required_sections(self) -> None:
        """Config should contain all required top-level sections."""
        config = load_config("config/settings.yaml")
        required_sections = [
            "project", "database", "forecasting",
            "nlp", "risk", "optimization", "simulation",
        ]
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"

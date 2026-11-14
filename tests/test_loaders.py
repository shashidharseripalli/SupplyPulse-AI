"""Tests for data loaders."""
import pytest
import pandas as pd
from pathlib import Path
from src.ingestion.loaders import load_csv


class TestCSVLoader:
    """Tests for CSV loading functionality."""

    def test_load_csv_returns_dataframe(self, tmp_csv: Path) -> None:
        """load_csv should return a pandas DataFrame."""
        df = load_csv(str(tmp_csv))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (20, 2)

    def test_load_csv_missing_file_raises(self) -> None:
        """load_csv should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent/data.csv")

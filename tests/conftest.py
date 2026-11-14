"""Shared test fixtures for SupplyPulse AI."""
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_time_series() -> pd.DataFrame:
    """Create a small sample time series DataFrame.
    
    Returns:
        DataFrame with 'date' and 'value' columns (20 rows).
    """
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    values = list(range(20))
    return pd.DataFrame({"date": dates, "value": values})


@pytest.fixture
def sample_config() -> dict:
    """Create a minimal config dict matching settings.yaml structure.
    
    Returns:
        Dictionary with key configuration sections.
    """
    return {
        "project": {"name": "SupplyPulse AI", "version": "0.1.0"},
        "database": {"url": "sqlite:///test.db", "pool_size": 5},
        "forecasting": {
            "horizon": 30,
            "seasonal_period": 7,
            "test_split_ratio": 0.2,
        },
        "nlp": {"model_name": "distilbert-base-uncased"},
        "risk": {"contamination": 0.05, "threshold": 0.5},
        "optimization": {"time_limit_seconds": 30},
        "simulation": {"num_scenarios": 100, "random_seed": 42},
    }


@pytest.fixture
def tmp_csv(tmp_path: Path, sample_time_series: pd.DataFrame) -> Path:
    """Write sample time series to a temporary CSV file.
    
    Args:
        tmp_path: Pytest temporary directory.
        sample_time_series: Sample DataFrame fixture.
        
    Returns:
        Path to the temporary CSV file.
    """
    filepath = tmp_path / "test_data.csv"
    sample_time_series.to_csv(filepath, index=False)
    return filepath

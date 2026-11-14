"""Data loaders for CSV, Parquet, and database sources.

Each loader validates input paths, wraps low-level I/O errors in
:class:`~src.common.exceptions.DataError`, and logs key diagnostics such as
the resulting DataFrame shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from src.common.db import get_engine
from src.common.exceptions import DataError
from src.common.logger import get_logger

logger = get_logger(__name__)


def load_csv(filepath: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a :class:`pandas.DataFrame`.

    Args:
        filepath: Path to the CSV file.
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_csv`.

    Returns:
        A DataFrame containing the parsed CSV data.

    Raises:
        DataError: If *filepath* does not exist or the file cannot be read.
    """
    path = Path(filepath)
    if not path.is_file():
        raise DataError(
            f"CSV file not found: '{path}'. "
            "Verify the file path and ensure the file exists."
        )

    logger.info("Loading CSV from '%s'.", path)
    try:
        df: pd.DataFrame = pd.read_csv(path, **kwargs)
    except Exception as exc:
        raise DataError(
            f"Failed to read CSV file '{path}': {exc}"
        ) from exc

    logger.info("Loaded CSV with shape %s from '%s'.", df.shape, path)
    return df


def load_parquet(filepath: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a Parquet file into a :class:`pandas.DataFrame`.

    Args:
        filepath: Path to the Parquet file.
        **kwargs: Additional keyword arguments forwarded to
            :func:`pandas.read_parquet`.

    Returns:
        A DataFrame containing the parsed Parquet data.

    Raises:
        DataError: If *filepath* does not exist or the file cannot be read.
    """
    path = Path(filepath)
    if not path.is_file():
        raise DataError(
            f"Parquet file not found: '{path}'. "
            "Verify the file path and ensure the file exists."
        )

    logger.info("Loading Parquet from '%s'.", path)
    try:
        df: pd.DataFrame = pd.read_parquet(path, **kwargs)
    except Exception as exc:
        raise DataError(
            f"Failed to read Parquet file '{path}': {exc}"
        ) from exc

    logger.info("Loaded Parquet with shape %s from '%s'.", df.shape, path)
    return df


def load_from_db(
    query: str,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Execute a SQL query and return the result as a :class:`pandas.DataFrame`.

    Args:
        query: SQL query string to execute.
        engine: Optional SQLAlchemy :class:`Engine`.  When ``None`` the
            default project engine from :func:`~src.common.db.get_engine` is
            used.

    Returns:
        A DataFrame containing the query results.

    Raises:
        DataError: If the query execution fails.
    """
    if engine is None:
        engine = get_engine()

    logger.info("Executing database query: %.120s…", query.strip())
    try:
        df: pd.DataFrame = pd.read_sql(query, con=engine)
    except Exception as exc:
        raise DataError(
            f"Database query failed: {exc}"
        ) from exc

    logger.info("Query returned DataFrame with shape %s.", df.shape)
    return df

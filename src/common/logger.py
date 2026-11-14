"""Centralized logging configuration for SupplyPulse AI.

Call :func:`get_logger` from any module to obtain a properly-configured
:class:`logging.Logger`.  On the very first invocation the module attempts to
load ``config/logging_config.yaml``; if the file is absent it falls back to
:func:`logging.basicConfig` with a sensible default format.
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Final

import yaml

_initialized: bool = False

_DEFAULT_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DEFAULT_LEVEL: Final[int] = logging.INFO
_LOGGING_CONFIG_PATH: Final[str] = "config/logging_config.yaml"
_LOGS_DIR: Final[str] = "logs"


def _initialize_logging() -> None:
    """Bootstrap the logging subsystem (called once).

    Reads ``config/logging_config.yaml`` when available.  If the file does not
    exist the function configures a basic StreamHandler so that downstream
    modules still get reasonable output.
    """
    global _initialized  # noqa: PLW0603

    os.makedirs(_LOGS_DIR, exist_ok=True)

    config_path = Path(_LOGGING_CONFIG_PATH)
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as fh:
            config_dict: dict = yaml.safe_load(fh)
        logging.config.dictConfig(config_dict)
    else:
        logging.basicConfig(level=_DEFAULT_LEVEL, format=_DEFAULT_FORMAT)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initializing the logging subsystem if needed.

    Args:
        name: Dot-separated logger name, typically ``__name__`` of the
            calling module.

    Returns:
        A :class:`logging.Logger` configured according to the project's
        logging settings.
    """
    if not _initialized:
        _initialize_logging()

    return logging.getLogger(name)

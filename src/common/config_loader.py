"""YAML configuration loader with environment-variable interpolation.

Usage::

    from src.common.config_loader import load_config

    cfg = load_config()                       # default path
    cfg = load_config("config/prod.yaml")     # explicit path

Environment variables are interpolated using the ``${VAR:default}`` syntax.
If *VAR* is set in the environment its value is substituted; otherwise the
literal *default* is used.  The pattern ``${VAR}`` (no default) resolves to
an empty string when *VAR* is unset.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from src.common.exceptions import ConfigError
from src.common.logger import get_logger

logger = get_logger(__name__)

_config_cache: dict[str, dict[str, Any]] = {}

_ENV_VAR_PATTERN: re.Pattern[str] = re.compile(
    r"\$\{(?P<var>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<default>[^}]*))?\}"
)


def _interpolate_env_vars(raw: str) -> str:
    """Replace ``${VAR:default}`` patterns with environment values.

    Args:
        raw: The raw string potentially containing interpolation tokens.

    Returns:
        The string with all ``${…}`` tokens resolved.
    """

    def _replacer(match: re.Match[str]) -> str:
        var_name: str = match.group("var")
        default: str | None = match.group("default")
        return os.environ.get(var_name, default if default is not None else "")

    return _ENV_VAR_PATTERN.sub(_replacer, raw)


def load_config(config_path: str = "config/settings.yaml") -> dict[str, Any]:
    """Load and cache a YAML configuration file.

    The file contents are read once per unique *config_path* and cached in a
    module-level dictionary for subsequent calls.

    Args:
        config_path: Filesystem path to the YAML configuration file.  Defaults
            to ``config/settings.yaml``.

    Returns:
        A dictionary representing the parsed YAML document with all
        environment-variable references resolved.

    Raises:
        FileNotFoundError: If *config_path* does not point to an existing file.
        ConfigError: If the YAML content cannot be parsed.
    """
    resolved_path: str = str(Path(config_path).resolve())

    if resolved_path in _config_cache:
        logger.debug("Returning cached config for '%s'.", resolved_path)
        return _config_cache[resolved_path]

    path = Path(resolved_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: '{resolved_path}'. "
            "Please ensure the file exists and the path is correct."
        )

    logger.info("Loading configuration from '%s'.", resolved_path)

    try:
        raw_text: str = path.read_text(encoding="utf-8")
        interpolated_text: str = _interpolate_env_vars(raw_text)
        config: dict[str, Any] = yaml.safe_load(interpolated_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Failed to parse YAML configuration at '{resolved_path}': {exc}"
        ) from exc

    _config_cache[resolved_path] = config
    logger.debug("Configuration cached for '%s'.", resolved_path)
    return config

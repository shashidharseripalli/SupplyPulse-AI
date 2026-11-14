"""Custom exception hierarchy for SupplyPulse AI.

Provides a structured set of domain-specific exceptions so that callers can
catch errors at the appropriate level of granularity.
"""

from __future__ import annotations


class SupplyPulseError(Exception):
    """Base exception for all SupplyPulse AI errors.

    Every custom exception in the project inherits from this class, making it
    easy to catch *any* application-level error with a single ``except`` clause
    when needed.
    """


class ConfigError(SupplyPulseError):
    """Raised when a configuration file is missing, malformed, or invalid.

    Examples include a missing YAML key, an unresolvable environment variable
    reference, or a settings file that cannot be parsed.
    """


class DataError(SupplyPulseError):
    """Raised when data loading or processing fails.

    Typical causes are missing data files, unexpected schemas, or corrupt
    records encountered during ingestion or transformation.
    """


class ModelError(SupplyPulseError):
    """Raised when model training or prediction encounters an error.

    This covers issues such as convergence failures, incompatible feature
    shapes, or attempts to predict with an unfitted model.
    """


class OptimizationError(SupplyPulseError):
    """Raised when an optimization solver fails to find a feasible solution.

    Common scenarios include infeasible constraints, solver time-outs, or
    numerical instability in the objective function.
    """

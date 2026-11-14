"""Monte-Carlo scenario simulation engine.

Generates and evaluates what-if scenarios for supply-chain disruptions,
demand shocks, and lead-time variability using Monte-Carlo sampling.
"""

from __future__ import annotations

from src.common.logger import get_logger

logger = get_logger(__name__)


class ScenarioEngine:
    """Monte-Carlo engine for supply-chain scenario simulation."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize the scenario engine.

        Args:
            random_seed: Seed for reproducible random sampling (default: 42).
        """
        self.random_seed = random_seed
        logger.info(
            "ScenarioEngine initialised with random_seed=%d", self.random_seed
        )

    def run_scenario(self, scenario_type: str, params: dict) -> dict:
        """Run a single what-if scenario simulation.

        Args:
            scenario_type: Kind of scenario to simulate. Supported values
                include ``"demand_spike"``, ``"supplier_failure"``, and
                ``"lead_time_increase"``.
            params: Scenario-specific parameters (e.g. magnitude,
                duration, affected SKUs).

        Returns:
            Dictionary with simulation results including keys such as
            ``"scenario_type"``, ``"impact_summary"``, ``"timeline"``,
            and ``"recommendations"``.
        """
        raise NotImplementedError("Will be implemented in Module 5")

    def run_batch(self, scenarios: list[dict]) -> list[dict]:
        """Run multiple scenarios in batch.

        Each entry in *scenarios* must contain ``"scenario_type"`` and
        ``"params"`` keys.

        Args:
            scenarios: List of scenario specification dicts.

        Returns:
            List of result dicts, one per scenario (same order).
        """
        raise NotImplementedError("Will be implemented in Module 5")

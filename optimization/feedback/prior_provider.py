"""
prior_provider.py
=================

Provides feedback confidence priors derived from OutcomeStore query history.
"""

from __future__ import annotations

from optimization.config import (
    OPTIMIZATION_FEEDBACK_ENABLED,
    OPTIMIZATION_FEEDBACK_MIN_SAMPLE_SIZE,
)
from optimization.feedback.interfaces import IPriorProvider
from optimization.feedback.outcome_store import OutcomeStore


class PriorProvider(IPriorProvider):
    """Provides empirical confidence adjustments based on recorded thumbnail outcomes."""

    def __init__(
        self,
        store: OutcomeStore | None = None,
        enabled: bool = OPTIMIZATION_FEEDBACK_ENABLED,
        min_samples: int = OPTIMIZATION_FEEDBACK_MIN_SAMPLE_SIZE,
    ) -> None:
        self.store = store if store is not None else OutcomeStore()
        self.enabled = enabled
        self.min_samples = min_samples

    def rule_confidence_prior(self, rule_id: str) -> float:
        """Return confidence delta for Module 9 decision rule."""
        if not self.enabled:
            return 0.0
        mean_delta, count = self.store.mean_delta_by_decision_rule(rule_id)
        if count < self.min_samples:
            return 0.0
        # Bound confidence nudge between -0.2 and +0.2
        return float(max(-0.2, min(0.2, mean_delta)))

    def hook_type_prior(self, hook_type: str) -> float:
        """Return confidence delta for headline hook type."""
        if not self.enabled:
            return 0.0
        mean_delta, count = self.store.mean_delta_by_hook_type(hook_type)
        if count < self.min_samples:
            return 0.0
        return float(max(-0.2, min(0.2, mean_delta)))

    def candidate_strategy_prior(self, strategy_name: str) -> float:
        """Return confidence delta for candidate strategy."""
        if not self.enabled:
            return 0.0
        mean_delta, count = self.store.mean_delta_by_candidate_strategy(strategy_name)
        if count < self.min_samples:
            return 0.0
        return float(max(-0.2, min(0.2, mean_delta)))

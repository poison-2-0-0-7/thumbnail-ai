"""
outcome_store.py
================

Read and query API over sharded OptimizationOutcome history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
from loguru import logger

from optimization.config import OPTIMIZATION_OUTCOMES_DIR
from optimization.feedback.outcome_recorder import OptimizationOutcome


class OutcomeStore:
    """Read and aggregate query interface over stored optimization outcomes."""

    def __init__(self, storage_dir: Path = OPTIMIZATION_OUTCOMES_DIR) -> None:
        self.storage_dir = storage_dir

    def load_all(self) -> list[OptimizationOutcome]:
        """Load all persisted outcomes from disk."""
        outcomes: list[OptimizationOutcome] = []
        if not self.storage_dir.exists():
            return outcomes

        for outcome_file in self.storage_dir.glob("*/outcome.json"):
            try:
                data = outcome_file.read_text(encoding="utf-8")
                outcome = OptimizationOutcome.model_validate_json(data)
                outcomes.append(outcome)
            except Exception as exc:
                logger.warning("Failed to load outcome file {file}: {exc}", file=outcome_file, exc=exc)
        return outcomes

    def query_by_niche(self, niche: str) -> list[OptimizationOutcome]:
        """Return all outcomes matching niche."""
        return [o for o in self.load_all() if o.niche == niche]

    def mean_delta_by_hook_type(self, hook_type: str, niche: str | None = None) -> tuple[float, int]:
        """Return (mean_delta, count) for a specific hook type."""
        all_outcomes = self.query_by_niche(niche) if niche else self.load_all()
        matching = [o.delta for o in all_outcomes if o.hook_type_used == hook_type]
        if not matching:
            return (0.0, 0)
        return (float(sum(matching) / len(matching)), len(matching))

    def mean_delta_by_decision_rule(self, rule_id: str, niche: str | None = None) -> tuple[float, int]:
        """Return (mean_delta, count) for a Module 9 decision rule."""
        all_outcomes = self.query_by_niche(niche) if niche else self.load_all()
        matching = [o.delta for o in all_outcomes if rule_id in o.decisions_applied]
        if not matching:
            return (0.0, 0)
        return (float(sum(matching) / len(matching)), len(matching))

    def mean_delta_by_candidate_strategy(self, strategy_name: str, niche: str | None = None) -> tuple[float, int]:
        """Return (mean_delta, count) for a candidate strategy."""
        all_outcomes = self.query_by_niche(niche) if niche else self.load_all()
        matching = [o.delta for o in all_outcomes if o.candidate_strategy_name == strategy_name]
        if not matching:
            return (0.0, 0)
        return (float(sum(matching) / len(matching)), len(matching))

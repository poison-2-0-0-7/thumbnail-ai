"""
optimization/feedback/interfaces.py
====================================

Interfaces for feedback priors and outcome query providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class IPriorProvider(ABC):
    """Abstract interface providing decision and strategy confidence priors derived from outcomes."""

    @abstractmethod
    def rule_confidence_prior(self, rule_id: str) -> float:
        """Return prior confidence delta for a Module 9 decision rule."""
        pass

    @abstractmethod
    def hook_type_prior(self, hook_type: str) -> float:
        """Return prior confidence delta for a headline hook type."""
        pass

    @abstractmethod
    def candidate_strategy_prior(self, strategy_name: str) -> float:
        """Return prior confidence delta for a candidate strategy."""
        pass

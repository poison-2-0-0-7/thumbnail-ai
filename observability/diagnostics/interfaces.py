"""
observability/diagnostics/interfaces.py
========================================

Abstract Base Classes (ABCs) for Diagnostic Engine components and rules in PORCE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class IDiagnosticRule(ABC):
    """
    Interface for deterministic diagnostic rules evaluated by RuleEngine.
    """

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Unique identifier for the rule (e.g. 'RULE-DEC-04')."""
        pass

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Human-readable name of the rule."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Rule category (e.g. 'latent_initialization', 'decision_honoring')."""
        pass

    @abstractmethod
    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        """
        Evaluate rule against TraceFacts and optional RuleContext.
        Returns a Finding if rule criteria/anomaly is met, or None/PASS finding.
        """
        pass

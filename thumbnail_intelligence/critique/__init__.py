"""
Intelligent Critique Engine Package (Phase 5.4 — Intelligent Critique Engine).
Analyzes winning thumbnail EvaluationResults and produces structured, rule-based ImprovementPlans and CritiqueReports.
No LLMs, no image rendering, no regeneration. ONLY rule-based visual critique & deterministic improvement planning.
"""

from thumbnail_intelligence.critique.engine import (
    CritiqueEngineError,
    IntelligentCritiqueEngine,
)
from thumbnail_intelligence.critique.models import (
    CritiqueProfile,
    CritiqueReport,
    ImpactLevel,
    ImplementationCost,
    ImprovementPlan,
    ImprovementSuggestion,
    Issue,
    IssueSeverity,
)
from thumbnail_intelligence.critique.rules import CritiqueRule, DefaultCritiqueRuleSet

__all__ = [
    "IntelligentCritiqueEngine",
    "CritiqueEngineError",
    "IssueSeverity",
    "ImpactLevel",
    "ImplementationCost",
    "Issue",
    "ImprovementSuggestion",
    "ImprovementPlan",
    "CritiqueProfile",
    "CritiqueReport",
    "CritiqueRule",
    "DefaultCritiqueRuleSet",
]

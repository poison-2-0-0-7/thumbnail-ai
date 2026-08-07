"""Renderer V2 Phase 2 Planning Module.

Deterministic intelligence layer responsible for deciding WHAT should be modified
across detected instances, background, lighting, and composition.
"""

from .planner_types import (
    EditAction,
    TargetCategory,
    ObjectEditChange,
    ScoreBreakdown,
    CompositionAnalysis,
    CompositionDirectives,
    EditPlanOutput,
)
from .saliency import SaliencyEngine
from .composition import CompositionEngine
from .scoring import ScoringEngine
from .planner_rules import PlannerRuleEngine
from .planner import EditPlanner

__all__ = [
    "EditAction",
    "TargetCategory",
    "ObjectEditChange",
    "ScoreBreakdown",
    "CompositionAnalysis",
    "CompositionDirectives",
    "EditPlanOutput",
    "SaliencyEngine",
    "CompositionEngine",
    "ScoringEngine",
    "PlannerRuleEngine",
    "EditPlanner",
]

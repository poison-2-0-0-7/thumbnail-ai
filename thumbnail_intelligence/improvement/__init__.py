"""
Automatic Improvement Engine Package (Phase 5.5 — Automatic Improvement Engine).
Converts an ImprovementPlan into a revised RenderExecutionPackage using targeted layer modifications.
Preserves unchanged layers without blindly re-rendering the full thumbnail from scratch.
Does NOT execute rendering directly; emits UpdatedRenderExecutionPackage.
"""

from thumbnail_intelligence.improvement.engine import (
    AutomaticImprovementEngine,
    ImprovementEngineError,
)
from thumbnail_intelligence.improvement.models import (
    ImprovementExecutionPlan,
    ImprovementStrategyType,
    LayerModification,
    ModificationReport,
    UpdatedRenderExecutionPackage,
)
from thumbnail_intelligence.improvement.planner import ModificationPlanner, PlannerValidationError

__all__ = [
    "AutomaticImprovementEngine",
    "ImprovementEngineError",
    "ImprovementStrategyType",
    "LayerModification",
    "ImprovementExecutionPlan",
    "ModificationReport",
    "UpdatedRenderExecutionPackage",
    "ModificationPlanner",
    "PlannerValidationError",
]

"""
Evaluation Engine Package (Phase 5.2 — Thumbnail Evaluation Engine).
Defines deterministic, explainable quality scoring for candidate thumbnails across 22 metrics.
"""

from thumbnail_intelligence.evaluation.engine import (
    EvaluationEngineError,
    ThumbnailEvaluationEngine,
)
from thumbnail_intelligence.evaluation.models import (
    EvaluationMetric,
    EvaluationProfile,
    EvaluationReport,
    EvaluationResult,
    EvaluationSet,
    MetricBreakdown,
)

__all__ = [
    "ThumbnailEvaluationEngine",
    "EvaluationEngineError",
    "EvaluationMetric",
    "MetricBreakdown",
    "EvaluationProfile",
    "EvaluationResult",
    "EvaluationReport",
    "EvaluationSet",
]

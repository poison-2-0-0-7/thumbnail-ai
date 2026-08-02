"""
optimization package — Thumbnail Quality Optimization Layer.
"""

from .config import (
    OPTIMIZATION_ACCEPTANCE_REPORT_ONLY,
    OPTIMIZATION_DEEP_SCORE,
    OPTIMIZATION_FEEDBACK_ENABLED,
    OPTIMIZATION_LOOP_ENABLED,
    OPTIMIZATION_MAX_IDENTITY_DRIFT,
    OPTIMIZATION_MAX_RETRIES,
    OPTIMIZATION_MIN_STRUCTURAL_SIMILARITY,
    OPTIMIZATION_MIN_WIN_MARGIN,
    OPTIMIZATION_OUTCOMES_DIR,
)
from .exceptions import (
    AcceptanceError,
    BaselineScoringError,
    ComparativeScoringError,
    FeedbackError,
    OptimizationBaseError,
    OrchestrationError,
)

__all__ = [
    "OPTIMIZATION_ACCEPTANCE_REPORT_ONLY",
    "OPTIMIZATION_DEEP_SCORE",
    "OPTIMIZATION_FEEDBACK_ENABLED",
    "OPTIMIZATION_LOOP_ENABLED",
    "OPTIMIZATION_MAX_IDENTITY_DRIFT",
    "OPTIMIZATION_MAX_RETRIES",
    "OPTIMIZATION_MIN_STRUCTURAL_SIMILARITY",
    "OPTIMIZATION_MIN_WIN_MARGIN",
    "OPTIMIZATION_OUTCOMES_DIR",
    "AcceptanceError",
    "BaselineScoringError",
    "ComparativeScoringError",
    "FeedbackError",
    "OptimizationBaseError",
    "OrchestrationError",
]

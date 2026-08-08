"""
Iterative Optimization Engine Package (Phase 5.6 — Iterative Optimization Engine).
Orchestrates closed-loop iterative thumbnail quality optimization across the full pipeline until convergence.

Pipeline Loop:
  Generate ➔ Evaluate ➔ Rank ➔ Critique ➔ Improve ➔ Render ➔ Evaluate Again ➔ Repeat
"""

from thumbnail_intelligence.optimization.convergence import ConvergenceDetector
from thumbnail_intelligence.optimization.engine import (
    IterativeOptimizationEngine,
    OptimizationEngineError,
)
from thumbnail_intelligence.optimization.models import (
    IterationResult,
    OptimizationHistory,
    OptimizationReport,
    OptimizationSession,
    StoppingPolicy,
    StoppingReason,
)

__all__ = [
    "IterativeOptimizationEngine",
    "OptimizationEngineError",
    "ConvergenceDetector",
    "StoppingReason",
    "StoppingPolicy",
    "IterationResult",
    "OptimizationHistory",
    "OptimizationReport",
    "OptimizationSession",
]

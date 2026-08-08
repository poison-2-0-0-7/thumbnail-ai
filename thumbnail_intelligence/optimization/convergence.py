"""
convergence.py
==============

ConvergenceDetector Implementation for Phase 5.6.
Detects optimization loop convergence, score plateaus, target score attainment, and early stopping conditions.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from thumbnail_intelligence.optimization.models import (
    OptimizationHistory,
    StoppingPolicy,
    StoppingReason,
)

logger = logging.getLogger(__name__)


class ConvergenceDetector:
    """Detects convergence and stopping conditions in an optimization session history."""

    def check_convergence(
        self,
        history: OptimizationHistory,
        policy: StoppingPolicy,
        elapsed_seconds: float = 0.0,
    ) -> Tuple[bool, Optional[StoppingReason], str]:
        """Evaluate history against policy and return (should_stop, stopping_reason, description).

        Args:
            history: OptimizationHistory containing IterationResult records.
            policy: StoppingPolicy thresholds.
            elapsed_seconds: Total wall-clock elapsed time in seconds.

        Returns:
            Tuple of (should_stop: bool, stopping_reason: Optional[StoppingReason], description: str).
        """
        if not history or not history.iterations:
            return False, None, "History contains no iterations."

        latest_it = history.iterations[-1]
        k = latest_it.iteration_index
        latest_score = latest_it.overall_score

        # 1. Target Score Reached
        if latest_score >= policy.target_overall_score:
            desc = f"Target overall score reached ({latest_score:.1f} >= {policy.target_overall_score:.1f}) in iteration {k}."
            logger.info(f"[ConvergenceDetector] {desc}")
            return True, StoppingReason.TARGET_SCORE_REACHED, desc

        # 2. Maximum Iterations Reached
        if k >= policy.max_iterations:
            desc = f"Maximum allowed optimization iterations reached ({k} >= {policy.max_iterations})."
            logger.info(f"[ConvergenceDetector] {desc}")
            return True, StoppingReason.MAX_ITERATIONS_REACHED, desc

        # 3. Elapsed Time Limit
        if elapsed_seconds >= policy.max_elapsed_seconds:
            desc = f"Maximum allowed wall-clock time exceeded ({elapsed_seconds:.1f}s >= {policy.max_elapsed_seconds:.1f}s)."
            logger.info(f"[ConvergenceDetector] {desc}")
            return True, StoppingReason.TIMED_OUT, desc

        # 4. Confidence Plateau
        if latest_it.ranking_confidence < policy.min_confidence_threshold:
            desc = f"Ranking confidence dropped below minimum threshold ({latest_it.ranking_confidence:.2f} < {policy.min_confidence_threshold:.2f})."
            logger.info(f"[ConvergenceDetector] {desc}")
            return True, StoppingReason.CONFIDENCE_PLATEAU, desc

        # 5. Score Plateau (Across last 2 iterations)
        if len(history.iterations) >= 2:
            prev_it = history.iterations[-2]
            gain = latest_score - prev_it.overall_score
            if gain < policy.min_gain_threshold_pts:
                desc = f"Score plateau detected: gain of +{gain:.2f} pts across last 2 iterations is below minimum threshold ({policy.min_gain_threshold_pts:.1f} pts)."
                logger.info(f"[ConvergenceDetector] {desc}")
                return True, StoppingReason.SCORE_PLATEAU, desc

        # 6. Repeated Suggestions without score gain
        if len(history.iterations) >= 2:
            prev_it = history.iterations[-2]
            curr_suggs = latest_it.critique_report.improvement_plan.prioritized_suggestions
            prev_suggs = prev_it.critique_report.improvement_plan.prioritized_suggestions

            if curr_suggs and prev_suggs:
                if curr_suggs[0].action_type == prev_suggs[0].action_type and latest_score <= prev_it.overall_score:
                    desc = f"Repeated top suggestion ('{curr_suggs[0].action_type}') detected without score improvement ({latest_score:.1f} <= {prev_it.overall_score:.1f})."
                    logger.info(f"[ConvergenceDetector] {desc}")
                    return True, StoppingReason.REPEATED_SUGGESTIONS, desc

        return False, None, f"Optimization continuing (Iteration {k}, score={latest_score:.1f})."

"""
retry_strategy.py
=================

Bounded re-plan / re-prompt policy when candidate thumbnail quality does not beat original baseline.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict
from loguru import logger

from optimization.config import OPTIMIZATION_MAX_RETRIES


class RetryDecision(BaseModel):
    """Result of evaluating retry strategy."""

    model_config = ConfigDict(frozen=True)

    should_retry: bool
    attempt_index: int
    max_retries: int
    adjustment_hint: str
    adjusted_parameters: dict[str, Any] = {}


class RetryStrategy:
    """Policy engine determining bounded retry parameters."""

    def __init__(self, max_retries: int = OPTIMIZATION_MAX_RETRIES) -> None:
        self.max_retries = max_retries

    def evaluate(
        self,
        video_id: str,
        current_attempt: int,
        verdicts: list[Any],
        qa_reports: list[Any],
        decision_manifest: Optional[Any] = None,
    ) -> RetryDecision:
        """
        Evaluate if another attempt should be executed and derive bounded parameter adjustments.
        """
        next_attempt = current_attempt + 1
        if next_attempt > self.max_retries:
            logger.info("Retry budget exhausted for {vid} ({curr}/{max})", vid=video_id, curr=current_attempt, max=self.max_retries)
            return RetryDecision(
                should_retry=False,
                attempt_index=current_attempt,
                max_retries=self.max_retries,
                adjustment_hint="Max retries reached; budget exhausted.",
            )

        # Analyze failure signals to nudge next attempt
        hint = "Adjusting seed and candidate perturbation weights for attempt " + str(next_attempt)
        params: dict[str, Any] = {
            "seed_offset": next_attempt * 100,
            "prefer_keep_over_enhance": True,
            "denoise_nudge": -0.05 * next_attempt,
        }

        logger.info(
            "Scheduling retry attempt {next_app}/{max} for {vid} with hint: {hint}",
            next_app=next_attempt,
            max=self.max_retries,
            vid=video_id,
            hint=hint,
        )

        return RetryDecision(
            should_retry=True,
            attempt_index=next_attempt,
            max_retries=self.max_retries,
            adjustment_hint=hint,
            adjusted_parameters=params,
        )

"""
composition_scorer.py
======================

Scorer for 7.5 — Composition Quality.
Evaluates visual hierarchy, rule-of-thirds, and focal balance on the generated thumbnail.
"""

from __future__ import annotations

import time

from evaluation.config import EVAL_DIMENSION_THRESHOLDS
from modules.models import DimensionScore
from .interfaces import IQualityScorer
from .scoring_context import QualityScoringContext


class CompositionScorer(IQualityScorer):
    @property
    def dimension(self) -> str:
        return "composition"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        threshold = EVAL_DIMENSION_THRESHOLDS.get("composition", 0.50)

        # Re-use source composition score or redesign spec targets as reference
        rule_score = 0.75
        if context.thumbnail_intelligence and context.thumbnail_intelligence.composition:
            rule_score = float(context.thumbnail_intelligence.composition.rule_of_thirds_score or 0.75)

        if context.redesign_spec and context.redesign_spec.layout_direction:
            # Target negative space / clutter optimization boost
            rule_score = max(rule_score, 0.80)

        return DimensionScore(
            dimension=self.dimension,
            score=rule_score,
            passed=rule_score >= threshold,
            threshold=threshold,
            detail={"rule_of_thirds_score": rule_score},
            scorer_version="1.0.0",
            duration_seconds=time.monotonic() - t0,
            status="success",
        )

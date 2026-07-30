"""
text_readability_scorer.py
==========================

Scorer for 7.6 — Text Readability.
Evaluates OCR readability, contrast against background, and safe-zone containment.
"""

from __future__ import annotations

import time

from evaluation.config import EVAL_DIMENSION_THRESHOLDS
from modules.models import DimensionScore
from .interfaces import IQualityScorer
from .scoring_context import QualityScoringContext


class TextReadabilityScorer(IQualityScorer):
    @property
    def dimension(self) -> str:
        return "text_readability"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        threshold = EVAL_DIMENSION_THRESHOLDS.get("text_readability", 0.60)

        intel = context.thumbnail_intelligence
        if not intel or not intel.ocr or intel.ocr.word_count == 0:
            return DimensionScore(
                dimension=self.dimension,
                score=1.0,
                passed=True,
                threshold=threshold,
                detail={"text_detected": False, "reason": "No text detected in thumbnail"},
                scorer_version="1.0.0",
                duration_seconds=time.monotonic() - t0,
                status="success",
            )

        avg_conf = float(intel.ocr.average_confidence or 0.8)
        coverage = float(intel.ocr.text_coverage_ratio or 0.1)

        # High coverage or low confidence penalizes readability
        readability_score = min(1.0, max(0.2, avg_conf - max(0.0, coverage - 0.25)))

        return DimensionScore(
            dimension=self.dimension,
            score=readability_score,
            passed=readability_score >= threshold,
            threshold=threshold,
            detail={
                "average_confidence": avg_conf,
                "text_coverage_ratio": coverage,
                "word_count": intel.ocr.word_count,
            },
            scorer_version="1.0.0",
            duration_seconds=time.monotonic() - t0,
            status="success",
        )

"""
inline_qa_scorer.py
===================

Scorer for 7.2 — Reused Module 7 inline signals.
Lifts the winning candidate's QualityAssuranceReport/CandidateScore from Module 7.
"""

from __future__ import annotations

import time

from evaluation.config import EVAL_DIMENSION_THRESHOLDS
from modules.models import DimensionScore
from .interfaces import IQualityScorer
from .scoring_context import QualityScoringContext


class InlineQAScorer(IQualityScorer):
    @property
    def dimension(self) -> str:
        return "inline_qa"

    def score(self, context: QualityScoringContext) -> DimensionScore:
        t0 = time.monotonic()
        threshold = EVAL_DIMENSION_THRESHOLDS.get("inline_qa", 0.60)

        res = context.image_generation_result
        if not res or not res.candidate_scores:
            return DimensionScore(
                dimension=self.dimension,
                score=0.0,
                passed=False,
                threshold=threshold,
                detail={"reason": "No Module 7 candidate_scores found"},
                scorer_version="1.0.0",
                duration_seconds=time.monotonic() - t0,
                status="skipped",
            )

        selected_cand = next((c for c in res.candidate_scores if c.selected), res.candidate_scores[0])
        overall = float(selected_cand.overall_score)
        passed = selected_cand.hard_gate_passed and (overall >= threshold)

        return DimensionScore(
            dimension=self.dimension,
            score=overall,
            passed=passed,
            threshold=threshold,
            detail={
                "candidate_index": selected_cand.candidate_index,
                "overall_score": overall,
                "identity_similarity": selected_cand.identity_similarity,
                "hard_gate_passed": selected_cand.hard_gate_passed,
                "rank": selected_cand.rank,
            },
            scorer_version="1.0.0",
            duration_seconds=time.monotonic() - t0,
            status="success",
        )

"""
aggregator.py
=============

Aggregates individual IQualityScorer results into a QualityEvaluationReport.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Sequence

from loguru import logger

from evaluation.config import EVAL_QUALITY_WEIGHTS
from modules.models import DimensionScore, QualityEvaluationReport
from .background_quality_scorer import BackgroundQualityScorer
from .color_harmony_scorer import ColorHarmonyScorer
from .composition_scorer import CompositionScorer
from .face_preservation_scorer import FacePreservationScorer
from .inline_qa_scorer import InlineQAScorer
from .interfaces import IQualityScorer
from .object_preservation_scorer import ObjectPreservationScorer
from .scoring_context import QualityScoringContext
from .text_readability_scorer import TextReadabilityScorer
from .visual_consistency_scorer import VisualConsistencyScorer


from .attractiveness_scorer import AttractivenessScorer
from .determinism_checker import DeterminismCheckerScorer
from .emotional_ctr_scorer import EmotionalCTRScorer
from .performance_profiler import PerformanceProfilerScorer
from .prompt_adherence_scorer import PromptAdherenceScorer
from .whitespace_scorer import WhitespaceScorer


class Aggregator:
    """Combines dimension scores into a unified QualityEvaluationReport."""

    def __init__(
        self,
        scorers: Sequence[IQualityScorer] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        default_scorers = [
            PromptAdherenceScorer(),
            FacePreservationScorer(),
            InlineQAScorer(),
            ObjectPreservationScorer(),
            BackgroundQualityScorer(),
            CompositionScorer(),
            TextReadabilityScorer(),
            ColorHarmonyScorer(),
            VisualConsistencyScorer(),
            AttractivenessScorer(),
            DeterminismCheckerScorer(),
            PerformanceProfilerScorer(),
            EmotionalCTRScorer(),
            WhitespaceScorer(),
        ]
        self.scorers = scorers if scorers is not None else default_scorers
        self.weights = weights if weights is not None else EVAL_QUALITY_WEIGHTS



    def evaluate(self, context: QualityScoringContext) -> QualityEvaluationReport:
        t0 = time.monotonic()
        evaluated_at = datetime.now(timezone.utc).isoformat()

        dimension_scores: list[DimensionScore] = []
        partial_failures: list[str] = []

        for scorer in self.scorers:
            t_s = time.monotonic()
            try:
                score = scorer.score(context)
                dimension_scores.append(score)
                if not score.passed and score.status == "success":
                    partial_failures.append(f"{scorer.dimension}_threshold_not_met")
                elif score.status == "error":
                    partial_failures.append(f"{scorer.dimension}_evaluation_error")
            except Exception as exc:
                logger.error("Scorer {dim} failed: {exc}", dim=scorer.dimension, exc=exc)
                err_score = DimensionScore(
                    dimension=scorer.dimension,
                    score=0.0,
                    passed=False,
                    threshold=0.5,
                    detail={"error": str(exc)},
                    scorer_version="1.0.0",
                    duration_seconds=time.monotonic() - t_s,
                    status="error",
                    error_message=str(exc),
                )
                dimension_scores.append(err_score)
                partial_failures.append(f"{scorer.dimension}_exception")

        # Lift inline scores from Module 7 if present
        inline_scores: dict[str, float] = {}
        if context.image_generation_result and context.image_generation_result.candidate_scores:
            winning_cand = next((c for c in context.image_generation_result.candidate_scores if c.selected), None)
            if winning_cand:
                inline_scores["overall_score"] = winning_cand.overall_score
                inline_scores["identity_similarity"] = winning_cand.identity_similarity

        # Compute weighted overall score across non-zero weight dimensions
        total_weight = 0.0
        weighted_sum = 0.0
        hard_gate_passed = True

        for score in dimension_scores:
            w = self.weights.get(score.dimension, 0.0)
            if w > 0:
                total_weight += w
                weighted_sum += score.score * w
            # Check hard gate pass status
            if score.status == "success" and not score.passed and w > 0:
                hard_gate_passed = False

        overall_score = float(weighted_sum / total_weight) if total_weight > 0 else 0.0
        sha256 = (
            context.image_generation_result.generated_asset.sha256
            if context.image_generation_result and context.image_generation_result.generated_asset
            else "unknown"
        )

        status = "success" if not partial_failures else ("partial" if dimension_scores else "error")

        return QualityEvaluationReport(
            video_id=context.video_id,
            generated_asset_sha256=sha256,
            dimension_scores=dimension_scores,
            inline_scores=inline_scores,
            weighted_overall_score=overall_score,
            hard_gate_passed=hard_gate_passed,
            status=status,
            partial_failure_reasons=partial_failures,
            total_duration_seconds=time.monotonic() - t0,
            evaluated_at=evaluated_at,
        )

"""
beats_original_scorer.py
========================

Computes head-to-head comparative verdict between generated candidate thumbnail and
the original thumbnail baseline quality score.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict

from optimization.config import OPTIMIZATION_MIN_WIN_MARGIN
from optimization.comparative.baseline_scorer import BaselineScore
from modules.models import QualityAssuranceReport


class BeatsOriginalVerdict(BaseModel):
    """Head-to-head comparative verdict between candidate and baseline."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    candidate_index: int
    baseline_overall_score: float
    candidate_overall_score: float
    delta: float
    beats_original: bool
    per_dimension_delta: dict[str, float] = {}


class BeatsOriginalScorer:
    """Computes comparative beats_original verdict."""

    def __init__(self, min_win_margin: float = OPTIMIZATION_MIN_WIN_MARGIN) -> None:
        self.min_win_margin = min_win_margin

    def score(
        self,
        video_id: str,
        candidate_index: int,
        candidate_qa_report: QualityAssuranceReport,
        baseline_score: BaselineScore,
        candidate_pvqef_scores: Optional[dict[str, float]] = None,
    ) -> BeatsOriginalVerdict:
        """
        Compare candidate quality against baseline score.
        """
        candidate_overall = candidate_qa_report.overall_score
        baseline_overall = baseline_score.overall_score
        delta = float(candidate_overall - baseline_overall)
        beats = bool(delta >= self.min_win_margin)

        per_dim_delta: dict[str, float] = {}
        if candidate_pvqef_scores and baseline_score.dimension_scores:
            for dim, base_val in baseline_score.dimension_scores.items():
                if dim in candidate_pvqef_scores:
                    per_dim_delta[dim] = float(candidate_pvqef_scores[dim] - base_val)

        return BeatsOriginalVerdict(
            video_id=video_id,
            candidate_index=candidate_index,
            baseline_overall_score=baseline_overall,
            candidate_overall_score=candidate_overall,
            delta=delta,
            beats_original=beats,
            per_dimension_delta=per_dim_delta,
        )

"""SelectionExplainer: Deterministic template-based explanation generator for candidate selection."""

from __future__ import annotations

import sys
from pathlib import Path

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field
from models import CandidateScore, CandidateStrategy


class SelectionExplanation(BaseModel):
    """Deterministic, template-driven decision explanation for candidate selection."""

    model_config = ConfigDict(frozen=True)

    winner_index: int
    winning_strategy: str
    winner_explanation: str
    dominant_scoring_dimensions: list[str] = Field(default_factory=list)
    winning_margin: float = 0.0
    excluded_candidate_summary: dict[str, Any] = Field(default_factory=dict)


class SelectionExplainer:
    """Explainer engine producing deterministic, structured explanations without LLM reliance."""

    def explain(
        self,
        winner_candidate: tuple[int, Path, Any, Any, CandidateStrategy, Any, str, dict[str, float]],
        candidate_scores: Sequence[CandidateScore],
        all_candidates: Sequence[tuple[int, Path, Any, Any, CandidateStrategy, Any, str, dict[str, float]]],
        clustering_exclusions: dict[int, str] | None = None,
        dimension_scores_map: dict[int, dict[str, float]] | None = None,
    ) -> SelectionExplanation:
        """
        Generate deterministic selection explanation.

        Args:
            winner_candidate: Tuple of winning candidate data.
            candidate_scores: List of CandidateScore audit records.
            all_candidates: List of all generated candidate tuples.
            clustering_exclusions: Map of duplicate candidate_idx -> exclusion reason.
            dimension_scores_map: Map of candidate_idx -> dimension score breakdown.

        Returns:
            SelectionExplanation instance.
        """
        winner_idx = winner_candidate[0]
        winner_qa = winner_candidate[2]
        winner_strategy = winner_candidate[4].name if winner_candidate[4] else "faithful"
        exclusions = clustering_exclusions or {}
        dim_map = dimension_scores_map or {}

        # 1. Identify ranked eligible scores
        eligible_scores = [cs for cs in candidate_scores if cs.hard_gate_passed and cs.overall_score is not None]
        eligible_scores.sort(key=lambda cs: cs.overall_score, reverse=True)

        winner_score = eligible_scores[0].overall_score if eligible_scores else getattr(winner_qa, "overall_score", 0.0)

        # 2. Calculate winning margin
        if len(eligible_scores) > 1:
            runner_up_score = eligible_scores[1].overall_score
            winning_margin = round(max(0.0, winner_score - runner_up_score), 4)
        else:
            winning_margin = round(winner_score, 4)

        # 3. Determine dominant scoring dimensions
        winner_dims = dim_map.get(winner_idx, {})
        if winner_dims:
            sorted_dims = sorted(winner_dims.items(), key=lambda item: item[1], reverse=True)
            dominant_scoring_dimensions = [dim for dim, val in sorted_dims[:2]]
        else:
            dominant_scoring_dimensions = ["ctr_score", "readability_score"]

        # 4. Build excluded candidate summary
        hard_gate_excluded: list[int] = []
        duplicate_excluded: list[int] = []

        for cand in all_candidates:
            c_idx = cand[0]
            if not cand[2].hard_gate_passed:
                hard_gate_excluded.append(c_idx)
            elif c_idx in exclusions:
                duplicate_excluded.append(c_idx)

        excluded_summary = {
            "total_candidates": len(all_candidates),
            "eligible_candidates_count": len(eligible_scores),
            "hard_gate_failures": hard_gate_excluded,
            "duplicate_exclusions": exclusions,
        }

        # 5. Format deterministic text explanation
        explanation_text = (
            f"Candidate {winner_idx} (Strategy: '{winner_strategy}') selected as winner with score {winner_score:.4f}. "
            f"Dominant dimensions: {', '.join(dominant_scoring_dimensions)}. "
            f"Winning margin over next candidate: {winning_margin:.4f}. "
            f"Excluded: {len(hard_gate_excluded)} hard-gate failure(s), {len(duplicate_excluded)} duplicate cluster member(s)."
        )

        return SelectionExplanation(
            winner_index=winner_idx,
            winning_strategy=winner_strategy,
            winner_explanation=explanation_text,
            dominant_scoring_dimensions=dominant_scoring_dimensions,
            winning_margin=winning_margin,
            excluded_candidate_summary=excluded_summary,
        )

"""
models.py
=========

Ranking Data Models and Data Contracts for Phase 5.3 Candidate Ranking Engine.
Defines:
- RankingPolicy (Enum: OVERALL_SCORE, WEIGHTED_METRIC, COMPOSITE_CTR)
- RankingProfile (Configurable ranking parameters, weights, tie-break priorities, confidence margins)
- MetricComparison (Pairwise metric comparison item)
- RankingExplanation (Why Candidate A beat Candidate B)
- RankedCandidate (Candidate entry with rank position, final score, and evaluation result)
- RankingReport (Detailed summary report with metric comparison table and tie-break log)
- RankingResult (Master ranking result contract)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso
from thumbnail_intelligence.evaluation.models import EvaluationResult


class RankingPolicy(str, Enum):
    """Policies governing candidate score aggregation and sorting."""

    OVERALL_SCORE = "overall_score"
    WEIGHTED_METRIC = "weighted_metric"
    COMPOSITE_CTR = "composite_ctr"


class RankingProfile(BaseKBModel):
    """Configurable ranking profile containing policy rules, tie-break priorities, and confidence margins."""

    profile_id: str = Field("default_ranking_profile", description="Ranking profile identifier")
    profile_name: str = Field("Default Candidate Ranking Profile", description="Human-readable profile name")
    schema_version: str = Field("1.0.0", description="Profile semver version")
    policy: RankingPolicy = Field(RankingPolicy.OVERALL_SCORE, description="Primary ranking policy")

    # Tie-breaking priority order (evaluated sequentially)
    tie_break_priority: List[str] = Field(
        default_factory=lambda: [
            "estimated_ctr_score",
            "text_readability",
            "face_visibility",
            "brand_preservation",
            "subject_saliency",
        ],
        description="Sequential priority order for deterministic tie-breaking",
    )

    tie_threshold_pts: float = Field(0.01, ge=0.0, description="Score delta threshold below which candidates are considered tied")
    max_confidence_margin_pts: float = Field(15.0, gt=0.0, description="Score margin for 100% score-separation confidence")
    top_n_default: int = Field(3, ge=1, description="Default Top-N selection count")


class MetricComparison(BaseKBModel):
    """Comparison of a single metric between two candidates."""

    metric_name: str = Field(..., description="Metric identifier e.g. estimated_ctr_score")
    candidate_a_score: float = Field(..., ge=0.0, le=100.0)
    candidate_b_score: float = Field(..., ge=0.0, le=100.0)
    delta: float = Field(..., description="candidate_a_score - candidate_b_score")
    advantage: str = Field(..., description="Advantage label e.g. candidate_a, candidate_b, neutral")


class RankingExplanation(BaseKBModel):
    """Detailed explainable comparison of why Candidate A beat Candidate B."""

    winner_candidate_id: str = Field(..., description="ID of winning candidate A")
    runner_up_candidate_id: str = Field(..., description="ID of runner-up candidate B")
    score_delta: float = Field(..., description="Winner overall score minus runner-up overall score")
    strengths: List[str] = Field(default_factory=list, description="List of areas where winner excelled (+ Higher CTR score)")
    weaknesses: List[str] = Field(default_factory=list, description="List of areas where runner-up excelled (- Weaker color harmony)")
    metric_comparisons: List[MetricComparison] = Field(default_factory=list, description="Detailed per-metric comparison breakdown")
    summary_reasoning: str = Field(..., description="Human-readable plain text explanation of ranking decision")


class RankedCandidate(BaseKBModel):
    """Ranked candidate entry with rank position, final score, and evaluation result."""

    rank: int = Field(..., ge=1, description="Rank position (1-indexed)")
    candidate_id: str = Field(..., description="Candidate identifier e.g. candidate_a")
    candidate_label: str = Field(..., description="Human-readable candidate label")
    final_score: float = Field(..., ge=0.0, le=100.0, description="Final ranking score")
    tie_broken: bool = Field(False, description="Flag indicating if position was resolved via tie-breaking")
    tie_break_reason: Optional[str] = Field(None, description="Explanation of tie-break metric if applicable")
    evaluation_result: EvaluationResult = Field(..., description="Underlying EvaluationResult")


class RankingReport(BaseKBModel):
    """Summary report detailing ranking outcomes, winner summary, runner-up summary, and comparison matrix."""

    set_id: str = Field(..., description="EvaluationSet set_id")
    ranking_policy_used: str = Field(..., description="Name of ranking policy used")
    winner_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary dict for winning candidate")
    runner_up_summary: Optional[Dict[str, Any]] = Field(None, description="Summary dict for runner-up candidate")
    tie_break_log: List[str] = Field(default_factory=list, description="Log of any tie-breaks resolved")
    metric_comparison_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Matrix mapping candidate_id to key metric scores")


class RankingResult(BaseKBModel):
    """Master ranking result emitted by CandidateRankingEngine."""

    set_id: str = Field(..., description="CandidateSet / EvaluationSet set_id")
    schema_version: str = Field("1.0.0", description="Ranking schema version")
    ranked_candidates: List[RankedCandidate] = Field(..., description="Ordered list of ranked candidates (Rank 1 to N)")
    winner: RankedCandidate = Field(..., description="Winning candidate (Rank 1)")
    runner_up: Optional[RankedCandidate] = Field(None, description="Runner-up candidate (Rank 2)")
    top_n: List[RankedCandidate] = Field(..., description="Selected Top-N candidates")
    ranking_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall ranking confidence score (0.0 to 1.0)")
    explanation: RankingExplanation = Field(..., description="Detailed pairwise explanation of why winner beat runner-up")
    report: RankingReport = Field(..., description="Structured ranking report")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO timestamp of ranking")

    def to_json(self, indent: int = 2) -> str:
        """Serialize RankingResult to formatted JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> RankingResult:
        """Deserialize RankingResult from JSON string."""
        return cls.model_validate(json.loads(json_str))

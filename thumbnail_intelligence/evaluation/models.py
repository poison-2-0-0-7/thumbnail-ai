"""
models.py
=========

Evaluation Data Models and Contracts for Phase 5.2 Thumbnail Evaluation Engine.
Defines:
- EvaluationMetric (Individual metric score, weight, confidence, reason, evidence)
- MetricBreakdown (Categorized metrics breakdown)
- EvaluationProfile (Configurable metric weights and thresholds)
- EvaluationResult (Per-candidate evaluation result)
- EvaluationReport (Evaluation set summary report)
- EvaluationSet (Collection of evaluation results for CandidateSet)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso


class EvaluationMetric(BaseKBModel):
    """Container for a single deterministic, explainable thumbnail metric score."""

    metric_name: str = Field(..., description="Unique metric identifier e.g. face_visibility, text_readability")
    category: str = Field(..., description="Metric category e.g. face, typography, composition, color, quality")
    score: float = Field(..., ge=0.0, le=100.0, description="Normalized metric score from 0.0 to 100.0")
    weight: float = Field(..., ge=0.0, description="Configured weight of this metric")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence level of measurement (0.0 to 1.0)")
    reason: str = Field(..., description="Human-readable explanation of WHY the score was given")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Underlying measured raw data/evidence")


class MetricBreakdown(BaseKBModel):
    """Categorized breakdown of evaluated metrics."""

    face_metrics: Dict[str, EvaluationMetric] = Field(default_factory=dict)
    typography_metrics: Dict[str, EvaluationMetric] = Field(default_factory=dict)
    composition_metrics: Dict[str, EvaluationMetric] = Field(default_factory=dict)
    color_metrics: Dict[str, EvaluationMetric] = Field(default_factory=dict)
    quality_metrics: Dict[str, EvaluationMetric] = Field(default_factory=dict)


class EvaluationProfile(BaseKBModel):
    """Configurable evaluation profile storing metric weights and non-magic thresholds."""

    profile_id: str = Field("default_eval_profile", description="Profile identifier")
    profile_name: str = Field("Default Thumbnail Evaluation Profile", description="Profile human-readable name")
    schema_version: str = Field("1.0.0", description="Profile semver version")

    # Configurable weights for all 22 metrics
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "face_visibility": 0.05,
            "face_size": 0.05,
            "face_position": 0.04,
            "eye_contact": 0.03,
            "emotion_strength": 0.04,
            "text_readability": 0.08,
            "font_contrast": 0.06,
            "subject_saliency": 0.06,
            "visual_hierarchy": 0.05,
            "rule_of_thirds": 0.04,
            "negative_space": 0.04,
            "composition_balance": 0.04,
            "background_clutter": 0.04,
            "color_harmony": 0.04,
            "color_contrast": 0.05,
            "brand_preservation": 0.03,
            "object_separation": 0.04,
            "typography_quality": 0.05,
            "thumbnail_clarity": 0.05,
            "visual_simplicity": 0.04,
            "mobile_readability": 0.06,
            "estimated_ctr_score": 0.07,
        }
    )

    # Configurable non-magic thresholds
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "ideal_face_size_min": 0.10,
            "ideal_face_size_max": 0.35,
            "ideal_negative_space_min": 0.15,
            "ideal_negative_space_max": 0.45,
            "wcag_contrast_min": 4.5,
            "min_font_size_px": 36.0,
            "max_ideal_words": 4.0,
            "min_clarity_laplacian": 80.0,
            "ideal_max_elements": 6.0,
            "rule_of_thirds_tolerance_pct": 0.15,
        }
    )


class EvaluationResult(BaseKBModel):
    """Complete evaluation result for a single candidate thumbnail."""

    candidate_id: str = Field(..., description="Candidate identifier e.g. candidate_a")
    candidate_label: str = Field(..., description="Candidate label")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Normalized overall weighted score (0.0 to 100.0)")
    weighted_score: float = Field(..., ge=0.0, le=100.0, description="Sum of (score * weight) / sum(weight)")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Overall confidence level")
    metrics: Dict[str, EvaluationMetric] = Field(..., description="All 22 metrics keyed by metric name")
    breakdown: MetricBreakdown = Field(..., description="Categorized metrics breakdown")
    evaluated_at: str = Field(default_factory=_utc_now_iso, description="ISO timestamp of evaluation")

    def to_json(self, indent: int = 2) -> str:
        """Serialize EvaluationResult to formatted JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> EvaluationResult:
        """Deserialize EvaluationResult from JSON string."""
        return cls.model_validate(json.loads(json_str))


class EvaluationReport(BaseKBModel):
    """Summary report for an EvaluationSet."""

    set_id: str = Field(..., description="CandidateSet set_id")
    total_candidates_evaluated: int = Field(..., ge=1, description="Number of candidates evaluated")
    top_scoring_candidate_id: str = Field(..., description="ID of the candidate with the highest overall score")
    top_score: float = Field(..., ge=0.0, le=100.0, description="Highest overall score in the set")
    average_overall_score: float = Field(..., ge=0.0, le=100.0, description="Average overall score across all candidates")
    candidate_scores: Dict[str, float] = Field(default_factory=dict, description="Summary mapping candidate_id to overall_score")


class EvaluationSet(BaseKBModel):
    """Collection of evaluation results for a CandidateSet."""

    set_id: str = Field(..., description="CandidateSet set_id")
    profile: EvaluationProfile = Field(..., description="EvaluationProfile used for scoring")
    results: List[EvaluationResult] = Field(..., description="EvaluationResult for every candidate")
    report: EvaluationReport = Field(..., description="Summary report")

    def get_result(self, candidate_id: str) -> Optional[EvaluationResult]:
        """Retrieve evaluation result for a specific candidate_id."""
        for r in self.results:
            if r.candidate_id == candidate_id:
                return r
        return None

    def to_json(self, indent: int = 2) -> str:
        """Serialize EvaluationSet to formatted JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> EvaluationSet:
        """Deserialize EvaluationSet from JSON string."""
        return cls.model_validate(json.loads(json_str))

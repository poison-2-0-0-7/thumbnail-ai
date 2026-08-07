"""
risk_models.py
==============

Domain models, risk taxonomies, severity and likelihood models, and candidate risk profiles
for the Risk Reasoning Engine (Phase 3.4F).
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import ConfigDict, Field

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceReference,
)
from thumbnail_intelligence.reasoning.models import (
    ReasoningRisk,
    RiskReasoningOutput,
)


class RiskCategory(str, Enum):
    """
    Extensible taxonomy of performance, visual, cognitive, and policy risks.
    """

    VISUAL_CLUTTER = "visual_clutter"
    WEAK_FOCAL_POINT = "weak_focal_point"
    COMPETING_SUBJECTS = "competing_subjects"
    POOR_CONTRAST = "poor_contrast"
    TINY_FACE = "tiny_face"
    HIDDEN_FACE = "hidden_face"
    WEAK_EMOTION = "weak_emotion"
    TEXT_OVERLOAD = "text_overload"
    UNREADABLE_TEXT = "unreadable_text"
    COLOR_COLLISION = "color_collision"
    BRAND_DRIFT = "brand_drift"
    CLICKBAIT_RISK = "clickbait_risk"
    VIEWER_FATIGUE = "viewer_fatigue"
    COMPETITOR_CONVERGENCE = "competitor_convergence"
    COPYRIGHT_RISK = "copyright_risk"
    PLATFORM_POLICY_RISK = "platform_policy_risk"
    LOW_CURIOSITY = "low_curiosity"
    WEAK_STORY = "weak_story"
    LOW_EMOTIONAL_HOOK = "low_emotional_hook"
    LOW_DISTINCTIVENESS = "low_distinctiveness"
    CUSTOM = "custom"


class RiskSeverity(str, Enum):
    """
    Impact severity level of a detected thumbnail risk.
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NEGLIGIBLE = "NEGLIGIBLE"


class RiskLikelihood(str, Enum):
    """
    Probability of the risk negatively impacting click-through or retention.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class DetectedRisk(BaseKBModel):
    """
    Grounded individual risk instance with category, severity, likelihood, and actionable mitigation.
    """

    risk_id: str = Field(
        default_factory=lambda: f"risk_{uuid.uuid4().hex[:8]}",
        description="Unique risk detection identifier",
    )
    category: RiskCategory = Field(description="Classified risk category")
    severity: RiskSeverity = Field(
        default=RiskSeverity.MEDIUM,
        description="Estimated negative impact severity",
    )
    likelihood: RiskLikelihood = Field(
        default=RiskLikelihood.MEDIUM,
        description="Probability of occurrence or viewer friction",
    )
    impact_score: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Normalized impact score in [0.0, 1.0]",
    )
    title: str = Field(description="Short human-readable risk title")
    description: str = Field(description="Detailed explanation of the risk mechanism")
    affected_element: str = Field(
        default="thumbnail_canvas",
        description="Visual or editorial element affected by this risk",
    )
    mitigation_suggestion: str = Field(
        default="",
        description="Actionable mitigation recommendation to neutralize the risk",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence supporting this risk deduction",
    )


class CandidateRiskProfile(BaseKBModel):
    """
    An individual candidate risk assessment evaluated during multi-hypothesis exploration.
    """

    candidate_id: str = Field(
        default_factory=lambda: f"cand_risk_{uuid.uuid4().hex[:8]}",
        description="Unique candidate risk profile identifier",
    )
    profile_name: str = Field(description="Descriptive name for this risk assessment perspective")
    detected_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="List of detected risks under this interpretation",
    )
    overall_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregated risk severity score in [0.0, 1.0]",
    )
    fatigue_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Audience trope fatigue score in [0.0, 1.0]",
    )
    competitor_convergence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity to competitors score in [0.0, 1.0]",
    )
    clickbait_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Misleading expectation score in [0.0, 1.0]",
    )
    fit_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Empirical accuracy fit score in [0.0, 1.0]",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Grounded confidence in this risk assessment",
    )
    pros: List[str] = Field(default_factory=list, description="Diagnostic strengths of this profile")
    cons: List[str] = Field(default_factory=list, description="Potential limitations or blindspots")
    rejection_rationale: Optional[str] = Field(
        default=None,
        description="Audit explanation if this candidate was evaluated and rejected",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Supporting grounding evidence references",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of evidence nodes backing this hypothesis",
    )


class RiskResult(RiskReasoningOutput):
    """
    Master output artifact of the RiskReasoner.
    Inherits from RiskReasoningOutput for 100% backward compatibility with ReasoningContext.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    visual_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="Risks related to clutter, contrast, focal competition, or small faces",
    )
    narrative_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="Risks related to weak story premise, low curiosity, or unclear stakes",
    )
    audience_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="Risks related to trope fatigue or cognitive overload",
    )
    brand_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="Risks related to brand drift, missing signature assets, or broken guidelines",
    )
    ctr_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="Risks related to weak click triggers or high expectation mismatch",
    )
    readability_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="Risks related to unreadable typography or small mobile scaling",
    )
    policy_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="Risks related to sensationalism, copyright, or platform policy guidelines",
    )
    attention_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="Risks related to competing focal points or chaotic gaze trajectory",
    )
    competition_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="Risks related to indistinguishable appearance from niche rivals",
    )
    cognitive_load_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Estimated cognitive processing load on mobile feed in [0.0, 1.0]",
    )
    overall_severity: RiskSeverity = Field(
        default=RiskSeverity.MEDIUM,
        description="Highest aggregate severity level detected",
    )
    overall_likelihood: RiskLikelihood = Field(
        default=RiskLikelihood.MEDIUM,
        description="Aggregate likelihood of viewer friction",
    )
    overall_impact: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Composite negative impact score on expected CTR in [0.0, 1.0]",
    )
    all_detected_risks: List[DetectedRisk] = Field(
        default_factory=list,
        description="Full consolidated list of all detected risks",
    )
    mitigation_suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable mitigation recommendations (alias for mitigation_strategies)",
    )
    primary_risk_profile: Optional[CandidateRiskProfile] = Field(
        default=None,
        description="Selected winning risk profile interpretation",
    )
    candidate_risk_profiles: List[CandidateRiskProfile] = Field(
        default_factory=list,
        description="All evaluated candidate risk profiles",
    )
    rejected_risk_profiles: List[CandidateRiskProfile] = Field(
        default_factory=list,
        description="Evaluated alternative risk profiles with rejection rationale",
    )
    selection_rationale: str = Field(
        default="",
        description="Explainable audit explanation why the primary risk profile was selected",
    )
    risk_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Propagated multi-signal risk confidence score",
    )
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Granular component confidence scores",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="All evidence node IDs backing risk conclusions",
    )

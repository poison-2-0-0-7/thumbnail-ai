"""
models.py
=========

Data Models and Contracts for Phase 5.4 Intelligent Critique Engine.
Defines:
- IssueSeverity (Enum: CRITICAL, MAJOR, MINOR, INFO)
- ImpactLevel (Enum: HIGH, MEDIUM, LOW)
- ImplementationCost (Enum: LOW, MEDIUM, HIGH)
- Issue (Detailed issue contract: metric_name, severity, confidence, evidence, affected_region, reason, suggested_fix, estimated_impact)
- ImprovementSuggestion (Deterministic action item with target element, parameter changes, expected gain, priority score)
- ImprovementPlan (Prioritized plan of improvement suggestions)
- CritiqueProfile (Configurable issue detection thresholds and rule parameters)
- CritiqueReport (Master critique report contract: executive summary, strengths, weaknesses, critical issues, improvement plan)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso


class IssueSeverity(str, Enum):
    """Classified severity levels for visual/quality issues."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class ImpactLevel(str, Enum):
    """Visual impact classification levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImplementationCost(str, Enum):
    """Estimated implementation complexity/cost."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Issue(BaseKBModel):
    """Detailed visual/quality issue detected on a candidate thumbnail."""

    issue_id: str = Field(..., description="Unique issue identifier e.g. issue_text_readability_01")
    metric_name: str = Field(..., description="Affected evaluation metric e.g. text_readability")
    severity: IssueSeverity = Field(..., description="Issue severity classification")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence score")
    affected_region: str = Field("canvas", description="Affected visual region e.g. hero_subject, headline_text, background")
    reason: str = Field(..., description="Human-readable explanation of WHY this is an issue")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Raw metric measurement evidence")
    suggested_fix: str = Field(..., description="Actionable fix recommendation")
    estimated_impact: ImpactLevel = Field(ImpactLevel.MEDIUM, description="Expected visual/CTR impact level")


class ImprovementSuggestion(BaseKBModel):
    """Actionable, deterministic parameter adjustment suggestion."""

    suggestion_id: str = Field(..., description="Unique suggestion identifier")
    action_type: str = Field(..., description="Action primitive e.g. scale_subject, boost_contrast, reduce_words")
    description: str = Field(..., description="Human-readable description of suggested fix e.g. Increase face scale by 15%")
    target_element: str = Field(..., description="Target element or layer e.g. hero_subject, headline_text")
    parameter_changes: Dict[str, Any] = Field(default_factory=dict, description="Specific parameter updates e.g. {'scale_multiplier': 1.15}")
    expected_ctr_gain: float = Field(0.0, ge=0.0, description="Estimated CTR / score lift in points")
    visual_impact: ImpactLevel = Field(ImpactLevel.MEDIUM, description="Visual impact level")
    implementation_cost: ImplementationCost = Field(ImplementationCost.LOW, description="Implementation complexity cost")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in suggestion effectiveness")
    priority_score: float = Field(0.0, description="Calculated priority score for sorting")


class ImprovementPlan(BaseKBModel):
    """Prioritized collection of deterministic improvement suggestions for a candidate."""

    plan_id: str = Field(..., description="Unique improvement plan identifier")
    candidate_id: str = Field(..., description="Target candidate identifier")
    prioritized_suggestions: List[ImprovementSuggestion] = Field(..., description="Ordered list of suggestions (highest priority first)")
    total_estimated_gain_pts: float = Field(0.0, ge=0.0, description="Cumulative estimated score lift in points")


class CritiqueProfile(BaseKBModel):
    """Configurable critique profile defining metric thresholds and issue rules."""

    profile_id: str = Field("default_critique_profile", description="Critique profile identifier")
    profile_name: str = Field("Default Intelligent Critique Profile", description="Human-readable profile name")
    schema_version: str = Field("1.0.0", description="Profile semver version")

    # Thresholds below which an evaluation metric triggers an issue
    critical_threshold_score: float = Field(50.0, description="Metric score below which issue is CRITICAL")
    major_threshold_score: float = Field(70.0, description="Metric score below which issue is MAJOR")
    minor_threshold_score: float = Field(85.0, description="Metric score below which issue is MINOR")


class CritiqueReport(BaseKBModel):
    """Master structured critique report for a candidate thumbnail."""

    report_id: str = Field(..., description="Unique critique report identifier")
    schema_version: str = Field("1.0.0", description="Critique schema version")
    candidate_id: str = Field(..., description="Target candidate identifier")
    candidate_label: str = Field(..., description="Target candidate label")
    overall_quality_score: float = Field(..., ge=0.0, le=100.0, description="Current overall quality score")

    executive_summary: str = Field(..., description="High-level plain text executive summary")
    strengths: List[str] = Field(default_factory=list, description="Key visual strengths of candidate")
    weaknesses: List[str] = Field(default_factory=list, description="Key visual weaknesses of candidate")
    critical_issues: List[Issue] = Field(default_factory=list, description="List of detected critical and major issues")
    improvement_plan: ImprovementPlan = Field(..., description="Prioritized improvement plan")
    estimated_overall_gain_pts: float = Field(0.0, ge=0.0, description="Estimated total overall score lift from applying plan")

    created_at: str = Field(default_factory=_utc_now_iso, description="ISO timestamp of critique generation")

    def to_json(self, indent: int = 2) -> str:
        """Serialize CritiqueReport to formatted JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> CritiqueReport:
        """Deserialize CritiqueReport from JSON string."""
        return cls.model_validate(json.loads(json_str))

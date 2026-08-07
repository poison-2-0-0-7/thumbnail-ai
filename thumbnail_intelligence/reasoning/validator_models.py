"""
validator_models.py
===================

Domain models, validation issue schemas, conflict taxonomies, readiness scoring models,
and the final grounded ValidatedReasoningPackage for the Strategic Reasoning Validator (Phase 3.4H).
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import ConfigDict, Field

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceReference,
    _utc_now_iso,
)
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.strategy_models import StrategyDecision


class ValidationStatus(str, Enum):
    """
    Overall execution status outcome of the Strategic Reasoning Validator.
    """

    PASSED = "PASSED"
    WARNINGS = "WARNINGS"
    FAILED = "FAILED"
    BLOCKING_ERRORS = "BLOCKING_ERRORS"


class ValidationIssueType(str, Enum):
    """
    Extensible taxonomy of strategic reasoning anomalies, defects, and contradictions.
    """

    CONTRADICTION = "contradiction"
    MISSING_OUTPUT = "missing_output"
    EMPTY_EVIDENCE = "empty_evidence"
    CONFIDENCE_MISMATCH = "confidence_mismatch"
    UNGROUNDED_REASONING = "ungrounded_reasoning"
    CIRCULAR_REASONING = "circular_reasoning"
    MISSING_DEPENDENCY = "missing_dependency"
    INVALID_REFERENCE = "invalid_reference"
    IMPOSSIBLE_COMBINATION = "impossible_combination"
    ORPHAN_OUTPUT = "orphan_output"
    CUSTOM = "custom"


class ValidationSeverity(str, Enum):
    """
    Impact severity level of a detected validation issue.
    """

    BLOCKING = "BLOCKING"
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class ConflictType(str, Enum):
    """
    Extensible taxonomy of cross-module strategic contradictions.
    """

    NARRATIVE_VS_PRIORITY = "narrative_vs_priority"
    AUDIENCE_VS_STRATEGY = "audience_vs_strategy"
    AUDIENCE_VS_BRAND = "audience_vs_brand"
    BRAND_VS_RISK = "brand_vs_risk"
    BRAND_VS_PRIORITY = "brand_vs_priority"
    CREATOR_VS_BRAND = "creator_vs_brand"
    CREATOR_VS_STRATEGY = "creator_vs_strategy"
    NARRATIVE_VS_STRATEGY = "narrative_vs_strategy"
    PRIORITY_VS_STRATEGY = "priority_vs_strategy"
    RISK_VS_STRATEGY = "risk_vs_strategy"
    GENERIC = "generic"


class ValidationIssue(BaseKBModel):
    """
    A single grounded validation anomaly or structural defect identified during verification.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    issue_id: str = Field(
        default_factory=lambda: f"issue_{uuid.uuid4().hex[:8]}",
        description="Unique validation issue identifier",
    )
    issue_type: ValidationIssueType = Field(
        default=ValidationIssueType.CONTRADICTION,
        description="Classification of the validation defect",
    )
    severity: ValidationSeverity = Field(
        default=ValidationSeverity.WARNING,
        description="Severity level of the issue",
    )
    reason: str = Field(description="Detailed explanation why this issue was raised")
    affected_module: str = Field(
        description="Module or cross-module pair affected by this issue (e.g. 'narrative_vs_priority')",
    )
    evidence_references: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence references associated with this issue",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of evidence nodes backing or involved in this issue",
    )
    suggested_resolution: str = Field(
        default="",
        description="Actionable recommendation for resolving or mitigating this issue",
    )


class DetectedConflict(BaseKBModel):
    """
    Structured representation of an explicit cross-module strategic contradiction.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    conflict_id: str = Field(
        default_factory=lambda: f"conflict_{uuid.uuid4().hex[:8]}",
        description="Unique conflict identifier",
    )
    conflict_type: ConflictType = Field(
        default=ConflictType.GENERIC,
        description="Taxonomy classification of the cross-module contradiction",
    )
    source_module_a: str = Field(description="First reasoner module in contradiction")
    source_module_b: str = Field(description="Second reasoner module in contradiction")
    claim_a: str = Field(description="Strategic assertion or claim from module A")
    claim_b: str = Field(description="Contradictory strategic assertion or claim from module B")
    severity: ValidationSeverity = Field(
        default=ValidationSeverity.CRITICAL,
        description="Severity of the contradiction",
    )
    description: str = Field(description="Comprehensive explanation of the contradiction mechanism")
    suggested_resolution: str = Field(
        default="",
        description="Actionable recommendation to resolve the strategic conflict",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Supporting evidence references",
    )


class ValidationTraceStep(BaseKBModel):
    """
    Audit trace log recording the execution of an individual validation rule or check.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    step_id: str = Field(
        default_factory=lambda: f"vstep_{uuid.uuid4().hex[:8]}",
        description="Unique trace step identifier",
    )
    check_name: str = Field(description="Name or title of the validation check executed")
    status: Literal["PASSED", "WARNING", "FAILED"] = Field(
        default="PASSED",
        description="Outcome status of the check",
    )
    duration_ms: float = Field(default=0.0, ge=0.0, description="Duration in milliseconds")
    details: str = Field(default="", description="Diagnostic summary or explanation")
    issues_found: int = Field(default=0, ge=0, description="Number of issues detected during check")
    timestamp: str = Field(default_factory=_utc_now_iso)


class ReasoningValidation(BaseKBModel):
    """
    Master output report artifact produced by the StrategicReasoningValidator.
    Contains overall status, consistency/readiness scores, blocking errors, detected conflicts,
    actionable resolutions, audit traces, and validation confidence.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    validation_id: str = Field(
        default_factory=lambda: f"val_{uuid.uuid4().hex[:8]}",
        description="Unique validation report identifier",
    )
    status: ValidationStatus = Field(
        default=ValidationStatus.PASSED,
        description="Overall validation outcome status",
    )
    consistency_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Internal logical consistency score across all reasoning outputs in [0.0, 1.0]",
    )
    readiness_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Composite readiness score for DesignBrief generation in [0.0, 1.0]",
    )
    ready_for_design_brief: bool = Field(
        default=True,
        description="Strict Boolean flag determining readiness for DesignBrief generation",
    )
    blocking_errors: List[ValidationIssue] = Field(
        default_factory=list,
        description="Critical blocking validation issues that prevent downstream generation",
    )
    warnings: List[ValidationIssue] = Field(
        default_factory=list,
        description="Non-blocking validation warnings and advisory items",
    )
    detected_conflicts: List[DetectedConflict] = Field(
        default_factory=list,
        description="Explicit cross-module strategic contradictions detected",
    )
    contradictions: List[ValidationIssue] = Field(
        default_factory=list,
        description="Contradiction validation issues",
    )
    resolution_suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations to resolve or mitigate detected issues",
    )
    validation_trace: List[ValidationTraceStep] = Field(
        default_factory=list,
        description="Chronological audit trace steps of all executed validation checks",
    )
    evidence_references: List[EvidenceReference] = Field(
        default_factory=list,
        description="Aggregated evidence references validated during verification",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Empirical confidence score in the validation process itself",
    )
    created_at: str = Field(default_factory=_utc_now_iso)


class ValidatedReasoningPackage(BaseKBModel):
    """
    Unified master artifact passed downstream to DesignBrief Generator.
    Combines ReasoningContext, StrategyDecision, and ReasoningValidation into a single verified package.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    package_id: str = Field(
        default_factory=lambda: f"pkg_{uuid.uuid4().hex[:8]}",
        description="Unique validated package identifier",
    )
    context: ReasoningContext = Field(description="Source ReasoningContext containing all 7 strategic facets")
    strategy_decision: Optional[StrategyDecision] = Field(
        default=None,
        description="Source StrategyDecision from StrategyRanker",
    )
    validation: ReasoningValidation = Field(
        description="Comprehensive strategic reasoning validation report",
    )
    ready_for_design_brief: bool = Field(
        default=True,
        description="Boolean gate dictating whether DesignBrief Generator can consume this package",
    )
    created_at: str = Field(default_factory=_utc_now_iso)

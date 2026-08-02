"""
observability/diagnostics/models.py
====================================

Frozen Pydantic data models for Findings, Rule Results, and Diagnostic Context in PORCE.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from observability.facts.models import FactCollection, TraceFacts
from observability.models import ArtifactRef, GenerationTraceRecord, LogLineRef, PipelineTrace

FindingSeverity = Literal["FAIL", "WARNING", "INFO", "PASS"]

FindingCategory = Literal[
    "latent_initialization",
    "conditioning",
    "decision_honoring",
    "asset_provenance",
    "prompt_consistency",
    "composition",
    "general",
]


class Finding(BaseModel):
    """
    Structured, architecture-defined Finding produced by a diagnostic rule.
    Represents a deterministic rule evaluation observation.
    """

    model_config = ConfigDict(frozen=True)

    finding_id: str
    rule_name: str = ""
    category: str = "general"
    severity: FindingSeverity
    confidence: float = 1.0
    affected_module: str
    root_cause: str
    recommended_action: str
    supporting_evidence: list[ArtifactRef | LogLineRef | dict[str, Any]] = Field(default_factory=list)
    supporting_facts: list[str] = Field(default_factory=list)
    related_artifacts: list[str] = Field(default_factory=list)
    evaluation_timestamp: str = ""
    rule_version: str = "1.0.0"


class FindingCollection(BaseModel):
    """
    Collection of all Findings produced for a single video_id execution trace.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    findings: list[Finding] = Field(default_factory=list)
    fail_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    pass_count: int = 0
    evaluated_at: str = ""


class RuleContext(BaseModel):
    """
    Execution context passed to IDiagnosticRule evaluation.
    """

    model_config = ConfigDict(frozen=True)

    facts: TraceFacts
    pipeline_trace: Optional[PipelineTrace] = None
    generation_trace: Optional[GenerationTraceRecord] = None
    fact_collection: Optional[FactCollection] = None


class RuleResult(BaseModel):
    """
    Evaluation result for a single IDiagnosticRule execution.
    """

    model_config = ConfigDict(frozen=True)

    rule_id: str
    rule_name: str
    passed: bool
    finding: Optional[Finding] = None

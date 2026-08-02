"""
observability/reporting/models.py
==================================

Frozen Pydantic data models for canonical RootCauseReport in PORCE.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from observability.diagnostics.models import Finding


class RootCauseReport(BaseModel):
    """
    Canonical, machine-readable root cause report for a single video_id execution trace.
    Consolidated output produced by Sprint 3C RootCauseAssembler.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    findings: list[Finding] = Field(default_factory=list)
    fail_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    pass_count: int = 0
    top_root_causes: list[str] = Field(default_factory=list)
    generated_from_trace_hash: str = ""
    engine_version: str = "1.0.0"
    status: Literal["success", "partial", "error"] = "success"
    generated_at: str = ""
    evidence_summary: dict[str, Any] = Field(default_factory=dict)

"""
reports.py
==========

Structured report data contracts for the Execution Engine (Phase 4.1).
Defines StageExecutionReport, RenderJobReport, CritiqueReport, and status enums.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso


class StageStatus(str, Enum):
    """Execution status for an individual stage or operation."""

    SUCCESS = "SUCCESS"
    SUCCESS_WITH_DEGRADATION = "SUCCESS_WITH_DEGRADATION"
    FAILED_RECOVERABLE = "FAILED_RECOVERABLE"
    FAILED_FATAL = "FAILED_FATAL"
    SKIPPED = "SKIPPED"


class RenderJobStatus(str, Enum):
    """Overall status for a RenderJob."""

    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    SUCCESS_WITH_DEGRADATION = "SUCCESS_WITH_DEGRADATION"
    FAILED_HUMAN_REVIEW = "FAILED_HUMAN_REVIEW"
    FAILED_FATAL = "FAILED_FATAL"
    CANCELLED = "CANCELLED"


class StageExecutionReport(BaseKBModel):
    """Detailed report emitted after the execution of each stage or operation."""

    stage: str = Field(description="Name or identifier of the stage executed")
    op_id: str = Field(default="", description="RenderOperation op_id associated with this stage")
    status: StageStatus = Field(default=StageStatus.SUCCESS, description="Execution status outcome")
    latency_s: float = Field(default=0.0, ge=0.0, description="Wall-clock latency in seconds")
    vram_peak_gb: float = Field(default=0.0, ge=0.0, description="Peak VRAM observed during stage in GB")
    validation_notes: List[str] = Field(default_factory=list, description="Validation notes and assertions")
    error_message: Optional[str] = Field(default=None, description="Detailed error message if failed")
    output_keys: List[str] = Field(default_factory=list, description="Keys produced or modified in workspace")
    timestamp: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC execution timestamp")


class CritiqueReport(BaseKBModel):
    """Critique report emitted on quality evaluation failures for retry targeting."""

    failing_dimension: str = Field(description="Quality metric that failed target threshold")
    hypothesis: str = Field(description="Targeted correction hypothesis for retry")
    implicated_stage: str = Field(description="Earliest execution stage implicated by failure")
    target_element_id: Optional[str] = Field(default=None, description="Target element ID if localized")
    suggested_param_adjustment: Dict[str, Any] = Field(
        default_factory=dict, description="Suggested parameter adjustments for retry"
    )


class RenderJobReport(BaseKBModel):
    """
    Comprehensive final job execution report returned by ExecutionEngine.execute().
    Summarizes job status, per-stage reports, overall timing, VRAM usage, and errors.
    """

    job_id: str = Field(description="Unique render job identifier")
    correlation_id: str = Field(description="Tracing correlation identifier")
    attempt: int = Field(default=1, ge=1, description="Monotonically increasing attempt counter")
    status: RenderJobStatus = Field(default=RenderJobStatus.SUCCESS, description="Final job status")
    stage_reports: List[StageExecutionReport] = Field(
        default_factory=list, description="Ordered list of stage execution reports"
    )
    total_latency_s: float = Field(default=0.0, ge=0.0, description="Total job duration in seconds")
    vram_peak_gb: float = Field(default=0.0, ge=0.0, description="Maximum peak VRAM observed across all stages")
    critique_report: Optional[CritiqueReport] = Field(default=None, description="Terminal critique report if failed quality")
    output_image_path: Optional[str] = Field(default=None, description="Path or sink key to final output image")
    validation_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of package, graph, workspace, and image validations"
    )
    errors: List[str] = Field(default_factory=list, description="List of fatal or non-fatal errors recorded")
    warnings: List[str] = Field(default_factory=list, description="List of degradation or warning notes")
    created_at: str = Field(default_factory=_utc_now_iso, description="Job completion ISO-8601 UTC timestamp")

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a lightweight dictionary summary of job execution results."""
        return {
            "job_id": self.job_id,
            "correlation_id": self.correlation_id,
            "attempt": self.attempt,
            "status": self.status.value,
            "total_latency_s": self.total_latency_s,
            "vram_peak_gb": self.vram_peak_gb,
            "stages_executed": len(self.stage_reports),
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "output_image_path": self.output_image_path,
        }

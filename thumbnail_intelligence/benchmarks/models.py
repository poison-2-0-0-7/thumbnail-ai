"""
models.py
=========

Data Models and Data Contracts for Phase 6.1 Benchmark & Evaluation Framework.
Defines:
- FailureCategory (Enum: POOR_FACE_EXTRACTION, TYPOGRAPHY_FAILURES, LOW_CONTRAST, WEAK_COMPOSITION, BACKGROUND_FAILURES, PIPELINE_FAILURES, OOM_FAILURES)
- DatasetItem (Benchmark dataset sample contract)
- BenchmarkResult (Single dataset item evaluation result)
- BenchmarkSummaryMetrics (Aggregated benchmark metrics)
- LeaderboardEntry (Ranked entry for leaderboard)
- Leaderboard (System leaderboard contract)
- BenchmarkReport (Exportable multi-format report)
- BenchmarkSession (Master benchmark session container)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso
from thumbnail_intelligence.optimization.models import OptimizationSession
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief


class FailureCategory(str, Enum):
    """Classified benchmark failure categories."""

    POOR_FACE_EXTRACTION = "poor_face_extraction"
    TYPOGRAPHY_FAILURES = "typography_failures"
    LOW_CONTRAST = "low_contrast"
    WEAK_COMPOSITION = "weak_composition"
    BACKGROUND_FAILURES = "background_failures"
    PIPELINE_FAILURES = "pipeline_failures"
    OOM_FAILURES = "oom_failures"
    NONE = "none"


class DatasetItem(BaseKBModel):
    """Single benchmark dataset sample."""

    item_id: str = Field(..., description="Unique dataset item identifier e.g. video_001")
    title: str = Field("Sample Video Title", description="Video title or headline text")
    category: str = Field("Tech", description="Content category e.g. Gaming, Tech, Vlogs")
    video_url: Optional[str] = Field(None, description="Optional YouTube video URL")
    original_thumbnail_path: Optional[str] = Field(None, description="Path to existing human-designed thumbnail")
    brief: Optional[DesignBrief] = Field(None, description="Optional pre-built DesignBrief")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional item metadata")


class BenchmarkResult(BaseKBModel):
    """Evaluation result for a single dataset sample."""

    item_id: str = Field(..., description="Dataset item identifier")
    success: bool = Field(True, description="Success flag")
    failure_category: FailureCategory = Field(FailureCategory.NONE, description="Failure category if unsuccessful")
    failure_reason: Optional[str] = Field(None, description="Detailed failure description")

    initial_score: float = Field(0.0, ge=0.0, le=100.0, description="Initial thumbnail score")
    final_score: float = Field(0.0, ge=0.0, le=100.0, description="Final optimized thumbnail score")
    score_gain_pts: float = Field(0.0, description="Final score minus initial score")
    ctr_prediction: float = Field(0.0, ge=0.0, le=100.0, description="Final estimated CTR score")
    iterations_required: int = Field(1, ge=1, description="Number of optimization iterations required")

    runtime_s: float = Field(0.0, ge=0.0, description="Wall-clock latency in seconds")
    peak_vram_gb: float = Field(0.0, ge=0.0, description="Peak GPU VRAM usage in GB")
    gpu_memory_used_mb: float = Field(0.0, ge=0.0, description="GPU memory allocated in MB")
    estimated_render_cost: str = Field("LOW", description="Render cost estimate (LOW, MEDIUM, HIGH)")

    original_image_path: Optional[str] = Field(None, description="Path to original input thumbnail")
    generated_image_path: Optional[str] = Field(None, description="Path to initial generated thumbnail")
    improved_image_path: Optional[str] = Field(None, description="Path to final improved thumbnail")
    visual_comparison_path: Optional[str] = Field(None, description="Path to side-by-side comparison artifact")
    optimization_session: Optional[OptimizationSession] = Field(None, description="Underlying OptimizationSession")


class BenchmarkSummaryMetrics(BaseKBModel):
    """Aggregated benchmark summary metrics across all dataset items."""

    total_samples: int = Field(..., ge=0, description="Total dataset items evaluated")
    successful_samples: int = Field(..., ge=0, description="Number of successful evaluations")
    failed_samples: int = Field(..., ge=0, description="Number of failed evaluations")
    success_rate_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of successful samples")
    failure_rate_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of failed samples")

    avg_initial_score: float = Field(0.0, ge=0.0, le=100.0, description="Average initial score")
    avg_final_score: float = Field(0.0, ge=0.0, le=100.0, description="Average final score")
    avg_score_improvement_pts: float = Field(0.0, description="Average score gain in points")
    avg_ctr_prediction: float = Field(0.0, ge=0.0, le=100.0, description="Average estimated CTR score")
    avg_iterations_required: float = Field(1.0, ge=1.0, description="Average iterations required")

    avg_runtime_s: float = Field(0.0, ge=0.0, description="Average runtime per sample in seconds")
    total_runtime_s: float = Field(0.0, ge=0.0, description="Total wall-clock runtime in seconds")
    avg_gpu_memory_mb: float = Field(0.0, ge=0.0, description="Average GPU memory used in MB")
    peak_vram_gb: float = Field(0.0, ge=0.0, description="Peak VRAM recorded across session in GB")
    optimization_efficiency: float = Field(0.0, ge=0.0, description="Score gain per second of runtime")

    failure_distribution: Dict[str, int] = Field(default_factory=dict, description="Count of failures per category")
    render_cost_distribution: Dict[str, int] = Field(default_factory=dict, description="Count of samples per render cost tier")


class LeaderboardEntry(BaseKBModel):
    """Single entry in the system leaderboard."""

    rank: int = Field(..., ge=1, description="Rank position (1-indexed)")
    model_or_pipeline: str = Field(..., description="Pipeline or model version name")
    avg_quality_score: float = Field(..., ge=0.0, le=100.0, description="Average final quality score")
    avg_ctr_score: float = Field(..., ge=0.0, le=100.0, description="Average estimated CTR score")
    success_rate_pct: float = Field(..., ge=0.0, le=100.0, description="Success rate percentage")
    avg_runtime_s: float = Field(..., ge=0.0, description="Average runtime in seconds")
    peak_vram_gb: float = Field(..., ge=0.0, description="Peak VRAM in GB")


class Leaderboard(BaseKBModel):
    """System leaderboard ranking pipelines or model versions."""

    leaderboard_id: str = Field(..., description="Unique leaderboard ID")
    schema_version: str = Field("1.0.0", description="Leaderboard schema version")
    entries: List[LeaderboardEntry] = Field(default_factory=list, description="Ordered leaderboard entries")
    updated_at: str = Field(default_factory=_utc_now_iso, description="ISO timestamp of update")

    def to_markdown(self) -> str:
        """Export leaderboard as Markdown table."""
        lines = [
            "# Thumbnail AI Benchmark Leaderboard",
            "",
            "| Rank | Model / Pipeline | Avg Quality Score | Avg CTR Score | Success Rate | Avg Runtime (s) | Peak VRAM (GB) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for e in self.entries:
            lines.append(
                f"| {e.rank} | **{e.model_or_pipeline}** | {e.avg_quality_score:.2f} | {e.avg_ctr_score:.2f} | {e.success_rate_pct:.1f}% | {e.avg_runtime_s:.2f}s | {e.peak_vram_gb:.2f} GB |"
            )
        return "\n".join(lines)


class BenchmarkReport(BaseKBModel):
    """Exportable multi-format benchmark report."""

    session_id: str = Field(..., description="Benchmark session ID")
    dataset_name: str = Field("default_dataset", description="Name of benchmark dataset")
    summary: BenchmarkSummaryMetrics = Field(..., description="Aggregated metrics summary")
    leaderboard: Leaderboard = Field(..., description="Leaderboard")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO timestamp")

    html_report_path: Optional[str] = Field(None, description="Path to generated HTML report")
    markdown_report_path: Optional[str] = Field(None, description="Path to generated Markdown report")
    json_report_path: Optional[str] = Field(None, description="Path to generated JSON report")
    csv_report_path: Optional[str] = Field(None, description="Path to generated CSV report")


class BenchmarkSession(BaseKBModel):
    """Master benchmark session container."""

    session_id: str = Field(..., description="Unique benchmark session ID")
    schema_version: str = Field("1.0.0", description="Benchmark schema version")
    dataset_name: str = Field(..., description="Dataset name")
    results: List[BenchmarkResult] = Field(default_factory=list, description="Results for all dataset items")
    summary: BenchmarkSummaryMetrics = Field(..., description="Aggregated summary metrics")
    report: Optional[BenchmarkReport] = Field(None, description="Exported BenchmarkReport")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO timestamp")

    def to_json(self, indent: int = 2) -> str:
        """Serialize BenchmarkSession to formatted JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> BenchmarkSession:
        """Deserialize BenchmarkSession from JSON string."""
        return cls.model_validate(json.loads(json_str))

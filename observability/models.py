"""
observability/models.py
=======================

Frozen Pydantic data models for the Pipeline Observability & Root Cause Engine (PORCE).
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    """
    Pointer to a single module artifact file (existing or expected/missing).
    """

    model_config = ConfigDict(frozen=True)

    module: str
    artifact_type: str
    path: Optional[str] = None
    exists: bool
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None


class ArtifactIndex(BaseModel):
    """
    Index of all discovered and expected artifacts for a single video_id.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    refs: list[ArtifactRef] = Field(default_factory=list)
    built_at: str


class LogLineRef(BaseModel):
    """
    Pointer to a specific line in a module log file.
    """

    model_config = ConfigDict(frozen=True)

    file_path: str
    line_number: int
    timestamp: Optional[str] = None
    level: str = "INFO"
    module: str = "unknown"
    message: str = ""
    raw_line: str = ""


class ModuleTraceEntry(BaseModel):
    """
    Observability trace entry for a single pipeline module execution.
    """

    model_config = ConfigDict(frozen=True)

    module: str
    stage_order: int
    status: Literal["success", "partial", "error", "skipped", "not_run"]
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    duration_source: Literal["exact", "log_derived", "unavailable"] = "unavailable"
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    log_lines: list[LogLineRef] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FragmentAttachmentRecord(BaseModel):
    """
    Record of a single workflow fragment attachment point and weight.
    """

    model_config = ConfigDict(frozen=True)

    fragment_name: str
    attach_point: str
    strength_or_weight: Optional[float] = None
    requested_capability: Optional[str] = None
    resolved_model: Optional[str] = None
    resolution_source: Optional[str] = None
    fallback_path: Optional[bool] = False
    compatibility_decision: Optional[str] = None


class GenerationTraceRecord(BaseModel):
    """
    Detailed execution trace for Module 7 image generation attempts.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    attempt_index: int = 0
    generation_id: Optional[str] = None
    workflow_template: str = ""
    workflow_hash: str = ""
    workflow_fragments: list[str] = Field(default_factory=list)
    fragments_attached: list[FragmentAttachmentRecord] = Field(default_factory=list)
    latent_source: Literal["noise", "vae_encoded_source"] = "noise"
    denoise: float = 1.0
    seed: int = 0
    scheduler: Optional[str] = None
    sampler: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    controlnet_enabled: bool = False
    ipadapter_enabled: bool = False
    edit_mode: Optional[str] = None
    generation_profile: Optional[str] = None
    controlnet_config: dict[str, Any] = Field(default_factory=dict)
    ipadapter_config: dict[str, Any] = Field(default_factory=dict)
    conditioning_assets: list[str] = Field(default_factory=list)
    asset_references: list[str] = Field(default_factory=list)
    mask_references: list[str] = Field(default_factory=list)
    composition_references: list[str] = Field(default_factory=list)
    generation_plan_reference: Optional[str] = None
    prompt_reference: Optional[str] = None
    negative_prompt_reference: Optional[str] = None
    source_thumbnail_path: Optional[str] = None
    edit_mask_paths: list[str] = Field(default_factory=list)
    execution_timestamps: dict[str, str] = Field(default_factory=dict)
    renderer_version: str = "1.0.0"
    model_version: Optional[str] = None
    lora_list: list[str] = Field(default_factory=list)
    vae: Optional[str] = None
    output_image_path: Optional[str] = None
    recorded_at: str = ""

    # Module 8 Optimization Layer trace fields
    baseline_score: Optional[float] = None
    candidate_scores: list[float] = Field(default_factory=list)
    beats_original: Optional[bool] = None
    winning_candidate_index: Optional[int] = None
    module7_selected_index: Optional[int] = None
    selection_agreed: Optional[bool] = None
    edit_magnitude: Optional[float] = None
    over_edited: Optional[bool] = None
    optimization_strategy_used: Optional[str] = None
    retry_attempt_count: int = 0

    # Module 9 Multi-Candidate Selection trace fields
    strategy_name: Optional[str] = None
    cluster_id: Optional[str] = None
    exclusion_reason: Optional[str] = None
    ranking_dimensions: Optional[dict[str, float]] = None
    selection_explanation: Optional[str] = None
    manual_override: bool = False

    # Module 10 Creator Style Learning trace fields
    creator_channel_id: Optional[str] = None
    style_signature_reference: Optional[str] = None
    style_embedding_similarity: Optional[float] = None
    style_profile_established: Optional[bool] = None
    style_bonus_applied: Optional[float] = None
    drift_detected: Optional[bool] = None
    drift_confidence: Optional[float] = None
    style_prompt_guidance_applied: Optional[bool] = None




class PipelineTrace(BaseModel):
    """
    Full end-to-end pipeline trace for a single video_id.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    modules: list[ModuleTraceEntry] = Field(default_factory=list)
    artifact_index: ArtifactIndex
    generation_trace: Optional[GenerationTraceRecord | dict[str, Any]] = None
    overall_status: Literal["success", "partial", "error"]
    assembled_at: str


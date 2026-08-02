"""
observability/facts/models.py
==============================

Frozen Pydantic models for the Facts Extraction Layer of PORCE.
All models represent strictly objective, reproducible observations (facts)
extracted from PipelineTrace and its associated generation trace records and artifacts.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class FactModel(BaseModel):
    """
    Atomic observation fact entry.
    """

    model_config = ConfigDict(frozen=True)

    fact_key: str
    category: str
    name: str
    value: Any = None
    data_type: str = "string"
    source_module: str = "pipeline"


class TraceFacts(BaseModel):
    """
    Structured container for deterministic, reproducible facts extracted from a PipelineTrace.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    extracted_at: str
    fact_version: str = "1.0.0"

    # Generation parameters & workflow selection
    workflow_selected: Optional[str] = None
    edit_mode: Optional[str] = None
    generation_profile: Optional[str] = None
    model_used: Optional[str] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    cfg: Optional[float] = None
    steps: Optional[int] = None
    denoise: Optional[float] = None
    latent_initialization_mode: Optional[str] = None  # e.g., "EmptyLatentImage", "noise", "vae_encoded_source"

    # ControlNet & IPAdapter & Masks
    controlnet_count: int = 0
    controlnet_configuration: dict[str, Any] = Field(default_factory=dict)
    controlnet_enabled: bool = False
    ipadapter_count: int = 0
    ipadapter_configuration: dict[str, Any] = Field(default_factory=dict)
    ipadapter_enabled: bool = False
    mask_count: int = 0
    edit_mask_paths: list[str] = Field(default_factory=list)

    # Assets & Workspace
    conditioning_assets: list[str] = Field(default_factory=list)
    background_assets: list[str] = Field(default_factory=list)
    foreground_assets: list[str] = Field(default_factory=list)
    composition_workspace: Optional[str] = None
    has_composition_workspace: bool = False

    # Plan & Prompt references
    generation_plan_reference: Optional[str] = None
    prompt_reference: Optional[str] = None
    negative_prompt_reference: Optional[str] = None
    positive_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None

    # Technical / System observations
    renderer_version: Optional[str] = None
    execution_timing: dict[str, Optional[float]] = Field(default_factory=dict)
    total_execution_time_seconds: Optional[float] = None
    artifact_availability: dict[str, bool] = Field(default_factory=dict)
    persisted_outputs: dict[str, Optional[str]] = Field(default_factory=dict)
    module_completion_status: dict[str, str] = Field(default_factory=dict)

    # Fragment details & state flags
    attached_fragment_count: int = 0
    attached_fragment_names: list[str] = Field(default_factory=list)
    source_thumbnail_exists: bool = False
    generated_thumbnail_exists: bool = False
    asset_extraction_enabled: bool = False
    decision_engine_enabled: bool = False
    thumbnail_planner_enabled: bool = False

    # Module 8 Optimization facts
    beats_original: Optional[bool] = None
    over_edited: Optional[bool] = None
    selection_agreed: Optional[bool] = None
    baseline_score: Optional[float] = None
    winning_candidate_index: Optional[int] = None
    module7_selected_index: Optional[int] = None
    edit_magnitude: Optional[float] = None


class FactCollection(BaseModel):
    """
    Collection wrapper for all facts extracted for a single video_id.
    Contains both the structured TraceFacts and individual FactModel atomic entries.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    trace_facts: TraceFacts
    atomic_facts: list[FactModel] = Field(default_factory=list)
    extracted_at: str
    fact_version: str = "1.0.0"

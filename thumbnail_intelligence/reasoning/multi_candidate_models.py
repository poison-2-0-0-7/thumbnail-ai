"""
multi_candidate_models.py
=========================

Data contracts and variation dimension models for Phase 5.1 Multi-Candidate Generation.
Defines:
- VariationDimension (Enum)
- VariationProfile (Strategic variation configuration)
- CandidateDescriptor (Candidate definition)
- CandidateMetadata (Generation execution & variation tracking metadata)
- CandidateResult (Individual generated candidate thumbnail result)
- CandidateSet (Set of generated thumbnail candidates: A, B, C, D, E)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso
from thumbnail_intelligence.reasoning.renderer_adapter_models import RenderExecutionPackage
from renderer_v2.execution.reports import RenderJobReport


class VariationDimension(str, Enum):
    """Strategic dimensions along which thumbnail candidate variations are systematically generated."""

    EMOTIONAL_EMPHASIS = "emotional_emphasis"
    CURIOSITY_EMPHASIS = "curiosity_emphasis"
    TYPOGRAPHY_EMPHASIS = "typography_emphasis"
    FACE_EMPHASIS = "face_emphasis"
    BACKGROUND_EMPHASIS = "background_emphasis"
    COLOR_EMPHASIS = "color_emphasis"
    COMPOSITION_EMPHASIS = "composition_emphasis"


class VariationProfile(BaseKBModel):
    """Strategic profile dictating deterministic transformations applied to a base RenderExecutionPackage."""

    profile_id: str = Field(..., description="Unique variation profile identifier e.g. profile_cand_a_emotional")
    profile_name: str = Field(..., description="Descriptive profile name e.g. High Emotion Hero")
    primary_dimension: VariationDimension = Field(..., description="Primary strategic variation dimension")
    secondary_dimension: Optional[VariationDimension] = Field(None, description="Optional secondary variation dimension")

    # Typography variations
    typography_scale_multiplier: float = Field(1.0, gt=0.0, description="Multiplier for headline text font size")
    font_color_hex: Optional[str] = Field(None, description="Override headline text color hex code")
    stroke_color_hex: Optional[str] = Field(None, description="Override headline text stroke color hex code")
    stroke_width_multiplier: float = Field(1.0, ge=0.0, description="Multiplier for text stroke width")
    pill_fill_hex: Optional[str] = Field(None, description="Override typography pill container fill color hex code")

    # Subject / Face variations
    subject_scale_multiplier: float = Field(1.0, gt=0.0, description="Multiplier for subject pixel bounding box scale")
    subject_x_offset_pct: float = Field(0.0, description="Normalized horizontal offset shift (-0.5 to +0.5)")
    subject_y_offset_pct: float = Field(0.0, description="Normalized vertical offset shift (-0.5 to +0.5)")

    # Lighting variations
    key_light_intensity_multiplier: float = Field(1.0, ge=0.0, le=2.0, description="Key light intensity multiplier")
    rim_light_enabled_override: Optional[bool] = Field(None, description="Override rim light flag")

    # Background & Color variations
    background_style_direction: Optional[str] = Field(None, description="Override background style prompt direction")
    dominant_colors_override: List[str] = Field(default_factory=list, description="Override dominant color palette hex codes")

    deterministic_seed: int = Field(42, description="Deterministic seed for reproducible generation")


class CandidateDescriptor(BaseKBModel):
    """Manifest descriptor for a single candidate variant before rendering."""

    candidate_id: str = Field(..., description="Candidate identifier e.g. candidate_a")
    candidate_label: str = Field(..., description="Human-readable candidate label e.g. Candidate A (Emotional Emphasis)")
    profile: VariationProfile = Field(..., description="Variation profile governing this candidate")
    package: RenderExecutionPackage = Field(..., description="Transformed RenderExecutionPackage for this candidate")


class CandidateMetadata(BaseKBModel):
    """Execution and variation tracking metadata for a generated CandidateSet."""

    set_id: str = Field(..., description="Unique candidate set identifier")
    generated_at: str = Field(default_factory=_utc_now_iso, description="ISO timestamp of candidate set generation")
    total_requested: int = Field(5, ge=1, description="Number of candidates requested")
    total_generated: int = Field(5, ge=1, description="Number of candidates successfully generated")
    variation_dimensions: List[str] = Field(default_factory=list, description="List of variation dimensions represented")
    strategy_summary: Dict[str, str] = Field(default_factory=dict, description="Summary mapping candidate_id to strategic focus")
    execution_latencies_s: Dict[str, float] = Field(default_factory=dict, description="Per-candidate render latency in seconds")


class CandidateResult(BaseKBModel):
    """Result of rendering a single candidate thumbnail."""

    candidate_id: str = Field(..., description="Candidate identifier e.g. candidate_a")
    candidate_label: str = Field(..., description="Human-readable candidate label")
    profile: VariationProfile = Field(..., description="Variation profile used")
    report: RenderJobReport = Field(..., description="RenderJobReport produced by ExecutionEngine")
    image_path: str = Field(..., description="File path to the rendered output thumbnail image")
    package: RenderExecutionPackage = Field(..., description="RenderExecutionPackage used for rendering")


class CandidateSet(BaseKBModel):
    """Immutable collection of generated thumbnail candidates (Candidate A, B, C, D, E)."""

    set_id: str = Field(..., description="Unique candidate set ID")
    candidates: List[CandidateResult] = Field(..., description="List of generated CandidateResult objects")
    metadata: CandidateMetadata = Field(..., description="Execution and variation metadata")

    def get_candidate(self, candidate_id: str) -> Optional[CandidateResult]:
        """Retrieve a specific candidate result by candidate_id."""
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                return c
        return None

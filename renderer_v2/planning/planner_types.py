"""Data schemas and types for Renderer V2 Phase 2 Edit Planner."""

from __future__ import annotations

from enum import Enum
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class EditAction(str, Enum):
    """Supported deterministic editing actions for detected objects and composition elements."""
    KEEP = "keep"
    REMOVE = "remove"
    MOVE = "move"
    RESIZE = "resize"
    RECOLOR = "recolor"
    RELIGHT = "relight"
    REPLACE = "replace"
    REGENERATE = "regenerate"
    BLUR = "blur"
    DESATURATE = "desaturate"
    ENHANCE = "enhance"


class TargetCategory(str, Enum):
    """Categorization for edit targets."""
    CREATOR_FACE = "creator_face"
    CREATOR_BODY = "creator_body"
    BACKGROUND = "background"
    LOGO = "logo"
    PRODUCT = "product"
    TYPOGRAPHY = "typography"
    GRAPHIC_OVERLAY = "graphic_overlay"
    CLUTTER = "clutter"
    OTHER = "other"


class ObjectEditChange(BaseModel):
    """Deterministic edit decision for an individual instance or scene element."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str = Field(description="Unique instance or region identifier, e.g. 'creator_0', 'background', 'logo_0'")
    action: EditAction = Field(description="The primary editing action to perform")
    reason: str = Field(description="Objective explanation for why this edit was chosen")
    target_category: TargetCategory = Field(default=TargetCategory.OTHER, description="Category of the target entity")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Deterministic numeric and geometric parameters for execution")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score for this decision")
    priority: int = Field(default=1, ge=1, le=10, description="Execution priority (1 = highest)")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ScoreBreakdown(BaseModel):
    """Objective 0-100 scores across all thumbnail visual dimensions."""
    model_config = ConfigDict(frozen=True)

    composition: float = Field(default=0.0, ge=0.0, le=100.0, description="Spatial layout, rule of thirds, balance score (0-100)")
    contrast: float = Field(default=0.0, ge=0.0, le=100.0, description="Luminance and color contrast separation (0-100)")
    subject_prominence: float = Field(default=0.0, ge=0.0, le=100.0, description="Visual dominance and scale of primary subject (0-100)")
    readability: float = Field(default=0.0, ge=0.0, le=100.0, description="Availability and contrast of uncluttered text safe zones (0-100)")
    visual_clutter: float = Field(default=0.0, ge=0.0, le=100.0, description="Cleanliness and lack of background noise (100 = minimal clutter)")
    background_quality: float = Field(default=0.0, ge=0.0, le=100.0, description="Background aesthetic and depth separation (0-100)")
    identity_preservation: float = Field(default=100.0, ge=0.0, le=100.0, description="Protection score for creator face and key brand assets (0-100)")
    text_placement: float = Field(default=0.0, ge=0.0, le=100.0, description="Optimal placement score for primary copy (0-100)")
    depth_usage: float = Field(default=0.0, ge=0.0, le=100.0, description="Foreground/background depth layering differentiation (0-100)")
    focus_hierarchy: float = Field(default=0.0, ge=0.0, le=100.0, description="Attention hierarchy clarity (0-100)")
    overall: float = Field(default=0.0, ge=0.0, le=100.0, description="Calibrated composite composition score (0-100)")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class CompositionAnalysis(BaseModel):
    """Quantitative composition metrics evaluated from the input scene."""
    model_config = ConfigDict(frozen=True)

    subject_scale: float = Field(description="Ratio of primary subject area to canvas area (0.0 - 1.0)")
    subject_position: Tuple[float, float] = Field(description="Normalized centroid coordinates (x, y) in [0.0, 1.0]")
    rule_of_thirds_alignment: float = Field(description="Proximity score to rule-of-thirds power points (0.0 - 1.0)")
    negative_space_ratio: float = Field(description="Fraction of canvas free from high saliency and subject clutter (0.0 - 1.0)")
    text_safe_zone_available: bool = Field(description="True if a compliant high-contrast safe zone is detected")
    text_safe_zones: List[Tuple[int, int, int, int]] = Field(default_factory=list, description="Candidate bounding boxes (xmin, ymin, xmax, ymax)")
    hierarchy_clarity: float = Field(description="Distinction between primary and secondary focal elements (0.0 - 1.0)")
    contrast_ratio: float = Field(description="Luminance contrast ratio between subject and background (>= 1.0)")
    visual_balance: float = Field(description="Horizontal and vertical visual mass equilibrium (0.0 - 1.0)")
    focus_score: float = Field(description="Prominence and isolation of the primary focal point (0.0 - 1.0)")
    attention_direction: str = Field(description="Directional flow of visual attention (e.g. 'left_to_right', 'center_outward')")
    color_harmony: str = Field(description="Color scheme classification (e.g. 'complementary', 'analogous', 'triadic')")
    ctr_improvement_potential: float = Field(description="Estimated headroom for CTR optimization (0.0 - 1.0)")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class CompositionDirectives(BaseModel):
    """Target composition specifications emitted for downstream rendering execution."""
    model_config = ConfigDict(frozen=True)

    target_subject_scale: float = Field(default=0.35, description="Recommended subject canvas area fraction")
    target_subject_position: Tuple[float, float] = Field(default=(0.7, 0.5), description="Target centroid coordinates (x, y)")
    rule_of_thirds_target: Tuple[float, float] = Field(default=(0.67, 0.5), description="Target power point")
    recommended_text_zone: Tuple[int, int, int, int] = Field(default=(40, 40, 600, 300), description="Primary text safe zone bbox")
    depth_layering_order: List[str] = Field(default_factory=list, description="Ordered layer IDs from back to front")
    lighting_direction: str = Field(default="top_left", description="Key light angle description")
    color_palette_target: List[str] = Field(default_factory=list, description="Target hex color codes")
    contrast_boost_factor: float = Field(default=1.15, description="Recommended contrast multiplier")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class EditPlanOutput(BaseModel):
    """The complete deterministic Edit Plan emitted by Phase 2."""
    model_config = ConfigDict(frozen=True)

    summary: str = Field(description="High-level strategic rationale explaining all proposed changes")
    composition_score: float = Field(ge=0.0, le=100.0, description="Overall baseline composition score (0-100)")
    target_composition_score: float = Field(ge=0.0, le=100.0, description="Projected composition score post-edits (0-100)")
    changes: List[ObjectEditChange] = Field(description="List of deterministic changes for detected objects and elements")
    scoring_breakdown: ScoreBreakdown = Field(description="Detailed 10-dimension objective score breakdown")
    composition_analysis: CompositionAnalysis = Field(description="Quantitative baseline composition metrics")
    composition_directives: CompositionDirectives = Field(description="Target directives for layer layout and relighting")
    locked_instances: List[str] = Field(default_factory=list, description="IDs of instances strictly locked from generative degradation")
    quality_targets: Dict[str, float] = Field(default_factory=dict, description="Target similarity and quality thresholds for validation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Contextual video metadata, archetype, and style parameters")

    def to_dict(self) -> Dict[str, Any]:
        """Convert EditPlanOutput to a plain Python dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize EditPlanOutput to a deterministic JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EditPlanOutput:
        """Create EditPlanOutput from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> EditPlanOutput:
        """Parse EditPlanOutput from a JSON string."""
        return cls.model_validate_json(json_str)

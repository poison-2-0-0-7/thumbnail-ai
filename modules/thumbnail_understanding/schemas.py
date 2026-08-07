"""
schemas.py
==========

Structured Pydantic data contracts for Thumbnail-AI V2 Thumbnail Understanding Engine.

Defines schemas for:
- Scene Elements (Phase 2, 4, 5)
- Scene Graph & Relationships (Phase 6, 7)
- Subject & Visual Hierarchy (Phase 3, 8)
- Composition Intelligence (Phase 9)
- Thumbnail Psychology Assessment (Phase 10)
- Weakness Analysis (Phase 11)
- Scene Decomposition & Editable Layers (Phase 12, 13, 14)
- AI Thumbnail Director & Professional Improvement Plan (Phase 15, 16)
- Top-level ThumbnailUnderstanding artifact (Phase 1)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from models import BoundingBox, GeminiReasoning


class ElementType(str, Enum):
    PERSON = "person"
    OBJECT = "object"
    TEXT = "text"
    LOGO = "logo"
    BACKGROUND = "background"
    PROP = "prop"
    EMOJI = "emoji"


class ElementRole(str, Enum):
    HERO = "hero"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUPPORTING = "supporting"
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    PROP = "prop"
    DISTRACTION = "distraction"
    DECORATIVE = "decorative"
    VISUAL_ANCHOR = "visual_anchor"


class EditabilityStatus(str, Enum):
    PRESERVE = "preserve"
    EDITABLE = "editable"
    REPLACE_ONLY = "replace_only"
    REMOVE_ONLY = "remove_only"


class SceneElement(BaseModel):
    """
    Structured representation of a single detected element in a thumbnail.
    Integrates object, face, text, and visual hierarchy properties.
    """

    model_config = ConfigDict(frozen=True)

    element_id: str
    element_type: ElementType
    category: str
    label: str
    semantic_description: str = ""
    bbox: BoundingBox
    polygon: list[tuple[float, float]] = Field(default_factory=list)
    mask_path: Optional[str] = None
    cutout_path: Optional[str] = None
    confidence: float = 1.0
    importance_rank: int = 1
    role: ElementRole = ElementRole.SUPPORTING
    preserve_score: float = 0.5
    replace_score: float = 0.0
    edit_priority: int = 1
    depth_level: float = 0.5  # 0.0 = frontmost, 1.0 = backmost
    occlusion_ratio: float = 0.0
    parent_id: Optional[str] = None
    children_ids: list[str] = Field(default_factory=list)
    identity_relevance: float = 0.0
    story_relevance: float = 0.5
    visual_relevance: float = 0.5
    editability: EditabilityStatus = EditabilityStatus.EDITABLE
    source_detector: str = "grounding_dino"
    provenance: str = ""

    # Face specific attributes (when element_type == ElementType.PERSON)
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    expression_intensity: float = 0.5
    head_pose: Optional[str] = None
    eye_direction: Optional[str] = None
    sharpness: float = 1.0
    lighting_quality: float = 0.5
    is_creator: bool = False

    @field_validator("element_id")
    @classmethod
    def id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("element_id must not be empty")
        return v.strip()


class SpatialRelation(str, Enum):
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    INSIDE = "inside"
    OVERLAPPING = "overlapping"
    OCCLUDING = "occluding"
    BEHIND = "behind"
    IN_FRONT_OF = "in_front_of"
    HOLDING = "holding"
    WEARING = "wearing"
    LOOKING_AT = "looking_at"
    INTERACTING_WITH = "interacting_with"
    TEXT_RELATIVE_TO = "text_relative_to"


class ElementRelationship(BaseModel):
    """Pairwise directional relationship between two scene elements."""

    model_config = ConfigDict(frozen=True)

    subject_element_id: str
    relation: SpatialRelation
    object_element_id: str
    confidence: float = 1.0
    spatial_direction: Optional[str] = None
    provenance: str = "spatial_reasoner"


class SceneGraph(BaseModel):
    """Graph structure containing all detected scene elements and their relationships."""

    model_config = ConfigDict(frozen=True)

    elements: list[SceneElement] = Field(default_factory=list)
    relationships: list[ElementRelationship] = Field(default_factory=list)
    hero_element_id: Optional[str] = None
    primary_subject_ids: list[str] = Field(default_factory=list)
    secondary_subject_ids: list[str] = Field(default_factory=list)


class VisualHierarchy(BaseModel):
    """Deterministic visual hierarchy and eye-flow analysis."""

    model_config = ConfigDict(frozen=True)

    reading_order: list[str] = Field(default_factory=list)
    first_attention_target: Optional[str] = None
    second_attention_target: Optional[str] = None
    dominant_subject_id: Optional[str] = None
    visual_anchors: list[str] = Field(default_factory=list)
    attention_competition_score: float = 0.0
    eye_flow_description: str = ""
    negative_space_ratio: float = 0.0
    text_safe_areas: list[BoundingBox] = Field(default_factory=list)
    visual_clutter_score: float = 0.0
    subject_separation_score: float = 0.0
    balance_score: float = 0.0
    focal_strength_score: float = 0.0
    hierarchy_basis: str = "size_contrast_position"


class CompositionIntelligence(BaseModel):
    """Actionable breakdown of compositional properties and visual structure."""

    model_config = ConfigDict(frozen=True)

    rule_of_thirds_score: float = 0.0
    subject_placement: str = "center"
    cropping_quality: float = 0.5
    balance: float = 0.5
    symmetry_asymmetry: str = "asymmetric"
    negative_space: float = 0.0
    depth_layers_count: int = 1
    camera_perspective: str = "medium_shot"
    subject_background_separation: float = 0.5
    contrast_distribution: float = 0.5
    lighting_distribution: float = 0.5
    visual_density: float = 0.5
    actionable_findings: list[str] = Field(default_factory=list)


class PsychologyDriver(BaseModel):
    """Individual psychological click driver assessment."""

    model_config = ConfigDict(frozen=True)

    driver: str
    strength: float = 0.5
    confidence: float = 0.8
    supporting_evidence: str = ""
    potential_improvement: str = ""
    associated_element_ids: list[str] = Field(default_factory=list)


class ThumbnailPsychologyAssessment(BaseModel):
    """Assessment of click drivers, curiosity gap, and emotional pull."""

    model_config = ConfigDict(frozen=True)

    ctr_potential_score: float = 0.5
    curiosity_gap_score: float = 0.5
    drivers: list[PsychologyDriver] = Field(default_factory=list)
    content_mismatch_detected: bool = False
    mismatch_explanation: Optional[str] = None
    overall_psychology_summary: str = ""


class WeaknessFinding(BaseModel):
    """Structured detection of a visual or narrative weakness in the thumbnail."""

    model_config = ConfigDict(frozen=True)

    weakness_type: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    confidence: float = 0.8
    evidence: str = ""
    affected_element_ids: list[str] = Field(default_factory=list)
    recommended_correction: str = ""


class WeaknessAnalysis(BaseModel):
    """Comprehensive collection of weaknesses detected in the thumbnail."""

    model_config = ConfigDict(frozen=True)

    findings: list[WeaknessFinding] = Field(default_factory=list)
    overall_risk_level: str = "medium"


class LayerCategory(str, Enum):
    BACKGROUND = "background"
    PRIMARY_SUBJECT = "primary_subject"
    SECONDARY_SUBJECT = "secondary_subject"
    FOREGROUND_OBJECT = "foreground_object"
    PROP = "prop"
    TEXT = "text"
    LOGO = "logo"
    EMOJI = "emoji"
    SHADOW = "shadow"
    LIGHTING = "lighting"
    DEPTH = "depth"


class SceneLayer(BaseModel):
    """Independent editable image layer decomposed from the source thumbnail."""

    model_config = ConfigDict(frozen=True)

    layer_id: str
    category: LayerCategory
    element_ref_id: Optional[str] = None
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    bounding_region: Optional[BoundingBox] = None
    polygon: list[tuple[float, float]] = Field(default_factory=list)
    depth_priority: int = 0  # 0 = background, higher = closer to camera
    relationships: list[str] = Field(default_factory=list)
    preservation_requirement: Literal["must_preserve", "preserve_if_possible", "replaceable", "removable"] = "replaceable"
    editability: str = "editable"
    source_provenance: str = ""


class DecomposedScene(BaseModel):
    """Complete multi-layer decomposition of the thumbnail frame."""

    model_config = ConfigDict(frozen=True)

    layers: list[SceneLayer] = Field(default_factory=list)
    background_plate_path: Optional[str] = None
    background_reconstructed: bool = False
    layer_count: int = 0


class AIThumbnailDirectorPlan(BaseModel):
    """Creative direction plan synthesized by the AI Director."""

    model_config = ConfigDict(frozen=True)

    creative_direction: str = ""
    story_analysis: str = ""
    elements_to_keep: list[str] = Field(default_factory=list)
    elements_to_change: list[str] = Field(default_factory=list)
    elements_to_remove: list[str] = Field(default_factory=list)
    elements_to_emphasize: list[str] = Field(default_factory=list)
    composition_strategy: str = ""
    redesign_aggressiveness: Literal["conservative", "moderate", "aggressive"] = "moderate"
    expected_ctr_improvement: str = ""


class ActionType(str, Enum):
    KEEP = "keep"
    REMOVE = "remove"
    REPLACE = "replace"
    ENHANCE = "enhance"
    REPOSITION = "reposition"
    RECOLOR = "recolor"
    RELIGHT = "relight"


class ImprovementAction(BaseModel):
    """Concrete structured action within the professional redesign plan."""

    model_config = ConfigDict(frozen=True)

    action_id: str
    action: ActionType
    target_element_id: str
    reason: str
    priority: int = 1
    confidence: float = 0.8
    expected_impact: str = ""
    expected_ctr_gain: float = 0.0
    identity_risk: Literal["low", "medium", "high"] = "low"
    visual_risk: Literal["low", "medium", "high"] = "low"
    dependencies: list[str] = Field(default_factory=list)
    preserve_requirements: list[str] = Field(default_factory=list)
    edit_method: str = "inpaint"
    fallback_action: str = "keep"


class ProfessionalImprovementPlan(BaseModel):
    """Ordered collection of improvement actions for execution."""

    model_config = ConfigDict(frozen=True)

    actions: list[ImprovementAction] = Field(default_factory=list)
    primary_goal: str = ""


class ThumbnailUnderstanding(BaseModel):
    """
    Top-level structured Thumbnail Understanding artifact (V2 Source of Truth).
    Completely describes scene, graph, hierarchy, composition, psychology,
    weaknesses, layers, director plan, and improvement plan.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    source_thumbnail_path: str
    scene_graph: SceneGraph
    hierarchy: VisualHierarchy
    composition: CompositionIntelligence
    psychology: ThumbnailPsychologyAssessment
    weaknesses: WeaknessAnalysis
    decomposed_scene: DecomposedScene
    director_plan: AIThumbnailDirectorPlan
    improvement_plan: ProfessionalImprovementPlan
    legacy_reasoning: Optional[GeminiReasoning] = None
    status: Literal["success", "partial", "error"] = "success"
    error_message: Optional[str] = None
    analyzed_at: str

    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()

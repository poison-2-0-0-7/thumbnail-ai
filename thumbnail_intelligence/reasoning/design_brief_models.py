"""
design_brief_models.py
======================

Strongly typed Pydantic data contracts for the DesignBrief Generator (Phase 3.5).
Defines the final, deterministic, renderer-independent DesignBrief emitted by the
Thumbnail Intelligence Engine.

The DesignBrief is the single creative input consumed downstream by:
- Renderer V2
- Execution Planner
- Layout Planner
- Future rendering engines

It contains ZERO renderer-specific parameters (no SD prompts, no ComfyUI nodes,
no BrushNet instructions, no SDXL parameters, no inpainting logic).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceReference,
    _utc_now_iso,
)


class BriefMetadata(BaseKBModel):
    """Metadata attributes for brief identification, versioning, and provenance."""

    brief_id: str = Field(
        default_factory=lambda: f"brief_{uuid.uuid4().hex[:8]}",
        description="Unique deterministic brief identifier",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Schema contract version for forward/backward compatibility",
    )
    created_at: str = Field(
        default_factory=_utc_now_iso,
        description="ISO 8601 UTC creation timestamp",
    )
    updated_at: str = Field(
        default_factory=_utc_now_iso,
        description="ISO 8601 UTC modification timestamp",
    )
    generator_id: str = Field(
        default="design_brief_generator_v1",
        description="Identifier of the emitting generator component",
    )


class NarrativeBrief(BaseKBModel):
    """Narrative and visual storytelling directions extracted from NarrativeReasoner."""

    primary_story: str = Field(
        default="",
        description="Core visual story premise and mystery hook",
    )
    supporting_story: str = Field(
        default="",
        description="Secondary narrative angle or context",
    )
    emotional_goal: str = Field(
        default="curiosity",
        description="Target emotional read (e.g., surprise, intrigue, awe, urgency)",
    )
    story_focus: str = Field(
        default="",
        description="Primary narrative element carries the story read",
    )
    narrative_type: str = Field(
        default="discovery",
        description="Classified narrative archetype (e.g. discovery, challenge, transformation)",
    )
    narrative_arc: str = Field(
        default="Mystery Arc",
        description="Dominant story arc progression stage",
    )


class AudienceBrief(BaseKBModel):
    """Target viewer intent, curiosity gaps, and cognitive friction guidelines."""

    primary_audience: str = Field(
        default="General Audience",
        description="Identified target viewer segment",
    )
    secondary_audience: List[str] = Field(
        default_factory=list,
        description="Secondary target segments or viewer sub-niches",
    )
    curiosity_trigger: str = Field(
        default="",
        description="Psychological curiosity gap or click motivation hook",
    )
    viewer_intent: str = Field(
        default="entertainment",
        description="Viewer motivation (e.g. learning, problem_solving, entertainment)",
    )
    cognitive_load: str = Field(
        default="medium",
        description="Optimal visual processing load (e.g. low, medium, high)",
    )


class CreatorBrief(BaseKBModel):
    """Creator identity, channel voice, and visual signature constraints."""

    creator_identity: str = Field(
        default="Default Creator",
        description="Channel handle, creator identity, or persona name",
    )
    style_constraints: List[str] = Field(
        default_factory=list,
        description="Signature elements and creator visual guidelines",
    )
    historical_consistency: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Historical brand consistency score across previous thumbnails",
    )
    creator_archetype: str = Field(
        default="educator",
        description="Persona archetype (e.g. educator, entertainer, authority)",
    )
    channel_voice: str = Field(
        default="",
        description="Channel tone of voice and visual identity summary",
    )


class BrandBrief(BaseKBModel):
    """Mandatory brand preservation rules, palettes, and prohibited elements."""

    required_elements: List[str] = Field(
        default_factory=list,
        description="Mandatory brand elements (e.g. logo, avatar, brand symbol)",
    )
    forbidden_changes: List[str] = Field(
        default_factory=list,
        description="Prohibited modifications or forbidden treatments",
    )
    brand_preservation_rules: List[str] = Field(
        default_factory=list,
        description="Explicit brand compliance directives",
    )
    brand_pillars: List[str] = Field(
        default_factory=list,
        description="Core brand positioning pillars",
    )


class CompositionBrief(BaseKBModel):
    """Visual hierarchy, focal points, canvas allocations, and spatial safe zones."""

    primary_subject: str = Field(
        default="Host Face",
        description="Dominant visual hero subject",
    )
    secondary_subject: str = Field(
        default="",
        description="Supporting visual element",
    )
    visual_hierarchy: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered hierarchy nodes with gaze order and canvas allocation",
    )
    negative_space: str = Field(
        default="balanced",
        description="Target negative space treatment (e.g. high_contrast_clean, uncluttered)",
    )
    safe_zones: List[str] = Field(
        default_factory=list,
        description="Reserved regions for UI overlays (e.g. timestamp_bottom_right, title_top_left)",
    )
    depth_treatment: str = Field(
        default="shallow_subject_forward",
        description="Visual depth and separation goal",
    )


class TypographyBrief(BaseKBModel):
    """Text priority, placement regions, character limits, and readability goals."""

    text_priority: str = Field(
        default="high",
        description="Relative visual priority of text overlay",
    )
    text_regions: List[str] = Field(
        default_factory=list,
        description="Target canvas regions for text placement",
    )
    maximum_characters: int = Field(
        default=25,
        ge=0,
        description="Maximum character budget to prevent feed clutter",
    )
    readability_targets: str = Field(
        default="high_mobile_contrast",
        description="Legibility target on small mobile feeds",
    )
    max_word_count: int = Field(
        default=4,
        ge=0,
        description="Maximum word count limit",
    )


class ColorBrief(BaseKBModel):
    """Color palette references, contrast targets, and brand color rules."""

    primary_palette: List[str] = Field(
        default_factory=list,
        description="Dominant color palette hex codes or color names",
    )
    accent_palette: List[str] = Field(
        default_factory=list,
        description="Accent and pop color codes",
    )
    contrast_targets: str = Field(
        default="high_luminance_separation",
        description="Required luminance contrast target (e.g. 4.5:1 ratio)",
    )
    brand_colors: List[str] = Field(
        default_factory=list,
        description="Official brand color palette",
    )


class LightingBrief(BaseKBModel):
    """Lighting mood, direction hints, and intensity goals."""

    mood: str = Field(
        default="high_key_dramatic",
        description="Visual lighting mood (e.g. cinematic, high_key, warm_studio)",
    )
    direction: str = Field(
        default="top_left_key_soft_rim",
        description="Directional hint for primary key light and rim lighting",
    )
    intensity: str = Field(
        default="punchy",
        description="Lighting contrast intensity (e.g. subtle, moderate, punchy)",
    )


class CameraBrief(BaseKBModel):
    """Subject scale, framing crop, perspective, and camera distance goals."""

    crop: str = Field(
        default="medium_close_up",
        description="Subject framing crop (e.g. extreme_close_up, close_up, medium_shot)",
    )
    perspective: str = Field(
        default="eye_level",
        description="Camera angle perspective (e.g. eye_level, low_angle, dynamic_three_quarter)",
    )
    zoom: str = Field(
        default="subject_focused",
        description="Lens zoom level hint",
    )
    subject_scale: str = Field(
        default="large_mobile_first",
        description="Relative subject scale on canvas",
    )


class ObjectsBrief(BaseKBModel):
    """Required, optional, and forbidden visual objects and props."""

    required_objects: List[str] = Field(
        default_factory=list,
        description="Mandatory visual objects that must be present",
    )
    optional_objects: List[str] = Field(
        default_factory=list,
        description="Supporting props or contextual objects",
    )
    forbidden_objects: List[str] = Field(
        default_factory=list,
        description="Strictly prohibited visual elements",
    )


class ExecutionConstraintsBrief(BaseKBModel):
    """Must preserve directives and allowed vs forbidden transformations."""

    must_preserve: List[str] = Field(
        default_factory=list,
        description="Identity and brand elements that must not be altered",
    )
    allowed_transformations: List[str] = Field(
        default_factory=list,
        description="Permissible visual enhancements and expressions",
    )
    forbidden_transformations: List[str] = Field(
        default_factory=list,
        description="Prohibited distortions, pose flips, or style shifts",
    )


class ValidationBrief(BaseKBModel):
    """Validation report summary, evidence grounding references, and readiness metrics."""

    strategy_id: str = Field(
        default="",
        description="ID of the winning strategy candidate",
    )
    evidence_references: List[EvidenceReference] = Field(
        default_factory=list,
        description="All evidence node references backing the brief",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Propagated decision confidence score",
    )
    validation_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Consistency score from StrategicReasoningValidator",
    )
    readiness_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Readiness score from StrategicReasoningValidator",
    )
    ready_for_design_brief: bool = Field(
        default=True,
        description="Boolean readiness flag",
    )
    detected_conflicts_count: int = Field(
        default=0,
        ge=0,
        description="Total cross-module conflicts detected during validation",
    )
    blocking_errors_count: int = Field(
        default=0,
        ge=0,
        description="Total blocking errors detected during validation",
    )


class DesignBrief(BaseKBModel):
    """
    Master DesignBrief data contract.
    The single, deterministic, strongly typed, renderer-independent creative contract
    produced by DesignBriefGenerator (Phase 3.5).
    """

    metadata: BriefMetadata = Field(
        default_factory=BriefMetadata,
        description="Brief identity, version, and UTC timestamps",
    )
    narrative: NarrativeBrief = Field(
        default_factory=NarrativeBrief,
        description="Story premise, emotional read, and narrative arc goals",
    )
    audience: AudienceBrief = Field(
        default_factory=AudienceBrief,
        description="Target viewer segment, intent, and curiosity triggers",
    )
    creator: CreatorBrief = Field(
        default_factory=CreatorBrief,
        description="Creator identity, style constraints, and brand equity rules",
    )
    brand: BrandBrief = Field(
        default_factory=BrandBrief,
        description="Required brand elements and prohibited changes",
    )
    composition: CompositionBrief = Field(
        default_factory=CompositionBrief,
        description="Visual hierarchy, primary/secondary subjects, and safe zones",
    )
    typography: TypographyBrief = Field(
        default_factory=TypographyBrief,
        description="Text overlay priorities, character limits, and legibility goals",
    )
    color: ColorBrief = Field(
        default_factory=ColorBrief,
        description="Primary and accent palettes, contrast targets, and brand colors",
    )
    lighting: LightingBrief = Field(
        default_factory=LightingBrief,
        description="Lighting mood, direction hints, and intensity",
    )
    camera: CameraBrief = Field(
        default_factory=CameraBrief,
        description="Crop framing, perspective angle, and subject scale",
    )
    objects: ObjectsBrief = Field(
        default_factory=ObjectsBrief,
        description="Required, optional, and forbidden visual objects",
    )
    execution_constraints: ExecutionConstraintsBrief = Field(
        default_factory=ExecutionConstraintsBrief,
        description="Must preserve directives and forbidden transformations",
    )
    validation: ValidationBrief = Field(
        default_factory=ValidationBrief,
        description="Audit metrics, evidence grounding references, and readiness scores",
    )

    # ---------------------------------------------------------------------------
    # Serialization Methods
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert DesignBrief to python dictionary representation."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DesignBrief:
        """Construct DesignBrief from dictionary representation."""
        return cls.model_validate(data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize DesignBrief to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> DesignBrief:
        """Deserialize DesignBrief from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        """Serialize DesignBrief to YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> DesignBrief:
        """Deserialize DesignBrief from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    # ---------------------------------------------------------------------------
    # Validation Method
    # ---------------------------------------------------------------------------

    def validate_brief(self) -> List[str]:
        """
        Validate completeness, strong typing, and evidence reference integrity.
        Returns a list of validation error descriptions (empty list if valid).
        """
        errors: List[str] = []

        if not self.metadata.brief_id:
            errors.append("Metadata brief_id must be non-empty.")
        if not self.metadata.schema_version:
            errors.append("Metadata schema_version must be specified.")

        if not self.narrative.primary_story:
            errors.append("Narrative primary_story is missing or empty.")

        if not self.composition.primary_subject:
            errors.append("Composition primary_subject is missing or empty.")

        if self.typography.maximum_characters < 0:
            errors.append("Typography maximum_characters cannot be negative.")

        if self.validation.confidence < 0.0 or self.validation.confidence > 1.0:
            errors.append("Validation confidence must be in range [0.0, 1.0].")

        return errors

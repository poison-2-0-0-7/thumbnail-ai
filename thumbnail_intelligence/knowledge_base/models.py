"""
models.py
=========

Structured Pydantic data contracts for the Thumbnail Intelligence Knowledge Base.
Defines foundational models for:
- KnowledgeEntry (unifying multi-modal knowledge record)
- CreatorProfile (cross-channel creator identity)
- ChannelProfile (per-channel visual & hook signatures)
- CompetitorProfile (competitor baseline and style signatures)
- ThumbnailPattern (composite compositional & lighting patterns)
- Archetype (named design templates with defining predicates)
- EvidenceReference (grounding reference for explainable intelligence)
- BrandConstraint (brand rules, palettes, and typography constraints)
- IdentityConstraint (face/instance locking and similarity preservation)
- VisualPattern (granular visual techniques and styling cues)
- DesignPattern (reusable visual & psychological patterns)
- DesignReason (grounded creative reasoning unit)
- ArchetypeMatch (audit-trail archetype classification result)
- DifferentiationSummary (competitive differentiation assessment)

Every model includes strict typing, schema validation, serialization, version support,
ISO-8601 UTC timestamps, and extensible metadata.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypeVar, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utc_now_iso() -> str:
    """Return current timestamp in ISO 8601 UTC format with explicit timezone."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class KnowledgeEntryType(str, Enum):
    """Categorization of entries stored in the unified Knowledge Base index."""

    ARCHETYPE_EXAMPLE = "archetype_example"
    HISTORICAL_THUMBNAIL = "historical_thumbnail"
    COMPETITOR_THUMBNAIL = "competitor_thumbnail"
    DESIGN_PATTERN = "design_pattern"
    VISUAL_PATTERN = "visual_pattern"
    THUMBNAIL_PATTERN = "thumbnail_pattern"
    CREATOR_PROFILE_ENTRY = "creator_profile_entry"


class EvidenceSourceType(str, Enum):
    """Source classification for auditable grounding evidence references."""

    SCENE_GRAPH_ELEMENT = "scene_graph_element"
    SCENE_GRAPH_RELATIONSHIP = "scene_graph_relationship"
    PSYCHOLOGY_DRIVER = "psychology_driver"
    KNOWLEDGE_ENTRY = "knowledge_entry"
    CREATOR_PROFILE_FIELD = "creator_profile_field"
    COMPETITOR_PROFILE_FIELD = "competitor_profile_field"
    ARCHETYPE_MATCH = "archetype_match"
    AUDIENCE_PATTERN = "audience_pattern"
    DESIGN_PATTERN = "design_pattern"
    VISUAL_PATTERN = "visual_pattern"
    OUTCOME_RECORD = "outcome_record"
    BRAND_RULE = "brand_rule"
    THUMBNAIL_STYLE_SIGNATURE = "thumbnail_style_signature"


class DesignReasonType(str, Enum):
    """Classification of creative and strategic reasoning justifications."""

    BRAND_CONSISTENCY = "brand_consistency"
    CTR_EVIDENCE = "ctr_evidence"
    COMPETITOR_DIFFERENTIATION = "competitor_differentiation"
    ARCHETYPE_ALIGNMENT = "archetype_alignment"
    AUDIENCE_PSYCHOLOGY = "audience_psychology"
    NARRATIVE_GROUNDING = "narrative_grounding"
    VISUAL_PATTERN_EVIDENCE = "visual_pattern_evidence"


class EvidenceGrade(str, Enum):
    """Confidence grade assigned to empirical evidence and observations."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    PATTERN_ONLY = "pattern_only"
    NONE = "none"


class CompetitorStatus(str, Enum):
    """Operational availability status of a tracked competitor channel."""

    ACTIVE = "active"
    UNAVAILABLE = "unavailable"
    STALE = "stale"


class PatternScope(str, Enum):
    """Domain scope for design and persuasion patterns."""

    AUDIENCE_PSYCHOLOGY = "audience_psychology"
    VISUAL_DESIGN = "visual_design"


T = TypeVar("T", bound="BaseKBModel")


# ---------------------------------------------------------------------------
# Base Knowledge Base Model
# ---------------------------------------------------------------------------


class BaseKBModel(BaseModel):
    """
    Abstract base model for all Knowledge Base data contracts.
    Enforces immutable frozen schemas, schema versioning, UTC timestamps, and metadata.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", validate_assignment=True)

    schema_version: str = Field(default="1.0.0", description="Semantic schema version of the model")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC creation timestamp")
    updated_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extension metadata")

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if not v or not isinstance(v, str) or "." not in v:
            raise ValueError(f"Invalid schema_version '{v}'. Expected semver format like '1.0.0'.")
        return v

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        if not v:
            return _utc_now_iso()
        try:
            # Validate ISO timestamp parseability
            datetime.fromisoformat(v)
        except Exception as e:
            raise ValueError(f"Invalid ISO-8601 timestamp '{v}': {e}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to a dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize model to a formatted JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        """Instantiate model from a dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls: type[T], json_str: str) -> T:
        """Instantiate model from a JSON string."""
        return cls.model_validate_json(json_str)


# ---------------------------------------------------------------------------
# Grounding & Reasoning Contracts
# ---------------------------------------------------------------------------


class EvidenceReference(BaseKBModel):
    """
    Direct pointer to concrete grounding evidence supporting a claim.
    Enforces the grounding gate requirement: every strategic recommendation
    must trace to an observable detection, historical outcome, or profile rule.
    """

    source_type: EvidenceSourceType = Field(description="Subsystem or entity type where evidence originates")
    source_id: str = Field(description="Unique identifier of the source entity (e.g. element_id, entry_id)")
    source_field: Optional[str] = Field(default=None, description="Specific field on the source entity being cited")
    excerpt_or_value: str = Field(default="", description="Literal quotation, numeric value, or summary of cited fact")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the validity of this evidence")

    @field_validator("source_id")
    @classmethod
    def validate_source_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source_id must not be empty")
        return v.strip()


class DesignReason(BaseKBModel):
    """
    Auditable design reason justifying a strategic creative decision.
    Enforces that the evidence list must be non-empty (§19.2 grounding gate).
    """

    reason_id: str = Field(description="Unique deterministic identifier for this design reason")
    claim: str = Field(description="Factual or strategic assertion (e.g. 'Increase hero face scale ratio')")
    reason_type: DesignReasonType = Field(description="Category of strategic justification")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score in [0.0, 1.0]")
    evidence: List[EvidenceReference] = Field(
        default_factory=list,
        min_length=1,
        description="Grounded evidence references supporting this claim (must have at least 1)",
    )
    target_element_id: Optional[str] = Field(default=None, description="Optional target scene element identifier")

    @field_validator("reason_id", "claim")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Knowledge Entry
# ---------------------------------------------------------------------------


class KnowledgeEntry(BaseKBModel):
    """
    Unified multimodal knowledge record indexed across the Knowledge Base.
    Used for archetype examples, historical thumbnail blueprints, competitor thumbnails,
    and design patterns, enabling uniform hybrid retrieval (§5.1).
    """

    entry_id: str = Field(description="Deterministic unique hash or identifier for this knowledge record")
    entry_type: KnowledgeEntryType = Field(description="Classification of this knowledge record")
    embedding: List[float] = Field(default_factory=list, description="512-dim OpenCLIP embedding vector")
    embedding_model: str = Field(default="OpenCLIP-ViT-B-32", description="Backbone model used to produce embedding")
    source_video_id: Optional[str] = Field(default=None, description="Video identifier for historical entries")
    source_channel_id: Optional[str] = Field(default=None, description="Channel identifier of origin")
    source_competitor_id: Optional[str] = Field(default=None, description="Competitor identifier for competitor entries")
    archetype_id: Optional[str] = Field(default=None, description="Associated archetype ID when classified")
    pattern_id: Optional[str] = Field(default=None, description="Associated design pattern ID when applicable")
    niche: str = Field(default="general", description="Content niche or topic domain")
    facets: Dict[str, Any] = Field(default_factory=dict, description="Structured queryable filter facets")
    outcome_ref: Optional[str] = Field(default=None, description="Reference to linked historical outcome record")
    superseded_by: Optional[str] = None

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("entry_id must not be empty")
        return v.strip()

    @field_validator("embedding")
    @classmethod
    def validate_embedding_floats(cls, v: List[float]) -> List[float]:
        for idx, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Embedding elements must be numeric floats. Found {type(val)} at index {idx}.")
        return v


# ---------------------------------------------------------------------------
# Creator & Channel Profiles
# ---------------------------------------------------------------------------


class CreatorProfile(BaseKBModel):
    """
    Cross-channel creator identity and high-level branding rules (§6.1).
    Aggregates multi-channel relationships for creators who manage multiple channels.
    """

    creator_id: str = Field(description="Stable unique creator identity")
    display_name: str = Field(description="Public display name or handle")
    channel_ids: List[str] = Field(default_factory=list, description="Associated channel identifiers (1..n)")
    primary_niche: str = Field(default="general", description="Primary niche domain of the creator")
    brand_rules: List[DesignReason] = Field(default_factory=list, description="Extracted stable brand rules")
    cross_channel_consistency_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Cross-channel style consistency score when 2+ channels exist"
    )

    @field_validator("creator_id", "display_name")
    @classmethod
    def validate_creator_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("creator_id and display_name must not be empty")
        return v.strip()


class ChannelProfile(BaseKBModel):
    """
    Per-channel visual profile and style signature wrapper (§6.2).
    Links directly to CreatorStyleEmbedding without duplicating vector storage.
    """

    channel_id: str = Field(description="YouTube channel identifier")
    creator_id: Optional[str] = Field(default=None, description="Parent creator identity link")
    niche: str = Field(default="general", description="Channel primary niche")
    style_embedding_ref: str = Field(description="Filepath or identifier reference to CreatorStyleEmbedding")
    profile_established: bool = Field(default=False, description="True if sample_count >= minimum sample threshold")
    sample_count: int = Field(default=0, ge=0, description="Number of observed thumbnail samples")
    archetype_affinity: Dict[str, float] = Field(default_factory=dict, description="archetype_id -> frequency in [0, 1]")
    dominant_hook_types: List[str] = Field(default_factory=list, description="Top observed hook types")
    brand_stability_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Brand stability [0, 1]")
    last_updated_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp of last profile update")

    @field_validator("channel_id")
    @classmethod
    def validate_channel_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("channel_id must not be empty")
        return v.strip()


class CompetitorProfile(BaseKBModel):
    """
    Competitor channel intelligence profile (§9.1).
    Maintains competitor style signature, dominant archetypes, and freshness status.
    """

    competitor_id: str = Field(description="Unique identifier for competitor channel")
    channel_name: str = Field(description="Name or title of competitor channel")
    niche: str = Field(default="general", description="Content niche of competitor")
    style_embedding: List[float] = Field(default_factory=list, description="512-dim centroid style embedding")
    dominant_archetypes: List[str] = Field(default_factory=list, description="Dominant archetype IDs observed")
    dominant_hook_types: List[str] = Field(default_factory=list, description="Dominant hook types used")
    color_palette_signature: List[str] = Field(default_factory=list, description="Hex color palette signature")
    text_density_avg: float = Field(default=0.0, ge=0.0, le=1.0, description="Average text area coverage fraction")
    sample_count: int = Field(default=0, ge=0, description="Number of ingested competitor thumbnails")
    status: CompetitorStatus = Field(default=CompetitorStatus.ACTIVE, description="Freshness status")
    last_ingested_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 timestamp of last snapshot")

    @field_validator("competitor_id", "channel_name")
    @classmethod
    def validate_competitor_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("competitor_id and channel_name must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------


class Archetype(BaseKBModel):
    """
    Named thumbnail archetype template definition (§7.1).
    Contains structural predicate rules, typical hooks, emotion, and centroid embedding.
    """

    archetype_id: str = Field(description="Deterministic ID (e.g. 'big_face_reaction', 'before_after_split')")
    name: str = Field(description="Human-readable archetype name")
    description: str = Field(description="Detailed description of archetype structure and purpose")
    defining_scene_graph_pattern: Dict[str, Any] = Field(
        default_factory=dict, description="Structured checkable predicates for structural matching"
    )
    typical_hook_types: List[str] = Field(default_factory=list, description="Hook types typically used")
    typical_emotion: Optional[str] = Field(default=None, description="Expected primary emotion")
    niches_observed_in: List[str] = Field(default_factory=list, description="Niches where archetype is prevalent")
    centroid_embedding: List[float] = Field(default_factory=list, description="OpenCLIP centroid vector")
    example_count: int = Field(default=0, ge=0, description="Number of curated and accumulated examples")
    version: str = Field(default="1.0.0", description="Semantic version of archetype definition")

    @field_validator("archetype_id", "name")
    @classmethod
    def validate_archetype_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("archetype_id and name must not be empty")
        return v.strip()


class ArchetypeMatch(BaseKBModel):
    """
    Explainable result of matching a thumbnail against the Archetype Library (§7.2).
    """

    video_id: str = Field(description="Analyzed video identifier")
    archetype_id: Optional[str] = Field(default=None, description="Matched archetype ID or None if generic")
    match_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Match confidence in [0.0, 1.0]")
    matched_via: Literal["embedding_similarity", "structural_predicate", "both", "none"] = Field(
        default="none", description="Method that established the match"
    )
    runner_up_archetype_ids: List[str] = Field(default_factory=list, description="Candidate runner-up archetype IDs")

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class BrandConstraint(BaseKBModel):
    """
    Brand constraints extracted from creator history and style profile (§10, renderer v2 §3.4).
    Constrains palette, typography, logo placement, and prohibited visual elements.
    """

    constraint_id: str = Field(description="Unique identifier for this brand constraint set")
    channel_id: Optional[str] = Field(default=None, description="Associated channel ID")
    creator_id: Optional[str] = Field(default=None, description="Associated creator ID")
    palette_ref: Optional[str] = Field(default=None, description="Palette reference or name")
    font_ref: Optional[str] = Field(default=None, description="Primary brand font reference")
    logo_placement: Optional[str] = Field(default=None, description="Required logo placement (e.g. 'bottom_right')")
    tone: Optional[str] = Field(default=None, description="Brand tone descriptor (e.g. 'high-energy, punchy')")
    prohibited_elements: List[str] = Field(default_factory=list, description="Forbidden visual assets or tropes")
    mandatory_elements: List[str] = Field(default_factory=list, description="Mandatory elements (e.g. logo, signature prop)")
    color_rules: Dict[str, Any] = Field(default_factory=dict, description="Specific color limits and requirements")
    typography_rules: Dict[str, Any] = Field(default_factory=dict, description="Word count and placement limits")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in constraint validity")
    evidence_refs: List[EvidenceReference] = Field(default_factory=list, description="Grounding evidence for brand rules")
    version: str = Field(default="1.0.0", description="Version of constraint definition")

    @field_validator("constraint_id")
    @classmethod
    def validate_constraint_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("constraint_id must not be empty")
        return v.strip()


class IdentityConstraint(BaseKBModel):
    """
    Identity protection constraints for creator face, body, and brand assets (renderer v2 §0, §3.4).
    Enforces pixel locking, pose preservation, and facial similarity thresholds.
    """

    constraint_id: str = Field(description="Unique identifier for identity constraint")
    creator_id: Optional[str] = Field(default=None, description="Creator ID")
    channel_id: Optional[str] = Field(default=None, description="Channel ID")
    locked_instances: List[str] = Field(
        default_factory=list, description="Instance identifiers that must NEVER pass through generative diffusion"
    )
    pose_change_allowed: bool = Field(default=False, description="Whether subject re-posing is permissible")
    expression_change_allowed: bool = Field(default=True, description="Whether facial expression adjustment is allowed")
    clothing_preservation_required: bool = Field(default=True, description="Whether clothing must be strictly preserved")
    face_similarity_threshold: float = Field(
        default=0.90, ge=0.0, le=1.0, description="Minimum InsightFace cosine similarity threshold"
    )
    evidence_refs: List[EvidenceReference] = Field(default_factory=list, description="Grounding references")
    version: str = Field(default="1.0.0", description="Version of identity constraint")

    @field_validator("constraint_id")
    @classmethod
    def validate_identity_constraint_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("constraint_id must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


class VisualPattern(BaseKBModel):
    """
    Granular visual design technique or aesthetic pattern (§15).
    Techniques like rim lighting, high-contrast vignette, split-screen contrast, or radial blur.
    """

    pattern_id: str = Field(description="Unique visual pattern identifier")
    name: str = Field(description="Human-readable name (e.g. 'rim_light_subject_edge')")
    description: str = Field(description="Technical and visual description of the technique")
    category: str = Field(default="composition", description="Category: 'lighting', 'contrast', 'layout', etc.")
    visual_techniques: List[str] = Field(default_factory=list, description="Specific graphics/lighting techniques")
    applicable_roles: List[str] = Field(default_factory=list, description="Applicable scene element roles")
    frequency_by_niche: Dict[str, float] = Field(default_factory=dict, description="Observed frequency per niche")
    centroid_embedding: List[float] = Field(default_factory=list, description="Visual centroid embedding")
    curated: bool = Field(default=True, description="True if hand-curated; False if frequency-mined candidate")
    evidence_grade: EvidenceGrade = Field(default=EvidenceGrade.PATTERN_ONLY, description="Empirical evidence grade")
    version: str = Field(default="1.0.0", description="Pattern version")

    @field_validator("pattern_id", "name")
    @classmethod
    def validate_pattern_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("pattern_id and name must not be empty")
        return v.strip()


class DesignPattern(BaseKBModel):
    """
    Reusable visual or psychological design pattern (§15.1).
    Distinguishes visual design patterns from audience psychology mechanisms.
    """

    pattern_id: str = Field(description="Unique pattern identifier")
    pattern_scope: PatternScope = Field(
        default=PatternScope.VISUAL_DESIGN, description="Scope: 'audience_psychology' or 'visual_design'"
    )
    name: str = Field(description="Pattern name (e.g. 'arrow_circle_callout')")
    description: str = Field(description="Detailed explanation of pattern utility")
    applicable_element_types: List[str] = Field(default_factory=list, description="Target element types")
    frequency_in_niche: Dict[str, float] = Field(default_factory=dict, description="Niche -> frequency in [0, 1]")
    curated: bool = Field(default=True, description="True if curated seed; False if proposed candidate")
    proposed_from_entry_ids: List[str] = Field(default_factory=list, description="Provenance entry IDs")
    version: str = Field(default="1.0.0", description="Pattern definition version")

    @field_validator("pattern_id", "name")
    @classmethod
    def validate_design_pattern_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("pattern_id and name must not be empty")
        return v.strip()


class ThumbnailPattern(BaseKBModel):
    """
    Composite thumbnail pattern integrating archetype, visual patterns, composition rules,
    and historical performance uplift data.
    """

    pattern_id: str = Field(description="Unique composite pattern ID")
    name: str = Field(description="Descriptive composite pattern name")
    description: str = Field(description="Comprehensive pattern description")
    archetype_id: Optional[str] = Field(default=None, description="Linked archetype ID")
    visual_pattern_ids: List[str] = Field(default_factory=list, description="Constituent visual pattern IDs")
    design_pattern_ids: List[str] = Field(default_factory=list, description="Constituent design pattern IDs")
    composition_rules: Dict[str, Any] = Field(default_factory=dict, description="Spatial and layout rules")
    lighting_rules: Dict[str, Any] = Field(default_factory=dict, description="Lighting and mood rules")
    typography_rules: Dict[str, Any] = Field(default_factory=dict, description="Typography styling rules")
    niche_affinity: Dict[str, float] = Field(default_factory=dict, description="Niche -> affinity score [0, 1]")
    historical_ctr_uplift_avg: Optional[float] = Field(default=None, description="Observed average CTR delta")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Pattern reliability confidence")
    evidence_refs: List[EvidenceReference] = Field(default_factory=list, description="Grounding evidence references")
    version: str = Field(default="1.0.0", description="Pattern definition version")

    @field_validator("pattern_id", "name")
    @classmethod
    def validate_thumbnail_pattern_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("pattern_id and name must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Differentiation & Competitive Intelligence
# ---------------------------------------------------------------------------


class DifferentiationSummary(BaseKBModel):
    """
    Structured differentiation summary contrasting a channel against its competitive set (§9.2).
    """

    channel_id: str = Field(description="Target channel ID")
    niche: str = Field(description="Content niche domain")
    competitor_ids_considered: List[str] = Field(default_factory=list, description="List of competitor IDs compared")
    shared_conventions: List[str] = Field(default_factory=list, description="Conventions shared with competitors")
    differentiating_factors: List[DesignReason] = Field(
        default_factory=list, description="Grounded differentiating advantages"
    )
    convergence_risk: Literal["low", "medium", "high"] = Field(
        default="low", description="Risk of visual indistinguishability"
    )
    computed_at: str = Field(default_factory=_utc_now_iso, description="ISO-8601 computation timestamp")

    @field_validator("channel_id", "niche")
    @classmethod
    def validate_diff_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("channel_id and niche must not be empty")
        return v.strip()

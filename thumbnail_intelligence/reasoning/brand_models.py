"""
brand_models.py
===============

Domain models, preservation directives, visual guardrails, and candidate brand interpretations
for the Brand Reasoning Engine (Phase 3.4D).
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import ConfigDict, Field

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceReference,
)
from thumbnail_intelligence.reasoning.models import BrandReasoningOutput


class BrandPreservationPriority(str, Enum):
    """
    Enforcement priority of visual brand elements in redesigns.
    """

    STRICT_MANDATORY = "strict_mandatory"
    HIGH_RECOMMENDED = "high_recommended"
    FLEXIBLE_ADVISORY = "flexible_advisory"


class VisualElementPreservation(BaseKBModel):
    """
    Grounded specification for a specific visual brand asset that must be preserved or constrained.
    """

    preservation_id: str = Field(
        default_factory=lambda: f"pres_{uuid.uuid4().hex[:8]}",
        description="Unique preservation directive identifier",
    )
    element_name: str = Field(description="Name or label of the visual brand element")
    element_type: str = Field(
        description="Classification (e.g. 'face', 'logo', 'color', 'typography', 'prop')",
    )
    preservation_priority: BrandPreservationPriority = Field(
        default=BrandPreservationPriority.HIGH_RECOMMENDED,
        description="Enforcement strictness level",
    )
    required_treatment: str = Field(
        description="Prescribed visual handling, scale, positioning, or lighting",
    )
    allowed_variation: str = Field(
        default="",
        description="Acceptable artistic or composition flexibility",
    )
    forbidden_change: str = Field(
        default="",
        description="Strictly prohibited modifications (e.g. facial distortion, color shifting)",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence for this preservation directive",
    )


class CandidateBrandInterpretation(BaseKBModel):
    """
    An individual candidate brand interpretation generated during multi-hypothesis exploration.
    """

    candidate_id: str = Field(
        default_factory=lambda: f"cand_br_{uuid.uuid4().hex[:8]}",
        description="Unique brand interpretation identifier",
    )
    interpretation_name: str = Field(description="Descriptive label for this brand angle")
    brand_pillars: List[str] = Field(
        default_factory=list,
        description="Core identity pillars supporting this interpretation",
    )
    fit_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Empirical fit score reflecting historical consistency and evidence",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Grounded confidence in this interpretation",
    )
    color_palette: List[str] = Field(
        default_factory=list,
        description="Associated hex color palette",
    )
    typography_preferences: str = Field(
        default="",
        description="Recommended font styling, stroke, and hierarchy",
    )
    recurring_subjects: List[str] = Field(
        default_factory=list,
        description="Key subjects or personas associated with this brand identity",
    )
    recurring_layout_patterns: List[str] = Field(
        default_factory=list,
        description="Recommended composition blueprints",
    )
    creator_signature_elements: List[str] = Field(
        default_factory=list,
        description="Signature elements emphasized in this interpretation",
    )
    required_preservations: List[str] = Field(
        default_factory=list,
        description="Mandatory preservation rules",
    )
    allowed_variations: List[str] = Field(
        default_factory=list,
        description="Permissible creative departures",
    )
    forbidden_changes: List[str] = Field(
        default_factory=list,
        description="Prohibited alterations",
    )
    pros: List[str] = Field(
        default_factory=list,
        description="Strategic advantages of adopting this brand interpretation",
    )
    cons: List[str] = Field(
        default_factory=list,
        description="Potential tradeoffs or limitations",
    )
    rejection_rationale: Optional[str] = Field(
        default=None,
        description="Audit explanation if this interpretation was evaluated and rejected",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Supporting grounding evidence references",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of evidence nodes backing this interpretation",
    )


class BrandResult(BrandReasoningOutput):
    """
    Master output artifact of the BrandReasoner.
    Inherits from BrandReasoningOutput for 100% backward compatibility with ReasoningContext.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    brand_identity: str = Field(
        default="",
        description="Core brand identity and positioning statement",
    )
    brand_pillars: List[str] = Field(
        default_factory=list,
        description="Foundational values and visual pillars of the channel brand",
    )
    visual_identity: Dict[str, Any] = Field(
        default_factory=dict,
        description="Comprehensive visual identity specs (palette, typography, lighting)",
    )
    logo_usage: str = Field(
        default="",
        description="Logo placement, sizing, and clearspace rules",
    )
    color_palette: List[str] = Field(
        default_factory=list,
        description="Mandatory and accent color palette hex codes",
    )
    typography_preferences: str = Field(
        default="",
        description="Typography guidelines, font weights, and text styling rules",
    )
    recurring_subjects: List[str] = Field(
        default_factory=list,
        description="Recurring characters, products, or props defining the brand",
    )
    recurring_layout_patterns: List[str] = Field(
        default_factory=list,
        description="Historical layout structures and composition templates",
    )
    creator_signature_elements: List[str] = Field(
        default_factory=list,
        description="Signature elements anchoring creator identity",
    )
    brand_constraints: List[str] = Field(
        default_factory=list,
        description="High-level brand constraints and guardrails",
    )
    required_preservations: List[VisualElementPreservation] = Field(
        default_factory=list,
        description="Grounded visual elements that must be preserved in any redesign",
    )
    allowed_variations: List[str] = Field(
        default_factory=list,
        description="Permissible creative variations for visual experimentation",
    )
    forbidden_changes: List[str] = Field(
        default_factory=list,
        description="Strictly prohibited visual changes",
    )
    primary_brand_interpretation: Optional[CandidateBrandInterpretation] = Field(
        default=None,
        description="Selected winning brand interpretation",
    )
    candidate_interpretations: List[CandidateBrandInterpretation] = Field(
        default_factory=list,
        description="All evaluated candidate brand interpretations",
    )
    rejected_interpretations: List[CandidateBrandInterpretation] = Field(
        default_factory=list,
        description="Evaluated alternative interpretations with rejection rationale",
    )
    selection_rationale: str = Field(
        default="",
        description="Explainable audit explanation why the primary brand interpretation was selected",
    )
    brand_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Propagated multi-signal brand confidence score",
    )
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Granular component confidence scores",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="All evidence node IDs backing brand conclusions",
    )

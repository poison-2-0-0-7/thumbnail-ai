"""
creator_models.py
=================

Domain models, visual identity contracts, and candidate persona hypotheses
for the Creator Reasoning Engine (Phase 3.4C).
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
from thumbnail_intelligence.reasoning.models import CreatorReasoningOutput


class CreatorArchetype(str, Enum):
    """
    Core creator archetype and persona style.
    """

    ENTERTAINER = "entertainer"
    EDUCATOR = "educator"
    CHALLENGER = "challenger"
    STORYTELLER = "storyteller"
    EXPERT_REVIEWER = "expert_reviewer"
    LIFESTYLE_VLOGGER = "lifestyle_vlogger"
    INVESTIGATOR = "investigator"
    COMMENTATOR = "commentator"
    CUSTOM = "custom"


class VisualIdentityStyle(BaseKBModel):
    """
    Extracted visual style signature and aesthetic constraints.
    """

    dominant_color_palette: List[str] = Field(
        default_factory=list,
        description="Signature hex color palette or dominant tone accents",
    )
    typography_style: str = Field(
        default="Bold sans-serif with high contrast drop shadow",
        description="Creator preferred thumbnail text styling",
    )
    face_framing_preference: str = Field(
        default="Close-up emotional hero framing on outer third",
        description="Preferred facial scale, expression intensity, and positioning",
    )
    lighting_preference: str = Field(
        default="High key with vibrant rim light separation",
        description="Preferred lighting setup and contrast treatment",
    )
    composition_rule: str = Field(
        default="Two-element split: high-emotion face + curiosity object",
        description="Dominant composition layout archetype",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence for visual identity preferences",
    )


class CandidateCreatorStyle(BaseKBModel):
    """
    An individual creator style interpretation generated during multi-hypothesis exploration.
    """

    candidate_id: str = Field(
        default_factory=lambda: f"cand_cre_{uuid.uuid4().hex[:8]}",
        description="Unique creator candidate identifier",
    )
    persona_name: str = Field(description="Descriptive persona and style angle")
    creator_archetype: CreatorArchetype = Field(description="Taxonomy classification of creator archetype")
    channel_voice: str = Field(description="Dominant tone and editorial voice")
    signature_elements: List[str] = Field(
        default_factory=list,
        description="Visual tropes, recurring props, or expressions",
    )
    brand_equity_anchors: List[str] = Field(
        default_factory=list,
        description="Visual assets anchoring channel brand recognition",
    )
    historical_thumbnail_style: str = Field(
        default="",
        description="Dominant thumbnail aesthetic observed in channel history",
    )
    fit_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Empirical alignment score with creator history and evidence",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Grounded confidence score in this creator interpretation",
    )
    visual_identity: Optional[VisualIdentityStyle] = Field(
        default=None,
        description="Structured visual identity directives",
    )
    pros: List[str] = Field(default_factory=list, description="Strengths of adopting this style interpretation")
    cons: List[str] = Field(default_factory=list, description="Potential constraints or risks")
    rejection_rationale: Optional[str] = Field(
        default=None,
        description="Audit explanation if this candidate was evaluated and rejected",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Supporting grounding evidence references",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of evidence nodes backing this interpretation",
    )


class CreatorResult(CreatorReasoningOutput):
    """
    Master output artifact of the CreatorReasoner.
    Inherits from CreatorReasoningOutput for 100% backward compatibility with ReasoningContext.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    creator_identity: str = Field(
        default="",
        description="Identified or inferred creator name / handle / channel identity",
    )
    creator_style: str = Field(
        default="",
        description="Comprehensive summary of creator visual and narrative style",
    )
    creator_brand: str = Field(
        default="",
        description="Core brand positioning and channel brand equity summary",
    )
    visual_identity: Optional[VisualIdentityStyle] = Field(
        default=None,
        description="Structured visual design rules and layout guidelines",
    )
    historical_thumbnail_style: str = Field(
        default="",
        description="Historical thumbnail visual blueprint and layout pattern",
    )
    historical_content_patterns: List[str] = Field(
        default_factory=list,
        description="Observed historical content topics, hook formats, and visual structures",
    )
    brand_consistency: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Historical brand consistency score across previous thumbnails",
    )
    visual_constraints: List[str] = Field(
        default_factory=list,
        description="Visual guardrails (e.g. prohibited colors, mandatory face presence)",
    )
    creator_preferences: List[str] = Field(
        default_factory=list,
        description="Documented or inferred creator creative preferences",
    )
    creator_strengths: List[str] = Field(
        default_factory=list,
        description="Creator key visual assets (e.g. highly expressive face, recognisable branding)",
    )
    creator_weaknesses: List[str] = Field(
        default_factory=list,
        description="Potential visual pitfalls to avoid (e.g. cluttered backgrounds, small typography)",
    )
    primary_creator_style: Optional[CandidateCreatorStyle] = Field(
        default=None,
        description="Selected winning creator style interpretation",
    )
    candidate_creator_styles: List[CandidateCreatorStyle] = Field(
        default_factory=list,
        description="Candidate style interpretations evaluated",
    )
    rejected_interpretations: List[CandidateCreatorStyle] = Field(
        default_factory=list,
        description="Evaluated alternative creator interpretations with rejection rationale",
    )
    selection_rationale: str = Field(
        default="",
        description="Explainable audit explanation why the primary creator interpretation was selected",
    )
    creator_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Propagated multi-signal creator confidence score",
    )
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Granular component confidence scores",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="All evidence node IDs backing creator conclusions",
    )

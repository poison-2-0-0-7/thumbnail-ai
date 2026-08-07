"""
narrative_models.py
===================

Domain models and extensible taxonomies for the Narrative Reasoning Engine (Phase 3.4B).
Defines candidate narratives, narrative types, narrative arc stages, visual focus candidates,
and the final grounded NarrativeResult.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import ConfigDict, Field, field_validator

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceReference,
    _utc_now_iso,
)
from thumbnail_intelligence.reasoning.models import NarrativeReasoningOutput


class NarrativeType(str, Enum):
    """
    Extensible taxonomy of video storytelling formats and narrative genres.
    """

    DISCOVERY = "discovery"
    CHALLENGE = "challenge"
    TRANSFORMATION = "transformation"
    TUTORIAL = "tutorial"
    REACTION = "reaction"
    COMPARISON = "comparison"
    REVIEW = "review"
    DOCUMENTARY = "documentary"
    COMPETITION = "competition"
    COMEDY = "comedy"
    STORYTELLING = "storytelling"
    EDUCATIONAL = "educational"
    VLOG = "vlog"
    INTERVIEW = "interview"
    NEWS = "news"
    CUSTOM = "custom"


class ArcStage(str, Enum):
    """
    Key narrative and psychological progression stages across a video's storyline.
    """

    BEGINNING = "beginning"
    CONFLICT = "conflict"
    PEAK = "peak"
    RESOLUTION = "resolution"
    MYSTERY = "mystery"
    REWARD = "reward"
    EXPECTATION = "expectation"
    SURPRISE = "surprise"
    FAILURE = "failure"
    SUCCESS = "success"


class ArcStep(BaseKBModel):
    """
    An individual grounded stage in the narrative progression arc.
    """

    step_id: str = Field(
        default_factory=lambda: f"arc_step_{uuid.uuid4().hex[:8]}",
        description="Unique arc step identifier",
    )
    stage: ArcStage = Field(description="Progression classification of this step")
    description: str = Field(description="Natural language summary of what occurs in this stage")
    emotional_intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated emotional arousal/intensity score in [0.0, 1.0]",
    )
    visual_cue: str = Field(
        default="",
        description="Concrete visual representation or focal imagery associated with this stage",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of supporting evidence nodes",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence references for this arc step",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Empirical confidence score in [0.0, 1.0]",
    )


class NarrativeArc(BaseKBModel):
    """
    Complete chronological and psychological narrative arc inferred for the video.
    """

    arc_name: str = Field(
        default="",
        description="Descriptive archetype name for this narrative arc",
    )
    primary_driver: str = Field(
        default="curiosity",
        description="Core psychological driver (e.g. 'curiosity', 'tension', 'surprise')",
    )
    stages: List[ArcStep] = Field(
        default_factory=list,
        description="Ordered sequence of grounded progression stages",
    )
    dominant_stage: ArcStage = Field(
        default=ArcStage.PEAK,
        description="The most visually compelling stage recommended for thumbnail depiction",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in the inferred narrative arc",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding references backing the overall arc",
    )


class VisualFocusCandidate(BaseKBModel):
    """
    A proposed visual focal point that should remain central in thumbnail redesign.
    """

    focus_id: str = Field(
        default_factory=lambda: f"foc_{uuid.uuid4().hex[:8]}",
        description="Unique visual focus candidate identifier",
    )
    element_name: str = Field(description="Name or label of the visual focal element")
    role_in_narrative: str = Field(
        description="Narrative role (e.g. 'Subject experiencing peak reaction', 'Core mystery object')",
    )
    visual_priority: Literal["PRIMARY", "SECONDARY", "TERTIARY"] = Field(
        default="PRIMARY",
        description="Recommended hierarchy ranking in visual composition",
    )
    recommended_treatment: str = Field(
        default="",
        description="Stylistic lighting, contrast, or framing recommendations",
    )
    source_node_id: Optional[str] = Field(
        default=None,
        description="Identifier of source evidence node if grounded directly",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in this visual focus recommendation",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Supporting evidence references",
    )


class CandidateNarrative(BaseKBModel):
    """
    An individual candidate narrative hypothesis generated during multi-hypothesis exploration.
    """

    candidate_id: str = Field(
        default_factory=lambda: f"cand_nar_{uuid.uuid4().hex[:8]}",
        description="Unique candidate identifier",
    )
    title: str = Field(description="Short descriptive title for this narrative angle")
    narrative_type: NarrativeType = Field(description="Taxonomy classification of this candidate")
    premise: str = Field(description="Underlying storyline premise or core theme")
    hook: str = Field(description="Attention-capturing viewer curiosity hook")
    emotional_tone: str = Field(default="", description="Dominant emotional resonance")
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Calculated composite fit score balancing evidence and archetype synergy",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Grounded confidence in this candidate narrative",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Supporting grounding evidence references",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of evidence nodes backing this hypothesis",
    )
    pros: List[str] = Field(default_factory=list, description="Strategic advantages of this narrative")
    cons: List[str] = Field(default_factory=list, description="Potential limitations or drawbacks")
    rejection_rationale: Optional[str] = Field(
        default=None,
        description="Explainable rationale if this candidate was evaluated and rejected",
    )


class NarrativeResult(NarrativeReasoningOutput):
    """
    Master output artifact of the NarrativeReasoner.
    Inherits from NarrativeReasoningOutput to preserve full backward compatibility with ReasoningContext.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    primary_narrative: Optional[CandidateNarrative] = Field(
        default=None,
        description="Selected winning candidate narrative hypothesis",
    )
    supporting_narratives: List[CandidateNarrative] = Field(
        default_factory=list,
        description="Secondary viable supporting narrative angles",
    )
    narrative_type: NarrativeType = Field(
        default=NarrativeType.DISCOVERY,
        description="Primary narrative format classification",
    )
    narrative_arc: Optional[NarrativeArc] = Field(
        default=None,
        description="Inferred multi-stage storyline progression arc",
    )
    story_summary: str = Field(
        default="",
        description="Comprehensive narrative digest synthesizing video storyline",
    )
    key_subjects: List[str] = Field(
        default_factory=list,
        description="Principal characters, creators, or subjects identified in the narrative",
    )
    key_events: List[str] = Field(
        default_factory=list,
        description="Key plot points, turning points, or actions identified in the narrative",
    )
    visual_focus_candidates: List[VisualFocusCandidate] = Field(
        default_factory=list,
        description="Focal points that should remain central in the redesign",
    )
    narrative_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Propagated multi-signal narrative confidence score",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="All evidence node IDs backing the primary and supporting narrative claims",
    )
    rejected_alternatives: List[CandidateNarrative] = Field(
        default_factory=list,
        description="Evaluated alternative hypotheses with explicit rejection explanations",
    )
    selection_rationale: str = Field(
        default="",
        description="Explainable audit explanation why the primary narrative was selected",
    )
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Granular component confidence scores (evidence, transcript, ocr, scene, agreement, penalty)",
    )

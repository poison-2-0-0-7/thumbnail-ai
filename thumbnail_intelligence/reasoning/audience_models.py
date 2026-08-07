"""
audience_models.py
==================

Domain models, viewer personas, cognitive load taxonomies, and candidate hypotheses
for the Audience Reasoning Engine (Phase 3.4C).
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
from thumbnail_intelligence.reasoning.models import AudienceReasoningOutput


class ViewerIntent(str, Enum):
    """
    Primary viewer intent driving click decisions.
    """

    ENTERTAINMENT = "entertainment"
    LEARNING = "learning"
    PROBLEM_SOLVING = "problem_solving"
    CURIOSITY_SEEKING = "curiosity_seeking"
    INSPIRATION = "inspiration"
    ESCAPISM = "escapism"
    PURCHASE_DECISION = "purchase_decision"
    NEWS_UPDATE = "news_update"
    CUSTOM = "custom"


class ViewerKnowledgeLevel(str, Enum):
    """
    Viewer familiarity and background knowledge level.
    """

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    GENERAL = "general"


class CognitiveLoadLevel(str, Enum):
    """
    Optimal thumbnail visual complexity and cognitive processing load.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ViewerPersona(BaseKBModel):
    """
    Concrete archetypal viewer persona representing a key segment.
    """

    persona_id: str = Field(
        default_factory=lambda: f"per_{uuid.uuid4().hex[:8]}",
        description="Unique persona identifier",
    )
    name: str = Field(description="Descriptive persona label (e.g. 'Curious Casual Scroller')")
    demographics_summary: str = Field(description="Target audience demographic profile")
    core_interest: str = Field(description="Primary topic or hook this persona cares about")
    click_trigger: str = Field(description="Visual or emotional stimulus that motivates clicking")
    skepticism_level: str = Field(
        default="medium",
        description="Viewer skepticism level ('low', 'medium', 'high')",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence references for this persona",
    )


class CandidateAudience(BaseKBModel):
    """
    An individual audience segment hypothesis generated during multi-hypothesis exploration.
    """

    candidate_id: str = Field(
        default_factory=lambda: f"cand_aud_{uuid.uuid4().hex[:8]}",
        description="Unique audience candidate identifier",
    )
    audience_segment: str = Field(description="Target audience segment label")
    intent: ViewerIntent = Field(description="Primary intent of this audience segment")
    knowledge_level: ViewerKnowledgeLevel = Field(
        default=ViewerKnowledgeLevel.GENERAL,
        description="Expected domain knowledge level",
    )
    cognitive_load: CognitiveLoadLevel = Field(
        default=CognitiveLoadLevel.MEDIUM,
        description="Optimal cognitive processing load",
    )
    fit_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Empirical fit score of this audience segment",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Grounded confidence score",
    )
    curiosity_triggers: List[str] = Field(
        default_factory=list,
        description="Curiosity gap mechanisms appealing to this segment",
    )
    psychological_hooks: List[str] = Field(
        default_factory=list,
        description="Psychological triggers capturing attention",
    )
    emotional_drivers: List[str] = Field(
        default_factory=list,
        description="Emotional motivators",
    )
    pain_points: List[str] = Field(
        default_factory=list,
        description="Viewer frustrations or pain points addressed",
    )
    reward_expectations: List[str] = Field(
        default_factory=list,
        description="Expected payoff from watching the video",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Supporting grounding evidence references",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of evidence nodes backing this audience hypothesis",
    )
    pros: List[str] = Field(default_factory=list, description="Strategic advantages of targeting this audience")
    cons: List[str] = Field(default_factory=list, description="Potential trade-offs or limitations")
    rejection_rationale: Optional[str] = Field(
        default=None,
        description="Audit explanation if this candidate was evaluated and rejected",
    )


class AudienceResult(AudienceReasoningOutput):
    """
    Master output artifact of the AudienceReasoner.
    Inherits from AudienceReasoningOutput for 100% backward compatibility with ReasoningContext.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    primary_audience: Optional[CandidateAudience] = Field(
        default=None,
        description="Selected winning audience hypothesis",
    )
    secondary_audiences: List[CandidateAudience] = Field(
        default_factory=list,
        description="Secondary viable audience segments",
    )
    viewer_intent: ViewerIntent = Field(
        default=ViewerIntent.ENTERTAINMENT,
        description="Primary viewer motivation",
    )
    viewer_knowledge_level: ViewerKnowledgeLevel = Field(
        default=ViewerKnowledgeLevel.GENERAL,
        description="Estimated audience domain sophistication",
    )
    viewer_motivation: str = Field(
        default="",
        description="Core underlying motivation for clicking",
    )
    viewer_emotional_drivers: List[str] = Field(
        default_factory=list,
        description="Key emotional states driving viewer action",
    )
    viewer_pain_points: List[str] = Field(
        default_factory=list,
        description="Viewer frustrations resolved by video content",
    )
    viewer_reward_expectations: List[str] = Field(
        default_factory=list,
        description="Anticipated viewer payoff (e.g. answer to mystery, emotional climax)",
    )
    viewer_personas: List[ViewerPersona] = Field(
        default_factory=list,
        description="Archetypal personas representing the target audience",
    )
    rejected_audiences: List[CandidateAudience] = Field(
        default_factory=list,
        description="Evaluated alternative audience hypotheses with rejection rationale",
    )
    selection_rationale: str = Field(
        default="",
        description="Explainable audit explanation why the primary audience was selected",
    )
    audience_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Propagated multi-signal audience confidence score",
    )
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Granular component confidence scores",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="All evidence node IDs backing audience conclusions",
    )

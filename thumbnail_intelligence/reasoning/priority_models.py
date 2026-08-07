"""
priority_models.py
==================

Domain models, visual hierarchy levels, attention flow models, and candidate hierarchies
for the Priority Reasoning Engine (Phase 3.4E).
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
from thumbnail_intelligence.reasoning.models import PriorityReasoningOutput


class HierarchyTier(str, Enum):
    """
    Tier level in the visual hierarchy determining gaze order and dominance.
    """

    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    SUPPRESSED = "suppressed"


class ElementPriorityLevel(str, Enum):
    """
    Strategic visual emphasis priority for specific asset categories.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class BackgroundPriority(str, Enum):
    """
    Background environment visual prominence.
    """

    CONTEXTUAL = "contextual"
    MINIMAL = "minimal"
    LOW = "low"
    MUTED = "muted"


class VisualHierarchyNode(BaseKBModel):
    """
    An individual element node in the grounded visual hierarchy.
    """

    node_id: str = Field(
        default_factory=lambda: f"hnode_{uuid.uuid4().hex[:8]}",
        description="Unique hierarchy node identifier",
    )
    element_name: str = Field(description="Name or label of the visual element")
    element_category: str = Field(
        description="Category (e.g. 'face', 'object', 'text', 'background', 'logo', 'graphic')",
    )
    tier: HierarchyTier = Field(
        default=HierarchyTier.PRIMARY,
        description="Assigned hierarchy tier",
    )
    importance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relative importance score in [0.0, 1.0]",
    )
    attention_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Estimated share of viewer initial attention in [0.0, 1.0]",
    )
    canvas_allocation_fraction: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Target canvas area fraction allocation in [0.0, 1.0]",
    )
    contrast_requirement: str = Field(
        default="Minimum 4.5:1 luminance separation with warm rim lighting",
        description="Contrast and edge separation directives",
    )
    gaze_order: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Estimated viewer fixation sequence (1 = first gaze attractor)",
    )
    non_compete_with: List[str] = Field(
        default_factory=list,
        description="Elements that must not compete or overlap with this node",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence for this hierarchy node",
    )


class AttentionFlowStep(BaseKBModel):
    """
    Sequential gaze trajectory step across the thumbnail canvas.
    """

    step_order: int = Field(ge=1, le=5, description="1-indexed fixation step")
    target_element: str = Field(description="Visual element capturing gaze at this step")
    visual_cue: str = Field(description="Salient visual trigger causing fixation")
    psychological_driver: str = Field(description="Cognitive trigger reinforcing comprehension")
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Supporting grounding evidence references",
    )


class CandidateHierarchy(BaseKBModel):
    """
    An individual candidate visual hierarchy interpretation evaluated during multi-hypothesis exploration.
    """

    candidate_id: str = Field(
        default_factory=lambda: f"cand_hier_{uuid.uuid4().hex[:8]}",
        description="Unique candidate hierarchy identifier",
    )
    hierarchy_name: str = Field(description="Descriptive name for this hierarchy strategy")
    primary_focus: str = Field(description="Primary dominant focal point")
    secondary_focus: str = Field(description="Secondary supporting story element")
    tertiary_focus: str = Field(description="Tertiary contextual background / headline text")
    fit_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Empirical fit score balancing narrative, audience, and brand synergy",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Grounded confidence in this hierarchy candidate",
    )
    attention_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized attention weights across elements",
    )
    canvas_allocations: Dict[str, float] = Field(
        default_factory=dict,
        description="Target canvas area fractions per element",
    )
    pros: List[str] = Field(default_factory=list, description="Advantages of this hierarchy")
    cons: List[str] = Field(default_factory=list, description="Tradeoffs or potential risks")
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
        description="IDs of evidence nodes backing this hypothesis",
    )


class PriorityResult(PriorityReasoningOutput):
    """
    Master output artifact of the PriorityReasoner.
    Inherits from PriorityReasoningOutput for 100% backward compatibility with ReasoningContext.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    primary_subject: str = Field(
        default="",
        description="Element receiving primary visual prominence and first viewer fixation",
    )
    secondary_subject: str = Field(
        default="",
        description="Supporting story hook or contrast item receiving second fixation",
    )
    supporting_subjects: List[str] = Field(
        default_factory=list,
        description="Tertiary or ambient elements providing context",
    )
    visual_hierarchy: List[VisualHierarchyNode] = Field(
        default_factory=list,
        description="Full ordered list of visual hierarchy nodes with weights and allocations",
    )
    importance_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Relative importance score per visual element",
    )
    attention_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Estimated gaze attention distribution summing to ~1.0",
    )
    canvas_allocation: Dict[str, float] = Field(
        default_factory=dict,
        description="Target canvas area fractions per element category",
    )
    text_priority: ElementPriorityLevel = Field(
        default=ElementPriorityLevel.MEDIUM,
        description="Visual emphasis priority for thumbnail text overlays",
    )
    face_priority: ElementPriorityLevel = Field(
        default=ElementPriorityLevel.HIGH,
        description="Visual emphasis priority for creator/character faces",
    )
    object_priority: ElementPriorityLevel = Field(
        default=ElementPriorityLevel.HIGH,
        description="Visual emphasis priority for story props or mystery objects",
    )
    background_priority: BackgroundPriority = Field(
        default=BackgroundPriority.MUTED,
        description="Visual prominence of the background environment",
    )
    color_importance: Dict[str, float] = Field(
        default_factory=dict,
        description="Relative contrast and saliency weights per key color",
    )
    contrast_priority: List[str] = Field(
        default_factory=list,
        description="Critical luminance and color contrast boundaries",
    )
    required_emphasis: List[str] = Field(
        default_factory=list,
        description="Directives describing what visual aspects must be emphasized",
    )
    suppressed_elements: List[str] = Field(
        default_factory=list,
        description="Elements that should be dimmed, blurred, or removed to avoid competition",
    )
    attention_flow: List[AttentionFlowStep] = Field(
        default_factory=list,
        description="Sequential 1-2-3 gaze trajectory across the canvas",
    )
    max_focal_points: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Maximum allowed competing visual focal points",
    )
    non_compete_rules: List[str] = Field(
        default_factory=list,
        description="Explicit rules preventing competing visual dominance",
    )
    primary_hierarchy_candidate: Optional[CandidateHierarchy] = Field(
        default=None,
        description="Selected winning visual hierarchy candidate",
    )
    candidate_hierarchies: List[CandidateHierarchy] = Field(
        default_factory=list,
        description="All evaluated candidate hierarchy interpretations",
    )
    rejected_hierarchies: List[CandidateHierarchy] = Field(
        default_factory=list,
        description="Evaluated alternative hierarchies with rejection rationale",
    )
    selection_rationale: str = Field(
        default="",
        description="Explainable audit explanation why the primary hierarchy was chosen",
    )
    priority_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Propagated multi-signal visual priority confidence score",
    )
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Granular component confidence scores",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="All evidence node IDs backing priority conclusions",
    )

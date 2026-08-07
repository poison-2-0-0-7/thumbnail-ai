"""
models.py
=========

Core domain models for the Strategic Reasoning Coordinator Foundation.
Defines strongly typed reasoning contracts, trace steps, decision tree representations,
reasoner output contracts, risk models, and ranked strategy representations.
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


class ReasonerType(str, Enum):
    """Classification of strategic reasoning modules."""

    NARRATIVE = "narrative"
    AUDIENCE = "audience"
    CREATOR = "creator"
    BRAND = "brand"
    PRIORITY = "priority"
    RISK = "risk"
    STRATEGY_RANKER = "strategy_ranker"
    VALIDATOR = "validator"
    DESIGN_BRIEF_GENERATOR = "design_brief_generator"
    EXECUTION_PLANNER = "execution_planner"
    SPATIAL_COMPOSITION_PLANNER = "spatial_composition_planner"
    RENDERER_ADAPTER = "renderer_adapter"
    CUSTOM = "custom"



class ReasonerContract(BaseKBModel):
    """
    Metadata contract defining a reasoner's identity, type, dependencies, and execution bounds.
    """

    name: str = Field(description="Unique reasoner identifier")
    reasoner_type: ReasonerType = Field(description="Classification of this reasoner")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Names of reasoners that must execute before this reasoner",
    )
    version: str = Field(default="1.0.0", description="Semantic version string")
    description: str = Field(default="", description="Human-readable description of reasoner purpose")
    is_mandatory: bool = Field(default=True, description="Whether coordinator halts if this reasoner fails")
    timeout_ms: float = Field(default=5000.0, gt=0.0, description="Max execution duration in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary configuration and flags")

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Reasoner name must not be empty")
        return v.strip()


class ReasoningTraceStep(BaseKBModel):
    """
    Immutable audit step recording the execution lifecycle of a single reasoner.
    """

    step_id: str = Field(
        default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}",
        description="Unique trace step identifier",
    )
    reasoner_name: str = Field(description="Name of the reasoner executed")
    action: str = Field(description="Action performed (e.g. 'execute', 'validate', 'merge')")
    status: Literal["SUCCESS", "FAILED", "SKIPPED", "VALIDATION_FAILED"] = Field(
        default="SUCCESS",
        description="Outcome of the trace step",
    )
    duration_ms: float = Field(default=0.0, ge=0.0, description="Execution duration in milliseconds")
    evidence_count: int = Field(default=0, ge=0, description="Number of evidence references produced")
    details: str = Field(default="", description="Diagnostic message, error, or explanation")
    timestamp: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp")


class ReasoningRisk(BaseKBModel):
    """
    Structured visual, audience, or brand risk assessment.
    """

    risk_id: str = Field(
        default_factory=lambda: f"risk_{uuid.uuid4().hex[:8]}",
        description="Unique risk identifier",
    )
    risk_type: str = Field(description="Category of risk (e.g. 'fatigue', 'convergence', 'clickbait')")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        default="MEDIUM",
        description="Severity level of the risk",
    )
    description: str = Field(description="Detailed explanation of the risk mechanism")
    mitigation: str = Field(default="", description="Actionable mitigation recommendation")
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence supporting this risk claim",
    )


class RankedStrategy(BaseKBModel):
    """
    Individual candidate thumbnail design strategy ranked by expected impact and evidence backing.
    """

    strategy_id: str = Field(
        default_factory=lambda: f"strat_{uuid.uuid4().hex[:8]}",
        description="Unique strategy identifier",
    )
    title: str = Field(description="Concise strategy title")
    description: str = Field(description="Strategic rationale and execution details")
    archetype_id: Optional[str] = Field(default=None, description="Associated archetype ID if grounded")
    expected_ctr_impact: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Expected CTR uplift score in [0.0, 1.0]",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Empirical confidence score in [0.0, 1.0]",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite ranking score balancing impact, confidence, and risk",
    )
    pros: List[str] = Field(default_factory=list, description="Strategic advantages")
    cons: List[str] = Field(default_factory=list, description="Strategic disadvantages or tradeoffs")
    tradeoffs: Dict[str, Any] = Field(default_factory=dict, description="Structured tradeoff metrics")
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding references for this strategy",
    )


class DecisionTreeNode(BaseKBModel):
    """
    An explainable decision node in the strategic reasoning decision tree.
    """

    node_id: str = Field(description="Unique decision node identifier")
    parent_id: Optional[str] = Field(default=None, description="Parent decision node ID")
    decision_type: str = Field(description="Domain of decision (e.g. 'archetype_selection', 'palette')")
    label: str = Field(description="Short human-readable label")
    chosen_option: str = Field(description="The selected strategic choice")
    alternative_options: List[str] = Field(
        default_factory=list,
        description="Alternative options that were considered and rejected",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Source evidence IDs justifying this decision",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in this decision",
    )
    rationale: str = Field(default="", description="Explainable audit rationale for the choice")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional decision context")


class DecisionTree(BaseKBModel):
    """
    Complete explainable decision tree capturing all strategic choices made during reasoning.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    tree_id: str = Field(
        default_factory=lambda: f"dtree_{uuid.uuid4().hex[:8]}",
        description="Unique decision tree identifier",
    )
    root_node_id: str = Field(default="", description="Root node of the decision tree")
    nodes: Dict[str, DecisionTreeNode] = Field(
        default_factory=dict,
        description="All decision tree nodes indexed by node_id",
    )
    created_at: str = Field(default_factory=_utc_now_iso)

    def add_node(self, node: DecisionTreeNode) -> None:
        """Add a decision node to the tree."""
        if not self.root_node_id:
            self.root_node_id = node.node_id
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[DecisionTreeNode]:
        """Retrieve a decision node by ID."""
        return self.nodes.get(node_id)

    def get_children(self, parent_id: str) -> List[DecisionTreeNode]:
        """Retrieve all immediate children of a given node."""
        return [node for node in self.nodes.values() if node.parent_id == parent_id]


# ---------------------------------------------------------------------------
# Reasoner Output Contracts
# ---------------------------------------------------------------------------


class NarrativeReasoningOutput(BaseKBModel):
    """
    Output contract for Narrative Reasoning.
    Captures visual storytelling hooks, emotional tone, and narrative framing.
    """

    story_hook: str = Field(default="", description="Core visual narrative hook")
    narrative_angle: str = Field(default="", description="Primary storytelling angle or premise")
    emotional_tone: str = Field(default="", description="Intended viewer emotional response")
    key_visual_metaphors: List[str] = Field(
        default_factory=list,
        description="Visual metaphors or symbolic visual elements",
    )
    scene_framing: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scene composition framing directives",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence supporting narrative claims",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in narrative reasoning",
    )
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Audit trace of narrative deduction steps",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AudienceReasoningOutput(BaseKBModel):
    """
    Output contract for Audience Reasoning.
    Captures psychological triggers, curiosity gaps, and audience expectations.
    """

    target_audience_segment: str = Field(default="", description="Primary target audience segment")
    curiosity_triggers: List[str] = Field(
        default_factory=list,
        description="Specific curiosity gap mechanisms",
    )
    psychological_hooks: List[str] = Field(
        default_factory=list,
        description="Cognitive and psychological triggers used to capture attention",
    )
    cognitive_load_level: str = Field(
        default="medium",
        description="Target cognitive load (low, medium, high)",
    )
    viewer_expectations: List[str] = Field(
        default_factory=list,
        description="Expectations the thumbnail establishes for the viewer",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence for audience psychology claims",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in audience reasoning",
    )
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Audit trace of audience deduction steps",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreatorReasoningOutput(BaseKBModel):
    """
    Output contract for Creator Reasoning.
    Captures creator intent, persona, consistency, and channel brand equity.
    """

    creator_persona: str = Field(default="", description="Creator archetype or persona identity")
    signature_elements: List[str] = Field(
        default_factory=list,
        description="Creator signature visual elements and recognizable tropes",
    )
    style_alignment_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Alignment with historical channel visual style",
    )
    channel_voice: str = Field(default="", description="Channel tone, voice, and positioning")
    brand_equity_anchors: List[str] = Field(
        default_factory=list,
        description="Visual assets anchoring channel brand equity",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence for creator style claims",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in creator reasoning",
    )
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Audit trace of creator deduction steps",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BrandReasoningOutput(BaseKBModel):
    """
    Output contract for Brand Reasoning.
    Captures enforced palette rules, typography guidelines, logo rules, and prohibitions.
    """

    color_palette_rules: List[str] = Field(
        default_factory=list,
        description="Mandatory color palette hex codes and application rules",
    )
    typography_rules: List[str] = Field(
        default_factory=list,
        description="Typography constraints, font weights, and text styling",
    )
    logo_rules: List[str] = Field(
        default_factory=list,
        description="Logo placement, sizing, and clearspace rules",
    )
    prohibited_elements: List[str] = Field(
        default_factory=list,
        description="Prohibited visual tropes, colors, or objects",
    )
    identity_lock_requirements: List[str] = Field(
        default_factory=list,
        description="Mandatory identity protection and face locking requirements",
    )
    compliance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall brand constraint compliance score",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence for brand rules",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in brand reasoning",
    )
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Audit trace of brand deduction steps",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PriorityReasoningOutput(BaseKBModel):
    """
    Output contract for Priority Reasoning.
    Captures visual hierarchy, focal points, layout weighting, and contrast allocations.
    """

    focal_element_hierarchy: List[str] = Field(
        default_factory=list,
        description="Ordered hierarchy of visual elements from primary to tertiary",
    )
    visual_weight_allocations: Dict[str, float] = Field(
        default_factory=dict,
        description="Relative visual weight percentages per element type",
    )
    composition_style: str = Field(default="", description="Target composition rule (rule of thirds, central, split)")
    contrast_priorities: List[str] = Field(
        default_factory=list,
        description="Key contrast boundaries and separation priorities",
    )
    lighting_priorities: List[str] = Field(
        default_factory=list,
        description="Lighting cues, key light directions, and rim lighting priorities",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence for visual priorities",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in priority reasoning",
    )
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Audit trace of priority deduction steps",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskReasoningOutput(BaseKBModel):
    """
    Output contract for Risk Reasoning.
    Captures fatigue scores, competitor convergence risk, policy risks, and mitigations.
    """

    fatigue_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Risk of audience visual fatigue from overused tropes",
    )
    competitor_convergence_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Risk of visual indistinguishability from niche competitors",
    )
    misleading_clickbait_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Risk of viewer bounce due to mismatched expectation vs video content",
    )
    identified_risks: List[ReasoningRisk] = Field(
        default_factory=list,
        description="List of specific identified risks with severities",
    )
    mitigation_strategies: List[str] = Field(
        default_factory=list,
        description="Actionable mitigation recommendations",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence for risk assessments",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in risk reasoning",
    )
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Audit trace of risk deduction steps",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategyRankingOutput(BaseKBModel):
    """
    Output contract for Strategy Ranking.
    Captures ordered candidate strategies, tradeoff analysis, and chosen recommendation.
    """

    candidate_strategies: List[RankedStrategy] = Field(
        default_factory=list,
        description="Ranked list of candidate thumbnail design strategies",
    )
    selected_strategy_id: Optional[str] = Field(
        default=None,
        description="ID of the highest-ranked chosen strategy",
    )
    ranking_rationale: str = Field(default="", description="Explainable rationale for strategy selection")
    tradeoff_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Comparative tradeoff metrics between candidate strategies",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence for ranking calculations",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in strategy ranking",
    )
    reasoning_trace: List[str] = Field(
        default_factory=list,
        description="Audit trace of strategy ranking calculations",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

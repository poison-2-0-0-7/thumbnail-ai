"""
context.py
==========

ReasoningContext container for the Strategic Reasoning Coordinator.
Serves as the unified, strongly typed, immutable contract encapsulating narrative,
audience, creator intent, brand constraints, visual priorities, risks, ranked strategies,
confidence breakdown, grounding evidence references, audit traces, and the decision tree.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from pydantic import ConfigDict, Field

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceReference,
    _utc_now_iso,
)
from thumbnail_intelligence.reasoning.models import (
    AudienceReasoningOutput,
    BrandReasoningOutput,
    CreatorReasoningOutput,
    DecisionTree,
    NarrativeReasoningOutput,
    PriorityReasoningOutput,
    ReasoningRisk,
    ReasoningTraceStep,
    RiskReasoningOutput,
    StrategyRankingOutput,
)


class ReasoningContext(BaseKBModel):
    """
    Unified master context artifact produced by the ReasoningCoordinator.
    Holds strongly typed placeholders for all reasoning facets and maintains full grounding.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    context_id: str = Field(
        default_factory=lambda: f"ctx_{uuid.uuid4().hex[:12]}",
        description="Unique context execution identifier",
    )
    graph_id: str = Field(
        default="",
        description="Identifier of the source NormalizedEvidenceGraph",
    )

    # Core Strategic Reasoning Facets
    narrative: Optional[NarrativeReasoningOutput] = Field(
        default=None,
        description="Visual storytelling, hooks, and narrative framing",
    )
    audience: Optional[AudienceReasoningOutput] = Field(
        default=None,
        description="Audience psychology, curiosity triggers, and expectations",
    )
    creator_intent: Optional[CreatorReasoningOutput] = Field(
        default=None,
        description="Creator persona, consistency, and channel style equity",
    )
    brand_constraints: Optional[BrandReasoningOutput] = Field(
        default=None,
        description="Enforced brand palette, typography, logo rules, and prohibitions",
    )
    visual_priorities: Optional[PriorityReasoningOutput] = Field(
        default=None,
        description="Visual hierarchy, focal point allocations, and contrast priorities",
    )
    risks: Optional[RiskReasoningOutput] = Field(
        default=None,
        description="Risk assessment, fatigue scores, and competitor convergence",
    )
    strategies: Optional[StrategyRankingOutput] = Field(
        default=None,
        description="Ranked candidate thumbnail design strategies and tradeoff analysis",
    )

    # Custom and extensible outputs
    custom_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs from custom or third-party registered reasoners",
    )

    # Holistic Confidence & Grounding
    overall_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Holistic confidence score aggregated across all executed reasoners",
    )
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-reasoner confidence scores indexed by reasoner name",
    )
    evidence_references: List[EvidenceReference] = Field(
        default_factory=list,
        description="Aggregated, deduplicated evidence references backing all reasoning claims",
    )

    # Auditability & Explainability
    reasoning_trace: List[ReasoningTraceStep] = Field(
        default_factory=list,
        description="Chronological audit log of all reasoner execution steps",
    )
    decision_tree: Optional[DecisionTree] = Field(
        default=None,
        description="Explainable decision tree capturing all strategic choices",
    )

    # Metadata & Telemetry
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution statistics, timing, and coordinator environment metadata",
    )
    created_at: str = Field(default_factory=_utc_now_iso)

    # -----------------------------------------------------------------------
    # Helper Inspection & Access Methods
    # -----------------------------------------------------------------------

    def get_active_reasoner_names(self) -> List[str]:
        """Return the names of all reasoners that have populated outputs in this context."""
        active: List[str] = []
        if self.narrative is not None:
            active.append("narrative")
        if self.audience is not None:
            active.append("audience")
        if self.creator_intent is not None:
            active.append("creator")
        if self.brand_constraints is not None:
            active.append("brand")
        if self.visual_priorities is not None:
            active.append("priority")
        if self.risks is not None:
            active.append("risk")
        if self.strategies is not None:
            active.append("strategy_ranker")
        active.extend(self.custom_outputs.keys())
        return active

    def get_evidence_count(self) -> int:
        """Return the total number of grounding evidence references present in this context."""
        return len(self.evidence_references)

    def has_slot(self, slot_name: str) -> bool:
        """Check if a specific reasoning slot is populated."""
        normalized = slot_name.lower().strip()
        slot_map = {
            "narrative": self.narrative,
            "audience": self.audience,
            "creator": self.creator_intent,
            "creator_intent": self.creator_intent,
            "brand": self.brand_constraints,
            "brand_constraints": self.brand_constraints,
            "priority": self.visual_priorities,
            "visual_priorities": self.visual_priorities,
            "risk": self.risks,
            "risks": self.risks,
            "strategies": self.strategies,
            "strategy_ranker": self.strategies,
        }
        if normalized in slot_map:
            return slot_map[normalized] is not None
        return normalized in self.custom_outputs

    def get_slot(self, slot_name: str) -> Optional[Any]:
        """Retrieve the output of a specific slot by name."""
        normalized = slot_name.lower().strip()
        slot_map = {
            "narrative": self.narrative,
            "audience": self.audience,
            "creator": self.creator_intent,
            "creator_intent": self.creator_intent,
            "brand": self.brand_constraints,
            "brand_constraints": self.brand_constraints,
            "priority": self.visual_priorities,
            "visual_priorities": self.visual_priorities,
            "risk": self.risks,
            "risks": self.risks,
            "strategies": self.strategies,
            "strategy_ranker": self.strategies,
        }
        if normalized in slot_map:
            return slot_map[normalized]
        return self.custom_outputs.get(normalized)

    def is_complete(self, required_slots: Optional[List[str]] = None) -> bool:
        """
        Verify whether all specified required slots are populated in this context.
        If required_slots is None, defaults to checking all core standard slots.
        """
        slots = required_slots or [
            "narrative",
            "audience",
            "creator",
            "brand",
            "priority",
            "risk",
            "strategies",
        ]
        return all(self.has_slot(s) for s in slots)

    def get_all_identified_risks(self) -> List[ReasoningRisk]:
        """Convenience method to retrieve all identified risks if the risk slot is populated."""
        if self.risks is not None:
            return self.risks.identified_risks
        return []

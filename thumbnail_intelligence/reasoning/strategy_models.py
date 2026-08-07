"""
strategy_models.py
==================

Domain models, strategy archetypes, multi-hypothesis candidate evaluation, and tradeoff models
for the Strategy Ranking Engine (Phase 3.4G).
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
from thumbnail_intelligence.reasoning.models import (
    RankedStrategy,
    StrategyRankingOutput,
)


class StrategyArchetype(str, Enum):
    """
    Core strategic creative archetypes for thumbnail concepts.
    """

    CURIOSITY = "curiosity"
    EMOTION = "emotion"
    TRANSFORMATION = "transformation"
    MYSTERY = "mystery"
    COMPARISON = "comparison"
    MINIMALIST = "minimalist"
    EDUCATIONAL = "educational"
    HIGH_ENERGY = "high_energy"
    CINEMATIC = "cinematic"
    REACTION = "reaction"
    CHALLENGE = "challenge"
    CUSTOM = "custom"


class StrategyCandidate(BaseKBModel):
    """
    An individual candidate thumbnail design strategy evaluated and ranked across multi-objective metrics.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    candidate_id: str = Field(
        default_factory=lambda: f"strat_cand_{uuid.uuid4().hex[:8]}",
        description="Unique strategy candidate identifier",
    )
    title: str = Field(description="Actionable strategic concept title")
    archetype: StrategyArchetype = Field(
        default=StrategyArchetype.CURIOSITY,
        description="Classified creative strategy archetype",
    )
    description: str = Field(description="Strategic rationale, visual framing, and narrative justification")
    expected_ctr_uplift: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated CTR uplift score in [0.0, 1.0]",
    )
    retention_alignment_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Alignment with viewer expectation and video retention in [0.0, 1.0]",
    )
    brand_equity_protection_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Preservation of creator visual identity in [0.0, 1.0]",
    )
    risk_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregated risk penalty score in [0.0, 1.0]",
    )
    composite_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Multi-objective Pareto composite score in [0.0, 1.0]",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Grounded confidence score in [0.0, 1.0]",
    )
    pros: List[str] = Field(default_factory=list, description="Core advantages of this strategy")
    cons: List[str] = Field(default_factory=list, description="Strategic tradeoffs and risks")
    execution_priorities: List[str] = Field(
        default_factory=list,
        description="Key visual elements and steps required to execute this strategy",
    )
    success_factors: List[str] = Field(
        default_factory=list,
        description="Empirical criteria determining whether this strategy will succeed",
    )
    failure_risks: List[str] = Field(
        default_factory=list,
        description="Potential failure modes if executed poorly",
    )
    rejection_rationale: Optional[str] = Field(
        default=None,
        description="Audit explanation if this strategy was evaluated and rejected or outranked",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Supporting grounding evidence references",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of evidence nodes backing this strategy",
    )

    def to_ranked_strategy(self) -> RankedStrategy:
        """Convert this candidate strategy to a standard RankedStrategy output model."""
        arch_id = self.archetype.value if hasattr(self.archetype, "value") else str(self.archetype)
        return RankedStrategy(
            strategy_id=self.candidate_id,
            title=self.title,
            description=self.description,
            archetype_id=arch_id,
            expected_ctr_impact=self.expected_ctr_uplift,
            confidence_score=self.confidence,
            overall_score=self.composite_score,
            pros=self.pros,
            cons=self.cons,
            tradeoffs={
                "retention_alignment": self.retention_alignment_score,
                "brand_equity_protection": self.brand_equity_protection_score,
                "risk_penalty": self.risk_penalty,
                "composite_score": self.composite_score,
            },
            evidence_refs=self.evidence_refs,
        )


class TradeoffAnalysis(BaseKBModel):
    """
    Structured trade-off metrics comparing candidate strategies.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    pareto_optimal_strategy_id: str = Field(description="ID of the Pareto-dominant strategy")
    ctr_vs_retention_tradeoff: str = Field(
        description="Analysis of CTR capture versus viewer bounce risk",
    )
    brand_vs_novelty_tradeoff: str = Field(
        description="Analysis of channel brand recognition versus creative departure",
    )
    cognitive_load_tradeoff: str = Field(
        description="Analysis of visual complexity versus mobile comprehension speed",
    )
    comparative_scores: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Comparative metric table across candidate strategies",
    )
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list,
        description="Grounding evidence for tradeoff deductions",
    )


class StrategyDecision(StrategyRankingOutput):
    """
    Master output artifact of the StrategyRanker.
    Combines every strategic reasoning facet into a single ranked decision.
    Inherits from StrategyRankingOutput for 100% backward compatibility with ReasoningContext.
    """

    model_config = ConfigDict(frozen=False, extra="ignore", validate_assignment=True)

    winning_strategy: Optional[StrategyCandidate] = Field(
        default=None,
        description="The chosen winning thumbnail design strategy",
    )
    alternative_strategies: List[StrategyCandidate] = Field(
        default_factory=list,
        description="Evaluated alternative strategies ranked in descending order",
    )
    tradeoff_analysis_detail: Optional[TradeoffAnalysis] = Field(
        default=None,
        description="Detailed multi-dimensional tradeoff analysis across candidates",
    )
    decision_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Propagated confidence score in the final strategic decision",
    )
    decision_rationale: str = Field(
        default="",
        description="Explainable audit explanation why the winning strategy was selected",
    )
    rejected_strategies: List[StrategyCandidate] = Field(
        default_factory=list,
        description="Alternative candidates rejected with explicit audit rationale",
    )
    execution_priorities: List[str] = Field(
        default_factory=list,
        description="Immediate visual execution priorities for the chosen concept",
    )
    success_factors: List[str] = Field(
        default_factory=list,
        description="Key performance indicators and execution requirements for success",
    )
    failure_risks: List[str] = Field(
        default_factory=list,
        description="Critical failure risks to monitor and avoid during rendering",
    )
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Granular component confidence scores across all strategic facets",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="All evidence node IDs backing the strategy decision",
    )

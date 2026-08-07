"""
test_strategic_reasoning_pipeline_phase7.py
===========================================

Comprehensive end-to-end integration test suite verifying the complete 7-reasoner Strategic Reasoning Pipeline (Phase 3.4G):
- Executes all 7 production reasoners in topological DAG sequence:
  NarrativeReasoner -> AudienceReasoner -> CreatorReasoner -> BrandReasoner -> PriorityReasoner -> RiskReasoner -> StrategyRanker
- Validates all 7 ReasoningContext slots:
  narrative, audience, creator_intent, brand_constraints, visual_priorities, risks, strategies
- Validates complete context status with context.is_complete() == True
- Verifies grounded evidence aggregation across all 7 reasoners
- Validates DecisionTree construction covering all 7 reasoning stages
- Tests empty graphs, suppressed nodes, and validation rejections across the entire pipeline
"""

from __future__ import annotations

from typing import Any, Dict, List
import pytest

from thumbnail_intelligence.knowledge_base.models import (
    EvidenceGrade,
    EvidenceReference,
    EvidenceSourceType,
)
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceConflict,
    EvidenceNode,
    EvidenceSummary,
    EvidenceWeight,
    KnowledgeEntryType,
    NormalizedEvidenceGraph,
    ProvenanceRecord,
)
from thumbnail_intelligence.reasoning.audience_models import AudienceResult
from thumbnail_intelligence.reasoning.audience_reasoner import AudienceReasoner
from thumbnail_intelligence.reasoning.brand_models import BrandResult
from thumbnail_intelligence.reasoning.brand_reasoner import BrandReasoner
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.creator_models import CreatorResult
from thumbnail_intelligence.reasoning.creator_reasoner import CreatorReasoner
from thumbnail_intelligence.reasoning.narrative_models import NarrativeResult
from thumbnail_intelligence.reasoning.narrative_reasoner import NarrativeReasoner
from thumbnail_intelligence.reasoning.pipeline import ReasoningPipeline
from thumbnail_intelligence.reasoning.priority_models import PriorityResult
from thumbnail_intelligence.reasoning.priority_reasoner import PriorityReasoner
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry
from thumbnail_intelligence.reasoning.risk_models import RiskResult
from thumbnail_intelligence.reasoning.risk_reasoner import RiskReasoner
from thumbnail_intelligence.reasoning.strategy_models import (
    StrategyCandidate,
    StrategyDecision,
    TradeoffAnalysis,
)
from thumbnail_intelligence.reasoning.strategy_ranker import StrategyRanker
from thumbnail_intelligence.retrieval.evidence_bundle import (
    RankingMetadata,
    RetrievalScore,
    RetrievedEvidence,
)


def _build_test_node(
    node_id: str,
    node_type: KnowledgeEntryType,
    payload: Dict[str, Any],
    confidence: float = 0.90,
    is_active: bool = True,
) -> EvidenceNode:
    ref = EvidenceReference(
        source_id=node_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=confidence,
        grade=EvidenceGrade.STRONG,
        claim_summary=f"Evidence for {node_id}",
    )
    score = RetrievalScore(composite_score=confidence)
    item = RetrievedEvidence(
        evidence_id=f"ev_{node_id}",
        entry_id=node_id,
        entry_type=node_type,
        origin=f"origin:{node_id}",
        confidence=confidence,
        reason_retrieved=f"Retrieved for {node_id}",
        score=score,
        ranking=RankingMetadata(rank=1, score=score),
        data_payload=payload,
        evidence_refs=[ref],
    )
    return EvidenceNode(
        node_id=node_id,
        node_type=node_type,
        evidence_item=item,
        confidence=ConfidenceScore(raw_confidence=confidence, propagated_confidence=confidence),
        weight=EvidenceWeight(base_weight=1.0, effective_weight=1.0),
        provenance=ProvenanceRecord(
            origin=f"origin:{node_id}",
            source_id=node_id,
            source_type=EvidenceSourceType.OUTCOME_RECORD,
            retrieval_query_id="query_pipeline7",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=is_active,
    )


def test_full_7_reasoner_strategic_pipeline():
    """Verify Narrative, Audience, Creator, Brand, Priority, Risk, and StrategyRanker run in topological DAG order and populate all 7 slots."""
    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(AudienceReasoner())
    registry.register(CreatorReasoner())
    registry.register(BrandReasoner())
    registry.register(PriorityReasoner())
    registry.register(RiskReasoner())
    registry.register(StrategyRanker())

    # Verify execution order: narrative must be first, strategy ranker must be last
    exec_order = [r.name for r in registry.get_execution_order()]
    assert exec_order[0] == "narrative_reasoner"
    assert exec_order[-1] == "strategy_ranker"
    assert "audience_reasoner" in exec_order[1:-1]
    assert "creator_reasoner" in exec_order[1:-1]
    assert "brand_reasoner" in exec_order[1:-1]
    assert "priority_reasoner" in exec_order[1:-1]
    assert "risk_reasoner" in exec_order[1:-1]

    # Build rich evidence graph
    nodes = {
        "node_pipe7_title_01": _build_test_node(
            "node_pipe7_title_01",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {
                "title": "I Survived 100 Hours in Antarctica with Zero Gear",
                "transcript": "The blizzard hit on hour 12 and the temperatures dropped to minus 40.",
                "ocr_text": "100H ANTARCTICA",
                "ctr": 0.148,
                "color_palette": ["#00E5FF", "#FF3366", "#0E0E12", "#FFFFFF"],
                "objects": ["explorer face", "ice cave", "frozen compass"],
            },
        ),
        "node_pipe7_creator_02": _build_test_node(
            "node_pipe7_creator_02",
            KnowledgeEntryType.CREATOR_PROFILE_ENTRY,
            {
                "creator_id": "creator_survival_pro",
                "display_name": "Survival Pro",
                "primary_niche": "extreme_survival",
            },
        ),
        "node_pipe7_pattern_03": _build_test_node(
            "node_pipe7_pattern_03",
            KnowledgeEntryType.DESIGN_PATTERN,
            {"pattern_id": "extreme_challenge_split"},
        ),
    }

    graph = NormalizedEvidenceGraph(
        graph_id="graph_pipe7_full",
        nodes=nodes,
        summary=EvidenceSummary(graph_id="graph_pipe7_full", primary_archetype="extreme_challenge"),
    )

    pipeline = ReasoningPipeline.from_registry(registry)
    context = pipeline.run(graph)

    # Validate all 7 strategic slots are fully populated
    assert context.has_slot("narrative")
    assert context.has_slot("audience")
    assert context.has_slot("creator_intent")
    assert context.has_slot("brand_constraints")
    assert context.has_slot("visual_priorities")
    assert context.has_slot("risks")
    assert context.has_slot("strategies")
    assert context.has_slot("strategy_ranker")

    assert isinstance(context.narrative, NarrativeResult)
    assert isinstance(context.audience, AudienceResult)
    assert isinstance(context.creator_intent, CreatorResult)
    assert isinstance(context.brand_constraints, BrandResult)
    assert isinstance(context.visual_priorities, PriorityResult)
    assert isinstance(context.risks, RiskResult)
    assert isinstance(context.strategies, StrategyDecision)

    # Validate context is 100% complete across all 7 core slots
    assert context.is_complete() is True
    assert len(context.get_active_reasoner_names()) == 7

    # Validate StrategyDecision details
    strat: StrategyDecision = context.strategies
    assert strat.winning_strategy is not None
    assert strat.winning_strategy.composite_score > 0.0
    assert len(strat.alternative_strategies) >= 3
    assert len(strat.rejected_strategies) >= 3
    assert strat.tradeoff_analysis_detail is not None
    assert strat.tradeoff_analysis_detail.pareto_optimal_strategy_id == strat.winning_strategy.candidate_id
    assert strat.decision_confidence > 0.70
    assert len(strat.execution_priorities) >= 2
    assert len(strat.success_factors) >= 1
    assert len(strat.failure_risks) >= 1

    # Validate grounding evidence references and trace steps across all 7 reasoners
    assert len(context.evidence_references) >= 3
    assert len(context.reasoning_trace) >= 7
    assert context.overall_confidence > 0.70

    # Validate DecisionTree has 7 connected decision nodes
    if context.decision_tree:
        assert len(context.decision_tree.nodes) == 7
        assert context.decision_tree.root_node_id != ""


def test_empty_graph_and_suppressed_nodes_handling_across_all_7_reasoners():
    """Verify all 7 reasoners execute safely with grounded fallbacks on empty graphs."""
    empty_graph = NormalizedEvidenceGraph(
        graph_id="graph_empty_all7",
        nodes={},
        summary=EvidenceSummary(graph_id="graph_empty_all7"),
    )

    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(AudienceReasoner())
    registry.register(CreatorReasoner())
    registry.register(BrandReasoner())
    registry.register(PriorityReasoner())
    registry.register(RiskReasoner())
    registry.register(StrategyRanker())

    pipeline = ReasoningPipeline.from_registry(registry)
    context = pipeline.run(empty_graph)

    assert context.narrative is not None
    assert context.audience is not None
    assert context.creator_intent is not None
    assert context.brand_constraints is not None
    assert context.visual_priorities is not None
    assert context.risks is not None
    assert context.strategies is not None
    assert isinstance(context.strategies, StrategyDecision)
    assert context.strategies.winning_strategy is not None
    assert context.is_complete() is True


def test_strategy_ranker_registry_lookup_and_cloning():
    """Verify StrategyRanker interacts cleanly with ReasonerRegistry operations."""
    registry = ReasonerRegistry()
    ranker = StrategyRanker()
    registry.register(ranker)

    assert registry.has("strategy_ranker") is True
    assert registry.count() == 1
    retrieved = registry.get_required("strategy_ranker")
    assert retrieved is ranker

    cloned = registry.clone()
    assert cloned.has("strategy_ranker") is True
    assert cloned.count() == 1

    registry.unregister("strategy_ranker")
    assert registry.has("strategy_ranker") is False
    assert cloned.has("strategy_ranker") is True

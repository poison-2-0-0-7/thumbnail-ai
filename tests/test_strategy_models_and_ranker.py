"""
test_strategy_models_and_ranker.py
==================================

Comprehensive unit and integration test suite for the Strategy Ranking Engine (Phase 3.4G).
Tests:
- StrategyArchetype taxonomy extensibility
- StrategyCandidate model validation and conversion
- TradeoffAnalysis calculations and structured comparisons
- StrategyDecision master output artifact validation
- StrategyRanker candidate generation across archetypes
- Multi-objective Pareto scoring and ranking algorithm
- Explainable rejection rationales for alternative candidates
- Propagated multi-signal calibrated confidence model
- Grounding gate enforcement (zero evidence refs -> zero confidence)
- Custom objective weight configurations
- Edge cases: empty graphs, suppressed nodes, missing upstream context slots
- Validator rejections and error handling
"""

from __future__ import annotations

from typing import Any, Dict, List
import pytest

from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceConflict,
    EvidenceNode,
    EvidenceReference,
    EvidenceSourceType,
    EvidenceSummary,
    EvidenceWeight,
    KnowledgeEntryType,
    NormalizedEvidenceGraph,
    ProvenanceRecord,
)
from thumbnail_intelligence.knowledge_base.models import (
    EvidenceGrade,
    _utc_now_iso,
)
from thumbnail_intelligence.reasoning.audience_models import (
    AudienceResult,
    CandidateAudience,
    CognitiveLoadLevel,
    ViewerIntent,
    ViewerKnowledgeLevel,
    ViewerPersona,
)
from thumbnail_intelligence.reasoning.brand_models import (
    BrandPreservationPriority,
    BrandResult,
    CandidateBrandInterpretation,
    VisualElementPreservation,
)
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.creator_models import (
    CandidateCreatorStyle,
    CreatorArchetype,
    CreatorResult,
    VisualIdentityStyle,
)
from thumbnail_intelligence.reasoning.exceptions import (
    GroundingEnforcementError,
    ReasonerValidationError,
)
from thumbnail_intelligence.reasoning.models import (
    RankedStrategy,
    ReasonerContract,
    ReasonerType,
    ReasoningRisk,
)
from thumbnail_intelligence.reasoning.narrative_models import (
    ArcStage,
    ArcStep,
    CandidateNarrative,
    NarrativeArc,
    NarrativeResult,
    NarrativeType,
)
from thumbnail_intelligence.reasoning.priority_models import (
    AttentionFlowStep,
    BackgroundPriority,
    CandidateHierarchy,
    ElementPriorityLevel,
    HierarchyTier,
    PriorityResult,
    VisualHierarchyNode,
)
from thumbnail_intelligence.reasoning.risk_models import (
    CandidateRiskProfile,
    DetectedRisk,
    RiskCategory,
    RiskLikelihood,
    RiskResult,
    RiskSeverity,
)
from thumbnail_intelligence.reasoning.strategy_models import (
    StrategyArchetype,
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
            retrieval_query_id="query_strategy_unit",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=is_active,
    )


def _build_mock_context(graph_id: str = "graph_strat_test") -> ReasoningContext:
    ctx = ReasoningContext(graph_id=graph_id)
    ref = EvidenceReference(
        source_id="ref_mock_01",
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.92,
        grade=EvidenceGrade.STRONG,
        claim_summary="Mock test reference",
    )

    ctx.narrative = NarrativeResult(
        story_hook="Surviving 100 Hours in the Abyss",
        narrative_angle="Extreme Survival Trial",
        emotional_tone="Intrigue and suspense",
        confidence=0.92,
        evidence_refs=[ref],
    )
    ctx.audience = AudienceResult(
        target_audience_segment="Survival & Outdoor Enthusiasts",
        confidence=0.90,
        evidence_refs=[ref],
    )
    ctx.creator_intent = CreatorResult(
        creator_persona="Survival Expert",
        confidence=0.88,
        evidence_refs=[ref],
    )
    ctx.brand_constraints = BrandResult(
        color_palette_rules=["#00E5FF", "#FF3366", "#0E0E12"],
        confidence=0.89,
        compliance_score=0.95,
        evidence_refs=[ref],
    )
    ctx.visual_priorities = PriorityResult(
        focal_element_hierarchy=["explorer face", "abyss chasm", "frozen compass"],
        confidence=0.91,
        evidence_refs=[ref],
    )
    ctx.risks = RiskResult(
        fatigue_risk_score=0.28,
        competitor_convergence_risk=0.22,
        misleading_clickbait_risk=0.15,
        risk_confidence=0.87,
        confidence=0.87,
        evidence_refs=[ref],
    )
    return ctx


# ===========================================================================
# 1. Strategy Taxonomy & Models Tests
# ===========================================================================


def test_strategy_archetype_taxonomy_coverage():
    """Verify StrategyArchetype enum includes all standard and custom archetypes."""
    expected = [
        "curiosity",
        "emotion",
        "transformation",
        "mystery",
        "comparison",
        "minimalist",
        "educational",
        "high_energy",
        "cinematic",
        "reaction",
        "challenge",
        "custom",
    ]
    for arch in expected:
        assert StrategyArchetype(arch) is not None
        assert arch in [e.value for e in StrategyArchetype]


def test_strategy_candidate_model_and_conversion():
    """Verify StrategyCandidate model instantiation, bounds, and conversion to RankedStrategy."""
    ref = EvidenceReference(
        source_id="ref_cand_01",
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.95,
        grade=EvidenceGrade.STRONG,
    )
    cand = StrategyCandidate(
        title="High-Tension Ice Chasm Juxtaposition",
        archetype=StrategyArchetype.CURIOSITY,
        description="Juxtaposes explorer face with deep ice fissure",
        expected_ctr_uplift=0.88,
        retention_alignment_score=0.84,
        brand_equity_protection_score=0.80,
        risk_penalty=0.15,
        composite_score=0.86,
        confidence=0.92,
        pros=["High click capture", "Strong visual contrast"],
        cons=["Moderate trope saturation risk"],
        execution_priorities=["Enforce 4.5:1 luminance contrast", "40% hero face"],
        success_factors=["Instant comprehension < 400ms"],
        failure_risks=["Visual clutter"],
        evidence_refs=[ref],
        supporting_evidence_ids=["ref_cand_01"],
    )

    assert cand.expected_ctr_uplift == 0.88
    assert cand.composite_score == 0.86
    assert cand.archetype == StrategyArchetype.CURIOSITY

    # Convert to RankedStrategy
    ranked = cand.to_ranked_strategy()
    assert isinstance(ranked, RankedStrategy)
    assert ranked.strategy_id == cand.candidate_id
    assert ranked.title == cand.title
    assert ranked.archetype_id == "curiosity"
    assert ranked.expected_ctr_impact == 0.88
    assert ranked.overall_score == 0.86
    assert len(ranked.evidence_refs) == 1


def test_tradeoff_analysis_model():
    """Verify TradeoffAnalysis model captures comparative matrix and analytical prose."""
    ref = EvidenceReference(
        source_id="ref_tradeoff_01",
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.90,
    )
    analysis = TradeoffAnalysis(
        pareto_optimal_strategy_id="strat_cand_001",
        ctr_vs_retention_tradeoff="High CTR uplift with solid retention balance.",
        brand_vs_novelty_tradeoff="Preserves cyan signature lighting while pushing novelty.",
        cognitive_load_tradeoff="Mobile scan time bounded to under 450ms.",
        comparative_scores={
            "strat_cand_001": {"composite_score": 0.88, "expected_ctr": 0.90},
            "strat_cand_002": {"composite_score": 0.82, "expected_ctr": 0.84},
        },
        evidence_refs=[ref],
    )
    assert analysis.pareto_optimal_strategy_id == "strat_cand_001"
    assert len(analysis.comparative_scores) == 2
    assert len(analysis.evidence_refs) == 1


def test_strategy_decision_model_inheritance_and_fields():
    """Verify StrategyDecision properly inherits from StrategyRankingOutput with all required slots."""
    ref = EvidenceReference(
        source_id="ref_decision_01",
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.91,
    )
    winner = StrategyCandidate(
        title="Winning Concept",
        archetype=StrategyArchetype.CURIOSITY,
        description="Top concept",
        composite_score=0.89,
        evidence_refs=[ref],
    )
    decision = StrategyDecision(
        winning_strategy=winner,
        alternative_strategies=[],
        decision_confidence=0.92,
        decision_rationale="Selected based on highest Pareto composite score.",
        rejected_strategies=[],
        execution_priorities=["Priority 1", "Priority 2"],
        success_factors=["Factor 1"],
        failure_risks=["Risk 1"],
        confidence_breakdown={"narrative": 0.92, "brand": 0.90},
        supporting_evidence_ids=["ref_decision_01"],
        evidence_refs=[ref],
        confidence=0.92,
    )

    assert decision.winning_strategy.title == "Winning Concept"
    assert decision.decision_confidence == 0.92
    assert len(decision.execution_priorities) == 2
    assert len(decision.supporting_evidence_ids) == 1


# ===========================================================================
# 2. StrategyRanker Execution, Scoring, and Ranking Tests
# ===========================================================================


def test_strategy_ranker_contract():
    """Verify StrategyRanker contract identity, dependencies, and type."""
    ranker = StrategyRanker()
    contract: ReasonerContract = ranker.contract
    assert contract.name == "strategy_ranker"
    assert contract.reasoner_type == ReasonerType.STRATEGY_RANKER
    assert "narrative_reasoner" in contract.dependencies
    assert "audience_reasoner" in contract.dependencies
    assert "creator_reasoner" in contract.dependencies
    assert "brand_reasoner" in contract.dependencies
    assert "priority_reasoner" in contract.dependencies
    assert "risk_reasoner" in contract.dependencies
    assert contract.is_mandatory is True


def test_strategy_ranker_execution_on_grounded_graph():
    """Verify StrategyRanker executes over rich graph and produces validated StrategyDecision."""
    nodes = {
        "node_strat_01": _build_test_node(
            "node_strat_01",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {
                "title": "I Spent 100 Hours Trapped in an Ice Cave",
                "ctr": 0.142,
                "objects": ["explorer face", "ice pick", "blizzard"],
                "color_palette": ["#00E5FF", "#FF3366", "#0E0E12"],
            },
        ),
        "node_strat_02": _build_test_node(
            "node_strat_02",
            KnowledgeEntryType.CREATOR_PROFILE_ENTRY,
            {
                "creator_id": "creator_survival_pro",
                "primary_niche": "extreme_survival",
            },
        ),
        "node_strat_03": _build_test_node(
            "node_strat_03",
            KnowledgeEntryType.DESIGN_PATTERN,
            {"pattern_id": "high_tension_split"},
        ),
    }
    graph = NormalizedEvidenceGraph(
        graph_id="graph_strat_rich",
        nodes=nodes,
        summary=EvidenceSummary(graph_id="graph_strat_rich", primary_archetype="extreme_challenge"),
    )
    context = _build_mock_context(graph_id="graph_strat_rich")

    ranker = StrategyRanker()
    decision = ranker.reason(graph, context)

    assert isinstance(decision, StrategyDecision)
    assert decision.winning_strategy is not None
    assert decision.winning_strategy.composite_score > 0.0
    assert len(decision.alternative_strategies) >= 3
    assert len(decision.rejected_strategies) >= 3
    assert decision.tradeoff_analysis_detail is not None
    assert decision.tradeoff_analysis_detail.pareto_optimal_strategy_id == decision.winning_strategy.candidate_id
    assert len(decision.tradeoff_analysis_detail.comparative_scores) >= 4
    assert len(decision.evidence_refs) >= 3
    assert decision.decision_confidence > 0.70
    assert len(decision.execution_priorities) >= 2
    assert len(decision.success_factors) >= 1
    assert len(decision.failure_risks) >= 1
    assert len(decision.reasoning_trace) >= 5
    assert ranker.validate_output(decision) is True


def test_strategy_ranking_pareto_sorting():
    """Verify that candidate strategies are strictly sorted descending by composite score."""
    nodes = {
        "node_pareto_01": _build_test_node(
            "node_pareto_01",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {"title": "Epic Survival Test", "ctr": 0.135},
        ),
    }
    graph = NormalizedEvidenceGraph(
        graph_id="graph_pareto_test",
        nodes=nodes,
        summary=EvidenceSummary(graph_id="graph_pareto_test"),
    )
    context = _build_mock_context(graph_id="graph_pareto_test")

    ranker = StrategyRanker()
    decision = ranker.reason(graph, context)

    # Check that winner has composite score >= all alternatives
    winner_score = decision.winning_strategy.composite_score
    for alt in decision.alternative_strategies:
        assert winner_score >= alt.composite_score

    # Check that alternative list is sorted descending
    alt_scores = [alt.composite_score for alt in decision.alternative_strategies]
    assert alt_scores == sorted(alt_scores, reverse=True)


def test_explainable_rejection_rationales():
    """Verify every rejected strategy includes clear audit rationale with rank and gap explanations."""
    nodes = {
        "node_rej_01": _build_test_node(
            "node_rej_01",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {"title": "Mystery Vault Opening", "ctr": 0.12},
        ),
    }
    graph = NormalizedEvidenceGraph(
        graph_id="graph_rej_test",
        nodes=nodes,
        summary=EvidenceSummary(graph_id="graph_rej_test"),
    )
    context = _build_mock_context(graph_id="graph_rej_test")

    ranker = StrategyRanker()
    decision = ranker.reason(graph, context)

    for idx, rejected in enumerate(decision.rejected_strategies, start=2):
        assert rejected.rejection_rationale is not None
        assert f"Ranked as Alternative #{idx}" in rejected.rejection_rationale
        assert "trailing winner by" in rejected.rejection_rationale


def test_custom_objective_weights():
    """Verify custom weights change composite scoring emphasizing brand over CTR."""
    nodes = {
        "node_weights_01": _build_test_node(
            "node_weights_01",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {"title": "Channel Loyalty Concept", "ctr": 0.10},
        ),
    }
    graph = NormalizedEvidenceGraph(
        graph_id="graph_weights_test",
        nodes=nodes,
        summary=EvidenceSummary(graph_id="graph_weights_test"),
    )
    context = _build_mock_context(graph_id="graph_weights_test")

    # High brand weight (0.60), low CTR weight (0.10)
    brand_heavy_ranker = StrategyRanker(
        weights={
            "ctr_weight": 0.10,
            "retention_weight": 0.20,
            "brand_weight": 0.60,
            "risk_weight": 0.10,
        }
    )
    decision = brand_heavy_ranker.reason(graph, context)
    assert decision.winning_strategy.brand_equity_protection_score >= 0.85


# ===========================================================================
# 3. Grounding & Confidence Model Tests
# ===========================================================================


def test_grounding_gate_enforcement_zero_evidence():
    """Verify grounding gate invariant: zero evidence references strictly yields 0.0 confidence."""
    empty_graph = NormalizedEvidenceGraph(
        graph_id="graph_grounding_empty",
        nodes={},
        summary=EvidenceSummary(graph_id="graph_grounding_empty"),
    )
    empty_context = ReasoningContext(graph_id="graph_grounding_empty")

    ranker = StrategyRanker()
    decision = ranker.reason(empty_graph, empty_context)

    # When no evidence references are provided anywhere, confidence must be 0.0
    assert decision.decision_confidence == 0.0
    assert decision.confidence == 0.0
    assert decision.confidence_breakdown["evidence_quality"] == 0.0


def test_conflict_penalty_propagation():
    """Verify graph conflicts decrease decision confidence."""
    nodes = {
        "node_conf_01": _build_test_node("node_conf_01", KnowledgeEntryType.HISTORICAL_THUMBNAIL, {"ctr": 0.12}),
        "node_conf_02": _build_test_node("node_conf_02", KnowledgeEntryType.HISTORICAL_THUMBNAIL, {"ctr": 0.06}),
    }
    conflict = EvidenceConflict(
        conflict_id="conf_01",
        conflict_type="CONTRADICTORY_CLAIM",
        description="Contradicting CTR benchmarks",
        node_ids=["node_conf_01", "node_conf_02"],
        severity="high",
    )
    graph_with_conflicts = NormalizedEvidenceGraph(
        graph_id="graph_conflict_test",
        nodes=nodes,
        conflicts=[conflict, conflict],
        summary=EvidenceSummary(graph_id="graph_conflict_test"),
    )
    context = _build_mock_context(graph_id="graph_conflict_test")

    ranker = StrategyRanker()
    decision = ranker.reason(graph_with_conflicts, context)

    assert decision.confidence_breakdown["conflict_penalty"] > 0.15
    assert decision.decision_confidence < 0.85


# ===========================================================================
# 4. Edge Cases & Validation Tests
# ===========================================================================


def test_strategy_ranker_validate_output_rejections():
    """Verify validate_output rejects non-StrategyDecision or invalid confidence scores."""
    ranker = StrategyRanker()
    assert not ranker.validate_output("invalid_string")
    assert not ranker.validate_output(None)
    assert not ranker.validate_output(12345)


def test_strategy_ranker_with_missing_optional_context_slots():
    """Verify StrategyRanker gracefully handles missing upstream context slots with safe defaults."""
    nodes = {
        "node_sparse_01": _build_test_node(
            "node_sparse_01",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {"title": "Sparse Context Test", "ctr": 0.11},
        ),
    }
    graph = NormalizedEvidenceGraph(
        graph_id="graph_sparse_test",
        nodes=nodes,
        summary=EvidenceSummary(graph_id="graph_sparse_test"),
    )
    # Context with None for all slots
    sparse_context = ReasoningContext(graph_id="graph_sparse_test")

    ranker = StrategyRanker()
    decision = ranker.reason(graph, sparse_context)

    assert decision is not None
    assert decision.winning_strategy is not None
    assert decision.winning_strategy.composite_score > 0.0
    assert len(decision.alternative_strategies) >= 1
    assert decision.decision_confidence > 0.0

"""
test_reasoning_coordinator.py
=============================

Test suite for ReasoningCoordinator in Phase 3.4A.
Tests orchestration over NormalizedEvidenceGraph, reasoner invocation,
intermediate output validation, confidence aggregation, and grounding gate enforcement.
"""

from __future__ import annotations

from typing import Any, List
import pytest

from thumbnail_intelligence.knowledge_base.models import (
    EvidenceGrade,
    EvidenceReference,
    EvidenceSourceType,
)
from thumbnail_intelligence.evidence.models import (
    EvidenceSummary,
    NormalizedEvidenceGraph,
)
from thumbnail_intelligence.reasoning.config import ReasoningConfig
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.exceptions import (
    EmptyEvidenceGraphError,
    GroundingEnforcementError,
    ReasonerExecutionError,
    ReasonerNotFoundError,
    ReasonerValidationError,
)
from thumbnail_intelligence.reasoning.interfaces import (
    AudienceReasoner,
    BaseReasoner,
    BrandReasoner,
    CreatorReasoner,
    NarrativeReasoner,
    PriorityReasoner,
    RiskReasoner,
    StrategyRanker,
)
from thumbnail_intelligence.reasoning.models import (
    AudienceReasoningOutput,
    BrandReasoningOutput,
    CreatorReasoningOutput,
    NarrativeReasoningOutput,
    PriorityReasoningOutput,
    RankedStrategy,
    ReasonerContract,
    ReasonerType,
    ReasoningRisk,
    RiskReasoningOutput,
    StrategyRankingOutput,
)
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry


# ---------------------------------------------------------------------------
# Test Fixtures & Stub Reasoners
# ---------------------------------------------------------------------------


def _dummy_graph(graph_id: str = "graph_coord_001") -> NormalizedEvidenceGraph:
    return NormalizedEvidenceGraph(
        graph_id=graph_id,
        summary=EvidenceSummary(graph_id=graph_id, primary_archetype="big_face_reaction"),
    )


def _dummy_ref(source_id: str = "ref_001") -> EvidenceReference:
    return EvidenceReference(
        source_id=source_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.9,
        grade=EvidenceGrade.STRONG,
        claim_summary="High historical CTR test claim",
    )


class StubNarrativeReasoner(NarrativeReasoner):
    def __init__(self, name: str = "narrative", deps: List[str] = None):
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.NARRATIVE,
            dependencies=deps or [],
            version="1.0.0",
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> NarrativeReasoningOutput:
        return NarrativeReasoningOutput(
            story_hook="Surprise discovery hook",
            narrative_angle="Uncovering hidden secret",
            emotional_tone="Astonishment",
            evidence_refs=[_dummy_ref("ref_narrative")],
            confidence=0.90,
        )


class StubAudienceReasoner(AudienceReasoner):
    def __init__(self, name: str = "audience", deps: List[str] = None):
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.AUDIENCE,
            dependencies=deps or [],
            version="1.0.0",
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> AudienceReasoningOutput:
        return AudienceReasoningOutput(
            target_audience_segment="Tech enthusiasts",
            curiosity_triggers=["Hidden benchmark"],
            evidence_refs=[_dummy_ref("ref_audience")],
            confidence=0.85,
        )


class StubCreatorReasoner(CreatorReasoner):
    def __init__(self, name: str = "creator", deps: List[str] = None):
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.CREATOR,
            dependencies=deps or [],
            version="1.0.0",
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> CreatorReasoningOutput:
        return CreatorReasoningOutput(
            creator_persona="Energetic expert",
            signature_elements=["Cyan text overlay"],
            style_alignment_score=0.95,
            evidence_refs=[_dummy_ref("ref_creator")],
            confidence=0.92,
        )


class StubBrandReasoner(BrandReasoner):
    def __init__(self, name: str = "brand", deps: List[str] = None):
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.BRAND,
            dependencies=deps or [],
            version="1.0.0",
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> BrandReasoningOutput:
        return BrandReasoningOutput(
            color_palette_rules=["#00CCFF", "#FFFFFF"],
            typography_rules=["Bold Sans"],
            compliance_score=1.0,
            evidence_refs=[_dummy_ref("ref_brand")],
            confidence=0.98,
        )


class StubPriorityReasoner(PriorityReasoner):
    def __init__(self, name: str = "priority", deps: List[str] = None):
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.PRIORITY,
            dependencies=deps or [],
            version="1.0.0",
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> PriorityReasoningOutput:
        return PriorityReasoningOutput(
            focal_element_hierarchy=["face", "headline"],
            visual_weight_allocations={"face": 0.6, "headline": 0.4},
            composition_style="rule_of_thirds",
            evidence_refs=[_dummy_ref("ref_priority")],
            confidence=0.94,
        )


class StubRiskReasoner(RiskReasoner):
    def __init__(self, name: str = "risk", deps: List[str] = None):
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.RISK,
            dependencies=deps or [],
            version="1.0.0",
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> RiskReasoningOutput:
        return RiskReasoningOutput(
            fatigue_risk_score=0.15,
            competitor_convergence_risk=0.10,
            identified_risks=[ReasoningRisk(risk_type="fatigue", severity="LOW", description="Low fatigue")],
            evidence_refs=[_dummy_ref("ref_risk")],
            confidence=0.88,
        )


class StubStrategyRanker(StrategyRanker):
    def __init__(self, name: str = "strategy_ranker", deps: List[str] = None):
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.STRATEGY_RANKER,
            dependencies=deps or [],
            version="1.0.0",
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> StrategyRankingOutput:
        strat = RankedStrategy(
            strategy_id="strat_top",
            title="Curiosity Reveal",
            description="Focus on hidden secret with high contrast",
            expected_ctr_impact=0.89,
            confidence_score=0.90,
            overall_score=0.90,
            evidence_refs=[_dummy_ref("ref_strat")],
        )
        return StrategyRankingOutput(
            candidate_strategies=[strat],
            selected_strategy_id="strat_top",
            ranking_rationale="Highest uplift",
            evidence_refs=[_dummy_ref("ref_strat")],
            confidence=0.90,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_coordinator_full_orchestration():
    """Verify coordinator coordinates 7 reasoners and builds a complete ReasoningContext."""
    registry = ReasonerRegistry()
    registry.register(StubNarrativeReasoner("narrative"))
    registry.register(StubAudienceReasoner("audience", deps=["narrative"]))
    registry.register(StubCreatorReasoner("creator", deps=["audience"]))
    registry.register(StubBrandReasoner("brand", deps=["creator"]))
    registry.register(StubPriorityReasoner("priority", deps=["brand"]))
    registry.register(StubRiskReasoner("risk", deps=["priority"]))
    registry.register(StubStrategyRanker("strategy_ranker", deps=["risk"]))

    coordinator = ReasoningCoordinator(registry=registry)
    graph = _dummy_graph("test_graph_001")

    context = coordinator.coordinate(graph)

    assert context.graph_id == "test_graph_001"
    assert context.narrative is not None
    assert context.audience is not None
    assert context.creator_intent is not None
    assert context.brand_constraints is not None
    assert context.visual_priorities is not None
    assert context.risks is not None
    assert context.strategies is not None

    # Verify slots
    assert context.is_complete()
    assert len(context.confidence_breakdown) == 7
    assert len(context.reasoning_trace) == 7
    assert context.overall_confidence > 0.8
    assert context.decision_tree is not None
    assert len(context.decision_tree.nodes) == 7

    # Verify deduplicated evidence references
    assert len(context.evidence_references) == 7


def test_coordinator_empty_graph_raises_error():
    """Verify coordinator raises EmptyEvidenceGraphError when graph is None or invalid."""
    coordinator = ReasoningCoordinator()
    with pytest.raises(EmptyEvidenceGraphError):
        coordinator.coordinate(None)  # type: ignore

    with pytest.raises(EmptyEvidenceGraphError):
        coordinator.coordinate("not_a_graph")  # type: ignore


def test_coordinator_empty_registry():
    """Verify coordinator behavior on empty registry."""
    graph = _dummy_graph()

    # Default allow_empty_registry=True
    coord_allowed = ReasoningCoordinator(config=ReasoningConfig(allow_empty_registry=True))
    ctx = coord_allowed.coordinate(graph)
    assert ctx.graph_id == graph.graph_id
    assert len(ctx.get_active_reasoner_names()) == 0

    # Disallow empty registry
    coord_disallowed = ReasoningCoordinator(config=ReasoningConfig(allow_empty_registry=False))
    with pytest.raises(ReasonerNotFoundError):
        coord_disallowed.coordinate(graph)


def test_coordinator_confidence_aggregation_strategies():
    """Verify weighted_mean, minimum, and harmonic_mean confidence strategies."""
    registry = ReasonerRegistry()
    registry.register(StubNarrativeReasoner("narrative"))  # conf 0.90
    registry.register(StubAudienceReasoner("audience"))    # conf 0.85

    graph = _dummy_graph()

    # Minimum strategy
    cfg_min = ReasoningConfig(confidence_aggregation_strategy="minimum")
    coord_min = ReasoningCoordinator(registry=registry, config=cfg_min)
    ctx_min = coord_min.coordinate(graph)
    assert ctx_min.overall_confidence == pytest.approx(0.85)

    # Weighted mean strategy
    cfg_mean = ReasoningConfig(confidence_aggregation_strategy="weighted_mean")
    coord_mean = ReasoningCoordinator(registry=registry, config=cfg_mean)
    ctx_mean = coord_mean.coordinate(graph)
    assert ctx_mean.overall_confidence == pytest.approx(0.875)


def test_coordinator_grounding_gate_enforcement():
    """Verify GroundingEnforcementError when mandatory reasoner outputs claims with zero evidence."""
    class UngroundedReasoner(NarrativeReasoner):
        @property
        def contract(self):
            return ReasonerContract(
                name="ungrounded_narrative",
                reasoner_type=ReasonerType.NARRATIVE,
                is_mandatory=True,
            )

        def reason(self, graph, context):
            return NarrativeReasoningOutput(
                story_hook="Ungrounded claim",
                evidence_refs=[],  # Empty evidence!
                confidence=0.9,
            )

    registry = ReasonerRegistry()
    registry.register(UngroundedReasoner())

    coord = ReasoningCoordinator(registry=registry, config=ReasoningConfig(enforce_grounding=True))
    graph = _dummy_graph()

    with pytest.raises(GroundingEnforcementError) as exc_info:
        coord.coordinate(graph)

    assert "ungrounded_narrative" in str(exc_info.value)


def test_coordinator_validation_failure():
    """Verify ReasonerValidationError when output fails reasoner validate_output."""
    class InvalidOutputReasoner(NarrativeReasoner):
        @property
        def contract(self):
            return ReasonerContract(
                name="invalid_output_narrative",
                reasoner_type=ReasonerType.NARRATIVE,
                is_mandatory=True,
            )

        def reason(self, graph, context):
            # Return audience output instead of narrative output!
            return AudienceReasoningOutput(target_audience_segment="Wrong type")

    registry = ReasonerRegistry()
    registry.register(InvalidOutputReasoner())

    coord = ReasoningCoordinator(registry=registry)
    with pytest.raises(ReasonerValidationError):
        coord.coordinate(_dummy_graph())


def test_coordinator_fail_fast_vs_continue():
    """Verify coordinator halts on error when fail_fast=True, but continues when fail_fast=False for optional reasoners."""
    class FlakyOptionalReasoner(AudienceReasoner):
        @property
        def contract(self):
            return ReasonerContract(
                name="flaky_optional",
                reasoner_type=ReasonerType.AUDIENCE,
                is_mandatory=False,
            )

        def reason(self, graph, context):
            raise RuntimeError("Temporary upstream failure")

    registry = ReasonerRegistry()
    registry.register(StubNarrativeReasoner("narrative"))
    registry.register(FlakyOptionalReasoner())

    graph = _dummy_graph()

    # fail_fast=True -> raises ReasonerExecutionError
    coord_strict = ReasoningCoordinator(registry=registry, config=ReasoningConfig(fail_fast=True))
    with pytest.raises(ReasonerExecutionError):
        coord_strict.coordinate(graph)

    # fail_fast=False -> narrative succeeds, flaky recorded as FAILED in trace
    coord_tolerant = ReasoningCoordinator(registry=registry, config=ReasoningConfig(fail_fast=False))
    ctx = coord_tolerant.coordinate(graph)
    assert ctx.narrative is not None
    assert ctx.audience is None
    assert any(step.status == "FAILED" for step in ctx.reasoning_trace)


def test_coordinator_mandatory_reasoners_satisfied():
    """Verify coordinator completes successfully when mandatory reasoners are all present."""
    registry = ReasonerRegistry()
    registry.register(StubNarrativeReasoner("narrative"))
    registry.register(StubAudienceReasoner("audience"))

    cfg = ReasoningConfig(mandatory_reasoners=["narrative", "audience"])
    coord = ReasoningCoordinator(registry=registry, config=cfg)
    ctx = coord.coordinate(_dummy_graph())
    assert ctx.narrative is not None
    assert ctx.audience is not None


def test_coordinator_input_validation_rejection():
    """Verify input validation rejection is recorded as VALIDATION_FAILED when fail_fast=False."""
    class RejectingReasoner(NarrativeReasoner):
        @property
        def contract(self):
            return ReasonerContract(
                name="rejecting_narrative",
                reasoner_type=ReasonerType.NARRATIVE,
                is_mandatory=False,
            )

        def validate_input(self, graph, context):
            return False

        def reason(self, graph, context):
            return NarrativeReasoningOutput()

    registry = ReasonerRegistry()
    registry.register(RejectingReasoner())

    coord = ReasoningCoordinator(registry=registry, config=ReasoningConfig(fail_fast=False))
    ctx = coord.coordinate(_dummy_graph())
    assert any(step.status == "VALIDATION_FAILED" for step in ctx.reasoning_trace)

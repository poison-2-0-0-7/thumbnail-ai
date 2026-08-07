"""
test_reasoning_registry.py
==========================

Comprehensive test suite for ReasonerRegistry in Phase 3.4A.
Tests registration, lookup, contract validation, missing dependencies,
circular dependency detection, and deterministic topological ordering.
"""

from __future__ import annotations

from typing import Any, List
import pytest

from thumbnail_intelligence.evidence.models import NormalizedEvidenceGraph
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.exceptions import (
    CircularDependencyError,
    DuplicateReasonerError,
    InvalidReasonerError,
    MissingDependencyError,
    ReasonerNotFoundError,
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
    ReasonerContract,
    ReasonerType,
    RiskReasoningOutput,
    StrategyRankingOutput,
)
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry


# ---------------------------------------------------------------------------
# Mock Reasoners for Testing
# ---------------------------------------------------------------------------


class MockNarrativeReasoner(NarrativeReasoner):
    def __init__(self, name: str = "narrative_test", deps: List[str] = None, ver: str = "1.0.0"):
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.NARRATIVE,
            dependencies=deps or [],
            version=ver,
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> NarrativeReasoningOutput:
        return NarrativeReasoningOutput(story_hook="Test story hook", confidence=0.9)


class MockAudienceReasoner(AudienceReasoner):
    def __init__(self, name: str = "audience_test", deps: List[str] = None):
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
        return AudienceReasoningOutput(target_audience_segment="Tech enthusiast", confidence=0.85)


class MockCreatorReasoner(CreatorReasoner):
    def __init__(self, name: str = "creator_test", deps: List[str] = None):
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
        return CreatorReasoningOutput(creator_persona="High energy", confidence=0.88)


class MockBrandReasoner(BrandReasoner):
    def __init__(self, name: str = "brand_test", deps: List[str] = None):
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
        return BrandReasoningOutput(color_palette_rules=["#FF0000"], confidence=0.95)


class MockPriorityReasoner(PriorityReasoner):
    def __init__(self, name: str = "priority_test", deps: List[str] = None):
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
        return PriorityReasoningOutput(focal_element_hierarchy=["face", "text"], confidence=0.92)


class MockRiskReasoner(RiskReasoner):
    def __init__(self, name: str = "risk_test", deps: List[str] = None):
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
        return RiskReasoningOutput(fatigue_risk_score=0.1, confidence=0.9)


class MockStrategyRanker(StrategyRanker):
    def __init__(self, name: str = "strategy_ranker_test", deps: List[str] = None):
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
        return StrategyRankingOutput(ranking_rationale="High CTR uplift", confidence=0.91)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_registry_basic_crud():
    """Verify registration, lookup, has, count, unregister, and clear."""
    reg = ReasonerRegistry()
    assert reg.count() == 0
    assert not reg.has("narrative_test")

    r1 = MockNarrativeReasoner("narrative_test")
    reg.register(r1)

    assert reg.count() == 1
    assert reg.has("narrative_test")
    assert reg.get("narrative_test") is r1
    assert reg.get_required("narrative_test") is r1
    assert reg.list_names() == ["narrative_test"]
    assert reg.list() == [r1]

    # Unregister
    removed = reg.unregister("narrative_test")
    assert removed is True
    assert reg.count() == 0
    assert reg.get("narrative_test") is None

    # Unregister nonexistent
    assert reg.unregister("nonexistent") is False

    # Clear
    reg.register(r1)
    reg.register(MockAudienceReasoner("audience_test"))
    assert reg.count() == 2
    reg.clear()
    assert reg.count() == 0


def test_registry_get_required_raises_not_found():
    """Verify get_required raises ReasonerNotFoundError with available candidates."""
    reg = ReasonerRegistry()
    reg.register(MockNarrativeReasoner("narrative_test"))

    with pytest.raises(ReasonerNotFoundError) as exc_info:
        reg.get_required("missing_reasoner")

    err = exc_info.value
    assert err.error_code == "REASONER_NOT_FOUND"
    assert "missing_reasoner" in str(err)
    assert "narrative_test" in err.context["available_reasoners"]


def test_registry_duplicate_registration():
    """Verify DuplicateReasonerError when override is False, and replacement when True."""
    reg = ReasonerRegistry()
    r1 = MockNarrativeReasoner("narrative_test", ver="1.0.0")
    reg.register(r1)

    r2 = MockNarrativeReasoner("narrative_test", ver="2.0.0")
    with pytest.raises(DuplicateReasonerError) as exc_info:
        reg.register(r2, override=False)

    assert exc_info.value.error_code == "DUPLICATE_REASONER_ERROR"
    assert reg.get("narrative_test").version == "1.0.0"

    # With override=True
    reg.register(r2, override=True)
    assert reg.get("narrative_test").version == "2.0.0"


def test_registry_validation_invalid_reasoner():
    """Verify contract and interface validation rejects invalid reasoners."""
    reg = ReasonerRegistry()

    # Not a BaseReasoner
    with pytest.raises(InvalidReasonerError):
        reg.register("not_a_reasoner")  # type: ignore

    # Self-dependency
    class SelfDepReasoner(NarrativeReasoner):
        @property
        def contract(self):
            return ReasonerContract(
                name="self_dep",
                reasoner_type=ReasonerType.NARRATIVE,
                dependencies=["self_dep"],
                version="1.0.0",
            )

        def reason(self, graph, context):
            return NarrativeReasoningOutput()

    with pytest.raises(InvalidReasonerError) as exc_info:
        reg.register(SelfDepReasoner())
    assert "cannot depend on itself" in str(exc_info.value)

    # Invalid SemVer
    class InvalidSemVerReasoner(NarrativeReasoner):
        @property
        def contract(self):
            return ReasonerContract(
                name="bad_ver",
                reasoner_type=ReasonerType.NARRATIVE,
                dependencies=[],
                version="invalid-version-string",
            )

        def reason(self, graph, context):
            return NarrativeReasoningOutput()

    with pytest.raises(InvalidReasonerError) as exc_info:
        reg.register(InvalidSemVerReasoner())
    assert "SemVer" in str(exc_info.value)


def test_registry_missing_dependency():
    """Verify check_dependencies and get_execution_order detect missing dependencies."""
    reg = ReasonerRegistry()
    reg.register(MockAudienceReasoner("audience_test", deps=["narrative_test"]))

    # narrative_test is missing
    with pytest.raises(MissingDependencyError) as exc_info:
        reg.get_execution_order()

    err = exc_info.value
    assert err.error_code == "MISSING_REASONER_DEPENDENCY"
    assert "narrative_test" in str(err)


def test_registry_topological_linear_ordering():
    """Verify linear dependency chain: A -> B -> C -> D."""
    reg = ReasonerRegistry()
    reg.register(MockStrategyRanker("D", deps=["C"]))
    reg.register(MockPriorityReasoner("C", deps=["B"]))
    reg.register(MockAudienceReasoner("B", deps=["A"]))
    reg.register(MockNarrativeReasoner("A", deps=[]))

    ordered = reg.get_execution_order()
    ordered_names = [r.name for r in ordered]
    assert ordered_names == ["A", "B", "C", "D"]


def test_registry_topological_diamond_ordering():
    """Verify diamond dependency: A -> (B, C) -> D."""
    reg = ReasonerRegistry()
    reg.register(MockStrategyRanker("D", deps=["B", "C"]))
    reg.register(MockCreatorReasoner("B", deps=["A"]))
    reg.register(MockBrandReasoner("C", deps=["A"]))
    reg.register(MockNarrativeReasoner("A", deps=[]))

    ordered = reg.get_execution_order()
    ordered_names = [r.name for r in ordered]
    assert ordered_names[0] == "A"
    assert set(ordered_names[1:3]) == {"B", "C"}
    assert ordered_names[3] == "D"


def test_registry_circular_dependency_direct():
    """Verify circular dependency detection on direct cycle: A -> B -> A."""
    reg = ReasonerRegistry()
    reg.register(MockNarrativeReasoner("A", deps=["B"]))
    reg.register(MockAudienceReasoner("B", deps=["A"]))

    with pytest.raises(CircularDependencyError) as exc_info:
        reg.get_execution_order()

    assert exc_info.value.error_code == "CIRCULAR_REASONER_DEPENDENCY"


def test_registry_circular_dependency_indirect():
    """Verify circular dependency detection on 3-node cycle: A -> B -> C -> A."""
    reg = ReasonerRegistry()
    reg.register(MockNarrativeReasoner("A", deps=["C"]))
    reg.register(MockAudienceReasoner("B", deps=["A"]))
    reg.register(MockCreatorReasoner("C", deps=["B"]))

    with pytest.raises(CircularDependencyError) as exc_info:
        reg.get_execution_order()

    assert exc_info.value.error_code == "CIRCULAR_REASONER_DEPENDENCY"


def test_registry_clone():
    """Verify clone creates an independent registry copy."""
    reg1 = ReasonerRegistry()
    reg1.register(MockNarrativeReasoner("narrative_test"))

    reg2 = reg1.clone()
    assert reg2.count() == 1
    assert reg2.has("narrative_test")

    reg2.register(MockAudienceReasoner("audience_test"))
    assert reg2.count() == 2
    assert reg1.count() == 1

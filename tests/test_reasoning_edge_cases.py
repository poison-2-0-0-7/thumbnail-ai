"""
test_reasoning_edge_cases.py
============================

Test suite for edge cases, failure modes, boundary conditions,
and error handling in the Strategic Reasoning Coordinator Foundation (Phase 3.4A).
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
    CoordinatorError,
    GroundingEnforcementError,
    ReasonerExecutionError,
    ReasonerNotFoundError,
    ReasonerValidationError,
)
from thumbnail_intelligence.reasoning.interfaces import BaseReasoner, NarrativeReasoner
from thumbnail_intelligence.reasoning.models import (
    NarrativeReasoningOutput,
    ReasonerContract,
    ReasonerType,
)
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry


def _make_graph(graph_id: str = "edge_graph_001") -> NormalizedEvidenceGraph:
    return NormalizedEvidenceGraph(
        graph_id=graph_id,
        summary=EvidenceSummary(graph_id=graph_id, primary_archetype="edge_case_archetype"),
    )


def _make_ref(source_id: str) -> EvidenceReference:
    return EvidenceReference(
        source_id=source_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.9,
        grade=EvidenceGrade.STRONG,
        claim_summary=f"Evidence for {source_id}",
    )


def test_deep_dependency_chain():
    """Verify registry and coordinator handle deep (10+ reasoners) linear dependency chains."""
    registry = ReasonerRegistry()
    chain_length = 12

    for i in range(chain_length):
        rname = f"reasoner_{i:02d}"
        deps = [f"reasoner_{i-1:02d}"] if i > 0 else []

        class ChainReasoner(BaseReasoner):
            def __init__(self, name: str, dependencies: List[str]):
                self._contract = ReasonerContract(
                    name=name,
                    reasoner_type=ReasonerType.CUSTOM,
                    dependencies=dependencies,
                    version="1.0.0",
                )

            @property
            def contract(self) -> ReasonerContract:
                return self._contract

            def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> dict:
                return {"step": self.name, "evidence_refs": [_make_ref(f"ref_{self.name}")]}

            def validate_output(self, output: Any) -> bool:
                return isinstance(output, dict)

        registry.register(ChainReasoner(rname, deps))

    order = registry.get_execution_order()
    assert len(order) == chain_length
    for idx, r in enumerate(order):
        assert r.name == f"reasoner_{idx:02d}"

    coordinator = ReasoningCoordinator(registry=registry)
    ctx = coordinator.coordinate(_make_graph())
    assert len(ctx.custom_outputs) == chain_length
    assert len(ctx.evidence_references) == chain_length


def test_min_confidence_threshold_filtering():
    """Verify outputs below min_confidence_threshold are skipped from merging."""
    class LowConfidenceReasoner(NarrativeReasoner):
        @property
        def contract(self):
            return ReasonerContract(
                name="low_conf_narrative",
                reasoner_type=ReasonerType.NARRATIVE,
                is_mandatory=False,
            )

        def reason(self, graph, context):
            return NarrativeReasoningOutput(
                story_hook="Weak speculative hook",
                evidence_refs=[_make_ref("ref_low_conf")],
                confidence=0.30,  # Below threshold
            )

    registry = ReasonerRegistry()
    registry.register(LowConfidenceReasoner())

    coord = ReasoningCoordinator(
        registry=registry,
        config=ReasoningConfig(min_confidence_threshold=0.50, fail_fast=False),
    )
    ctx = coord.coordinate(_make_graph())

    # Slot should remain None because confidence was below 0.50
    assert ctx.narrative is None
    assert any(step.status == "SKIPPED" for step in ctx.reasoning_trace)


def test_max_trace_steps_truncation():
    """Verify reasoning_trace respects max_trace_steps config limit."""
    registry = ReasonerRegistry()
    for i in range(5):
        class StepReasoner(BaseReasoner):
            def __init__(self, name):
                self._contract = ReasonerContract(
                    name=name,
                    reasoner_type=ReasonerType.CUSTOM,
                    version="1.0.0",
                )

            @property
            def contract(self):
                return self._contract

            def reason(self, graph, context):
                return {"id": self.name}

            def validate_output(self, output):
                return True

        registry.register(StepReasoner(f"r_{i}"))

    coord = ReasoningCoordinator(
        registry=registry,
        config=ReasoningConfig(max_trace_steps=3),
    )
    ctx = coord.coordinate(_make_graph())
    assert len(ctx.reasoning_trace) == 3


def test_harmonic_mean_confidence_edge_cases():
    """Verify harmonic mean calculation with zero and positive values."""
    coord = ReasoningCoordinator(
        config=ReasoningConfig(confidence_aggregation_strategy="harmonic_mean")
    )
    # Empty
    assert coord._aggregate_confidence({}) == 1.0

    # Single value
    assert coord._aggregate_confidence({"r1": 0.8}) == pytest.approx(0.8)

    # Multi values
    res = coord._aggregate_confidence({"r1": 0.5, "r2": 1.0})
    expected = 2.0 / (1.0 / 0.5 + 1.0 / 1.0)  # 2.0 / 3.0 = 0.666666...
    assert res == pytest.approx(expected)

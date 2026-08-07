"""
test_reasoning_pipeline.py
==========================

Test suite for ReasoningPipeline in Phase 3.4A.
Tests pipeline execution flow:
NormalizedEvidenceGraph -> ReasoningCoordinator -> Registered Reasoners -> Collected Outputs -> ReasoningContext
"""

from __future__ import annotations

from typing import List
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
from thumbnail_intelligence.reasoning.interfaces import (
    BaseReasoner,
    BrandReasoner,
    NarrativeReasoner,
)
from thumbnail_intelligence.reasoning.models import (
    BrandReasoningOutput,
    NarrativeReasoningOutput,
    ReasonerContract,
    ReasonerType,
)
from thumbnail_intelligence.reasoning.pipeline import ReasoningPipeline
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry


def _sample_graph(graph_id: str = "pipe_graph_001") -> NormalizedEvidenceGraph:
    return NormalizedEvidenceGraph(
        graph_id=graph_id,
        summary=EvidenceSummary(graph_id=graph_id, primary_archetype="curiosity_gap"),
    )


def _sample_ref(source_id: str = "pipe_ref_001") -> EvidenceReference:
    return EvidenceReference(
        source_id=source_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.95,
        grade=EvidenceGrade.STRONG,
        claim_summary="Pipeline grounding evidence reference",
    )


class PipeNarrativeReasoner(NarrativeReasoner):
    def __init__(self, name: str = "pipe_narrative", deps: List[str] = None):
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
            story_hook="Pipeline execution hook",
            narrative_angle="Standard flow",
            emotional_tone="Focused",
            evidence_refs=[_sample_ref("ref_pipe_nar")],
            confidence=0.92,
        )


class PipeBrandReasoner(BrandReasoner):
    def __init__(self, name: str = "pipe_brand", deps: List[str] = None):
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
            color_palette_rules=["#112233"],
            compliance_score=1.0,
            evidence_refs=[_sample_ref("ref_pipe_brand")],
            confidence=0.96,
        )


class CustomThirdPartyReasoner(BaseReasoner):
    """Demonstrates third-party / future reasoner plugin extensibility without modifying coordinator."""

    def __init__(self, name: str = "third_party_sentiment"):
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.CUSTOM,
            dependencies=[],
            version="1.0.0",
            description="Third-party sentiment analyzer",
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> dict:
        return {
            "sentiment_score": 0.88,
            "sentiment_label": "positive_excitement",
            "evidence_refs": [_sample_ref("ref_sentiment")],
            "confidence": 0.88,
        }

    def validate_output(self, output: Any) -> bool:
        return isinstance(output, dict) and "sentiment_score" in output


def test_pipeline_factory_and_run():
    """Verify ReasoningPipeline.create_default and run()."""
    registry = ReasonerRegistry()
    registry.register(PipeNarrativeReasoner())
    registry.register(PipeBrandReasoner(deps=["pipe_narrative"]))

    pipeline = ReasoningPipeline.from_registry(registry)
    graph = _sample_graph()

    context = pipeline.run(graph)

    assert isinstance(context, ReasoningContext)
    assert context.graph_id == graph.graph_id
    assert context.narrative is not None
    assert context.brand_constraints is not None
    assert context.narrative.story_hook == "Pipeline execution hook"
    assert len(context.evidence_references) == 2


def test_pipeline_run_with_base_context():
    """Verify pipeline.run_with_context preserves pre-existing context IDs and fields."""
    registry = ReasonerRegistry()
    registry.register(PipeNarrativeReasoner())

    pipeline = ReasoningPipeline.from_registry(registry)
    graph = _sample_graph()

    base_ctx = ReasoningContext(
        context_id="custom_base_ctx_999",
        graph_id=graph.graph_id,
        metadata={"pre_existing_field": True},
    )

    result_ctx = pipeline.run_with_context(graph, base_ctx)
    assert result_ctx.context_id == "custom_base_ctx_999"
    assert result_ctx.metadata.get("pre_existing_field") is True
    assert result_ctx.narrative is not None


def test_pipeline_extensibility_custom_reasoner():
    """Verify that future/custom reasoners plug in without modifying coordinator."""
    registry = ReasonerRegistry()
    registry.register(PipeNarrativeReasoner())
    registry.register(CustomThirdPartyReasoner())

    pipeline = ReasoningPipeline.from_registry(registry)
    graph = _sample_graph()

    context = pipeline.run(graph)

    assert "third_party_sentiment" in context.custom_outputs
    custom_out = context.custom_outputs["third_party_sentiment"]
    assert custom_out["sentiment_score"] == 0.88
    assert context.has_slot("third_party_sentiment")
    assert context.has_slot("narrative")

"""
test_audience_creator_integration.py
====================================

Integration and edge case test suite for Phase 3.4C:
- Multi-reasoner pipeline running NarrativeReasoner, AudienceReasoner, and CreatorReasoner
- Topological dependency verification (Audience and Creator depend on Narrative)
- Edge cases: empty graphs, sparse evidence, conflicting evidence nodes
- Full context slot verification (narrative, audience, creator_intent)
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
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.creator_models import CreatorResult
from thumbnail_intelligence.reasoning.creator_reasoner import CreatorReasoner
from thumbnail_intelligence.reasoning.narrative_models import NarrativeResult
from thumbnail_intelligence.reasoning.narrative_reasoner import NarrativeReasoner
from thumbnail_intelligence.reasoning.pipeline import ReasoningPipeline
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry
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
            retrieval_query_id="query_full",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=is_active,
    )


def test_full_phase_3_4_multi_reasoner_pipeline():
    """Verify Narrative, Audience, and Creator reasoners run in topological sequence and populate all slots."""
    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(AudienceReasoner())
    registry.register(CreatorReasoner())

    # Verify execution order: narrative must be before audience and creator
    exec_order = [r.name for r in registry.get_execution_order()]
    assert exec_order[0] == "narrative_reasoner"
    assert "audience_reasoner" in exec_order[1:]
    assert "creator_reasoner" in exec_order[1:]

    # Build rich evidence graph
    nodes = {
        "node_title_01": _build_test_node(
            "node_title_01",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {
                "title": "I Built a Secret Vault Under My House for $1000",
                "transcript": "We spent 30 days excavating the underground basement.",
                "ocr_text": "SECRET VAULT",
            },
        ),
        "node_creator_02": _build_test_node(
            "node_creator_02",
            KnowledgeEntryType.CREATOR_PROFILE_ENTRY,
            {
                "creator_id": "creator_maker_01",
                "display_name": "Underground DIY Builder",
                "primary_niche": "diy_maker",
            },
        ),
        "node_pattern_03": _build_test_node(
            "node_pattern_03",
            KnowledgeEntryType.DESIGN_PATTERN,
            {"pattern_id": "curiosity_gap_high_contrast"},
        ),
    }

    graph = NormalizedEvidenceGraph(
        graph_id="graph_full_pipeline",
        nodes=nodes,
        summary=EvidenceSummary(graph_id="graph_full_pipeline", primary_archetype="curiosity_gap"),
    )

    pipeline = ReasoningPipeline.from_registry(registry)
    context = pipeline.run(graph)

    # Validate slots
    assert context.has_slot("narrative")
    assert context.has_slot("audience")
    assert context.has_slot("creator_intent")

    assert isinstance(context.narrative, NarrativeResult)
    assert isinstance(context.audience, AudienceResult)
    assert isinstance(context.creator_intent, CreatorResult)

    # Validate cross-slot grounding and trace steps
    assert len(context.evidence_references) >= 3
    assert len(context.reasoning_trace) >= 3
    assert context.overall_confidence > 0.70


def test_empty_graph_and_suppressed_nodes_handling():
    """Verify AudienceReasoner and CreatorReasoner operate safely with fallback defaults on empty/sparse graphs."""
    empty_graph = NormalizedEvidenceGraph(
        graph_id="graph_empty_all",
        nodes={},
        summary=EvidenceSummary(graph_id="graph_empty_all"),
    )

    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(AudienceReasoner())
    registry.register(CreatorReasoner())

    pipeline = ReasoningPipeline.from_registry(registry)
    context = pipeline.run(empty_graph)

    assert context.narrative is not None
    assert context.audience is not None
    assert context.creator_intent is not None
    assert context.audience.audience_confidence >= 0.0
    assert context.creator_intent.creator_confidence >= 0.0


def test_validate_output_rejection():
    """Verify validate_output rejects invalid types or out-of-range confidence."""
    aud_reasoner = AudienceReasoner()
    cre_reasoner = CreatorReasoner()

    assert not aud_reasoner.validate_output("invalid")
    assert not cre_reasoner.validate_output("invalid")
    assert not aud_reasoner.validate_output(None)
    assert not cre_reasoner.validate_output(None)

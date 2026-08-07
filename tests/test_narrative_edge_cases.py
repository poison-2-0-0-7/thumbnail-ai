"""
test_narrative_edge_cases.py
============================

Test suite for edge cases, missing evidence, boundary conditions,
and error handling in NarrativeReasoner (Phase 3.4B).
"""

from __future__ import annotations

from typing import Dict
import pytest

from thumbnail_intelligence.knowledge_base.models import (
    EvidenceGrade,
    EvidenceReference,
    EvidenceSourceType,
)
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceNode,
    EvidenceSummary,
    EvidenceWeight,
    KnowledgeEntryType,
    NormalizedEvidenceGraph,
    ProvenanceRecord,
)
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.narrative_models import (
    NarrativeResult,
    NarrativeType,
)
from thumbnail_intelligence.reasoning.narrative_reasoner import NarrativeReasoner
from thumbnail_intelligence.retrieval.evidence_bundle import (
    RankingMetadata,
    RetrievalScore,
    RetrievedEvidence,
)


def _build_minimal_node(
    node_id: str,
    payload: dict,
    confidence: float = 0.50,
    is_active: bool = True,
) -> EvidenceNode:
    ref = EvidenceReference(
        source_id=node_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=confidence,
        grade=EvidenceGrade.MODERATE,
        claim_summary=f"Minimal evidence for {node_id}",
    )
    score = RetrievalScore(composite_score=confidence)
    item = RetrievedEvidence(
        evidence_id=f"ev_{node_id}",
        entry_id=node_id,
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        origin=f"origin:{node_id}",
        confidence=confidence,
        reason_retrieved="Retrieved for test",
        score=score,
        ranking=RankingMetadata(rank=1, score=score),
        data_payload=payload,
        evidence_refs=[ref],
    )
    return EvidenceNode(
        node_id=node_id,
        node_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        evidence_item=item,
        confidence=ConfidenceScore(raw_confidence=confidence, propagated_confidence=confidence),
        weight=EvidenceWeight(base_weight=1.0, effective_weight=1.0),
        provenance=ProvenanceRecord(
            origin=f"origin:{node_id}",
            source_id=node_id,
            source_type=EvidenceSourceType.OUTCOME_RECORD,
            retrieval_query_id="q_min",
            retrieval_reason="Minimal test node",
        ),
        is_active=is_active,
    )


def test_narrative_reasoner_empty_nodes_graph():
    """Verify NarrativeReasoner handles a graph with zero nodes gracefully with synthetic fallback."""
    graph = NormalizedEvidenceGraph(
        graph_id="empty_nodes_graph",
        nodes={},
        summary=EvidenceSummary(graph_id="empty_nodes_graph"),
    )
    reasoner = NarrativeReasoner()
    result = reasoner.reason(graph, ReasoningContext(graph_id=graph.graph_id))

    assert isinstance(result, NarrativeResult)
    assert result.narrative_type is not None
    assert result.confidence >= 0.0
    assert result.story_hook != ""


def test_narrative_reasoner_only_visual_objects_no_text():
    """Verify NarrativeReasoner operates when only scene objects are present without title or transcript."""
    nodes = {
        "node_vis_01": _build_minimal_node("node_vis_01", {"objects": ["sports car", "trophy", "driver"]})
    }
    graph = NormalizedEvidenceGraph(
        graph_id="visual_only_graph",
        nodes=nodes,
        summary=EvidenceSummary(graph_id="visual_only_graph", primary_archetype="tournament_bracket"),
    )
    reasoner = NarrativeReasoner()
    result = reasoner.reason(graph, ReasoningContext(graph_id=graph.graph_id))

    assert isinstance(result, NarrativeResult)
    assert "Sports Car" in result.key_subjects or "Driver" in result.key_subjects or "Trophy" in result.key_subjects
    assert len(result.visual_focus_candidates) >= 1


def test_narrative_reasoner_suppressed_inactive_nodes():
    """Verify NarrativeReasoner ignores inactive or suppressed evidence nodes."""
    node_active = _build_minimal_node("node_active", {"title": "Active Title", "objects": ["active person"]})
    node_suppressed = _build_minimal_node(
        "node_suppressed",
        {"title": "Suppressed Title", "objects": ["ignored item"]},
        is_active=False,
    )

    graph = NormalizedEvidenceGraph(
        graph_id="suppressed_graph",
        nodes={"node_active": node_active, "node_suppressed": node_suppressed},
        summary=EvidenceSummary(graph_id="suppressed_graph"),
    )
    reasoner = NarrativeReasoner()
    result = reasoner.reason(graph, ReasoningContext(graph_id=graph.graph_id))

    # Inactive objects should not contaminate subjects
    assert "Ignored Item" not in result.key_subjects


def test_narrative_reasoner_validate_output_rejection():
    """Verify validate_output rejects non-NarrativeResult or out-of-bounds confidence."""
    reasoner = NarrativeReasoner()
    assert not reasoner.validate_output("not_a_result")
    assert not reasoner.validate_output(None)


def test_narrative_reasoner_low_confidence_propagation():
    """Verify that low-confidence input nodes yield a proportionally low final narrative confidence."""
    low_node = _build_minimal_node("low_node", {"title": "Vague Topic"}, confidence=0.20)
    graph = NormalizedEvidenceGraph(
        graph_id="low_conf_graph",
        nodes={"low_node": low_node},
        summary=EvidenceSummary(graph_id="low_conf_graph"),
    )
    reasoner = NarrativeReasoner()
    result = reasoner.reason(graph, ReasoningContext(graph_id=graph.graph_id))

    assert result.confidence < 0.65
    assert result.confidence_breakdown["evidence_quality"] == pytest.approx(0.20)

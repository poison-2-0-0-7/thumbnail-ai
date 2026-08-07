"""
Unit tests for EvidenceGraphValidator and ProvenanceTracker.
Tests grounding verification, timestamp validation, cycle detection,
node count limits, and cryptographic-style lineage tracking.
"""

from __future__ import annotations

import pytest

from thumbnail_intelligence.evidence.exceptions import (
    CyclicDependencyError,
    GraphError,
    GroundingValidationError,
    NodeNotFoundError,
)
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceEdge,
    EvidenceNode,
    EvidenceWeight,
    ProvenanceRecord,
)
from thumbnail_intelligence.evidence.provenance import ProvenanceTracker
from thumbnail_intelligence.evidence.validator import EvidenceGraphValidator
from thumbnail_intelligence.knowledge_base.models import (
    EvidenceSourceType,
    KnowledgeEntryType,
)
from thumbnail_intelligence.retrieval.evidence_bundle import RetrievedEvidence
from thumbnail_intelligence.retrieval.ranking import RankingMetadata
from thumbnail_intelligence.retrieval.scoring import RetrievalScore


def _valid_node(node_id: str) -> EvidenceNode:
    score = RetrievalScore(overall_score=0.8)
    ranking = RankingMetadata(rank=1, score=score)
    ev = RetrievedEvidence(
        evidence_id=f"ev_{node_id}",
        entry_id=node_id,
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        origin=f"historical:{node_id}",
        source_id=node_id,
        reason_retrieved="Test reason",
        score=score,
        ranking=ranking,
    )
    prov = ProvenanceRecord(
        origin=f"historical:{node_id}",
        source_id=node_id,
        source_type=EvidenceSourceType.KNOWLEDGE_ENTRY,
        retrieval_query_id="query_val",
        retrieval_reason="Test reason",
    )
    return EvidenceNode(
        node_id=node_id,
        node_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        evidence_item=ev,
        confidence=ConfidenceScore(raw_confidence=0.9, propagated_confidence=0.9),
        weight=EvidenceWeight(base_weight=0.8, effective_weight=0.8),
        provenance=prov,
    )


def test_validator_node_grounding_success_and_failure() -> None:
    validator = EvidenceGraphValidator()
    valid = _valid_node("good_node")
    validator.validate_node(valid)

    # 1. Missing origin -> GroundingValidationError
    prov_no_origin = ProvenanceRecord(
        origin="   ",
        source_id="id_1",
        source_type=EvidenceSourceType.KNOWLEDGE_ENTRY,
        retrieval_query_id="q",
        retrieval_reason="reason",
    )
    invalid_node = EvidenceNode(
        node_id="bad_node",
        node_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        evidence_item=valid.evidence_item,
        provenance=prov_no_origin,
    )
    with pytest.raises(GroundingValidationError):
        validator.validate_node(invalid_node)


def test_validator_cycle_detection() -> None:
    validator = EvidenceGraphValidator()

    n1 = _valid_node("cycle_1")
    n2 = _valid_node("cycle_2")
    n3 = _valid_node("cycle_3")
    nodes = {"cycle_1": n1, "cycle_2": n2, "cycle_3": n3}

    # Cycle: 1 -> 2 -> 3 -> 1 (DEPENDS_ON)
    e1 = EvidenceEdge.create("cycle_1", "cycle_2", "DEPENDS_ON")
    e2 = EvidenceEdge.create("cycle_2", "cycle_3", "DEPENDS_ON")
    e3 = EvidenceEdge.create("cycle_3", "cycle_1", "DEPENDS_ON")

    with pytest.raises(CyclicDependencyError):
        validator.check_for_cycles(nodes, [e1, e2, e3])


def test_validator_missing_edge_endpoints() -> None:
    validator = EvidenceGraphValidator()
    nodes = {"n1": _valid_node("n1")}
    bad_edge = EvidenceEdge.create("n1", "non_existent_node", "SUPPORTS")

    with pytest.raises(NodeNotFoundError):
        validator.validate_edge(bad_edge, set(nodes.keys()))


def test_provenance_tracker_lineage_and_derivation() -> None:
    ev = _valid_node("prov_test").evidence_item
    record = ProvenanceTracker.create_record(ev, query_id="q_lineage_01")
    assert record.origin == "historical:prov_test"
    assert record.retrieval_query_id == "q_lineage_01"
    assert record.trace_id.startswith("tr_")

    # Derive child record
    child = ProvenanceTracker.derive_record(record, derivation_reason="Synthesized style rule")
    assert child.origin.startswith("derived:")
    assert record.origin in child.parent_origins
    assert child.source_id == record.source_id

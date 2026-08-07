"""
Unit tests for Evidence Normalization Engine edge cases and failure resilience.
Tests empty retrieval bundles, single-item graphs, boundary confidences,
and extreme graph topology conditions.
"""

from __future__ import annotations

import pytest

from thumbnail_intelligence.evidence.config import EvidenceNormalizationConfig
from thumbnail_intelligence.evidence.exceptions import GraphError
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceConflict,
    EvidenceNode,
    EvidenceWeight,
    NormalizedEvidenceGraph,
    ProvenanceRecord,
)
from thumbnail_intelligence.evidence.normalizer import EvidenceNormalizer
from thumbnail_intelligence.evidence.validator import EvidenceGraphValidator
from thumbnail_intelligence.knowledge_base.models import (
    EvidenceSourceType,
    KnowledgeEntryType,
)
from thumbnail_intelligence.retrieval.evidence_bundle import (
    EvidenceBundle,
    RetrievalResult,
    RetrievedEvidence,
)
from thumbnail_intelligence.retrieval.query import RetrievalQuery
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


def test_evidence_normalizer_empty_bundle() -> None:
    normalizer = EvidenceNormalizer()
    empty_bundle = EvidenceBundle(query_id="q_empty", items=[])
    result = RetrievalResult(
        query=RetrievalQuery(query_id="q_empty"),
        bundle=empty_bundle,
        status="empty",
    )

    graph = normalizer.normalize(result)
    assert isinstance(graph, NormalizedEvidenceGraph)
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0
    assert graph.statistics.valid_nodes_count == 0
    assert graph.statistics.average_confidence == 0.0


def test_evidence_validator_max_node_limit() -> None:
    validator = EvidenceGraphValidator()
    nodes = {f"node_{i}": _valid_node(f"node_{i}") for i in range(10)}

    with pytest.raises(GraphError):
        validator.validate_graph(nodes, [], max_nodes=5)

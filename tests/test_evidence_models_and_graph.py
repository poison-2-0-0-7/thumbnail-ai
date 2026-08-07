"""
Unit tests for EvidenceNode, EvidenceEdge, EvidenceCluster, and EvidenceGraph.
Tests model instantiation, edge creation, graph indexing, adjacency queries,
and topological sorting on dependency DAGs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from thumbnail_intelligence.evidence.graph import EvidenceGraph
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceCluster,
    EvidenceEdge,
    EvidenceNode,
    EvidenceWeight,
    NormalizedEvidenceGraph,
    ProvenanceRecord,
)
from thumbnail_intelligence.knowledge_base.models import (
    EvidenceSourceType,
    KnowledgeEntryType,
)
from thumbnail_intelligence.retrieval.evidence_bundle import RetrievedEvidence
from thumbnail_intelligence.retrieval.ranking import RankingMetadata
from thumbnail_intelligence.retrieval.scoring import RetrievalScore


def _create_sample_node(node_id: str, node_type: KnowledgeEntryType = KnowledgeEntryType.HISTORICAL_THUMBNAIL) -> EvidenceNode:
    score = RetrievalScore(overall_score=0.85)
    ranking = RankingMetadata(rank=1, score=score)
    ev = RetrievedEvidence(
        evidence_id=f"ev_{node_id}",
        entry_id=node_id,
        entry_type=node_type,
        origin=f"{node_type.value}:{node_id}",
        source_id=node_id,
        reason_retrieved="High visual similarity and channel affinity",
        score=score,
        ranking=ranking,
    )
    prov = ProvenanceRecord(
        origin=f"{node_type.value}:{node_id}",
        source_id=node_id,
        source_type=EvidenceSourceType.KNOWLEDGE_ENTRY,
        retrieval_query_id="query_sample",
        retrieval_reason="Test reason",
    )
    return EvidenceNode(
        node_id=node_id,
        node_type=node_type,
        evidence_item=ev,
        confidence=ConfidenceScore(raw_confidence=0.90, propagated_confidence=0.90),
        weight=EvidenceWeight(base_weight=0.85, effective_weight=0.85),
        provenance=prov,
    )


def test_evidence_node_and_edge_creation() -> None:
    node1 = _create_sample_node("node_01")
    node2 = _create_sample_node("node_02")

    assert node1.node_id == "node_01"
    assert node1.is_active is True

    # Create directed edge
    edge = EvidenceEdge.create(
        source_id=node1.node_id,
        target_id=node2.node_id,
        relation_type="SUPPORTS",
        weight=1.0,
        confidence=0.90,
        explanation="node_01 supports node_02 hypothesis",
    )
    assert edge.source_node_id == "node_01"
    assert edge.target_node_id == "node_02"
    assert edge.relation_type == "SUPPORTS"


def test_evidence_graph_indexing_and_adjacency() -> None:
    graph = EvidenceGraph(graph_id="test_graph_01")

    n1 = _create_sample_node("n1", KnowledgeEntryType.HISTORICAL_THUMBNAIL)
    n2 = _create_sample_node("n2", KnowledgeEntryType.ARCHETYPE_EXAMPLE)
    n3 = _create_sample_node("n3", KnowledgeEntryType.VISUAL_PATTERN)

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    assert len(graph.nodes) == 3
    assert graph.get_node("n1") is not None
    assert graph.get_node("missing") is None

    # Add edges: n1 -> n2 (SUPPORTS), n3 -> n2 (DEPENDS_ON)
    e1 = EvidenceEdge.create("n1", "n2", "SUPPORTS")
    e2 = EvidenceEdge.create("n3", "n2", "DEPENDS_ON")
    graph.add_edge(e1)
    graph.add_edge(e2)

    # Outgoing and incoming checks
    assert len(graph.get_outgoing("n1")) == 1
    assert len(graph.get_incoming("n2")) == 2
    assert len(graph.get_supporting_evidence("n2")) == 1
    assert graph.get_supporting_evidence("n2")[0].node_id == "n1"

    # Filter nodes by type
    arch_nodes = graph.get_nodes_by_type(KnowledgeEntryType.ARCHETYPE_EXAMPLE)
    assert len(arch_nodes) == 1
    assert arch_nodes[0].node_id == "n2"


def test_evidence_graph_topological_sort() -> None:
    graph = EvidenceGraph(graph_id="dag_graph")

    # DAG: A -> B -> C (A depends on B, B depends on C)
    nA = _create_sample_node("nA")
    nB = _create_sample_node("nB")
    nC = _create_sample_node("nC")

    graph.add_node(nA)
    graph.add_node(nB)
    graph.add_node(nC)

    e1 = EvidenceEdge.create("nA", "nB", "DEPENDS_ON")
    e2 = EvidenceEdge.create("nB", "nC", "DEPENDS_ON")
    graph.add_edge(e1)
    graph.add_edge(e2)

    order = graph.topological_sort()
    assert len(order) == 3
    assert order.index("nA") < order.index("nB") < order.index("nC")

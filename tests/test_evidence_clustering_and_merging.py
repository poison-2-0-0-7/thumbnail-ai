"""
Unit tests for EvidenceClusterer and EvidenceMerger.
Tests domain clustering, cohesion calculation, duplicate merging,
and edge re-pointing.
"""

from __future__ import annotations

import pytest

from thumbnail_intelligence.evidence.clustering import EvidenceClusterer
from thumbnail_intelligence.evidence.merger import EvidenceMerger
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceEdge,
    EvidenceNode,
    EvidenceWeight,
    ProvenanceRecord,
)
from thumbnail_intelligence.knowledge_base.models import (
    EvidenceSourceType,
    KnowledgeEntryType,
)
from thumbnail_intelligence.retrieval.evidence_bundle import RetrievedEvidence
from thumbnail_intelligence.retrieval.ranking import RankingMetadata
from thumbnail_intelligence.retrieval.scoring import RetrievalScore


def _create_node_with_emb(node_id: str, node_type: KnowledgeEntryType, vec: list[float], conf: float = 0.9) -> EvidenceNode:
    score = RetrievalScore(overall_score=conf)
    ranking = RankingMetadata(rank=1, score=score)
    ev = RetrievedEvidence(
        evidence_id=f"ev_{node_id}",
        entry_id=node_id,
        entry_type=node_type,
        origin=f"{node_type.value}:{node_id}",
        source_id=node_id,
        reason_retrieved="Test reason",
        score=score,
        ranking=ranking,
        data_payload={"embedding": vec},
    )
    prov = ProvenanceRecord(
        origin=f"{node_type.value}:{node_id}",
        source_id=node_id,
        source_type=EvidenceSourceType.KNOWLEDGE_ENTRY,
        retrieval_query_id="query_cluster",
        retrieval_reason="Test reason",
    )
    return EvidenceNode(
        node_id=node_id,
        node_type=node_type,
        evidence_item=ev,
        confidence=ConfidenceScore(raw_confidence=conf, propagated_confidence=conf),
        weight=EvidenceWeight(base_weight=conf, effective_weight=conf),
        provenance=prov,
    )


def test_evidence_clustering_domain_and_cohesion() -> None:
    n1 = _create_node_with_emb("arch_01", KnowledgeEntryType.ARCHETYPE_EXAMPLE, [1.0, 0.0, 0.0, 0.0], 0.95)
    n2 = _create_node_with_emb("arch_02", KnowledgeEntryType.ARCHETYPE_EXAMPLE, [0.9, 0.1, 0.0, 0.0], 0.85)
    n3 = _create_node_with_emb("hist_01", KnowledgeEntryType.HISTORICAL_THUMBNAIL, [0.0, 1.0, 0.0, 0.0], 0.90)

    nodes = {n1.node_id: n1, n2.node_id: n2, n3.node_id: n3}
    clusters, edges = EvidenceClusterer.cluster_evidence(nodes, threshold=0.70)

    assert len(clusters) == 2
    arch_cluster = next(c for c in clusters if c.cluster_type == "archetype")
    assert arch_cluster.central_node_id == "arch_01"
    assert len(arch_cluster.node_ids) == 2
    assert arch_cluster.cohesion_score > 0.80

    # Verify PART_OF_CLUSTER edge from arch_02 to arch_01
    part_edges = [e for e in edges if e.relation_type == "PART_OF_CLUSTER"]
    assert len(part_edges) == 1
    assert part_edges[0].source_node_id == "arch_02"
    assert part_edges[0].target_node_id == "arch_01"


def test_evidence_merging_duplicates_and_edge_remapping() -> None:
    # Two nodes with same source_id
    n1 = _create_node_with_emb("node_dup_1", KnowledgeEntryType.HISTORICAL_THUMBNAIL, [1.0, 0.0], 0.80)
    object.__setattr__(n1.provenance, "source_id", "video_common_123")

    n2 = _create_node_with_emb("node_dup_2", KnowledgeEntryType.HISTORICAL_THUMBNAIL, [1.0, 0.0], 0.92)
    object.__setattr__(n2.provenance, "source_id", "video_common_123")

    nodes = {n1.node_id: n1, n2.node_id: n2}
    merged, id_map = EvidenceMerger.merge_duplicates(nodes)

    assert len(merged) == 1
    # n2 should be canonical because it has higher confidence (0.92 vs 0.80)
    assert "node_dup_2" in merged
    assert id_map["node_dup_1"] == "node_dup_2"

    # Remap an edge pointing to node_dup_1
    edge = EvidenceEdge.create("source_node", "node_dup_1", "SUPPORTS")
    remapped = EvidenceMerger.remap_edges([edge], id_map)
    assert len(remapped) == 1
    assert remapped[0].target_node_id == "node_dup_2"

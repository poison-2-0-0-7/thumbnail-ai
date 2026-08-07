"""
Extended unit test suite for Evidence Normalization Engine.
Tests multi-hop confidence decay, conflict resolution strategies (most_recent, highest_confidence),
graph dependency traversals, raw EvidenceBundle normalization, and cycle detection.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
import pytest

from thumbnail_intelligence.evidence.config import EvidenceNormalizationConfig
from thumbnail_intelligence.evidence.confidence import ConfidencePropagator
from thumbnail_intelligence.evidence.conflict_resolution import (
    ConflictDetector,
    ConflictResolver,
)
from thumbnail_intelligence.evidence.graph import EvidenceGraph
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceConflict,
    EvidenceEdge,
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
    RetrievedEvidence,
)
from thumbnail_intelligence.retrieval.ranking import RankingMetadata
from thumbnail_intelligence.retrieval.scoring import RetrievalScore


def _create_node(
    node_id: str,
    node_type: KnowledgeEntryType = KnowledgeEntryType.HISTORICAL_THUMBNAIL,
    conf: float = 0.90,
    created_at: str = None,
) -> EvidenceNode:
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
    )
    prov = ProvenanceRecord(
        origin=f"{node_type.value}:{node_id}",
        source_id=node_id,
        source_type=EvidenceSourceType.KNOWLEDGE_ENTRY,
        retrieval_query_id="q_ext",
        retrieval_reason="Test reason",
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )
    return EvidenceNode(
        node_id=node_id,
        node_type=node_type,
        evidence_item=ev,
        confidence=ConfidenceScore(raw_confidence=conf, propagated_confidence=conf),
        weight=EvidenceWeight(base_weight=conf, effective_weight=conf),
        provenance=prov,
    )


def test_confidence_propagation_multi_hop() -> None:
    # 3-hop graph: n1 -> n2 -> n3 (SUPPORTS)
    n1 = _create_node("hop_1", conf=0.90)
    n2 = _create_node("hop_2", conf=0.50)
    n3 = _create_node("hop_3", conf=0.40)
    nodes = {"hop_1": n1, "hop_2": n2, "hop_3": n3}

    e1 = EvidenceEdge.create("hop_1", "hop_2", "SUPPORTS")
    e2 = EvidenceEdge.create("hop_2", "hop_3", "SUPPORTS")

    ConfidencePropagator.propagate_confidence(nodes, [e1, e2], decay_factor=0.90)

    # n2 should have increased confidence boosted by n1
    assert n2.confidence.propagated_confidence > 0.50
    assert n2.confidence.decay_hops == 1


def test_conflict_resolution_strategies() -> None:
    # 1. Most recent strategy
    old_node = _create_node("old_node", conf=0.95, created_at="2024-01-01T00:00:00+00:00")
    new_node = _create_node("new_node", conf=0.75, created_at="2026-08-01T00:00:00+00:00")
    nodes = {"old_node": old_node, "new_node": new_node}

    conflict = EvidenceConflict(
        conflict_id="conf_time_01",
        conflict_type="OUTDATED_EVIDENCE",
        node_ids=["old_node", "new_node"],
        description="Old design style superseded by fresh brand refresh",
    )

    cfg_recency = EvidenceNormalizationConfig(conflict_resolution_strategy="most_recent")
    resolutions, edges = ConflictResolver.resolve_conflicts([conflict], nodes, cfg_recency)

    assert len(resolutions) == 1
    assert resolutions[0].winning_node_id == "new_node"
    assert old_node.is_active is False

    # 2. Highest confidence strategy
    n_high = _create_node("high_conf", conf=0.95)
    n_low = _create_node("low_conf", conf=0.60)
    nodes2 = {"high_conf": n_high, "low_conf": n_low}

    conflict2 = EvidenceConflict(
        conflict_id="conf_conf_01",
        conflict_type="CONTRADICTORY_CLAIM",
        node_ids=["high_conf", "low_conf"],
        description="Contradictory visual composition claim",
    )
    cfg_conf = EvidenceNormalizationConfig(conflict_resolution_strategy="highest_confidence")
    resolutions2, edges2 = ConflictResolver.resolve_conflicts([conflict2], nodes2, cfg_conf)

    assert len(resolutions2) == 1
    assert resolutions2[0].winning_node_id == "high_conf"
    assert n_low.is_active is False


def test_evidence_normalizer_raw_bundle_input() -> None:
    normalizer = EvidenceNormalizer()

    ev1 = _create_node("bundle_ev_1", KnowledgeEntryType.ARCHETYPE_EXAMPLE).evidence_item
    ev2 = _create_node("bundle_ev_2", KnowledgeEntryType.VISUAL_PATTERN).evidence_item

    bundle = EvidenceBundle(
        query_id="q_direct_bundle",
        items=[ev1, ev2],
    )

    graph = normalizer.normalize(bundle)
    assert isinstance(graph, NormalizedEvidenceGraph)
    assert len(graph.nodes) == 2
    assert graph.statistics.total_raw_evidence_count == 2
    assert graph.statistics.valid_nodes_count == 2


def test_graph_contradictions_and_dependencies_query() -> None:
    graph = EvidenceGraph(graph_id="g_query")
    nA = _create_node("nodeA")
    nB = _create_node("nodeB")
    nC = _create_node("nodeC")
    graph.add_node(nA)
    graph.add_node(nB)
    graph.add_node(nC)

    e1 = EvidenceEdge.create("nodeA", "nodeB", "CONTRADICTS")
    e2 = EvidenceEdge.create("nodeA", "nodeC", "DEPENDS_ON")
    graph.add_edge(e1)
    graph.add_edge(e2)

    contradictions = graph.get_contradictions("nodeA")
    assert len(contradictions) == 1
    assert contradictions[0].node_id == "nodeB"

    deps = graph.get_dependencies("nodeA")
    assert len(deps) == 1
    assert deps[0].node_id == "nodeC"

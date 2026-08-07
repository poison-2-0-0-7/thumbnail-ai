"""
Unit tests for ConflictDetector and ConflictResolver.
Tests detection of brand constraint violations and mutually exclusive archetypes,
and verifies explainable deterministic resolution strategies.
"""

from __future__ import annotations

import pytest

from thumbnail_intelligence.evidence.config import EvidenceNormalizationConfig
from thumbnail_intelligence.evidence.conflict_resolution import (
    ConflictDetector,
    ConflictResolver,
)
from thumbnail_intelligence.evidence.exceptions import UnresolvableConflictError
from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
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


def _create_node(
    node_id: str,
    node_type: KnowledgeEntryType,
    confidence: float = 0.90,
    data_payload: dict = None,
) -> EvidenceNode:
    score = RetrievalScore(overall_score=confidence)
    ranking = RankingMetadata(rank=1, score=score)
    payload = data_payload or {}
    ev = RetrievedEvidence(
        evidence_id=f"ev_{node_id}",
        entry_id=node_id,
        entry_type=node_type,
        origin=f"{node_type.value}:{node_id}",
        source_id=node_id,
        reason_retrieved="Test reason",
        score=score,
        ranking=ranking,
        data_payload=payload,
    )
    prov = ProvenanceRecord(
        origin=f"{node_type.value}:{node_id}",
        source_id=node_id,
        source_type=EvidenceSourceType.KNOWLEDGE_ENTRY,
        retrieval_query_id="query_conflict",
        retrieval_reason="Test reason",
    )
    return EvidenceNode(
        node_id=node_id,
        node_type=node_type,
        evidence_item=ev,
        confidence=ConfidenceScore(raw_confidence=confidence, propagated_confidence=confidence),
        weight=EvidenceWeight(base_weight=confidence, effective_weight=confidence),
        provenance=prov,
        metadata=payload,
    )


def test_conflict_detection_brand_vs_pattern() -> None:
    # 1. Brand rule prohibiting red neon
    brand_node = _create_node(
        node_id="brand_rule_01",
        node_type=KnowledgeEntryType.CREATOR_PROFILE_ENTRY,
        data_payload={"prohibited_tropes": ["red neon", "clickbait arrows"]},
    )
    object.__setattr__(brand_node.provenance, "source_type", EvidenceSourceType.BRAND_RULE)

    # 2. Visual pattern promoting red neon
    pattern_node = _create_node(
        node_id="pat_red_neon",
        node_type=KnowledgeEntryType.VISUAL_PATTERN,
        data_payload={"name": "Vibrant Red Neon Edge Glow", "description": "Adds glowing red neon contours"},
    )

    nodes = {brand_node.node_id: brand_node, pattern_node.node_id: pattern_node}
    conflicts = ConflictDetector.detect_conflicts(nodes)
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "BRAND_CONSTRAINT_VIOLATION"
    assert "red neon" in conflicts[0].description


def test_conflict_resolution_brand_dominance() -> None:
    brand_node = _create_node(
        node_id="brand_rule_01",
        node_type=KnowledgeEntryType.CREATOR_PROFILE_ENTRY,
        confidence=0.85,
        data_payload={"prohibited_tropes": ["flashy border"]},
    )
    object.__setattr__(brand_node.provenance, "source_type", EvidenceSourceType.BRAND_RULE)

    pattern_node = _create_node(
        node_id="pat_flashy",
        node_type=KnowledgeEntryType.VISUAL_PATTERN,
        confidence=0.95,
        data_payload={"name": "Flashy border styling"},
    )

    nodes = {brand_node.node_id: brand_node, pattern_node.node_id: pattern_node}
    conflicts = ConflictDetector.detect_conflicts(nodes)
    assert len(conflicts) == 1

    cfg = EvidenceNormalizationConfig(conflict_resolution_strategy="brand_dominance")
    resolutions, edges = ConflictResolver.resolve_conflicts(conflicts, nodes, cfg)

    assert len(resolutions) == 1
    assert resolutions[0].winning_node_id == "brand_rule_01"
    assert pattern_node.is_active is False
    assert "Suppressed" in pattern_node.suppression_reason

    # Verify SUPERSEDES edge from winner to loser
    sup_edges = [e for e in edges if e.relation_type == "SUPERSEDES"]
    assert len(sup_edges) == 1
    assert sup_edges[0].source_node_id == "brand_rule_01"
    assert sup_edges[0].target_node_id == "pat_flashy"

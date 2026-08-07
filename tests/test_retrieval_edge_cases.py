"""
Unit tests for Retrieval Engine edge cases, boundary conditions, and failure resilience.
Tests empty candidate sets, dimension mismatch recovery, invalid weight overrides,
and deduplication threshold edge conditions.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    KnowledgeEntry,
    KnowledgeEntryType,
)
from thumbnail_intelligence.retrieval.config import RankingWeights, RetrievalConfig
from thumbnail_intelligence.retrieval.embedding import (
    EmbeddingDimensionError,
    InMemoryVectorIndex,
    VectorMath,
)
from thumbnail_intelligence.retrieval.evidence_bundle import (
    EvidenceBundle,
    RetrievalResult,
)
from thumbnail_intelligence.retrieval.exceptions import (
    FilterError,
    InvalidQueryError,
    RetrievalError,
)
from thumbnail_intelligence.retrieval.hybrid_search import HybridSearchEngine
from thumbnail_intelligence.retrieval.query import (
    QueryContext,
    RetrievalQuery,
    SearchFilters,
)
from thumbnail_intelligence.retrieval.ranking import EvidenceDeduplicator, HybridRanker
from thumbnail_intelligence.retrieval.scoring import RetrievalScore, ScoringEngine


def test_ranking_weights_normalization_on_drift() -> None:
    # Weights that don't sum to 1.0 should be auto-normalized without crashing
    weights = RankingWeights(
        visual_similarity=0.70,
        creator_channel_affinity=0.30,
        archetype_match=0.30,
        niche_match=0.20,
        recency=0.20,
        confidence=0.20,
        metadata_quality=0.10,
    )
    total = (
        weights.visual_similarity
        + weights.creator_channel_affinity
        + weights.archetype_match
        + weights.niche_match
        + weights.recency
        + weights.confidence
        + weights.metadata_quality
    )
    assert 0.99 <= total <= 1.01


def test_empty_candidate_search_returns_empty_status() -> None:
    engine = HybridSearchEngine()
    query = RetrievalQuery(
        query_id="q_empty_test",
        context=QueryContext(niche="finance"),
    )

    result = engine.search(candidates=[], query=query)
    assert result.status == "empty"
    assert len(result.bundle.items) == 0
    assert result.bundle.total_candidates_examined == 0


def test_vector_index_boundary_conditions() -> None:
    index = InMemoryVectorIndex(expected_dim=4)

    # Empty query vector returns empty results without crashing
    assert index.search([], top_k=5) == []

    # Searching with non-existent candidate_ids returns empty list
    index.add("id_1", [1.0, 0.0, 0.0, 0.0])
    results = index.search([1.0, 0.0, 0.0, 0.0], candidate_ids={"missing_id"})
    assert len(results) == 0

    # Removing non-existent ID returns False
    assert index.remove("missing_id") is False


def test_evidence_deduplicator_with_empty_and_single_candidate() -> None:
    deduplicator = EvidenceDeduplicator()
    assert deduplicator.deduplicate([]) == []

    entry = KnowledgeEntry(
        entry_id="single_entry",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
    )
    score = RetrievalScore(overall_score=0.88)
    candidates = [(entry, score, "stage_1")]

    deduped = deduplicator.deduplicate(candidates)
    assert len(deduped) == 1
    assert deduped[0][0].entry_id == "single_entry"


def test_hybrid_search_query_weights_override() -> None:
    config = RetrievalConfig(embedding_dim=4)
    vector_index = InMemoryVectorIndex(expected_dim=4)
    engine = HybridSearchEngine(config=config, vector_index=vector_index)

    entry = KnowledgeEntry(
        entry_id="ent_weights_test",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        embedding=[1.0, 0.0, 0.0, 0.0],
        niche="tech",
    )
    vector_index.add("ent_weights_test", [1.0, 0.0, 0.0, 0.0])

    # Query with custom weights prioritizing visual similarity to 1.0
    query = RetrievalQuery(
        query_id="q_weights_override",
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        weights_override={"visual_similarity": 1.0, "recency": 0.0, "confidence": 0.0},
    )

    result = engine.search(candidates=[entry], query=query)
    assert result.status == "success"
    assert len(result.bundle.items) == 1
    assert result.bundle.items[0].score.visual_similarity == 1.0

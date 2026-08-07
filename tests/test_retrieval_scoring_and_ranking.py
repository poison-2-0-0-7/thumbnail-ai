"""
Unit tests for ScoringEngine, EvidenceDeduplicator, and HybridRanker.
Tests multi-signal explainable scoring, recency decay, semantic deduplication,
and top-K ranking metadata generation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
import pytest

from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    EvidenceGrade,
    KnowledgeEntry,
    KnowledgeEntryType,
)
from thumbnail_intelligence.retrieval.config import RankingWeights
from thumbnail_intelligence.retrieval.query import QueryContext, RetrievalQuery
from thumbnail_intelligence.retrieval.ranking import (
    EvidenceDeduplicator,
    HybridRanker,
)
from thumbnail_intelligence.retrieval.scoring import RetrievalScore, ScoringEngine


def test_scoring_engine_recency_time_decay() -> None:
    # 0 days old -> recency approx 1.0
    now_iso = datetime.now(timezone.utc).isoformat()
    score_now = ScoringEngine.compute_recency_score(now_iso, half_life_days=90.0)
    assert score_now >= 0.98

    # 90 days old -> recency approx 0.5 (half-life)
    from datetime import timedelta

    ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    score_90 = ScoringEngine.compute_recency_score(ninety_days_ago, half_life_days=90.0)
    assert 0.45 <= score_90 <= 0.55


def test_scoring_engine_candidate_evaluation() -> None:
    engine = ScoringEngine()

    query = RetrievalQuery(
        query_id="q_score_test",
        query_embedding=[0.5] * 512,
        context=QueryContext(
            channel_id="channel_123",
            niche="gaming",
            archetype_id="big_face_reaction",
        ),
    )

    # 1. Matching candidate
    matching_entry = KnowledgeEntry(
        entry_id="entry_match",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        source_channel_id="channel_123",
        niche="gaming",
        archetype_id="big_face_reaction",
        facets={"resolution": "4K", "ctr": 0.12},
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    score = engine.score_candidate(
        entry=matching_entry,
        query=query,
        visual_similarity=0.92,
    )

    assert score.overall_score > 0.80
    assert score.visual_similarity == 0.92
    assert score.creator_channel_similarity == 1.0
    assert score.archetype_similarity == 1.0
    assert score.niche_match_score == 1.0
    assert "Overall" in score.explanation


def test_evidence_deduplication_exact_and_semantic() -> None:
    deduplicator = EvidenceDeduplicator()

    score_high = RetrievalScore(overall_score=0.95)
    score_low = RetrievalScore(overall_score=0.70)

    # Two items with the same entry_id
    entry1 = KnowledgeEntry(
        entry_id="duplicate_id",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    entry2 = KnowledgeEntry(
        entry_id="duplicate_id",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )

    candidates = [
        (entry1, score_low, "stage_1"),
        (entry2, score_high, "stage_2"),
    ]

    deduped = deduplicator.deduplicate(candidates, threshold=0.95)
    assert len(deduped) == 1
    # Higher score should be retained
    assert deduped[0][1].overall_score == 0.95


def test_hybrid_ranker_ordering_and_top_k() -> None:
    ranker = HybridRanker()

    query = RetrievalQuery(
        query_id="q_rank_test",
        top_k=2,
        deduplicate=True,
    )

    c1 = (
        KnowledgeEntry(entry_id="c1", entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL),
        RetrievalScore(overall_score=0.50),
        "stage_1",
    )
    c2 = (
        KnowledgeEntry(entry_id="c2", entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL),
        RetrievalScore(overall_score=0.90),
        "stage_2",
    )
    c3 = (
        KnowledgeEntry(entry_id="c3", entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL),
        RetrievalScore(overall_score=0.75),
        "stage_3",
    )

    ranked = ranker.rank([c1, c2, c3], query=query)
    # top_k = 2
    assert len(ranked) == 2
    # #1 should be c2 (0.90)
    assert ranked[0][0].entry_id == "c2"
    assert ranked[0][2].rank == 1
    # #2 should be c3 (0.75)
    assert ranked[1][0].entry_id == "c3"
    assert ranked[1][2].rank == 2

"""
Unit tests for RetrievalQuery, QueryContext, SearchFilters, and MetadataFilterEngine.
Tests schema validation, deterministic stage 1 filter evaluation, date range boundaries,
and explainable filter rejection diagnostics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    EvidenceGrade,
    KnowledgeEntry,
    KnowledgeEntryType,
)
from thumbnail_intelligence.retrieval.exceptions import InvalidQueryError
from thumbnail_intelligence.retrieval.filters import MetadataFilterEngine
from thumbnail_intelligence.retrieval.query import (
    QueryContext,
    RetrievalQuery,
    SearchFilters,
)


def test_retrieval_query_validation_and_defaults() -> None:
    query = RetrievalQuery(
        query_id="query_001",
        query_embedding=[0.1] * 512,
        text_query="high energy reaction face",
        context=QueryContext(niche="gaming", channel_id="channel_123"),
        filters=SearchFilters(niche="gaming"),
        top_k=5,
        min_similarity=0.5,
    )
    assert query.query_id == "query_001"
    assert len(query.query_embedding) == 512
    assert query.top_k == 5
    assert query.min_similarity == 0.5

    # Compute cache key is deterministic
    key1 = query.compute_cache_key()
    key2 = query.compute_cache_key()
    assert key1 == key2
    assert len(key1) == 64


def test_retrieval_query_validation_errors() -> None:
    # Empty query_id
    with pytest.raises(ValidationError):
        RetrievalQuery(query_id="   ")

    # Invalid top_k
    with pytest.raises(ValidationError):
        RetrievalQuery(query_id="q_1", top_k=0)

    # NaN in embedding
    with pytest.raises(ValidationError):
        RetrievalQuery(query_id="q_1", query_embedding=[float("nan")] * 512)


def test_metadata_filter_engine_predicates() -> None:
    engine = MetadataFilterEngine()

    entry = KnowledgeEntry(
        entry_id="entry_001",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        source_channel_id="channel_abc",
        niche="gaming",
        facets={"resolution": "1080p", "has_creator": True},
        created_at="2026-08-01T12:00:00+00:00",
    )

    # 1. Matching filters
    matching_filters = SearchFilters(
        entry_types=[KnowledgeEntryType.HISTORICAL_THUMBNAIL],
        niche="gaming",
        channel_id="channel_abc",
        custom_facets={"has_creator": True},
    )
    assert engine.matches(entry, matching_filters) is True

    # 2. Mismatched entry type
    mismatch_type = SearchFilters(
        entry_types=[KnowledgeEntryType.ARCHETYPE_EXAMPLE],
    )
    assert engine.matches(entry, mismatch_type) is False

    # 3. Mismatched niche
    mismatch_niche = SearchFilters(
        niche="cooking",
    )
    assert engine.matches(entry, mismatch_niche) is False

    # 4. Excluded ID filter
    exclude_filter = SearchFilters(
        exclude_ids=["entry_001"],
    )
    assert engine.matches(entry, exclude_filter) is False


def test_metadata_filter_date_ranges_and_facets() -> None:
    engine = MetadataFilterEngine()

    entry = KnowledgeEntry(
        entry_id="entry_date_test",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        created_at="2026-08-05T00:00:00+00:00",
        facets={"season": "summer"},
    )

    # In range
    in_range = SearchFilters(
        date_from="2026-08-01T00:00:00+00:00",
        date_to="2026-08-10T00:00:00+00:00",
        custom_facets={"season": "summer"},
    )
    assert engine.matches(entry, in_range) is True

    # Out of range (too early)
    out_of_range = SearchFilters(
        date_from="2026-08-06T00:00:00+00:00",
    )
    assert engine.matches(entry, out_of_range) is False

    # Mismatched facet value
    facet_mismatch = SearchFilters(
        custom_facets={"season": "winter"},
    )
    assert engine.matches(entry, facet_mismatch) is False


def test_filter_rejection_explanation() -> None:
    engine = MetadataFilterEngine()

    entry = KnowledgeEntry(
        entry_id="entry_explain",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        niche="fitness",
    )

    filters = SearchFilters(
        entry_types=[KnowledgeEntryType.ARCHETYPE_EXAMPLE],
        niche="tech",
        exclude_ids=["entry_explain"],
    )

    reasons = engine.explain_filter_rejection(entry, filters)
    assert len(reasons) >= 2
    assert any("exclude_ids" in r for r in reasons)
    assert any("entry_type" in r for r in reasons)

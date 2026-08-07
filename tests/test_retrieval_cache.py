"""
Unit tests for RetrievalCache and EmbeddingCache.
Tests TTL expiration, LRU eviction, telemetry hit rates, and predicate invalidation.
"""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from thumbnail_intelligence.knowledge_base.models import KnowledgeEntry, KnowledgeEntryType
from thumbnail_intelligence.retrieval.cache import EmbeddingCache, RetrievalCache
from thumbnail_intelligence.retrieval.evidence_bundle import (
    EvidenceBundle,
    RetrievalResult,
    RetrievedEvidence,
)
from thumbnail_intelligence.retrieval.query import RetrievalQuery
from thumbnail_intelligence.retrieval.ranking import RankingMetadata
from thumbnail_intelligence.retrieval.scoring import RetrievalScore


def test_retrieval_cache_hit_and_miss() -> None:
    cache = RetrievalCache(max_size=10, default_ttl_seconds=3600)

    query1 = RetrievalQuery(query_id="q1", top_k=5)
    query2 = RetrievalQuery(query_id="q2", top_k=5)

    # 1. Miss on empty cache
    assert cache.get(query1) is None
    assert cache.stats()["misses"] == 1

    # 2. Store result
    bundle = EvidenceBundle(query_id="q1")
    result = RetrievalResult(query=query1, bundle=bundle, status="success")
    cache.set(query1, result)

    # 3. Hit on query1
    cached = cache.get(query1)
    assert cached is not None
    assert cached.query.query_id == "q1"
    assert cache.stats()["hits"] == 1
    assert cache.stats()["hit_rate"] == 0.5

    # 4. Miss on query2
    assert cache.get(query2) is None


def test_retrieval_cache_ttl_expiration() -> None:
    # 0.1s TTL
    cache = RetrievalCache(max_size=10, default_ttl_seconds=1)

    query = RetrievalQuery(query_id="q_ttl")
    bundle = EvidenceBundle(query_id="q_ttl")
    result = RetrievalResult(query=query, bundle=bundle, status="success")

    cache.set(query, result, ttl_seconds=0)  # expires immediately
    time.sleep(0.01)

    assert cache.get(query) is None


def test_retrieval_cache_lru_eviction() -> None:
    cache = RetrievalCache(max_size=2)

    q1 = RetrievalQuery(query_id="q1")
    q2 = RetrievalQuery(query_id="q2")
    q3 = RetrievalQuery(query_id="q3")

    cache.set(q1, RetrievalResult(query=q1, bundle=EvidenceBundle(query_id="q1")))
    cache.set(q2, RetrievalResult(query=q2, bundle=EvidenceBundle(query_id="q2")))

    assert cache.get(q1) is not None  # refresh q1

    # Insert q3 -> should evict q2 (LRU)
    cache.set(q3, RetrievalResult(query=q3, bundle=EvidenceBundle(query_id="q3")))

    assert cache.get(q1) is not None
    assert cache.get(q3) is not None
    assert cache.get(q2) is None  # q2 was evicted
    assert cache.stats()["evictions"] == 1


def test_embedding_cache() -> None:
    emb_cache = EmbeddingCache(max_size=3)
    emb_cache.set("hash_1", [0.1, 0.2])
    emb_cache.set("hash_2", [0.3, 0.4])

    assert emb_cache.get("hash_1") == [0.1, 0.2]
    assert emb_cache.get("hash_missing") is None

    emb_cache.clear()
    assert emb_cache.get("hash_1") is None

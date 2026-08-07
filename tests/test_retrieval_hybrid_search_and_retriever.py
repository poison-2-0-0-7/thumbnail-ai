"""
Unit tests for MetadataSearchEngine, HybridSearchEngine, and KnowledgeRetriever facade.
Tests end-to-end multi-stage retrieval, repository integration, IndexHook vector synchronization,
domain evidence partitioning, and convenience retrieval methods.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from thumbnail_intelligence.knowledge_base.config import KnowledgeBaseConfig
from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    CompetitorProfile,
    CreatorProfile,
    DesignPattern,
    KnowledgeEntry,
    KnowledgeEntryType,
    VisualPattern,
)
from thumbnail_intelligence.knowledge_base.repository import KnowledgeBaseRepository
from thumbnail_intelligence.retrieval.cache import RetrievalCache
from thumbnail_intelligence.retrieval.config import RetrievalConfig
from thumbnail_intelligence.retrieval.embedding import InMemoryVectorIndex
from thumbnail_intelligence.retrieval.hybrid_search import HybridSearchEngine
from thumbnail_intelligence.retrieval.metadata_search import MetadataSearchEngine
from thumbnail_intelligence.retrieval.query import (
    QueryContext,
    RetrievalQuery,
    SearchFilters,
)
from thumbnail_intelligence.retrieval.retriever import KnowledgeRetriever


def test_metadata_search_engine_keyword_matching() -> None:
    search_engine = MetadataSearchEngine()

    arch = Archetype(
        archetype_id="reaction_shock_01",
        name="Shocked Creator Reaction",
        description="High intensity reaction face expressing surprise",
        typical_hook_types=["reaction", "shock", "curiosity"],
    )

    query = RetrievalQuery(
        query_id="q_kw",
        text_query="shocked reaction face",
    )

    results = search_engine.search(candidates=[arch], query=query)
    assert len(results) == 1
    matched_entry, score, terms = results[0]
    assert score > 0.5
    assert "shocked" in terms or "reaction" in terms or "face" in terms


def test_hybrid_search_engine_workflow() -> None:
    config = RetrievalConfig(embedding_dim=4, default_top_k=5)
    vector_index = InMemoryVectorIndex(expected_dim=4)
    cache = RetrievalCache()

    engine = HybridSearchEngine(
        config=config,
        vector_index=vector_index,
        cache=cache,
    )

    # Candidate 1: Gaming entry
    entry1 = KnowledgeEntry(
        entry_id="hist_1",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        embedding=[1.0, 0.0, 0.0, 0.0],
        niche="gaming",
        facets={"resolution": "4K"},
    )
    vector_index.add("hist_1", [1.0, 0.0, 0.0, 0.0])

    # Candidate 2: Cooking entry
    entry2 = KnowledgeEntry(
        entry_id="hist_2",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        embedding=[0.0, 1.0, 0.0, 0.0],
        niche="cooking",
    )
    vector_index.add("hist_2", [0.0, 1.0, 0.0, 0.0])

    # Query targeting gaming with vector close to entry1
    query = RetrievalQuery(
        query_id="q_hybrid_test",
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        filters=SearchFilters(niche="gaming"),
        top_k=2,
    )

    result = engine.search(candidates=[entry1, entry2], query=query)
    assert result.status == "success"
    assert len(result.bundle.items) == 1
    top_item = result.bundle.items[0]
    assert top_item.entry_id == "hist_1"
    assert top_item.score.visual_similarity == 1.0
    assert result.bundle.statistics.filter_latency_ms >= 0.0


def test_knowledge_retriever_repository_integration_and_hooks(tmp_path: Path) -> None:
    kb_config = KnowledgeBaseConfig(base_dir=tmp_path)
    repo = KnowledgeBaseRepository(config=kb_config)

    retrieval_config = RetrievalConfig(embedding_dim=512)
    retriever = KnowledgeRetriever(
        repository=repo,
        config=retrieval_config,
    )

    # 1. Seed repository with default archetypes and patterns
    seeded_archetypes = repo.seed_default_archetypes()
    repo.seed_default_patterns()

    # Verify vector index received seeded archetypes
    assert retriever.vector_index.count() >= len(seeded_archetypes)

    # 2. Retrieve archetypes via helper
    arch_evidence = retriever.retrieve_archetypes(niche="entertainment", top_k=3)
    assert len(arch_evidence) >= 1
    assert any(e.entry_id == "big_face_reaction" for e in arch_evidence)

    # 3. Register a new entry in repository and verify IndexHook syncs it automatically
    new_entry = KnowledgeEntry(
        entry_id="hook_entry_01",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        embedding=[0.1] * 512,
        niche="tech",
        source_channel_id="channel_tech_01",
    )
    repo.entries.register(new_entry)

    # Vector index must now contain hook_entry_01
    assert retriever.vector_index.get("hook_entry_01") is not None

    # 4. Retrieve creator history
    history_items = retriever.retrieve_creator_history(
        channel_id="channel_tech_01",
        embedding=[0.1] * 512,
        top_k=2,
    )
    assert len(history_items) == 1
    assert history_items[0].entry_id == "hook_entry_01"

    # 5. Retrieve for video
    bundle = retriever.retrieve_for_video(
        video_id="vid_test_100",
        context=QueryContext(niche="tech", channel_id="channel_tech_01"),
        embedding=[0.1] * 512,
        top_k=5,
    )
    assert len(bundle.items) >= 1
    assert len(bundle.historical_evidence) >= 1

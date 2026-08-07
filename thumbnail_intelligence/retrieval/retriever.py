"""
retriever.py
============

High-level KnowledgeRetriever facade integrating KnowledgeBaseRepository,
HybridSearchEngine, InMemoryVectorIndex, and RetrievalCache.

Implements the IndexHook protocol to automatically keep in-memory vector indexes
and retrieval caches in sync with repository mutations.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    CompetitorProfile,
    CreatorProfile,
    DesignPattern,
    KnowledgeEntry,
    KnowledgeEntryType,
    VisualPattern,
)
from thumbnail_intelligence.knowledge_base.registry import IndexHook
from thumbnail_intelligence.knowledge_base.repository import KnowledgeBaseRepository
from thumbnail_intelligence.retrieval.cache import RetrievalCache
from thumbnail_intelligence.retrieval.config import RetrievalConfig
from thumbnail_intelligence.retrieval.embedding import InMemoryVectorIndex
from thumbnail_intelligence.retrieval.evidence_bundle import (
    EvidenceBundle,
    RetrievalResult,
    RetrievedEvidence,
)
from thumbnail_intelligence.retrieval.hybrid_search import HybridSearchEngine
from thumbnail_intelligence.retrieval.query import (
    QueryContext,
    RetrievalQuery,
    SearchFilters,
)


class KnowledgeRetriever(IndexHook[Any]):
    """
    Central facade for all evidence retrieval across the Thumbnail Intelligence Engine.
    Coordinates vector similarity, hard metadata filters, explainable multi-signal scoring,
    and automatic index synchronization.
    """

    def __init__(
        self,
        repository: KnowledgeBaseRepository,
        config: Optional[RetrievalConfig] = None,
        vector_index: Optional[InMemoryVectorIndex] = None,
        cache: Optional[RetrievalCache] = None,
    ) -> None:
        self.repository = repository
        self.config = config or RetrievalConfig()
        self.vector_index = vector_index or InMemoryVectorIndex(expected_dim=self.config.embedding_dim)
        self.cache = cache or (RetrievalCache(max_size=self.config.max_cache_size, default_ttl_seconds=self.config.cache_ttl_seconds) if self.config.cache_enabled else None)
        self.search_engine = HybridSearchEngine(
            config=self.config,
            vector_index=self.vector_index,
            cache=self.cache,
        )

        # Attach self as IndexHook to all repository registries for automatic synchronization
        self.repository.entries.register_index_hook(self)
        self.repository.archetypes.register_index_hook(self)
        self.repository.competitor_profiles.register_index_hook(self)
        self.repository.visual_patterns.register_index_hook(self)
        self.repository.design_patterns.register_index_hook(self)

        # Bootstrap index from current repository state
        self.sync_from_repository()

    def sync_from_repository(self) -> None:
        """Synchronize in-memory vector index with all currently persisted repository entries."""
        # Index entries
        for entry in self.repository.entries.list():
            if entry.embedding and len(entry.embedding) == self.config.embedding_dim:
                self.vector_index.add(entry.entry_id, entry.embedding)

        # Index archetypes
        for arch in self.repository.archetypes.list():
            if arch.centroid_embedding and len(arch.centroid_embedding) == self.config.embedding_dim:
                self.vector_index.add(arch.archetype_id, arch.centroid_embedding)

        # Index competitors
        for comp in self.repository.competitor_profiles.list():
            if comp.style_embedding and len(comp.style_embedding) == self.config.embedding_dim:
                self.vector_index.add(comp.competitor_id, comp.style_embedding)

        # Index visual patterns
        for vp in self.repository.visual_patterns.list():
            if vp.centroid_embedding and len(vp.centroid_embedding) == self.config.embedding_dim:
                self.vector_index.add(vp.pattern_id, vp.centroid_embedding)

    # -----------------------------------------------------------------------
    # IndexHook Callbacks
    # -----------------------------------------------------------------------

    def on_registered(self, entry: Any) -> None:
        """Automatically index newly registered repository entries and invalidate cache."""
        eid = getattr(entry, "entry_id", getattr(entry, "archetype_id", getattr(entry, "competitor_id", getattr(entry, "pattern_id", None))))
        vec = getattr(entry, "embedding", getattr(entry, "centroid_embedding", getattr(entry, "style_embedding", None)))
        if eid and vec and len(vec) == self.config.embedding_dim:
            self.vector_index.add(str(eid), vec)
        if self.cache:
            self.cache.clear()

    def on_updated(self, entry: Any) -> None:
        """Update vector index and invalidate cache upon entry modification."""
        self.on_registered(entry)

    def on_removed(self, entry_id: str) -> None:
        """Remove vector from index and invalidate cache upon deletion."""
        self.vector_index.remove(entry_id)
        if self.cache:
            self.cache.clear()

    # -----------------------------------------------------------------------
    # Retrieval API Methods
    # -----------------------------------------------------------------------

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Execute unified hybrid retrieval for a query over all candidate entries in repository.
        """
        all_candidates: List[Any] = []
        all_candidates.extend(self.repository.entries.list())
        all_candidates.extend(self.repository.archetypes.list())
        all_candidates.extend(self.repository.competitor_profiles.list())
        all_candidates.extend(self.repository.visual_patterns.list())
        all_candidates.extend(self.repository.design_patterns.list())

        return self.search_engine.search(candidates=all_candidates, query=query)

    def retrieve_for_video(
        self,
        video_id: str,
        context: QueryContext,
        embedding: Optional[List[float]] = None,
        top_k: int = 8,
    ) -> EvidenceBundle:
        """
        Retrieve comprehensive, multi-domain evidence bundle for an analyzed video.
        """
        qid = f"query_{uuid.uuid4().hex[:10]}"
        query = RetrievalQuery(
            query_id=qid,
            query_embedding=embedding or [],
            context=context,
            filters=SearchFilters(
                niche=context.niche,
            ),
            top_k=top_k,
        )
        res = self.retrieve(query)
        return res.bundle

    def retrieve_archetypes(
        self,
        niche: str = "general",
        embedding: Optional[List[float]] = None,
        top_k: int = 4,
    ) -> List[RetrievedEvidence]:
        """
        Retrieve top matching Archetype definitions for a niche and optional thumbnail vector.
        """
        query = RetrievalQuery(
            query_id=f"arch_query_{uuid.uuid4().hex[:8]}",
            query_embedding=embedding or [],
            context=QueryContext(niche=niche),
            filters=SearchFilters(
                niche=niche,
                entry_types=[KnowledgeEntryType.ARCHETYPE_EXAMPLE],
            ),
            top_k=top_k,
        )
        archetype_candidates = self.repository.archetypes.list()
        res = self.search_engine.search(candidates=archetype_candidates, query=query)
        return res.bundle.items

    def retrieve_creator_history(
        self,
        channel_id: str,
        embedding: Optional[List[float]] = None,
        top_k: int = 6,
    ) -> List[RetrievedEvidence]:
        """
        Retrieve historical thumbnail designs for a specific creator channel.
        """
        query = RetrievalQuery(
            query_id=f"history_query_{uuid.uuid4().hex[:8]}",
            query_embedding=embedding or [],
            context=QueryContext(channel_id=channel_id),
            filters=SearchFilters(
                channel_id=channel_id,
                entry_types=[KnowledgeEntryType.HISTORICAL_THUMBNAIL],
            ),
            top_k=top_k,
        )
        entries = self.repository.entries.list(filter_fn=lambda e: e.source_channel_id == channel_id)
        res = self.search_engine.search(candidates=entries, query=query)
        return res.bundle.items

    def retrieve_competitor_intelligence(
        self,
        niche: str,
        embedding: Optional[List[float]] = None,
        top_k: int = 6,
    ) -> List[RetrievedEvidence]:
        """
        Retrieve competitor benchmarks and dominant styles for a niche.
        """
        query = RetrievalQuery(
            query_id=f"comp_query_{uuid.uuid4().hex[:8]}",
            query_embedding=embedding or [],
            context=QueryContext(niche=niche),
            filters=SearchFilters(
                niche=niche,
                entry_types=[KnowledgeEntryType.COMPETITOR_THUMBNAIL],
            ),
            top_k=top_k,
        )
        competitors = self.repository.competitor_profiles.list(filter_fn=lambda c: c.niche == niche or c.niche == "general")
        res = self.search_engine.search(candidates=competitors, query=query)
        return res.bundle.items

    def retrieve_patterns(
        self,
        niche: str,
        top_k: int = 6,
    ) -> List[RetrievedEvidence]:
        """
        Retrieve visual and design patterns relevant to a niche.
        """
        query = RetrievalQuery(
            query_id=f"pattern_query_{uuid.uuid4().hex[:8]}",
            context=QueryContext(niche=niche),
            filters=SearchFilters(niche=niche),
            top_k=top_k,
        )
        patterns = []
        patterns.extend(self.repository.visual_patterns.list())
        patterns.extend(self.repository.design_patterns.list())
        res = self.search_engine.search(candidates=patterns, query=query)
        return res.bundle.items

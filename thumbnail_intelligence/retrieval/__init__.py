"""
retrieval
=========

Hybrid Retrieval Engine for the Thumbnail Intelligence Engine as specified in
docs/thumbnail_intelligence_architecture.md §16.

Provides:
- Multi-stage hybrid search combining hard metadata filtering, lexical matching,
  and vector cosine similarity.
- Explainable multi-signal candidate scoring and ranking.
- Semantic and ID evidence deduplication.
- Thread-safe LRU/TTL caching.
- Pluggable abstract embedding provider interfaces.
- Bounded EvidenceBundle outputs with transparent provenance.
"""

from __future__ import annotations

from thumbnail_intelligence.retrieval.cache import EmbeddingCache, RetrievalCache
from thumbnail_intelligence.retrieval.config import RankingWeights, RetrievalConfig
from thumbnail_intelligence.retrieval.embedding import (
    EmbeddingProvider,
    InMemoryVectorIndex,
    MockEmbeddingProvider,
    VectorMath,
)
from thumbnail_intelligence.retrieval.evidence_bundle import (
    EvidenceBundle,
    RetrievalResult,
    RetrievedEvidence,
    SearchStatistics,
)
from thumbnail_intelligence.retrieval.exceptions import (
    CacheError,
    DeduplicationError,
    EmbeddingDimensionError,
    FilterError,
    InvalidQueryError,
    ProviderNotFoundError,
    RankingError,
    RetrievalError,
)
from thumbnail_intelligence.retrieval.filters import MetadataFilterEngine
from thumbnail_intelligence.retrieval.hybrid_search import HybridSearchEngine
from thumbnail_intelligence.retrieval.metadata_search import MetadataSearchEngine
from thumbnail_intelligence.retrieval.query import (
    QueryContext,
    RetrievalQuery,
    SearchFilters,
)
from thumbnail_intelligence.retrieval.ranking import (
    EvidenceDeduplicator,
    HybridRanker,
    RankingMetadata,
)
from thumbnail_intelligence.retrieval.retriever import KnowledgeRetriever
from thumbnail_intelligence.retrieval.scoring import RetrievalScore, ScoringEngine

__all__ = [
    # Config & Weights
    "RetrievalConfig",
    "RankingWeights",
    # Exceptions
    "RetrievalError",
    "InvalidQueryError",
    "FilterError",
    "RankingError",
    "EmbeddingDimensionError",
    "CacheError",
    "ProviderNotFoundError",
    "DeduplicationError",
    # Query & Filters
    "RetrievalQuery",
    "QueryContext",
    "SearchFilters",
    "MetadataFilterEngine",
    # Embedding & Vector Index
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "InMemoryVectorIndex",
    "VectorMath",
    # Scoring & Ranking
    "RetrievalScore",
    "ScoringEngine",
    "RankingMetadata",
    "EvidenceDeduplicator",
    "HybridRanker",
    # Search Engines & Bundles
    "MetadataSearchEngine",
    "HybridSearchEngine",
    "EvidenceBundle",
    "RetrievedEvidence",
    "RetrievalResult",
    "SearchStatistics",
    # Caching & Facade
    "RetrievalCache",
    "EmbeddingCache",
    "KnowledgeRetriever",
]

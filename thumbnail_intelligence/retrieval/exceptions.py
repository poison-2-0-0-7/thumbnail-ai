"""
exceptions.py
=============

Structured exception hierarchy for the Hybrid Retrieval Engine.
Provides domain-specific exceptions for query parsing, filtering, ranking,
embedding dimensionality, caching, and deduplication.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from thumbnail_intelligence.knowledge_base.exceptions import KnowledgeBaseError


class RetrievalError(KnowledgeBaseError):
    """Base exception for all errors occurring within the Retrieval Engine."""

    default_error_code: str = "RETRIEVAL_ERROR"


class InvalidQueryError(RetrievalError):
    """Raised when a RetrievalQuery is malformed, has invalid bounds, or cannot be executed."""

    default_error_code = "RETRIEVAL_INVALID_QUERY"


class FilterError(RetrievalError):
    """Raised when evaluating a SearchFilters predicate fails or encounters invalid criteria."""

    default_error_code = "RETRIEVAL_FILTER_ERROR"


class RankingError(RetrievalError):
    """Raised when score aggregation, weight normalization, or candidate ranking fails."""

    default_error_code = "RETRIEVAL_RANKING_ERROR"


class EmbeddingDimensionError(RetrievalError):
    """Raised when a query or index embedding does not match expected vector dimensionality."""

    default_error_code = "RETRIEVAL_EMBEDDING_DIM_ERROR"


class CacheError(RetrievalError):
    """Raised when retrieval cache operations (lookup, eviction, serialization) fail."""

    default_error_code = "RETRIEVAL_CACHE_ERROR"


class ProviderNotFoundError(RetrievalError):
    """Raised when a requested embedding provider or search backend is not registered."""

    default_error_code = "RETRIEVAL_PROVIDER_NOT_FOUND"


class DeduplicationError(RetrievalError):
    """Raised when candidate deduplication encounters conflicting or corrupted evidence."""

    default_error_code = "RETRIEVAL_DEDUPLICATION_ERROR"

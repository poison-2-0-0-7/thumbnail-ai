"""
cache.py
========

Thread-safe in-memory retrieval caching with TTL expiration, LRU eviction,
and future embedding vector caching.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from thumbnail_intelligence.retrieval.evidence_bundle import RetrievalResult
from thumbnail_intelligence.retrieval.query import RetrievalQuery


class RetrievalCache:
    """
    LRU and TTL-enabled cache storing RetrievalResult objects.
    Reduces redundant retrieval and scoring overhead across identical or recurrent queries.
    """

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        # key -> (result, expiry_timestamp)
        self._cache: collections.OrderedDict[str, Tuple[RetrievalResult, float]] = collections.OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, query: RetrievalQuery) -> Optional[RetrievalResult]:
        """
        Lookup cached RetrievalResult for query.
        Returns None if missing or expired.
        """
        key = query.compute_cache_key()
        now = time.time()

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            result, expiry = self._cache[key]
            if now > expiry:
                # Expired entry
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end for LRU refresh
            self._cache.move_to_end(key)
            self._hits += 1

            # Stamp cache_hit flag on returned statistics
            stats = result.bundle.statistics
            object.__setattr__(stats, "cache_hit", True)
            return result

    def set(
        self,
        query: RetrievalQuery,
        result: RetrievalResult,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Cache a RetrievalResult with optional customized TTL.
        """
        key = query.compute_cache_key()
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = time.time() + ttl

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (result, expiry)

            # Evict oldest entries if over capacity
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

    def invalidate(self, filter_fn: Optional[Callable[[str, RetrievalResult], bool]] = None) -> int:
        """
        Invalidate cache entries matching filter predicate or all if None.
        Returns number of invalidated entries.
        """
        with self._lock:
            if filter_fn is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            to_delete = [
                key for key, (res, _) in self._cache.items() if filter_fn(key, res)
            ]
            for key in to_delete:
                del self._cache[key]
            return len(to_delete)

    def clear(self) -> None:
        """Clear all entries in cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> Dict[str, Any]:
        """Return operational cache telemetry metrics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "evictions": self._evictions,
            }


class EmbeddingCache:
    """
    Dedicated cache for storing computed text and visual embedding vectors.
    """

    def __init__(self, max_size: int = 5000) -> None:
        self.max_size = max_size
        self._cache: collections.OrderedDict[str, List[float]] = collections.OrderedDict()
        self._lock = threading.RLock()

    def get(self, content_hash: str) -> Optional[List[float]]:
        with self._lock:
            if content_hash in self._cache:
                self._cache.move_to_end(content_hash)
                return self._cache[content_hash]
            return None

    def set(self, content_hash: str, vector: List[float]) -> None:
        with self._lock:
            if content_hash in self._cache:
                self._cache.move_to_end(content_hash)
            self._cache[content_hash] = vector
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

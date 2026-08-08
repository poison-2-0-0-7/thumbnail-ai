"""
cache.py
========

ModelCache for lazy loading, reference counting, LRU eviction, and pinning in Phase 4.4.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Set
from renderer_v2.runtime.models import BaseModelAdapter, ModelDescriptor, ModelState

logger = logging.getLogger(__name__)


class ModelCacheError(RuntimeError):
    """Exception raised for cache operations or eviction failures."""
    pass


class ModelCache:
    """Thread-safe model cache managing active adapters, reference counts, and LRU eviction."""

    def __init__(self, max_loaded_models: int = 5) -> None:
        self.max_loaded_models = max_loaded_models
        self._adapters: Dict[str, BaseModelAdapter] = {}
        self._ref_counts: Dict[str, int] = {}
        self._pinned: Set[str] = set()
        self._lock = threading.RLock()

    def get_adapter(self, model_name: str) -> Optional[BaseModelAdapter]:
        """Retrieve adapter from cache if present."""
        with self._lock:
            return self._adapters.get(model_name)

    def put_adapter(self, adapter: BaseModelAdapter, is_pinned: bool = False) -> None:
        """Store loaded adapter in cache and track pinning status."""
        with self._lock:
            name = adapter.model_name
            self._adapters[name] = adapter
            if name not in self._ref_counts:
                self._ref_counts[name] = 0
            if is_pinned or adapter.descriptor.is_pinned:
                self._pinned.add(name)

    def acquire(self, model_name: str) -> BaseModelAdapter:
        """Increment reference count and set state to IN_USE."""
        with self._lock:
            if model_name not in self._adapters:
                raise ModelCacheError(f"Model '{model_name}' is not present in ModelCache.")
            
            adapter = self._adapters[model_name]
            self._ref_counts[model_name] = self._ref_counts.get(model_name, 0) + 1
            adapter.state = ModelState.IN_USE
            adapter.last_used_timestamp = time.time()
            return adapter

    def release(self, model_name: str) -> None:
        """Decrement reference count and transition to READY if zero active references."""
        with self._lock:
            if model_name in self._ref_counts:
                self._ref_counts[model_name] = max(0, self._ref_counts[model_name] - 1)
                if self._ref_counts[model_name] == 0:
                    if model_name in self._adapters:
                        self._adapters[model_name].state = ModelState.READY

    def get_ref_count(self, model_name: str) -> int:
        """Get active handle reference count for a model."""
        with self._lock:
            return self._ref_counts.get(model_name, 0)

    def pin_model(self, model_name: str) -> None:
        """Pin a model so it is protected from LRU cache eviction."""
        with self._lock:
            self._pinned.add(model_name)

    def unpin_model(self, model_name: str) -> None:
        """Unpin a model."""
        with self._lock:
            self._pinned.discard(model_name)

    def is_pinned(self, model_name: str) -> bool:
        """Check if model is pinned."""
        with self._lock:
            return model_name in self._pinned

    def evict_lru(self, target_freed_count: int = 1) -> List[str]:
        """Evict least recently used models that are READY (ref_count == 0) and not pinned."""
        with self._lock:
            evicted: List[str] = []

            for _ in range(target_freed_count):
                candidates = [
                    name
                    for name, adapter in self._adapters.items()
                    if self._ref_counts.get(name, 0) == 0
                    and name not in self._pinned
                    and not adapter.descriptor.is_pinned
                ]

                if not candidates:
                    break

                # Sort by last_used_timestamp ascending (oldest first)
                candidates.sort(key=lambda n: self._adapters[n].last_used_timestamp)
                lru_name = candidates[0]

                adapter = self._adapters.pop(lru_name)
                self._ref_counts.pop(lru_name, None)
                adapter.unload()
                evicted.append(lru_name)
                logger.info(f"LRU evicted model '{lru_name}' from ModelCache")

            return evicted

    def remove_adapter(self, model_name: str) -> Optional[BaseModelAdapter]:
        """Remove and unload a specific model from cache."""
        with self._lock:
            adapter = self._adapters.pop(model_name, None)
            self._ref_counts.pop(model_name, None)
            self._pinned.discard(model_name)
            if adapter is not None:
                adapter.unload()
            return adapter

    def clear(self) -> None:
        """Unload and clear all cached models."""
        with self._lock:
            for name, adapter in list(self._adapters.items()):
                adapter.unload()
            self._adapters.clear()
            self._ref_counts.clear()
            self._pinned.clear()

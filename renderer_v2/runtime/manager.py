"""
manager.py
==========

ModelRuntimeManager for Phase 4.4.
Centralized manager owning model lifecycles, caching, memory tracking, device placement, and health checks.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Type
from renderer_v2.runtime.cache import ModelCache, ModelCacheError
from renderer_v2.runtime.device import DeviceManager
from renderer_v2.runtime.health import HealthCheckResult, HealthMonitor
from renderer_v2.runtime.memory import MemoryTracker
from renderer_v2.runtime.models import (
    BaseModelAdapter,
    GenericModelAdapter,
    ModelDescriptor,
    ModelHandle,
    ModelState,
)
from renderer_v2.runtime.registry import ModelRegistry, ModelRegistryError

logger = logging.getLogger(__name__)


class ModelRuntimeManagerError(RuntimeError):
    """Exception raised for ModelRuntimeManager operational failures."""
    pass


class ModelRuntimeManager:
    """Central manager owning model lifecycle, caching, device placement, and VRAM budget."""

    def __init__(
        self,
        max_vram_gb: float = 12.0,
        max_loaded_models: int = 5,
    ) -> None:
        self.registry = ModelRegistry()
        self.cache = ModelCache(max_loaded_models=max_loaded_models)
        self.device_manager = DeviceManager()
        self.memory_tracker = MemoryTracker(max_vram_gb=max_vram_gb)
        self.health_monitor = HealthMonitor()
        self._lock = threading.RLock()

    def register_model(
        self,
        descriptor: ModelDescriptor,
        loader_fn: Optional[Callable[[], Any]] = None,
        adapter_cls: Optional[Type[BaseModelAdapter]] = None,
    ) -> None:
        """Register a model descriptor and optional loader factory."""
        with self._lock:
            self.registry.register_model(descriptor, loader_fn, adapter_cls)

    def acquire_model(self, model_name: str, device: Optional[str] = None) -> ModelHandle:
        """Acquire a model handle for inference. Performs lazy loading, LRU eviction, and reference counting."""
        with self._lock:
            if not self.registry.is_registered(model_name):
                raise ModelRuntimeManagerError(f"Model '{model_name}' is not registered in ModelRegistry.")

            descriptor = self.registry.get_descriptor(model_name)
            target_device = self.device_manager.resolve_device(device or descriptor.required_device)

            adapter = self.cache.get_adapter(model_name)
            if adapter is None:
                # Evict LRU models if cache is full
                if len(self.cache._adapters) >= self.cache.max_loaded_models:
                    self.cache.evict_lru(1)
                    self.memory_tracker.force_garbage_collection()

                # Instantiate adapter
                adapter_cls = self.registry.get_adapter_cls(model_name)
                loader_fn = self.registry.get_loader(model_name)

                if adapter_cls is GenericModelAdapter:
                    adapter = GenericModelAdapter(descriptor, loader_fn=loader_fn)
                else:
                    adapter = adapter_cls(descriptor)

                adapter.device = target_device

                # Attempt model loading
                try:
                    adapter.load()
                except Exception as e:
                    # On OOM or load failure, attempt LRU eviction and retry once
                    logger.warning(f"Failed to load model '{model_name}' ({e}); attempting LRU cache eviction...")
                    self.cache.evict_lru(2)
                    self.memory_tracker.force_garbage_collection()
                    try:
                        adapter.load()
                    except Exception as retry_err:
                        raise ModelRuntimeManagerError(f"Failed to load model '{model_name}' after eviction retry: {retry_err}") from retry_err

                self.cache.put_adapter(adapter, is_pinned=descriptor.is_pinned)
                self.memory_tracker.record_model_allocation(model_name, descriptor.estimated_vram_gb)

            # Acquire reference
            self.cache.acquire(model_name)
            return ModelHandle(manager=self, adapter=adapter)

    def release_model(self, model_name: str) -> None:
        """Release an active handle reference for a model."""
        with self._lock:
            self.cache.release(model_name)
            if self.cache.get_ref_count(model_name) == 0:
                logger.debug(f"Model '{model_name}' has zero active references; state is READY")

    def unload_model(self, model_name: str) -> None:
        """Explicitly unload a model from memory/GPU."""
        with self._lock:
            adapter = self.cache.remove_adapter(model_name)
            if adapter is not None:
                self.memory_tracker.record_model_deallocation(model_name)
                self.memory_tracker.force_garbage_collection()

    def unload_all(self) -> None:
        """Unload all models and clear memory cache."""
        with self._lock:
            self.cache.clear()
            self.memory_tracker.reset_stats()
            self.memory_tracker.force_garbage_collection()

    def pin_model(self, model_name: str) -> None:
        """Pin model to protect it from LRU eviction."""
        with self._lock:
            self.cache.pin_model(model_name)

    def unpin_model(self, model_name: str) -> None:
        """Unpin model."""
        with self._lock:
            self.cache.unpin_model(model_name)

    def get_model_state(self, model_name: str) -> ModelState:
        """Return current lifecycle state of a model."""
        with self._lock:
            if not self.registry.is_registered(model_name):
                return ModelState.UNREGISTERED
            adapter = self.cache.get_adapter(model_name)
            if adapter is None:
                return ModelState.REGISTERED
            return adapter.state

    def health_check(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Run health check on a specific model or all registered models."""
        with self._lock:
            if model_name is not None:
                if not self.registry.is_registered(model_name):
                    return {"model_name": model_name, "is_healthy": False, "error": "Not registered"}
                descriptor = self.registry.get_descriptor(model_name)
                adapter = self.cache.get_adapter(model_name) or GenericModelAdapter(descriptor)
                res = self.health_monitor.check_adapter_health(adapter)
                return res.to_dict()
            
            # Health check all registered models
            results: Dict[str, Any] = {}
            for desc in self.registry.list_models():
                name = desc.model_name
                adapter = self.cache.get_adapter(name) or GenericModelAdapter(desc)
                res = self.health_monitor.check_adapter_health(adapter)
                results[name] = res.to_dict()
            return results

    def get_memory_status(self) -> Dict[str, Any]:
        """Return memory status summary."""
        with self._lock:
            return self.memory_tracker.get_memory_status()

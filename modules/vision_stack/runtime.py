"""Runtime coordination primitives for AI Vision Stack V2.1."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from .exceptions import VisionStackRuntimeError
from .loader import ModelLoader, build_registry, load_config
from .models import (
    RegisteredVisionModel,
    RuntimeBootstrapMetadata,
    VisionModelLifecycleState,
    VisionStackConfig,
)
from .registry import ModelRegistry
from .resources import GPUResourceManager


T = TypeVar("T")


@dataclass
class VisionStackRuntime:
    """Shared runtime state required by the sequential execution architecture."""

    registry: ModelRegistry
    gpu_lock: threading.RLock = field(default_factory=threading.RLock)
    thumbnails_processed_by_worker: int = 0
    accepting_work: bool = True
    worker_restart_threshold: int = 1_000

    def record_thumbnail_processed(self) -> int:
        """Increment and return this worker's processed-thumbnail count."""
        self.thumbnails_processed_by_worker += 1
        return self.thumbnails_processed_by_worker

    @property
    def restart_required(self) -> bool:
        """Return whether the configured worker restart threshold has been reached."""
        return self.thumbnails_processed_by_worker >= self.worker_restart_threshold


class RuntimeManager:
    """Own V2.1 runtime bootstrap, registry, lifecycle, and scheduling state."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        checkpoint_root: Path | None = None,
        validate_checkpoints: bool = True,
        worker_restart_threshold: int = 1_000,
    ) -> None:
        self.config_path = config_path
        self.loader = ModelLoader(checkpoint_root)
        self.validate_checkpoints = validate_checkpoints
        self.worker_restart_threshold = worker_restart_threshold
        self._lock = threading.RLock()
        self.config: VisionStackConfig | None = None
        self.bootstrap_metadata: RuntimeBootstrapMetadata | None = None
        self.runtime: VisionStackRuntime | None = None
        self.gpu_resources: GPUResourceManager | None = None

    def bootstrap(self) -> RuntimeBootstrapMetadata:
        """Load config, validate checkpoint metadata, and register models."""
        with self._lock:
            self.config = load_config(self.config_path)
            self.bootstrap_metadata = self.loader.bootstrap_metadata(
                self.config,
                validate_exists=self.validate_checkpoints,
            )
            registry = build_registry(self.config, metadata=self.bootstrap_metadata)
            self.runtime = VisionStackRuntime(
                registry=registry,
                worker_restart_threshold=self.worker_restart_threshold,
            )
            self.gpu_resources = GPUResourceManager(registry, self.runtime.gpu_lock)
            return self.bootstrap_metadata

    @property
    def registry(self) -> ModelRegistry:
        """Return the active model registry."""
        if self.runtime is None:
            raise VisionStackRuntimeError("Runtime has not been bootstrapped")
        return self.runtime.registry

    def reserve_model(self, model_name: str):
        """Reserve the shared GPU slot for a future wrapper implementation."""
        if self.gpu_resources is None:
            raise VisionStackRuntimeError("Runtime has not been bootstrapped")
        return self.gpu_resources.reserve(model_name)

    def run_sequential(
        self,
        model_names: Iterable[str],
        operation: Callable[[RegisteredVisionModel], T],
    ) -> tuple[T, ...]:
        """Run metadata-only scheduled operations under the GPU reservation lock."""
        if self.runtime is None:
            raise VisionStackRuntimeError("Runtime has not been bootstrapped")
        if not self.runtime.accepting_work:
            raise VisionStackRuntimeError("Runtime is draining and is not accepting new work")

        results: list[T] = []
        for model_name in model_names:
            with self.reserve_model(model_name) as model:
                results.append(operation(model))
        return tuple(results)

    def mark_thumbnail_processed(self) -> int:
        """Record one completed thumbnail and return the worker count."""
        if self.runtime is None:
            raise VisionStackRuntimeError("Runtime has not been bootstrapped")
        return self.runtime.record_thumbnail_processed()

    def begin_graceful_drain(self) -> None:
        """Stop accepting new work while allowing in-flight work to complete."""
        if self.runtime is None:
            raise VisionStackRuntimeError("Runtime has not been bootstrapped")
        self.runtime.accepting_work = False

    def evict_all(self) -> None:
        """Move all non-registered models to the evicted lifecycle state."""
        for model in self.registry.all_models():
            if model.lifecycle_state == VisionModelLifecycleState.GPU_ACTIVE:
                model = self.registry.transition(
                    model.name,
                    VisionModelLifecycleState.CPU_CACHED,
                )
            if model.lifecycle_state == VisionModelLifecycleState.CPU_CACHED:
                self.registry.transition(
                    model.name,
                    VisionModelLifecycleState.EVICTED,
                )

    def shutdown(self) -> None:
        """Drain work and evict lifecycle state without touching CUDA."""
        self.begin_graceful_drain()
        self.evict_all()

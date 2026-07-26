"""GPU resource coordination for the V2.1 sequential runtime contract."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Iterator

from .exceptions import VisionStackResourceError
from .models import RegisteredVisionModel, VisionModelFallback, VisionModelLifecycleState
from .registry import ModelRegistry


class GPUResourceManager:
    """Enforce single-active-model GPU ownership.

    This class manages runtime state and lifecycle bookkeeping only. It does
    not execute CUDA operations or move model weights.
    """

    def __init__(self, registry: ModelRegistry, gpu_lock: threading.RLock | None = None) -> None:
        self.registry = registry
        self._gpu_lock = gpu_lock or threading.RLock()
        self._state_lock = threading.RLock()
        self._active_model_name: str | None = None

    @property
    def active_model_name(self) -> str | None:
        """Return the model currently holding the GPU reservation."""
        with self._state_lock:
            return self._active_model_name

    @contextmanager
    def reserve(self, model_name: str) -> Iterator[RegisteredVisionModel]:
        """Reserve the shared GPU slot for one model at a time."""
        model = self.registry.get(model_name)
        with self._gpu_lock:
            with self._state_lock:
                if self._active_model_name is not None:
                    raise VisionStackResourceError(
                        f"GPU is already reserved by {self._active_model_name}"
                    )
                self._active_model_name = model_name
            try:
                if model.lifecycle_state == VisionModelLifecycleState.REGISTERED:
                    model = self.registry.transition(
                        model_name,
                        VisionModelLifecycleState.CPU_CACHED,
                    )
                if model.lifecycle_state != VisionModelLifecycleState.CPU_CACHED:
                    raise VisionStackResourceError(
                        "GPU reservation requires a registered or CPU-cached model: "
                        f"{model_name} is {model.lifecycle_state.value}"
                    )
                active = self.registry.transition(
                    model_name,
                    VisionModelLifecycleState.GPU_ACTIVE,
                )
                self.registry.update_runtime_state(
                    model_name,
                    {
                        "gpu_reserved": True,
                        "cuda_executed": False,
                        "weights_loaded": False,
                    },
                )
                yield active
            finally:
                self._release(model_name)

    def cpu_fallback_metadata(self, model_name: str) -> dict[str, object]:
        """Return the configured CPU fallback policy for one model."""
        model = self.registry.get(model_name)
        fallback = model.config.fallback
        return {
            "model_name": model_name,
            "configured_device": model.config.device,
            "fallback": fallback.value,
            "cpu_fallback_available": model.config.device == "cpu"
            or fallback in {
                VisionModelFallback.CPU_FALLBACK,
                VisionModelFallback.CPU_TILED_PROCESSING,
            },
            "cuda_executed": False,
        }

    def _release(self, model_name: str) -> None:
        current = self.registry.get(model_name)
        if current.lifecycle_state == VisionModelLifecycleState.GPU_ACTIVE:
            self.registry.transition(model_name, VisionModelLifecycleState.CPU_CACHED)
        self.registry.update_runtime_state(
            model_name,
            {
                "gpu_reserved": False,
                "cuda_executed": False,
                "weights_loaded": False,
            },
        )
        with self._state_lock:
            if self._active_model_name == model_name:
                self._active_model_name = None

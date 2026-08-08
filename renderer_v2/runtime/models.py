"""
models.py
=========

Model Runtime Data Models and Abstract Interfaces for Phase 4.4.
Defines:
- ModelState (Enum)
- ModelDescriptor (Schema)
- BaseModelAdapter (Abstract Base Class)
- GenericModelAdapter (Wrapper Adapter)
- ModelHandle (Acquired Handle with Context Manager)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    """Lifecycle states of an AI model in Model Runtime Manager."""
    UNREGISTERED = "UNREGISTERED"
    REGISTERED = "REGISTERED"
    LOADING = "LOADING"
    READY = "READY"
    IN_USE = "IN_USE"
    OFFLOADED = "OFFLOADED"
    UNLOADING = "UNLOADING"
    FAILED = "FAILED"


class ModelDescriptor(BaseModel):
    """Configuration descriptor for a registered AI model."""
    model_name: str = Field(..., description="Unique model identifier")
    version: str = Field("1.0.0", description="Model version string")
    framework: str = Field("pytorch", description="Framework (pytorch, diffusers, onnx, opencv)")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint file on disk")
    required_device: str = Field("cuda", description="Preferred compute device (cuda, cpu)")
    estimated_vram_gb: float = Field(1.0, ge=0.0, description="Estimated peak VRAM memory requirement in GB")
    dependencies: List[str] = Field(default_factory=list, description="Other model dependencies required")
    supported_tasks: List[str] = Field(default_factory=list, description="Supported tasks (detection, segmentation, inpaint, depth, etc.)")
    is_pinned: bool = Field(False, description="If True, model is protected from LRU cache eviction")

    model_config = {"frozen": False, "arbitrary_types_allowed": True}


class BaseModelAdapter(ABC):
    """Abstract interface for all model runtime adapters managed by ModelRuntimeManager."""

    def __init__(self, descriptor: ModelDescriptor) -> None:
        self.descriptor = descriptor
        self.state = ModelState.REGISTERED
        self._device = descriptor.required_device
        self._model_instance: Any = None
        self.last_used_timestamp: float = time.time()

    @property
    def model_name(self) -> str:
        return self.descriptor.model_name

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, new_device: str) -> None:
        self._device = new_device

    @property
    def model_instance(self) -> Any:
        return self._model_instance

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory/GPU and transition state to READY."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model weights, free VRAM/RAM and transition state to UNREGISTERED or OFFLOADED."""
        pass

    @abstractmethod
    def warmup(self) -> bool:
        """Execute a zero-tensor dummy inference pass to warm up CUDA kernels."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Perform health and integrity check on loaded model instance."""
        pass

    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Perform inference pass on input data."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up temporary caches or buffers."""
        pass


class GenericModelAdapter(BaseModelAdapter):
    """Generic wrapper adapter for raw model instances or loader callables."""

    def __init__(self, descriptor: ModelDescriptor, loader_fn: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(descriptor)
        self.loader_fn = loader_fn

    def load(self) -> None:
        self.state = ModelState.LOADING
        try:
            if self.loader_fn is not None:
                self._model_instance = self.loader_fn()
            self.state = ModelState.READY
            self.last_used_timestamp = time.time()
            logger.info(f"Loaded model '{self.model_name}' into state {self.state}")
        except Exception as e:
            self.state = ModelState.FAILED
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise

    def unload(self) -> None:
        self.state = ModelState.UNLOADING
        self._model_instance = None
        self.state = ModelState.OFFLOADED
        logger.info(f"Unloaded model '{self.model_name}'")

    def warmup(self) -> bool:
        self.last_used_timestamp = time.time()
        return True

    def health_check(self) -> bool:
        if self.state == ModelState.FAILED:
            return False
        if self._model_instance is None and self.loader_fn is None:
            return False
        return True

    def predict(self, inputs: Any, **kwargs) -> Any:
        if self._model_instance is None:
            self.load()
        self.last_used_timestamp = time.time()
        if hasattr(self._model_instance, "predict"):
            return self._model_instance.predict(inputs, **kwargs)
        elif callable(self._model_instance):
            return self._model_instance(inputs, **kwargs)
        return self._model_instance

    def cleanup(self) -> None:
        pass


class ModelHandle:
    """RAII handle to an acquired model from ModelRuntimeManager.

    Supports context manager (`with runtime.acquire_model('SAM2') as handle: ...`).
    Automates reference release on exit.
    """

    def __init__(self, manager: Any, adapter: BaseModelAdapter) -> None:
        self.manager = manager
        self.adapter = adapter
        self._is_released = False

    @property
    def model_name(self) -> str:
        return self.adapter.model_name

    @property
    def descriptor(self) -> ModelDescriptor:
        return self.adapter.descriptor

    @property
    def state(self) -> ModelState:
        return self.adapter.state

    @property
    def device(self) -> str:
        return self.adapter.device

    @property
    def instance(self) -> Any:
        return self.adapter.model_instance

    def predict(self, inputs: Any, **kwargs) -> Any:
        """Delegate prediction directly to the wrapped adapter."""
        if self._is_released:
            raise RuntimeError(f"Cannot invoke predict on released handle for model '{self.model_name}'")
        return self.adapter.predict(inputs, **kwargs)

    def release(self) -> None:
        """Release handle reference back to ModelRuntimeManager."""
        if not self._is_released:
            self._is_released = True
            if self.manager is not None:
                self.manager.release_model(self.model_name)

    def __enter__(self) -> ModelHandle:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

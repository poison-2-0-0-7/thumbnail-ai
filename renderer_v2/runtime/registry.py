"""
registry.py
===========

Model Registry for Phase 4.4 Model Runtime Manager.
Maintains model descriptors, framework metadata, dependencies, and loader factories for:
- GroundingDINO
- SAM2
- BiRefNet
- SDXL
- BrushNet
- CodeFormer
- GFPGAN
- DepthAnything
- Future models
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Type
from renderer_v2.runtime.models import BaseModelAdapter, GenericModelAdapter, ModelDescriptor

logger = logging.getLogger(__name__)


class ModelRegistryError(RuntimeError):
    """Exception raised for model registry operation errors."""
    pass


class ModelRegistry:
    """Central registry storing model descriptors, factory functions, and adapter classes."""

    def __init__(self) -> None:
        self._descriptors: Dict[str, ModelDescriptor] = {}
        self._loaders: Dict[str, Callable[[], Any]] = {}
        self._adapters: Dict[str, Type[BaseModelAdapter]] = {}
        self._lock = threading.RLock()

        # Pre-register core models
        self._register_default_models()

    def _register_default_models(self) -> None:
        """Pre-populate registry with standard Thumbnail AI model descriptors."""
        defaults = [
            ModelDescriptor(
                model_name="GroundingDINO",
                version="1.0.0",
                framework="pytorch",
                required_device="cuda",
                estimated_vram_gb=1.5,
                supported_tasks=["detection", "zero_shot_object_detection"],
            ),
            ModelDescriptor(
                model_name="SAM2",
                version="2.0.0",
                framework="pytorch",
                required_device="cuda",
                estimated_vram_gb=2.0,
                supported_tasks=["segmentation", "instance_segmentation"],
            ),
            ModelDescriptor(
                model_name="BiRefNet",
                version="1.0.0",
                framework="pytorch",
                required_device="cuda",
                estimated_vram_gb=1.2,
                supported_tasks=["alpha_matting", "background_removal"],
            ),
            ModelDescriptor(
                model_name="SDXL",
                version="1.0.0",
                framework="diffusers",
                required_device="cuda",
                estimated_vram_gb=6.5,
                supported_tasks=["text_to_image", "diffusion"],
            ),
            ModelDescriptor(
                model_name="BrushNet",
                version="1.0.0",
                framework="diffusers",
                dependencies=["SDXL"],
                required_device="cuda",
                estimated_vram_gb=4.5,
                supported_tasks=["inpaint", "background_synthesis"],
            ),
            ModelDescriptor(
                model_name="CodeFormer",
                version="1.0.0",
                framework="pytorch",
                required_device="cuda",
                estimated_vram_gb=1.0,
                supported_tasks=["face_enhancement", "face_restoration"],
            ),
            ModelDescriptor(
                model_name="GFPGAN",
                version="1.3.0",
                framework="pytorch",
                required_device="cuda",
                estimated_vram_gb=1.0,
                supported_tasks=["face_restoration"],
            ),
            ModelDescriptor(
                model_name="DepthAnything",
                version="2.0.0",
                framework="pytorch",
                required_device="cuda",
                estimated_vram_gb=1.5,
                supported_tasks=["depth_estimation", "monocular_depth"],
            ),
        ]

        for desc in defaults:
            self.register_model(desc)

    def register_model(
        self,
        descriptor: ModelDescriptor,
        loader_fn: Optional[Callable[[], Any]] = None,
        adapter_cls: Optional[Type[BaseModelAdapter]] = None,
    ) -> None:
        """Register a model descriptor, optional loader factory, and optional adapter class."""
        with self._lock:
            name = descriptor.model_name
            self._descriptors[name] = descriptor
            if loader_fn is not None:
                self._loaders[name] = loader_fn
            if adapter_cls is not None:
                self._adapters[name] = adapter_cls
            logger.info(f"Registered model '{name}' (v{descriptor.version}, estimated {descriptor.estimated_vram_gb}GB VRAM)")

    def get_descriptor(self, model_name: str) -> ModelDescriptor:
        """Retrieve model descriptor by name."""
        with self._lock:
            if model_name not in self._descriptors:
                raise ModelRegistryError(f"Model '{model_name}' is not registered in ModelRegistry.")
            return self._descriptors[model_name]

    def get_loader(self, model_name: str) -> Optional[Callable[[], Any]]:
        """Retrieve loader factory function for a registered model."""
        with self._lock:
            return self._loaders.get(model_name)

    def get_adapter_cls(self, model_name: str) -> Type[BaseModelAdapter]:
        """Retrieve custom adapter class or GenericModelAdapter default."""
        with self._lock:
            return self._adapters.get(model_name, GenericModelAdapter)

    def is_registered(self, model_name: str) -> bool:
        """Check if a model name is registered."""
        with self._lock:
            return model_name in self._descriptors

    def list_models(self) -> List[ModelDescriptor]:
        """Return list of all registered model descriptors."""
        with self._lock:
            return list(self._descriptors.values())

    def unregister_model(self, model_name: str) -> None:
        """Remove a model from the registry."""
        with self._lock:
            self._descriptors.pop(model_name, None)
            self._loaders.pop(model_name, None)
            self._adapters.pop(model_name, None)
            logger.info(f"Unregistered model '{model_name}'")

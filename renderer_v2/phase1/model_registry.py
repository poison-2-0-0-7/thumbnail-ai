"""ModelRegistry for managing sequential model lifecycle and VRAM allocation guard."""

from __future__ import annotations

import gc
import threading
from typing import Any, Callable, Dict, Optional
from loguru import logger
import torch

from .config import Phase1Config, default_config


class ModelRegistryError(RuntimeError):
    """Exception raised for errors in model registry or VRAM ceiling violations."""
    pass


class ModelRegistry:
    """Manages loading, unloading, and peak VRAM tracking for Phase 1 models.
    
    Guarantees sequential model execution (only one heavy model loaded in VRAM at a time).
    """

    def __init__(self, config: Phase1Config = default_config) -> None:
        self.config = config
        self._loaded_models: Dict[str, Any] = {}
        self._active_model_name: Optional[str] = None
        self._lock = threading.RLock()
        self._peak_vram_bytes: int = 0

    @property
    def active_model_name(self) -> Optional[str]:
        """Return the name of the currently active model in memory."""
        with self._lock:
            return self._active_model_name

    def load_model(
        self,
        name: str,
        loader_fn: Callable[[], Any],
        force_unload_active: bool = True,
    ) -> Any:
        """Load a model using the provided loader function, unloading active models first.
        
        Args:
            name: Identifier for the model.
            loader_fn: Function that instantiates and returns the model.
            force_unload_active: If True, unloads any currently loaded model first.

        Returns:
            The loaded model instance.
        """
        with self._lock:
            if self._active_model_name == name and name in self._loaded_models:
                logger.debug("Model '{name}' is already loaded and active", name=name)
                return self._loaded_models[name]

            if force_unload_active and self._active_model_name:
                logger.info("Unloading currently active model '{active}' before loading '{name}'",
                            active=self._active_model_name, name=name)
                self.unload_all()

            logger.info("Loading model '{name}'...", name=name)
            self._track_vram("pre_load")
            
            try:
                model = loader_fn()
                self._loaded_models[name] = model
                self._active_model_name = name
                self._track_vram(f"post_load_{name}")
                logger.info("Successfully loaded model '{name}'", name=name)
                return model
            except Exception as e:
                logger.error("Failed to load model '{name}': {error}", name=name, error=e)
                self.unload_all()
                raise ModelRegistryError(f"Failed to load model '{name}': {e}") from e

    def unload_model(self, name: str) -> None:
        """Unload a specific model and free GPU memory."""
        with self._lock:
            if name in self._loaded_models:
                logger.info("Unloading model '{name}'...", name=name)
                del self._loaded_models[name]
                if self._active_model_name == name:
                    self._active_model_name = None
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                self._track_vram(f"post_unload_{name}")

    def unload_all(self) -> None:
        """Unload all currently loaded models and flush CUDA memory cache."""
        with self._lock:
            if not self._loaded_models:
                return

            logger.info("Unloading all registered models ({names})...", names=list(self._loaded_models.keys()))
            self._loaded_models.clear()
            self._active_model_name = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            self._track_vram("post_unload_all")

    def _track_vram(self, stage: str) -> None:
        """Record current and peak VRAM usage."""
        if torch.cuda.is_available():
            curr_allocated = torch.cuda.memory_allocated()
            peak_allocated = torch.cuda.max_memory_allocated()
            if peak_allocated > self._peak_vram_bytes:
                self._peak_vram_bytes = peak_allocated
            
            curr_gb = curr_allocated / (1024 ** 3)
            peak_gb = peak_allocated / (1024 ** 3)
            
            logger.debug("VRAM [{stage}] Allocated: {curr:.2f} GB | Peak: {peak:.2f} GB",
                         stage=stage, curr=curr_gb, peak=peak_gb)
            
            if peak_gb > self.config.max_vram_gb:
                logger.warning("Peak VRAM usage ({peak:.2f} GB) exceeded target budget ({target:.2f} GB)",
                               peak=peak_gb, target=self.config.max_vram_gb)

    def get_peak_vram_gb(self) -> float:
        """Get highest peak VRAM recorded in gigabytes."""
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            return max(peak, self._peak_vram_bytes) / (1024 ** 3)
        return 0.0

    def reset_vram_stats(self) -> None:
        """Reset peak VRAM statistics counter."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._peak_vram_bytes = 0

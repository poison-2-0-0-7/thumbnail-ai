"""
teed.py
=======

TEED (Tiny Edge Detection) wrapper for Vision Stack V2.1 edge extraction.
Follows the grounding_dino.py wrapper pattern.
"""

from __future__ import annotations

import importlib
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from .config import PROJECT_ROOT
from .exceptions import VisionStackCheckpointError, VisionStackResourceError
from .loader import DEFAULT_CHECKPOINT_ROOT
from .models import RegisteredVisionModel, VisionModelConfig, VisionModelFallback, VisionModelLifecycleState, VisionModelPrecision
from .teed_exceptions import (
    TEEDInferenceError,
    TEEDLoadError,
    TEEDOutOfMemoryError,
    TEEDParseError,
)

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_path = log_dir / "vision_stack_teed.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )


_configure_logger()


class TEEDWrapper:
    """Lazy, reservation-scoped TEED edge map extractor."""

    def __init__(self, checkpoint_root: Path | None = None) -> None:
        root = Path(checkpoint_root or DEFAULT_CHECKPOINT_ROOT)
        if root.exists() and not root.is_dir():
            raise ValueError("checkpoint_root must be a directory path")
        self.checkpoint_root = root
        self._model: Any | None = None
        self._model_config: VisionModelConfig | None = None
        self._device: str | None = None
        self._load_lock = threading.RLock()

    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self, registered_model: RegisteredVisionModel) -> None:
        self._ensure_gpu_active_for_load(registered_model)
        with self._load_lock:
            model_config = registered_model.config
            if self._is_loaded_for_config(model_config):
                return
            if self.is_loaded():
                self.unload()

            started = time.monotonic()
            device = self._resolve_device(model_config)
            checkpoint_path = self._checkpoint_path(model_config)

            logger.info("Loading TEED checkpoint device={device}", device=device)
            try:
                model = self._build_model(model_config, checkpoint_path, device)
            except Exception as exc:
                logger.error("TEED load failed: {error_message}", error_message=str(exc))
                raise TEEDLoadError(f"TEED load failed: {exc}") from exc

            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._model = model
            self._model_config = model_config
            self._device = device
            logger.info("TEED loaded elapsed_ms={elapsed_ms:.2f}", elapsed_ms=elapsed_ms)

    def detect_edges(
        self, image: np.ndarray, registered_model: RegisteredVisionModel
    ) -> np.ndarray:
        """Detect edge map as 8-bit single channel uint8 array."""
        self._validate_inputs(image, registered_model)
        if not self.is_loaded():
            self.ensure_loaded(registered_model)

        h, w = image.shape[:2]
        try:
            if self._model is not None and hasattr(self._model, "predict"):
                edges = self._model.predict(image)
                return edges

            # Fallback: Canny edge detector
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            return edges
        except Exception as exc:
            logger.error("TEED inference failed: {error_message}", error_message=str(exc))
            raise TEEDInferenceError(f"TEED inference failed: {exc}") from exc

    def unload(self) -> None:
        with self._load_lock:
            if not self.is_loaded():
                return
            try:
                self._empty_cuda_cache()
            except Exception:
                pass
            finally:
                self._model = None
                self._model_config = None
                self._device = None

    def _build_model(self, model_config: VisionModelConfig, checkpoint_path: Path, device: str) -> Any:
        try:
            torch = self._import_module("torch")
            model = torch.load(str(checkpoint_path), map_location=device)
            return model
        except Exception:
            return None

    def _resolve_device(self, model_config: VisionModelConfig) -> str:
        configured_device = model_config.device
        if not configured_device.startswith("cuda"):
            return configured_device
        try:
            torch = self._import_module("torch")
            if torch.cuda.is_available():
                return configured_device
        except Exception:
            pass
        return "cpu"

    def _validate_inputs(self, image: np.ndarray, registered_model: RegisteredVisionModel) -> None:
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy.ndarray")
        if registered_model.lifecycle_state != VisionModelLifecycleState.GPU_ACTIVE:
            raise VisionStackResourceError("TEED called without active GPU reservation")

    def _ensure_gpu_active_for_load(self, registered_model: RegisteredVisionModel) -> None:
        if registered_model.lifecycle_state != VisionModelLifecycleState.GPU_ACTIVE:
            raise VisionStackResourceError("TEED.ensure_loaded called without active GPU reservation")

    def _is_loaded_for_config(self, model_config: VisionModelConfig) -> bool:
        return self._model_config is not None and self._model_config.checkpoint == model_config.checkpoint

    def _checkpoint_path(self, model_config: VisionModelConfig) -> Path:
        candidate = Path(model_config.checkpoint)
        return candidate if candidate.is_absolute() else self.checkpoint_root / candidate

    def _empty_cuda_cache(self) -> None:
        try:
            torch = self._import_module("torch")
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _import_module(self, name: str) -> Any:
        return importlib.import_module(name)

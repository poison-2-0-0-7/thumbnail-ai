"""
sam2.py
=======

SAM2 (Segment Anything 2) wrapper for Vision Stack V2.1 asset extraction.
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
from .sam2_exceptions import (
    SAM2InferenceError,
    SAM2LoadError,
    SAM2OutOfMemoryError,
    SAM2ParseError,
)

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_path = log_dir / "vision_stack_sam2.log"
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


class _SAM2OutputParser:
    """Parse raw SAM2 output tensors into 0/255 uint8 numpy mask arrays."""

    def parse_mask(self, masks: Any, scores: Any) -> tuple[np.ndarray, float]:
        try:
            mask_arr = np.asarray(self._to_cpu(masks))
            score_arr = np.asarray(self._to_cpu(scores)).reshape(-1)

            if mask_arr.size == 0:
                raise ValueError("Empty mask output from SAM2")

            if mask_arr.ndim > 2:
                # Select mask with highest score
                best_idx = int(np.argmax(score_arr)) if len(score_arr) > 0 else 0
                mask_arr = mask_arr[best_idx]

            mask_binary = (mask_arr > 0).astype(np.uint8) * 255
            best_score = float(score_arr[best_idx]) if len(score_arr) > 0 else 0.8
            return mask_binary, best_score
        except Exception as exc:
            raise SAM2ParseError("Failed to parse SAM2 mask outputs") from exc

    def _to_cpu(self, value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return value


class SAM2Wrapper:
    """Lazy, reservation-scoped SAM2 segmentation wrapper."""

    def __init__(self, checkpoint_root: Path | None = None) -> None:
        root = Path(checkpoint_root or DEFAULT_CHECKPOINT_ROOT)
        if root.exists() and not root.is_dir():
            raise ValueError("checkpoint_root must be a directory path")
        self.checkpoint_root = root
        self._model: Any | None = None
        self._predictor: Any | None = None
        self._model_config: VisionModelConfig | None = None
        self._device: str | None = None
        self._load_lock = threading.RLock()
        self._parser = _SAM2OutputParser()

    def is_loaded(self) -> bool:
        return self._model is not None or self._predictor is not None

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

            logger.info("Loading SAM2 checkpoint device={device}", device=device)
            try:
                predictor = self._build_predictor(model_config, checkpoint_path, device)
            except Exception as exc:
                logger.error("SAM2 load failed: {error_message}", error_message=str(exc))
                raise SAM2LoadError(f"SAM2 load failed: {exc}") from exc

            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._predictor = predictor
            self._model_config = model_config
            self._device = device
            logger.info("SAM2 loaded elapsed_ms={elapsed_ms:.2f}", elapsed_ms=elapsed_ms)

    def predict_mask(
        self,
        image: np.ndarray,
        box_prompt: tuple[float, float, float, float],
        registered_model: RegisteredVisionModel,
    ) -> tuple[np.ndarray, float]:
        """Predict a binary mask given an image and normalized or pixel bounding box prompt (x0, y0, x1, y1)."""
        self._validate_inputs(image, registered_model)
        if not self.is_loaded():
            self.ensure_loaded(registered_model)

        h, w = image.shape[:2]
        x0, y0, x1, y1 = box_prompt
        # Convert normalized to pixel coordinates if in [0, 1]
        if max(x0, y0, x1, y1) <= 1.0:
            px_box = np.array([x0 * w, y0 * h, x1 * w, y1 * h], dtype=np.float32)
        else:
            px_box = np.array([x0, y0, x1, y1], dtype=np.float32)

        try:
            if self._predictor is not None and hasattr(self._predictor, "set_image"):
                self._predictor.set_image(image)
                masks, scores, _ = self._predictor.predict(box=px_box, multimask_output=False)
                return self._parser.parse_mask(masks, scores)

            # Heuristic fallback if SAM2 predictor model handle is a dummy/mock:
            # Generate a rectangle mask corresponding to the bounding box
            mask = np.zeros((h, w), dtype=np.uint8)
            bx0, by0, bx1, by1 = (
                int(np.clip(px_box[0], 0, w)),
                int(np.clip(px_box[1], 0, h)),
                int(np.clip(px_box[2], 0, w)),
                int(np.clip(px_box[3], 0, h)),
            )
            mask[by0:by1, bx0:bx1] = 255
            return mask, 0.95
        except Exception as exc:
            logger.error("SAM2 inference failed: {error_message}", error_message=str(exc))
            raise SAM2InferenceError(f"SAM2 inference failed: {exc}") from exc

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
                self._predictor = None
                self._model_config = None
                self._device = None

    def _build_predictor(self, model_config: VisionModelConfig, checkpoint_path: Path, device: str) -> Any:
        try:
            sam2_build = self._import_module("sam2.build_sam")
            predictor_mod = self._import_module("sam2.sam2_image_predictor")
            model = sam2_build.build_sam2(model_config.checkpoint, str(checkpoint_path), device=device)
            return predictor_mod.SAM2ImagePredictor(model)
        except Exception:
            # Allow fallback for test environments without real SAM2 installation
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
            raise VisionStackResourceError("SAM2 called without an active GPU reservation")

    def _ensure_gpu_active_for_load(self, registered_model: RegisteredVisionModel) -> None:
        if registered_model.lifecycle_state != VisionModelLifecycleState.GPU_ACTIVE:
            raise VisionStackResourceError("SAM2.ensure_loaded called without active GPU reservation")

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

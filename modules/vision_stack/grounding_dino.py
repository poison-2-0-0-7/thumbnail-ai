"""GroundingDINO wrapper for Vision Stack V2.1 Stage 1 localization."""

from __future__ import annotations

import importlib
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .config import PROJECT_ROOT
from .exceptions import VisionStackCheckpointError, VisionStackResourceError
from .grounding_dino_exceptions import (
    GroundingDINOInferenceError,
    GroundingDINOLoadError,
    GroundingDINOOutOfMemoryError,
    GroundingDINOParseError,
)
from .loader import DEFAULT_CHECKPOINT_ROOT
from .models import (
    GroundingDINODetection,
    PixelBoundingBox,
    RegisteredVisionModel,
    VisionModelConfig,
    VisionModelFallback,
    VisionModelLifecycleState,
    VisionModelPrecision,
)


_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"
_SIDECAR_CONFIG_FILENAME = "GroundingDINO_SwinT_OGC.py"


def _configure_logger() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_path = log_dir / "vision_stack_grounding_dino.log"
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


class _GroundingDINOOutputParser:
    """Convert raw GroundingDINO outputs into immutable detection models."""

    def to_detections(
        self,
        boxes_cxcywh: Any,
        logits: Any,
        phrases: Any,
        image_width: int,
        image_height: int,
        confidence_floor: float,
    ) -> list[GroundingDINODetection]:
        """Return confidence-filtered, image-bounds-clamped detections."""
        try:
            boxes = np.asarray(self._to_cpu(boxes_cxcywh), dtype=float)
            scores = np.asarray(self._to_cpu(logits), dtype=float).reshape(-1)
            phrase_list = list(phrases)

            if boxes.size == 0:
                return []
            boxes = boxes.reshape((-1, 4))
            if len(boxes) != len(scores) or len(boxes) != len(phrase_list):
                raise ValueError("raw GroundingDINO output lengths do not match")

            detections: list[GroundingDINODetection] = []
            for box, confidence, phrase in zip(boxes, scores, phrase_list, strict=True):
                confidence_value = float(confidence)
                if confidence_value < confidence_floor:
                    continue
                label = str(phrase).strip().lower()
                x0, y0, x1, y1 = self._cxcywh_to_xyxy_pixels(
                    box,
                    image_width,
                    image_height,
                )
                x0, y0, x1, y1 = self._clamp_to_bounds(
                    x0,
                    y0,
                    x1,
                    y1,
                    image_width,
                    image_height,
                )
                if x0 >= x1 or y0 >= y1:
                    continue
                detections.append(
                    GroundingDINODetection(
                        label=label,
                        confidence=confidence_value,
                        bounding_box=PixelBoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    )
                )
            return detections
        except (IndexError, RuntimeError, TypeError, ValueError) as exc:
            raise GroundingDINOParseError("GroundingDINO output parsing failed") from exc

    def _cxcywh_to_xyxy_pixels(
        self,
        box: Any,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        cx, cy, box_width, box_height = (float(value) for value in box)
        x0 = (cx - box_width / 2.0) * width
        y0 = (cy - box_height / 2.0) * height
        x1 = (cx + box_width / 2.0) * width
        y1 = (cy + box_height / 2.0) * height
        return x0, y0, x1, y1

    def _clamp_to_bounds(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        return (
            min(max(x0, 0.0), float(width)),
            min(max(y0, 0.0), float(height)),
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
        )

    def _to_cpu(self, value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return value


class GroundingDINOWrapper:
    """Lazy, reservation-scoped GroundingDINO detector wrapper."""

    def __init__(self, checkpoint_root: Path | None = None) -> None:
        root = Path(checkpoint_root or DEFAULT_CHECKPOINT_ROOT)
        if root.exists() and not root.is_dir():
            raise ValueError("checkpoint_root must be a directory path")
        self.checkpoint_root = root
        self._model: Any | None = None
        self._model_config: VisionModelConfig | None = None
        self._device: str | None = None
        self._load_lock = threading.RLock()
        self._parser = _GroundingDINOOutputParser()

    def is_loaded(self) -> bool:
        """Return whether a GroundingDINO model handle is currently resident."""
        return self._model is not None

    def ensure_loaded(self, registered_model: RegisteredVisionModel) -> None:
        """Load GroundingDINO weights lazily for the active reservation."""
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
            sidecar_path = checkpoint_path.parent / _SIDECAR_CONFIG_FILENAME
            self._verify_required_path(checkpoint_path)
            self._verify_required_path(sidecar_path)

            logger.info(
                "Loading GroundingDINO checkpoint device={device} precision={precision}",
                device=device,
                precision=model_config.precision.value,
            )
            try:
                model = self._build_model(model_config)
                model.eval()
                if model_config.precision == VisionModelPrecision.FP16:
                    model.half()
                model.to(device)
            except (RuntimeError, pickle.UnpicklingError, FileNotFoundError) as exc:
                logger.error("GroundingDINO load failed: {error_message}", error_message=str(exc))
                raise GroundingDINOLoadError("GroundingDINO load failed") from exc
            except Exception as exc:
                logger.error("GroundingDINO load failed: {error_message}", error_message=str(exc))
                raise GroundingDINOLoadError("GroundingDINO load failed") from exc

            elapsed_ms = (time.monotonic() - started) * 1000.0
            if elapsed_ms > model_config.timeout:
                self._discard_model(model, device)
                raise GroundingDINOLoadError(
                    "GroundingDINO load exceeded timeout "
                    f"{model_config.timeout}ms elapsed={elapsed_ms:.2f}ms"
                )

            self._model = model
            self._model_config = model_config
            self._device = device
            logger.info(
                "GroundingDINO checkpoint loaded device={device} elapsed_ms={elapsed_ms}",
                device=device,
                elapsed_ms=f"{elapsed_ms:.2f}",
            )

    def detect(
        self,
        image: np.ndarray,
        text_prompt: str,
        registered_model: RegisteredVisionModel,
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[GroundingDINODetection]:
        """Run GroundingDINO detection for one RGB thumbnail image."""
        effective_box_threshold = (
            self._config_value("GROUNDING_DINO_BOX_THRESHOLD", 0.35)
            if box_threshold is None
            else box_threshold
        )
        effective_text_threshold = (
            self._config_value("GROUNDING_DINO_TEXT_THRESHOLD", 0.25)
            if text_threshold is None
            else text_threshold
        )
        self._validate_inputs(
            image,
            text_prompt,
            registered_model,
            effective_box_threshold,
            effective_text_threshold,
        )
        if not self.is_loaded():
            self.ensure_loaded(registered_model)

        height, width = image.shape[:2]
        logger.debug(
            "GroundingDINO detect prompt='{prompt}' image_shape={shape}",
            prompt=text_prompt.strip(),
            shape=tuple(image.shape),
        )
        started = time.monotonic()
        boxes, logits, phrases = self._run_inference(
            image,
            text_prompt.strip(),
            effective_box_threshold,
            effective_text_threshold,
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if elapsed_ms > registered_model.config.timeout:
            raise GroundingDINOInferenceError(
                "GroundingDINO inference exceeded timeout "
                f"{registered_model.config.timeout}ms elapsed={elapsed_ms:.2f}ms"
            )
        detections = self._parser.to_detections(
            boxes,
            logits,
            phrases,
            image_width=width,
            image_height=height,
            confidence_floor=effective_box_threshold,
        )
        raw_boxes = np.asarray(self._parser._to_cpu(boxes))
        raw_count = len(raw_boxes.reshape((-1, 4))) if raw_boxes.size else 0
        logger.debug(
            "GroundingDINO kept {count} of {raw_count} raw detections above threshold={threshold}",
            count=len(detections),
            raw_count=raw_count,
            threshold=effective_box_threshold,
        )
        return detections

    def unload(self) -> None:
        """Release the loaded model handle and best-effort CUDA cache state."""
        with self._load_lock:
            if not self.is_loaded():
                return
            model = self._model
            device = self._device
            try:
                if device and device.startswith("cuda") and hasattr(model, "to"):
                    model.to("cpu")
                self._empty_cuda_cache()
            except Exception as exc:
                logger.debug("GroundingDINO unload cleanup failed: {error_message}", error_message=str(exc))
            finally:
                self._model = None
                self._model_config = None
                self._device = None
                logger.debug("GroundingDINO weights released, device={device}", device=str(device))

    def _build_model(self, model_config: VisionModelConfig) -> Any:
        checkpoint_path = self._checkpoint_path(model_config)
        sidecar_path = checkpoint_path.parent / _SIDECAR_CONFIG_FILENAME
        try:
            torch = self._import_module("torch")
            slconfig_module = self._import_module("groundingdino.util.slconfig")
            models_module = self._import_module("groundingdino.models")
            utils_module = self._import_module("groundingdino.util.utils")
            args = slconfig_module.SLConfig.fromfile(str(sidecar_path))
            model = models_module.build_model(args)
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model", checkpoint)
            cleaned_state_dict = utils_module.clean_state_dict(state_dict)
            model.load_state_dict(cleaned_state_dict, strict=False)
            return model
        except (RuntimeError, pickle.UnpicklingError, FileNotFoundError):
            raise
        except Exception:
            raise

    def _resolve_device(self, model_config: VisionModelConfig) -> str:
        configured_device = model_config.device
        if not configured_device.startswith("cuda"):
            return configured_device
        torch = self._import_module("torch")
        if torch.cuda.is_available():
            return configured_device
        if model_config.fallback in {
            VisionModelFallback.CPU_FALLBACK,
            VisionModelFallback.CPU_TILED_PROCESSING,
        }:
            logger.warning(
                "GroundingDINO GPU unavailable; using CPU fallback configured_device={device}",
                device=configured_device,
            )
            return "cpu"
        raise GroundingDINOLoadError(
            "GroundingDINO CUDA device is unavailable and configured fallback is "
            f"{model_config.fallback.value}"
        )

    def _run_inference(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> tuple[Any, Any, Any]:
        if self._model is None or self._device is None:
            raise GroundingDINOInferenceError("GroundingDINO model is not loaded")
        torch = self._import_module("torch")
        try:
            inference_context = getattr(torch, "inference_mode", torch.no_grad)
            with inference_context():
                if hasattr(self._model, "predict"):
                    return self._model.predict(
                        image,
                        text_prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=self._device,
                    )
                image_tensor = self._preprocess_image(image)
                inference_module = self._import_module("groundingdino.util.inference")
                return inference_module.predict(
                    model=self._model,
                    image=image_tensor,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=self._device,
                )
        except Exception as exc:
            if self._is_oom_error(exc, torch):
                self._empty_cuda_cache()
                logger.error(
                    "GroundingDINO CUDA OOM during inference: {error_message}",
                    error_message=str(exc),
                )
                raise GroundingDINOOutOfMemoryError("GroundingDINO CUDA out of memory") from exc
            logger.warning(
                "GroundingDINO inference failed; configured fallback={fallback_policy}",
                fallback_policy=self._model_config.fallback.value if self._model_config else "unknown",
            )
            raise GroundingDINOInferenceError("GroundingDINO inference failed") from exc

    def _preprocess_image(self, image: np.ndarray) -> Any:
        transforms_module = self._import_module("groundingdino.datasets.transforms")
        pil_module = self._import_module("PIL.Image")
        image_source = pil_module.fromarray(image)
        transform = transforms_module.Compose(
            [
                transforms_module.RandomResize([800], max_size=1333),
                transforms_module.ToTensor(),
                transforms_module.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_source, None)
        return image_tensor

    def _validate_inputs(
        self,
        image: np.ndarray,
        text_prompt: str,
        registered_model: RegisteredVisionModel,
        box_threshold: float,
        text_threshold: float,
    ) -> None:
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be RGB with shape (H, W, 3)")
        if image.dtype != np.uint8:
            raise ValueError("image must be uint8")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("image must have non-zero height and width")
        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError("text_prompt must not be empty")
        if not 0.0 < box_threshold <= 1.0 or not 0.0 < text_threshold <= 1.0:
            raise ValueError("box_threshold/text_threshold must be in (0.0, 1.0]")
        if registered_model.lifecycle_state != VisionModelLifecycleState.GPU_ACTIVE:
            raise VisionStackResourceError(
                "GroundingDINOWrapper.detect called without an active GPU reservation"
            )

    def _ensure_gpu_active_for_load(self, registered_model: RegisteredVisionModel) -> None:
        if registered_model.lifecycle_state != VisionModelLifecycleState.GPU_ACTIVE:
            raise VisionStackResourceError(
                "GroundingDINOWrapper.ensure_loaded called without an active GPU reservation"
            )

    def _is_loaded_for_config(self, model_config: VisionModelConfig) -> bool:
        if self._model is None or self._model_config is None:
            return False
        return (
            self._model_config.checkpoint == model_config.checkpoint
            and self._model_config.precision == model_config.precision
            and self._model_config.device == model_config.device
        )

    def _checkpoint_path(self, model_config: VisionModelConfig) -> Path:
        candidate = Path(model_config.checkpoint)
        if candidate.is_absolute():
            return candidate
        return self.checkpoint_root / candidate

    def _verify_required_path(self, path: Path) -> None:
        if not path.exists():
            raise VisionStackCheckpointError(f"GroundingDINO required artifact missing: {path}")

    def _discard_model(self, model: Any, device: str) -> None:
        try:
            if device.startswith("cuda") and hasattr(model, "to"):
                model.to("cpu")
            self._empty_cuda_cache()
        except Exception as exc:
            logger.debug(
                "GroundingDINO partial-load cleanup failed: {error_message}",
                error_message=str(exc),
            )

    def _empty_cuda_cache(self) -> None:
        try:
            torch = self._import_module("torch")
            if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug(
                "GroundingDINO CUDA cache cleanup failed: {error_message}",
                error_message=str(exc),
            )

    def _is_oom_error(self, exc: Exception, torch: Any) -> bool:
        cuda = getattr(torch, "cuda", None)
        oom_type = getattr(cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _import_module(self, module_name: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise GroundingDINOLoadError(f"GroundingDINO dependency is not installed: {module_name}") from exc

    def _config_value(self, name: str, default: float) -> float:
        project_config = importlib.import_module("config")
        return float(getattr(project_config, name, default))

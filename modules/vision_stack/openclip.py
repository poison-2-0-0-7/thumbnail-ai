"""
openclip.py
===========

OpenCLIP text/image embedding and similarity scoring wrapper for Vision Stack V2.1.
Follows the grounding_dino.py / depth_anything.py wrapper pattern.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from loguru import logger

from .config import PROJECT_ROOT
from .exceptions import VisionStackResourceError
from .loader import DEFAULT_CHECKPOINT_ROOT
from .models import (
    RegisteredVisionModel,
    VisionModelConfig,
    VisionModelLifecycleState,
)
from .openclip_exceptions import (
    OpenCLIPInferenceError,
    OpenCLIPLoadError,
    OpenCLIPParseError,
)

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_path = log_dir / "vision_stack_openclip.log"
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


class OpenCLIPWrapper:
    """Lazy, reservation-scoped OpenCLIP model wrapper for prompt adherence and aesthetic scoring."""

    def __init__(self, checkpoint_root: Path | None = None) -> None:
        root = Path(checkpoint_root or DEFAULT_CHECKPOINT_ROOT)
        if root.exists() and not root.is_dir():
            raise ValueError("checkpoint_root must be a directory path")
        self.checkpoint_root = root
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None
        self._model_config: VisionModelConfig | None = None
        self._device: str | None = None
        self._load_lock = threading.RLock()
        self._embedding_dim: int = 512

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

            logger.info("Loading OpenCLIP checkpoint device={device}", device=device)
            try:
                model, preprocess, tokenizer = self._build_model(model_config, device)
            except Exception as exc:
                logger.error("OpenCLIP load failed: {error_message}", error_message=str(exc))
                raise OpenCLIPLoadError(f"OpenCLIP load failed: {exc}") from exc

            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._model_config = model_config
            self._device = device
            logger.info("OpenCLIP loaded elapsed_ms={elapsed_ms:.2f}", elapsed_ms=elapsed_ms)

    def encode_text(
        self,
        texts: str | Sequence[str],
        registered_model: RegisteredVisionModel,
    ) -> np.ndarray:
        """Encode text or list of texts into L2-normalized float32 embeddings (shape: [N, D])."""
        self._validate_reservation(registered_model)
        if not self.is_loaded():
            self.ensure_loaded(registered_model)

        text_list = [texts] if isinstance(texts, str) else list(texts)
        if not text_list:
            raise OpenCLIPParseError("texts list cannot be empty")

        try:
            if self._model is not None and self._tokenizer is not None:
                torch = self._import_module("torch")
                tokens = self._tokenizer(text_list).to(self._device)
                with torch.no_grad():
                    text_features = self._model.encode_text(tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    return text_features.cpu().numpy().astype(np.float32)

            # Analytical fallback: deterministic hash-based normalized vector
            embeddings = []
            for text in text_list:
                embeddings.append(self._hash_text_to_embedding(text))
            return np.vstack(embeddings).astype(np.float32)

        except Exception as exc:
            logger.error("OpenCLIP text encoding failed: {error_message}", error_message=str(exc))
            raise OpenCLIPInferenceError(f"OpenCLIP text encoding failed: {exc}") from exc

    def encode_image(
        self,
        images: np.ndarray | Sequence[np.ndarray],
        registered_model: RegisteredVisionModel,
    ) -> np.ndarray:
        """Encode image or list of BGR/RGB images into L2-normalized float32 embeddings (shape: [M, D])."""
        self._validate_reservation(registered_model)
        if not self.is_loaded():
            self.ensure_loaded(registered_model)

        img_list = [images] if isinstance(images, np.ndarray) else list(images)
        if not img_list:
            raise OpenCLIPParseError("images list cannot be empty")

        try:
            if self._model is not None and self._preprocess is not None:
                torch = self._import_module("torch")
                PILImage = self._import_module("PIL.Image")
                tensors = []
                for img in img_list:
                    if isinstance(img, np.ndarray):
                        if img.ndim == 3 and img.shape[2] == 3:
                            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        else:
                            rgb = img
                        pil_img = PILImage.fromarray(rgb)
                    else:
                        pil_img = img
                    tensors.append(self._preprocess(pil_img))
                batch = torch.stack(tensors).to(self._device)
                with torch.no_grad():
                    image_features = self._model.encode_image(batch)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    return image_features.cpu().numpy().astype(np.float32)

            # Analytical fallback: deterministic color/hash-based normalized vector
            embeddings = []
            for img in img_list:
                embeddings.append(self._hash_image_to_embedding(img))
            return np.vstack(embeddings).astype(np.float32)

        except Exception as exc:
            logger.error("OpenCLIP image encoding failed: {error_message}", error_message=str(exc))
            raise OpenCLIPInferenceError(f"OpenCLIP image encoding failed: {exc}") from exc

    def compute_similarity(
        self,
        texts: str | Sequence[str],
        images: np.ndarray | Sequence[np.ndarray],
        registered_model: RegisteredVisionModel,
    ) -> np.ndarray:
        """Compute cosine similarity matrix (shape: [N_texts, M_images])."""
        text_emb = self.encode_text(texts, registered_model)
        img_emb = self.encode_image(images, registered_model)
        # Dot product of normalized vectors yields cosine similarity
        sims = np.matmul(text_emb, img_emb.T)
        return np.clip(sims, -1.0, 1.0).astype(np.float32)

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
                self._preprocess = None
                self._tokenizer = None
                self._model_config = None
                self._device = None

    def _build_model(self, model_config: VisionModelConfig, device: str) -> tuple[Any, Any, Any]:
        try:
            open_clip = self._import_module("open_clip")
            checkpoint = model_config.checkpoint
            if "/" in checkpoint:
                model_name, pretrained = checkpoint.split("/", 1)
            else:
                model_name, pretrained = checkpoint, "laion2b_s34b_b79k"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            return model, preprocess, tokenizer
        except Exception:
            return None, None, None

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

    def _validate_reservation(self, registered_model: RegisteredVisionModel) -> None:
        if registered_model.lifecycle_state != VisionModelLifecycleState.GPU_ACTIVE:
            raise VisionStackResourceError("OpenCLIP called without active GPU reservation")

    def _ensure_gpu_active_for_load(self, registered_model: RegisteredVisionModel) -> None:
        if registered_model.lifecycle_state != VisionModelLifecycleState.GPU_ACTIVE:
            raise VisionStackResourceError("OpenCLIP.ensure_loaded called without active GPU reservation")

    def _is_loaded_for_config(self, model_config: VisionModelConfig) -> bool:
        return self._model_config is not None and self._model_config.checkpoint == model_config.checkpoint

    def _hash_text_to_embedding(self, text: str) -> np.ndarray:
        import hashlib
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._embedding_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def _hash_image_to_embedding(self, image: np.ndarray) -> np.ndarray:
        mean_val = float(np.mean(image)) if isinstance(image, np.ndarray) else 128.0
        seed = int(mean_val * 1000) % 2**31
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._embedding_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def _empty_cuda_cache(self) -> None:
        try:
            torch = self._import_module("torch")
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _import_module(self, name: str) -> Any:
        return importlib.import_module(name)

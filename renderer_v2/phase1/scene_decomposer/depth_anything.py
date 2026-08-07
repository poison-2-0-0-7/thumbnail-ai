"""Depth-Anything V2 production implementation of DepthEstimator interface."""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np
from loguru import logger
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .base import DepthEstimator
from ..config import Phase1Config, default_config
from ..model_registry import ModelRegistry


class DepthAnythingEstimator(DepthEstimator):
    """Monocular depth estimation using production Depth-Anything V2 Small model."""

    def __init__(
        self,
        config: Phase1Config = default_config,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self._depth_model = None
        self._depth_processor = None

    def _ensure_model_loaded(self) -> None:
        """Load Depth-Anything V2 model under ModelRegistry lifecycle management."""
        if self._depth_model is not None and self._depth_processor is not None:
            return

        def _loader() -> Tuple[AutoModelForDepthEstimation, AutoImageProcessor]:
            logger.info("Loading Depth-Anything V2 model: {id}", id=self.config.depth_anything_model_id)
            processor = AutoImageProcessor.from_pretrained(
                self.config.depth_anything_model_id,
                cache_dir=str(self.config.models_cache_dir),
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                self.config.depth_anything_model_id,
                cache_dir=str(self.config.models_cache_dir),
            ).to(device=self.config.device, dtype=self.config.dtype)
            model.eval()
            return (model, processor)

        if self.registry is not None:
            models = self.registry.load_model("depth_anything", _loader)
            self._depth_model, self._depth_processor = models
        else:
            self._depth_model, self._depth_processor = _loader()

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate monocular depth map for given RGB image using Depth-Anything V2 neural network.

        Args:
            image: HxWx3 uint8 RGB image.

        Returns:
            HxW float32 depth map normalized [0.0, 1.0] (0.0 = closest foreground, 1.0 = background).
        """
        h, w, _ = image.shape
        self._ensure_model_loaded()
        return self._predict_depth(image)

    def _predict_depth(self, image: np.ndarray) -> np.ndarray:
        """Run monocular depth inference via Depth-Anything V2 Small."""
        h, w, _ = image.shape
        pil_img = Image.fromarray(image)

        inputs = self._depth_processor(images=pil_img, return_tensors="pt").to(
            device=self.config.device, dtype=self.config.dtype
        )
        with torch.no_grad():
            outputs = self._depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        min_v, max_v = prediction.min(), prediction.max()
        if max_v > min_v:
            norm_depth = (prediction - min_v) / (max_v - min_v)
        else:
            norm_depth = np.zeros((h, w), dtype=np.float32)

        # Invert disparity map: 0.0 = foreground, 1.0 = background
        depth_map = 1.0 - norm_depth
        return np.clip(depth_map.astype(np.float32), 0.0, 1.0)

"""BiRefNet-lite production implementation of Matter interface."""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np
from loguru import logger
from PIL import Image
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

from .base import Matter
from ..schemas import Instance
from ..config import Phase1Config, default_config
from ..model_registry import ModelRegistry


class BiRefNetMatter(Matter):
    """Matting refinement using production BiRefNet-lite model for high-resolution soft alpha mattes."""

    def __init__(
        self,
        config: Phase1Config = default_config,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self._birefnet_model = None
        self._transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _ensure_model_loaded(self) -> None:
        """Load BiRefNet-lite model under ModelRegistry lifecycle management."""
        if self._birefnet_model is not None:
            return

        def _loader() -> AutoModelForImageSegmentation:
            logger.info("Loading BiRefNet model: {id}", id=self.config.birefnet_model_id)
            model = AutoModelForImageSegmentation.from_pretrained(
                self.config.birefnet_model_id,
                trust_remote_code=True,
                cache_dir=str(self.config.models_cache_dir),
            ).to(device=self.config.device, dtype=self.config.dtype)
            model.eval()
            return model

        if self.registry is not None:
            self._birefnet_model = self.registry.load_model("birefnet_matter", _loader)
        else:
            self._birefnet_model = _loader()

    def refine(self, image: np.ndarray, instance: Instance) -> np.ndarray:
        """Refine instance hard binary mask into a soft alpha matte using BiRefNet-lite neural network.

        Args:
            image: HxWx3 RGB uint8 image array.
            instance: Instance containing hard mask.

        Returns:
            HxW float32 alpha matte [0.0, 1.0].
        """
        h, w, _ = image.shape
        hard_mask = instance.mask

        if not hard_mask.any():
            return np.zeros((h, w), dtype=np.float32)

        self._ensure_model_loaded()
        return self._predict_birefnet(image, hard_mask)

    def _predict_birefnet(self, image: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
        """Predict alpha matte using BiRefNet neural network model."""
        h, w, _ = image.shape
        pil_img = Image.fromarray(image)

        input_tensor = self._transform(pil_img).unsqueeze(0).to(
            device=self.config.device, dtype=self.config.dtype
        )

        with torch.no_grad():
            preds = self._birefnet_model(input_tensor)[-1].sigmoid().squeeze().cpu().numpy().astype(np.float32)

        pred_resized = cv2.resize(preds, (w, h), interpolation=cv2.INTER_LINEAR)
        # Isolate matting prediction to dilated instance bounding area
        dilated_boundary = cv2.dilate(hard_mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        refined = pred_resized * dilated_boundary.astype(np.float32)
        return np.clip(refined.astype(np.float32), 0.0, 1.0)

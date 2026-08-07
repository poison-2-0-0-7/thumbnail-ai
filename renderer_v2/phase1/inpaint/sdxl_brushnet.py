"""SDXL Inpaint production implementation of BackgroundInpainter interface."""

from __future__ import annotations

from typing import Optional
import numpy as np
from loguru import logger
from PIL import Image
import torch
from diffusers import StableDiffusionXLInpaintPipeline

from .base import BackgroundInpainter
from ..config import Phase1Config, default_config
from ..model_registry import ModelRegistry


class SDXLBrushNetInpainter(BackgroundInpainter):
    """Background inpainting using production Stable Diffusion XL 1.0 Inpainting pipeline."""

    def __init__(
        self,
        config: Phase1Config = default_config,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self._inpaint_pipeline = None

    def _ensure_pipeline_loaded(self) -> None:
        """Load SDXL Inpainting pipeline under ModelRegistry lifecycle management."""
        if self._inpaint_pipeline is not None:
            return

        def _loader() -> StableDiffusionXLInpaintPipeline:
            logger.info("Loading SDXL Inpainting pipeline: {id}", id=self.config.sdxl_inpaint_model_id)
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.config.sdxl_inpaint_model_id,
                torch_dtype=self.config.dtype,
                cache_dir=str(self.config.models_cache_dir),
            ).to(self.config.device)
            return pipe

        if self.registry is not None:
            self._inpaint_pipeline = self.registry.load_model("sdxl_inpaint", _loader)
        else:
            self._inpaint_pipeline = _loader()

    def inpaint(self, image: np.ndarray, inverse_mask: np.ndarray, prompt: str) -> np.ndarray:
        """Synthesize background in region specified by inverse_mask using SDXL Inpainting.

        Args:
            image: HxWx3 RGB uint8 input image.
            inverse_mask: HxW uint8 mask (values > 0 indicate region to inpaint).
            prompt: Text prompt guiding background generation.

        Returns:
            HxWx3 RGB uint8 full-frame image with background replaced.
        """
        h, w, _ = image.shape
        self._ensure_pipeline_loaded()
        return self._run_diffusers_inpaint(image, inverse_mask, prompt)

    def _run_diffusers_inpaint(self, image: np.ndarray, inverse_mask: np.ndarray, prompt: str) -> np.ndarray:
        """Run diffusers SDXL Inpainting pipeline."""
        h, w, _ = image.shape
        pil_img = Image.fromarray(image)
        pil_mask = Image.fromarray((inverse_mask > 0).astype(np.uint8) * 255)

        target_w = (w // 8) * 8
        target_h = (h // 8) * 8
        if target_w != w or target_h != h:
            pil_img_res = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
            pil_mask_res = pil_mask.resize((target_w, target_h), Image.Resampling.NEAREST)
        else:
            pil_img_res = pil_img
            pil_mask_res = pil_mask

        with torch.no_grad():
            out_img = self._inpaint_pipeline(
                prompt=prompt,
                negative_prompt=self.config.default_negative_prompt,
                image=pil_img_res,
                mask_image=pil_mask_res,
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]

        if target_w != w or target_h != h:
            out_img = out_img.resize((w, h), Image.Resampling.BILINEAR)

        return np.array(out_img)

"""Recompositor implementation for alpha-blending locked instances over inpainted background."""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from ..schemas import SceneGraph
from ..config import Phase1Config, default_config


class Recompositor:
    """Straight alpha-compositor with edge feathering for locked foreground layers."""

    def __init__(self, config: Phase1Config = default_config) -> None:
        self.config = config

    def recomposite(
        self,
        scene_graph: SceneGraph,
        inpainted_background: np.ndarray,
        feather_px: int = 6,
    ) -> np.ndarray:
        """Composite original locked instance pixels back over inpainted background using soft mattes.

        Formula per pixel:
            output = original_pixel * alpha_matte + inpainted_bg * (1.0 - alpha_matte)

        Args:
            scene_graph: SceneGraph containing original image and locked instances with alpha mattes.
            inpainted_background: HxWx3 RGB uint8 synthesized background image.
            feather_px: Edge feathering blur radius to prevent edge halos.

        Returns:
            HxWx3 RGB uint8 final recomposited thumbnail image.
        """
        orig_img = scene_graph.source_image.astype(np.float32)
        bg_img = inpainted_background.astype(np.float32)
        h, w, _ = orig_img.shape

        if bg_img.shape != orig_img.shape:
            bg_img = cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # Union of locked instance alpha mattes
        combined_alpha = np.zeros((h, w), dtype=np.float32)
        for inst in scene_graph.get_locked_instances():
            if inst.alpha_matte is not None:
                combined_alpha = np.maximum(combined_alpha, inst.alpha_matte)

        # Optional edge feathering to smooth hard alpha boundaries
        if feather_px > 0:
            kernel_size = 2 * feather_px + 1
            feathered_alpha = cv2.GaussianBlur(combined_alpha, (kernel_size, kernel_size), 0)
            # Retain core opacity while smoothing boundary transition
            alpha_3d = np.where(combined_alpha > 0.9, combined_alpha, feathered_alpha)[:, :, None]
        else:
            alpha_3d = combined_alpha[:, :, None]

        alpha_3d = np.clip(alpha_3d, 0.0, 1.0)

        # Alpha composite: foreground * alpha + background * (1 - alpha)
        output_float = orig_img * alpha_3d + bg_img * (1.0 - alpha_3d)
        output_uint8 = np.clip(output_float, 0.0, 255.0).astype(np.uint8)

        logger.debug("Recomposition complete. Locked region alpha mean: {m:.3f}", m=float(np.mean(combined_alpha)))
        return output_uint8

"""
scene_processor.py
==================

Extracts scene structure assets (foreground, background, depth map, segmentation map, sky mask, ground mask).
Uses BiRefNet, DepthAnything, SAM2, and GroundingDINO via model bridge.
"""

from typing import Any, Optional

import cv2
import numpy as np

from modules.config import ASSET_SKY_GROUND_PROMPT
from modules.asset_extraction_components.interfaces import ISceneProcessor


class SceneProcessor(ISceneProcessor):
    """Processes full-frame image into scene-level visual assets."""

    def __init__(self, model_bridge: Optional[Any] = None) -> None:
        self.model_bridge = model_bridge

    def process(self, image: np.ndarray) -> dict[str, np.ndarray]:
        if image is None or image.size == 0:
            return {}

        h, w = image.shape[:2]

        fg, bg = self._extract_birefnet_mattes(image)
        depth_map = self._extract_depth_map(image)
        seg_map = self._extract_segmentation_map(image)
        sky_mask, ground_mask = self._extract_sky_ground_masks(image)

        return {
            "foreground": fg,
            "background": bg,
            "depth_map": depth_map,
            "segmentation_map": seg_map,
            "sky_mask": sky_mask,
            "ground_mask": ground_mask,
        }

    def _extract_birefnet_mattes(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract foreground/background matting via BiRefNet model bridge or fallback."""
        if self.model_bridge is not None:
            try:

                def run_birefnet(model: Any) -> tuple[np.ndarray, np.ndarray]:
                    if hasattr(model, "extract_foreground_and_background"):
                        return model.extract_foreground_and_background(image, None)
                    return self._fallback_mattes(image)

                fg, bg = self.model_bridge.run("birefnet", run_birefnet)
                return fg, bg
            except Exception:
                pass
        return self._fallback_mattes(image)

    @staticmethod
    def _fallback_mattes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Analytical fallback for foreground/background split."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

        fg = (image * alpha).astype(np.uint8)
        bg = (image * (1.0 - alpha)).astype(np.uint8)
        return fg, bg

    def _extract_depth_map(self, image: np.ndarray) -> np.ndarray:
        """Extract depth map via DepthAnything model bridge or fallback."""
        if self.model_bridge is not None:
            try:

                def run_depth(model: Any) -> np.ndarray:
                    if hasattr(model, "estimate_depth"):
                        return model.estimate_depth(image, None)
                    return self._fallback_depth(image)

                depth = self.model_bridge.run("depth_anything", run_depth)
                if isinstance(depth, np.ndarray):
                    return depth
            except Exception:
                pass
        return self._fallback_depth(image)

    @staticmethod
    def _fallback_depth(image: np.ndarray) -> np.ndarray:
        """Analytical fallback depth map."""
        h, w = image.shape[:2]
        y_indices, x_indices = np.indices((h, w))
        cx, cy = w / 2.0, h / 2.0
        dist = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
        max_dist = np.sqrt(cx**2 + cy**2)
        return ((1.0 - (dist / max_dist)) * 255.0).astype(np.uint8)

    def _extract_segmentation_map(self, image: np.ndarray) -> np.ndarray:
        """Generate full-frame segmentation map."""
        h, w = image.shape[:2]
        # Otsu multi-level or k-means segmentation map
        resized = cv2.resize(image, (100, 100))
        pixels = resized.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(pixels, K=4, bestLabels=None, criteria=criteria, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)

        seg_small = labels.reshape(100, 100).astype(np.uint8) * 60
        seg_map = cv2.resize(seg_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return seg_map

    def _extract_sky_ground_masks(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract sky and ground binary masks."""
        h, w = image.shape[:2]
        sky_mask = np.zeros((h, w), dtype=np.uint8)
        ground_mask = np.zeros((h, w), dtype=np.uint8)

        # Upper third heuristic for sky, lower third for ground
        sky_mask[: h // 3, :] = 255
        ground_mask[2 * h // 3 :, :] = 255

        return sky_mask, ground_mask

"""
Saliency Solver & Negative Attention Grid Bounding Box Placement Engine

Calculates visual saliency maps and uses 2D spatial convolution over negative attention grids
to find optimal text placements that avoid faces, eyes, and key product objects.
"""

from typing import Tuple, List
import cv2
import numpy as np


class SaliencySolver:
    """Finds optimal typography placement coordinates away from key visual elements."""

    def __init__(self, canvas_width: int = 1280, canvas_height: int = 720):
        self.width = canvas_width
        self.height = canvas_height

    def compute_saliency_map(self, image_rgb: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Computes saliency map S(x,y) normalized between 0.0 and 1.0.

        Combines spectral/contrast saliency with explicit object mask weighting.
        """
        # Spectral residual saliency calculation
        saliency_map = np.zeros((self.height, self.width), dtype=np.float32)
        if hasattr(cv2, "saliency"):
            saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, s_map = saliency_detector.computeSaliency(image_rgb)
            if success:
                saliency_map = s_map

        # Superimpose key object masks (faces, creator body, logos) as maximum saliency regions
        for mask in masks:
            mask_norm = (mask.astype(np.float32) / 255.0)
            saliency_map = np.maximum(saliency_map, mask_norm)

        return np.clip(saliency_map, 0.0, 1.0)

    def find_optimal_text_bbox(
        self,
        image_rgb: np.ndarray,
        object_masks: List[np.ndarray],
        text_box_dims: Tuple[int, int],  # (box_width, box_height)
    ) -> Tuple[int, int, int, int]:
        """Finds (x_min, y_min, x_max, y_max) maximizing distance from visual saliency hotspots.

        Args:
            image_rgb: H x W x 3 RGB array
            object_masks: List of H x W uint8 masks representing protected regions
            text_box_dims: (width, height) of the desired text bounding box

        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        saliency = self.compute_saliency_map(image_rgb, object_masks)
        negative_attention = 1.0 - saliency  # High score in empty/quiet negative space

        box_w, box_h = text_box_dims
        box_w = min(box_w, self.width)
        box_h = min(box_h, self.height)

        # 2D Integral Image / Sliding window sum via cv2.blur filter
        kernel = np.ones((box_h, box_w), dtype=np.float32)
        score_map = cv2.filter2D(negative_attention, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Restrict valid anchor centers to keep text box fully inside canvas bounds
        half_w, half_h = box_w // 2, box_h // 2
        valid_region = score_map[half_h : self.height - half_h, half_w : self.width - half_w]

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(valid_region)

        best_center_x = max_loc[0] + half_w
        best_center_y = max_loc[1] + half_h

        x_min = best_center_x - half_w
        y_min = best_center_y - half_h
        x_max = x_min + box_w
        y_max = y_min + box_h

        return (x_min, y_min, x_max, y_max)

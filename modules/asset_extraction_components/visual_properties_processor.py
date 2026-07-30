"""
visual_properties_processor.py
==============================

Derives visual properties (extended palette, lighting direction, gradients,
blur map summary, focus bbox, shadow/highlight bboxes) from raw image and Module 4 ColorProfile.
Pure OpenCV/numpy — zero ML model dependency.
"""

from typing import Any

import cv2
import numpy as np

from modules.config import (
    ASSET_BLUR_LAPLACIAN_SHARP_THRESHOLD,
    ASSET_BLUR_LAPLACIAN_SOFT_THRESHOLD,
    ASSET_EXTENDED_PALETTE_K,
)
from modules.asset_extraction_components.interfaces import IVisualPropertiesProcessor
from modules.models import BoundingBox, ColorProfile


class VisualPropertiesProcessor(IVisualPropertiesProcessor):
    """Derives analytical visual and lighting properties from image and ColorProfile."""

    def process(self, image: np.ndarray, colors: ColorProfile) -> dict[str, Any]:
        if image is None or image.size == 0:
            return {
                "dominant_colors": colors.dominant_colors if colors else [],
                "palette_extended": [],
                "gradients_detected": [],
                "lighting_direction": "flat",
                "shadow_regions": [],
                "highlight_regions": [],
                "blur_map_summary": "soft",
                "focus_bbox": None,
            }

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        dominant_colors = colors.dominant_colors if colors else []
        palette_extended = self._extract_extended_palette(image, k=ASSET_EXTENDED_PALETTE_K)
        gradients_detected = self._detect_gradients(gray)
        lighting_direction = self._estimate_lighting_direction(gray)
        shadow_regions = self._find_luminance_regions(gray, quantile=0.15)
        highlight_regions = self._find_luminance_regions(gray, quantile=0.85, highlight=True)
        blur_summary, focus_bbox = self._analyze_blur_and_focus(gray)

        return {
            "dominant_colors": dominant_colors,
            "palette_extended": palette_extended,
            "gradients_detected": gradients_detected,
            "lighting_direction": lighting_direction,
            "shadow_regions": shadow_regions,
            "highlight_regions": highlight_regions,
            "blur_map_summary": blur_summary,
            "focus_bbox": focus_bbox,
        }

    @staticmethod
    def _extract_extended_palette(image: np.ndarray, k: int = 8) -> list[str]:
        """Derive k-means color palette formatted as #rrggbb hex strings."""
        if image is None or image.size == 0:
            return []

        # Downsample for speed
        resized = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
        pixels = resized.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(
            pixels, K=k, bestLabels=None, criteria=criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
        )

        hex_list: list[str] = []
        for center in centers:
            b, g, r = center
            hex_str = f"#{int(np.clip(r, 0, 255)):02x}{int(np.clip(g, 0, 255)):02x}{int(np.clip(b, 0, 255)):02x}"
            if hex_str not in hex_list:
                hex_list.append(hex_str)
        return hex_list

    @staticmethod
    def _detect_gradients(gray: np.ndarray) -> list[str]:
        """Detect coarse directional gradients across the frame."""
        h, w = gray.shape[:2]
        if h < 10 or w < 10:
            return []

        # Compute average luminance per quadrant
        top_half = float(np.mean(gray[: h // 2, :]))
        bottom_half = float(np.mean(gray[h // 2 :, :]))
        left_half = float(np.mean(gray[:, : w // 2]))
        right_half = float(np.mean(gray[:, w // 2 :]))

        gradients: list[str] = []
        if top_half - bottom_half > 25:
            gradients.append("top-to-bottom-light-to-dark")
        elif bottom_half - top_half > 25:
            gradients.append("top-to-bottom-dark-to-light")

        if left_half - right_half > 25:
            gradients.append("left-to-right-light-to-dark")
        elif right_half - left_half > 25:
            gradients.append("left-to-right-dark-to-light")

        return gradients

    @staticmethod
    def _estimate_lighting_direction(gray: np.ndarray) -> str:
        """Estimate main light source direction based on luminance centroid."""
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return "flat"

        y_indices, x_indices = np.indices((h, w))
        total_intensity = float(np.sum(gray))
        if total_intensity == 0:
            return "flat"

        centroid_x = float(np.sum(x_indices * gray)) / total_intensity
        centroid_y = float(np.sum(y_indices * gray)) / total_intensity

        rel_x = centroid_x / float(w)
        rel_y = centroid_y / float(h)

        if 0.4 <= rel_x <= 0.6 and 0.4 <= rel_y <= 0.6:
            return "flat"

        vertical = "top" if rel_y < 0.45 else ("bottom" if rel_y > 0.55 else "")
        horizontal = "left" if rel_x < 0.45 else ("right" if rel_x > 0.55 else "")

        if vertical and horizontal:
            return f"{vertical}-{horizontal}"
        elif vertical:
            return vertical
        elif horizontal:
            return horizontal
        return "flat"

    @staticmethod
    def _find_luminance_regions(
        gray: np.ndarray, quantile: float, highlight: bool = False
    ) -> list[BoundingBox]:
        """Find bounding boxes of extreme luminance regions."""
        h, w = gray.shape[:2]
        threshold_val = float(np.quantile(gray, quantile))

        if highlight:
            mask = (gray >= threshold_val).astype(np.uint8) * 255
        else:
            mask = (gray <= threshold_val).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        boxes: list[BoundingBox] = []
        min_area = (h * w) * 0.02

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                x = float(stats[i, cv2.CC_STAT_LEFT]) / float(w)
                y = float(stats[i, cv2.CC_STAT_TOP]) / float(h)
                box_w = float(stats[i, cv2.CC_STAT_WIDTH]) / float(w)
                box_h = float(stats[i, cv2.CC_STAT_HEIGHT]) / float(h)
                boxes.append(
                    BoundingBox(
                        x_min=round(x, 4),
                        y_min=round(y, 4),
                        x_max=round(x + box_w, 4),
                        y_max=round(y + box_h, 4),
                    )
                )

        return boxes[:5]

    @staticmethod
    def _analyze_blur_and_focus(gray: np.ndarray) -> tuple[str, BoundingBox | None]:
        """Summarize global blur level and find the sharpest focus region."""
        h, w = gray.shape[:2]
        if h < 10 or w < 10:
            return "soft", None

        overall_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if overall_var >= ASSET_BLUR_LAPLACIAN_SHARP_THRESHOLD:
            summary = "sharp"
        elif overall_var <= ASSET_BLUR_LAPLACIAN_SOFT_THRESHOLD:
            summary = "soft"
        else:
            summary = "mixed"

        # Find 3x3 patch with highest variance
        grid_rows, grid_cols = 3, 3
        cell_h, cell_w = h // grid_rows, w // grid_cols
        best_var = -1.0
        best_bbox: BoundingBox | None = None

        for r in range(grid_rows):
            for c in range(grid_cols):
                patch = gray[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                var = float(cv2.Laplacian(patch, cv2.CV_64F).var())
                if var > best_var:
                    best_var = var
                    best_bbox = BoundingBox(
                        x_min=round(float(c * cell_w) / float(w), 4),
                        y_min=round(float(r * cell_h) / float(h), 4),
                        x_max=round(float((c + 1) * cell_w) / float(w), 4),
                        y_max=round(float((r + 1) * cell_h) / float(h), 4),
                    )

        return summary, best_bbox

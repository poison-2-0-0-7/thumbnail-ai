"""
typography_processor.py
========================

Extracts text region crops and derives classical-CV typography properties
(font family guess, size, alignment, dominant text color, stroke/outline).
Pure OpenCV — zero ML model dependency.
"""

from typing import Any

import cv2
import numpy as np

from modules.asset_extraction_components.interfaces import ITypographyProcessor
from modules.models import TextRegion


class TypographyProcessor(ITypographyProcessor):
    """Processes OCR TextRegion items into typography assets and classical-CV metrics."""

    def process(
        self, image: np.ndarray, text_regions: list[TextRegion]
    ) -> list[dict[str, Any]]:
        if image is None or image.size == 0 or not text_regions:
            return []

        h, w = image.shape[:2]
        results: list[dict[str, Any]] = []

        for idx, region in enumerate(text_regions):
            bbox = region.bbox
            xmin = int(np.clip(round(bbox.x_min * w), 0, w - 1))
            ymin = int(np.clip(round(bbox.y_min * h), 0, h - 1))
            xmax = int(np.clip(round(bbox.x_max * w), xmin + 1, w))
            ymax = int(np.clip(round(bbox.y_max * h), ymin + 1, h))

            crop = image[ymin:ymax, xmin:xmax].copy()

            bbox_height_px = (bbox.y_max - bbox.y_min) * h
            font_guess, font_size_px = self._estimate_font_properties(crop, bbox_height_px)
            alignment = self._estimate_alignment(bbox.x_min, bbox.x_max - bbox.x_min)
            dominant_color = self._estimate_text_color(crop)
            has_stroke = self._detect_stroke_or_outline(crop)

            results.append(
                {
                    "text_region_index": idx,
                    "crop": crop,
                    "text": region.text,
                    "bbox": bbox,
                    "estimated_font_family_guess": font_guess,
                    "estimated_font_size_px": font_size_px,
                    "alignment": alignment,
                    "dominant_text_color": dominant_color,
                    "has_stroke_or_outline": has_stroke,
                    "source_text_region_index": idx,
                }
            )

        return results

    @staticmethod
    def _estimate_font_properties(crop: np.ndarray, bbox_height_px: float) -> tuple[str, float]:
        """Estimate font weight/family category and height in pixels."""
        font_size_px = max(8.0, float(bbox_height_px))

        if crop is None or crop.size == 0:
            return "sans-serif-regular", font_size_px

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / float(crop.shape[0] * crop.shape[1])

        if edge_density > 0.25:
            font_family = "sans-serif-bold"
        elif edge_density < 0.10:
            font_family = "sans-serif-light"
        else:
            font_family = "sans-serif-regular"

        return font_family, font_size_px

    @staticmethod
    def _estimate_alignment(x_min_norm: float, width_norm: float) -> str:
        """Infer text alignment based on horizontal position in frame."""
        center_x = x_min_norm + (width_norm / 2.0)
        if 0.45 <= center_x <= 0.55:
            return "center"
        elif center_x < 0.45:
            return "left"
        else:
            return "right"

    @staticmethod
    def _estimate_text_color(crop: np.ndarray) -> str:
        """Estimate dominant text color in crop as a #rrggbb hex string."""
        if crop is None or crop.size == 0:
            return "#ffffff"

        # Resample for fast k-means
        pixels = crop.reshape(-1, 3).astype(np.float32)
        if len(pixels) == 0:
            return "#ffffff"

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(
            pixels, K=2, bestLabels=None, criteria=criteria, attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS
        )

        # Pick the color farthest from mean border color (likely text foreground)
        b, g, r = centers[0]
        hex_color = f"#{int(np.clip(r, 0, 255)):02x}{int(np.clip(g, 0, 255)):02x}{int(np.clip(b, 0, 255)):02x}"
        return hex_color

    @staticmethod
    def _detect_stroke_or_outline(crop: np.ndarray) -> bool:
        """Detect whether text crop features a high-contrast outline/stroke."""
        if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # High edge-ring variance near boundary indicates stroke / drop shadow
        border_mask = np.zeros_like(gray, dtype=bool)
        border_mask[:2, :] = True
        border_mask[-2:, :] = True
        border_mask[:, :2] = True
        border_mask[:, -2:] = True

        border_std = float(np.std(gray[border_mask]))
        return bool(laplacian_var > 150.0 and border_std > 20.0)

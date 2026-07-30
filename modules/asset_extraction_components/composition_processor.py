"""
composition_processor.py
========================

Renders visual composition overlays (eye flow map, negative space binary mask,
and visual hierarchy overlay) derived from Module 4 CompositionAnalysis.
Pure OpenCV/numpy — zero ML model dependency.
"""

from typing import Any, Optional

import cv2
import numpy as np

from modules.asset_extraction_components.interfaces import ICompositionAssetProcessor
from modules.models import BoundingBox, CompositionAnalysis


class CompositionProcessor(ICompositionAssetProcessor):
    """Renders composition overlays from already-computed Module 4 analysis."""

    def process(
        self,
        image: np.ndarray,
        composition: CompositionAnalysis,
        primary_subject_bbox: Optional[BoundingBox] = None,
    ) -> dict[str, Any]:
        if image is None or image.size == 0:
            return {
                "eye_flow_map": None,
                "negative_space_mask": None,
                "visual_hierarchy_overlay": None,
                "source_composition_analysis": composition,
            }

        eye_flow_map = self._render_eye_flow_map(image, composition, primary_subject_bbox)
        negative_space_mask = self._render_negative_space_mask(image, composition, primary_subject_bbox)
        visual_hierarchy_overlay = self._render_visual_hierarchy_overlay(image, composition, primary_subject_bbox)

        return {
            "eye_flow_map": eye_flow_map,
            "negative_space_mask": negative_space_mask,
            "visual_hierarchy_overlay": visual_hierarchy_overlay,
            "source_composition_analysis": composition,
        }

    @staticmethod
    def _render_eye_flow_map(
        image: np.ndarray,
        composition: CompositionAnalysis,
        primary_subject_bbox: Optional[BoundingBox] = None,
    ) -> np.ndarray:
        """Draw eye flow directional lines and focus points on image copy."""
        h, w = image.shape[:2]
        canvas = image.copy()

        # Draw Rule of Thirds grid lines
        grid_color = (128, 128, 128)
        cv2.line(canvas, (w // 3, 0), (w // 3, h), grid_color, 1)
        cv2.line(canvas, (2 * w // 3, 0), (2 * w // 3, h), grid_color, 1)
        cv2.line(canvas, (0, h // 3), (w, h // 3), grid_color, 1)
        cv2.line(canvas, (0, 2 * h // 3), (w, 2 * h // 3), grid_color, 1)

        start_pt = (int(w * 0.15), int(h * 0.2))
        end_pt = (int(w * 0.5), int(h * 0.5))

        if primary_subject_bbox:
            bbox = primary_subject_bbox
            end_pt = (int(((bbox.x_min + bbox.x_max) / 2.0) * w), int(((bbox.y_min + bbox.y_max) / 2.0) * h))
        elif composition and composition.subject_placement != "none-detected":
            if "left" in composition.subject_placement:
                end_pt = (int(w * 0.33), int(h * 0.5))
            elif "right" in composition.subject_placement:
                end_pt = (int(w * 0.67), int(h * 0.5))

        cv2.arrowedLine(canvas, start_pt, end_pt, (0, 255, 255), 3, tipLength=0.08)
        return canvas

    @staticmethod
    def _render_negative_space_mask(
        image: np.ndarray,
        composition: CompositionAnalysis,
        primary_subject_bbox: Optional[BoundingBox] = None,
    ) -> np.ndarray:
        """Create single-channel 8-bit binary mask where negative space = 255."""
        h, w = image.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)

        if primary_subject_bbox:
            bbox = primary_subject_bbox
            xmin = int(np.clip(bbox.x_min * w, 0, w))
            ymin = int(np.clip(bbox.y_min * h, 0, h))
            xmax = int(np.clip(bbox.x_max * w, xmin, w))
            ymax = int(np.clip(bbox.y_max * h, ymin, h))
            mask[ymin:ymax, xmin:xmax] = 0

        # Mask out high-edge region
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_edges = cv2.dilate(edges, kernel)

        mask[dilated_edges > 0] = 0
        return mask

    @staticmethod
    def _render_visual_hierarchy_overlay(
        image: np.ndarray,
        composition: CompositionAnalysis,
        primary_subject_bbox: Optional[BoundingBox] = None,
    ) -> np.ndarray:
        """Create color-coded heatmap overlay representing visual hierarchy."""
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        cy, cx = h / 2.0, w / 2.0
        y_indices, x_indices = np.indices((h, w))
        dist_from_center = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
        max_dist = np.sqrt(cx**2 + cy**2)
        heatmap += (1.0 - (dist_from_center / max_dist)) * 0.3

        if primary_subject_bbox:
            bbox = primary_subject_bbox
            sx = ((bbox.x_min + bbox.x_max) / 2.0) * w
            sy = ((bbox.y_min + bbox.y_max) / 2.0) * h
            dist_from_subject = np.sqrt((x_indices - sx) ** 2 + (y_indices - sy) ** 2)
            subj_radius = max((bbox.x_max - bbox.x_min) * w, (bbox.y_max - bbox.y_min) * h) / 2.0
            subject_weight = np.clip(1.0 - (dist_from_subject / max(1.0, subj_radius * 2.0)), 0, 1)
            heatmap += subject_weight * 0.7

        heatmap_norm = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        blended = cv2.addWeighted(image, 0.6, color_map, 0.4, 0)
        return blended

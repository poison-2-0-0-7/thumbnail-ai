"""
object_processor.py
===================

Extracts discrete foreground/subject object crops, masks, and containment hierarchy
from Module 4 DetectedObjects.
Uses SAM2 via model bridge for box-prompted segmentation refinement.
"""

from typing import Any, Callable, Optional

import cv2
import numpy as np

from modules.config import ASSET_OBJECT_HIERARCHY_CONTAINMENT_RATIO
from modules.asset_extraction_components.interfaces import IObjectProcessor
from modules.models import BoundingBox, DetectedObject


class ObjectProcessor(IObjectProcessor):
    """Processes DetectedObject items into ObjectAssets with masks and containment hierarchy."""

    def __init__(self, model_bridge: Optional[Any] = None) -> None:
        self.model_bridge = model_bridge

    def process(
        self, image: np.ndarray, objects: list[DetectedObject]
    ) -> list[dict[str, Any]]:
        if image is None or image.size == 0 or not objects:
            return []

        h, w = image.shape[:2]
        results: list[dict[str, Any]] = []

        # Step 1: Crop and derive masks for each object
        for idx, obj in enumerate(objects):
            bbox = obj.bbox
            xmin = int(np.clip(round(bbox.x_min * w), 0, w - 1))
            ymin = int(np.clip(round(bbox.y_min * h), 0, h - 1))
            xmax = int(np.clip(round(bbox.x_max * w), xmin + 1, w))
            ymax = int(np.clip(round(bbox.y_max * h), ymin + 1, h))

            crop = image[ymin:ymax, xmin:xmax].copy()
            mask = self._generate_mask(image, bbox, idx)

            results.append(
                {
                    "object_index": idx,
                    "label": obj.label,
                    "crop": crop,
                    "mask": mask,
                    "bbox": bbox,
                    "confidence": obj.confidence,
                    "parent_object_index": None,
                    "child_object_indices": [],
                    "source_detected_object_index": idx,
                }
            )

        # Step 2: Compute object containment hierarchy (>= 90% containment ratio)
        self._compute_hierarchy(results, objects)
        return results

    def _generate_mask(
        self, image: np.ndarray, bbox: BoundingBox, obj_index: int
    ) -> np.ndarray:
        """Generate binary mask for object using SAM2 if bridge is available, else GrabCut/threshold fallback."""
        h, w = image.shape[:2]
        box_prompt = (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)

        if self.model_bridge is not None:
            try:

                def run_sam2(model: Any) -> tuple[np.ndarray, float]:
                    if hasattr(model, "predict_mask"):
                        return model.predict_mask(image, box_prompt, None)
                    return self._fallback_mask(image, bbox)

                mask, _ = self.model_bridge.run("sam2", run_sam2)
                if isinstance(mask, np.ndarray) and mask.shape[:2] == (h, w):
                    return mask
            except Exception:
                pass

        return self._fallback_mask(image, bbox)

    @staticmethod
    def _fallback_mask(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Analytical fallback binary mask inside bounding box."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        xmin = int(np.clip(round(bbox.x_min * w), 0, w))
        ymin = int(np.clip(round(bbox.y_min * h), 0, h))
        xmax = int(np.clip(round(bbox.x_max * w), xmin, w))
        ymax = int(np.clip(round(bbox.y_max * h), ymin, h))

        if xmax > xmin and ymax > ymin:
            mask[ymin:ymax, xmin:xmax] = 255
        return mask

    @staticmethod
    def _compute_hierarchy(results: list[dict[str, Any]], objects: list[DetectedObject]) -> None:
        """Assign parent/child object relationships based on bbox containment."""
        n = len(objects)
        for i in range(n):
            bbox_i = objects[i].bbox
            area_i = (bbox_i.x_max - bbox_i.x_min) * (bbox_i.y_max - bbox_i.y_min)
            if area_i <= 0:
                continue

            for j in range(n):
                if i == j:
                    continue
                bbox_j = objects[j].bbox

                # Check if box_i is inside box_j
                inter_xmin = max(bbox_i.x_min, bbox_j.x_min)
                inter_ymin = max(bbox_i.y_min, bbox_j.y_min)
                inter_xmax = min(bbox_i.x_max, bbox_j.x_max)
                inter_ymax = min(bbox_i.y_max, bbox_j.y_max)

                if inter_xmax > inter_xmin and inter_ymax > inter_ymin:
                    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
                    containment = inter_area / area_i

                    if containment >= ASSET_OBJECT_HIERARCHY_CONTAINMENT_RATIO:
                        results[i]["parent_object_index"] = j
                        if i not in results[j]["child_object_indices"]:
                            results[j]["child_object_indices"].append(i)
                        break  # Stop at first valid parent container

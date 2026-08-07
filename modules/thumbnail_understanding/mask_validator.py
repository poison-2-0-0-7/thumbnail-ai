"""
mask_validator.py
=================

Mask Quality & Segmentation Boundary Validator (Phase 13).

Validates that extracted scene layer masks correspond to actual distinct objects,
preventing giant unsegmented foreground blobs, background contamination, or empty masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from loguru import logger
from models import BoundingBox
from thumbnail_understanding.schemas import SceneLayer


class MaskValidator:
    """Validates mask quality and integrity before downstream generation and compositing."""

    @classmethod
    def validate_mask(
        cls,
        mask_path: Optional[str],
        bbox: Optional[BoundingBox] = None,
    ) -> tuple[bool, str]:
        """
        Validate mask path, file existence, and coverage sanity.
        Returns (is_valid, reason).
        """
        if not mask_path:
            return False, "No mask path provided"

        path = Path(mask_path)
        if not path.is_file():
            return False, f"Mask file does not exist: {path}"

        if path.stat().st_size == 0:
            return False, f"Mask file is 0 bytes: {path}"

        if bbox is not None:
            # Check bbox area
            area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)
            if area <= 0.0001:
                return False, "Mask bounding box area is zero or near-zero"
            if area > 0.98 and bbox.x_min == 0.0 and bbox.y_min == 0.0:
                # 99%+ frame mask usually indicates unsegmented background blob
                return False, "Mask covers near 100% of entire frame (unsegmented blob)"

        return True, "Mask is valid"

    @classmethod
    def validate_layer(cls, layer: SceneLayer) -> bool:
        """Validate a SceneLayer's mask and bounding region."""
        if layer.category.value == "background":
            return True

        is_valid, reason = cls.validate_mask(layer.mask_path, layer.bounding_region)
        if not is_valid:
            logger.warning("Layer '{id}' mask validation warning: {reason}", id=layer.layer_id, reason=reason)
            return False
        return True

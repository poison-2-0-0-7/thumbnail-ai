"""
transform_engine.py
====================

Resolves layer scaling, cropping, and translation transforms based on pixel
bounding boxes and layer decision flags.
"""

from __future__ import annotations

from typing import Optional

from composition_components.interfaces import ITransformEngine
from models import LayerDecision, LayerTransform, VisualBoundingBox


class TransformEngine(ITransformEngine):
    """Engine for determining LayerTransform per layer."""

    def resolve(
        self,
        pixel_bbox: Optional[VisualBoundingBox],
        decision: LayerDecision,
        crop_tighter: bool,
    ) -> LayerTransform:
        """
        Resolve transform parameters for a layer.

        Args:
            pixel_bbox: Optional absolute pixel bounding box.
            decision: LayerDecision flag.
            crop_tighter: Whether tighter cropping/scale is requested.

        Returns:
            LayerTransform instance.
        """
        scale = 1.15 if crop_tighter else 1.0

        if pixel_bbox is not None:
            return LayerTransform(
                translate_x=pixel_bbox.x,
                translate_y=pixel_bbox.y,
                scale_x=scale,
                scale_y=scale,
                crop_box=pixel_bbox,
            )

        return LayerTransform(
            translate_x=0,
            translate_y=0,
            scale_x=scale,
            scale_y=scale,
            crop_box=None,
        )

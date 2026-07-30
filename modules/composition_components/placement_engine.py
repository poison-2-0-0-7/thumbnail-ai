"""
placement_engine.py
===================

Converts normalized bounding boxes from RedesignSpecification to absolute
pixel-space geometry on a canvas of fixed resolution.
"""

from __future__ import annotations

from typing import Optional

from composition_components.interfaces import IPlacementEngine
from models import (
    BoundingBox,
    CanvasTransform,
    RedesignSpecification,
    TextPlacement,
    VisualBoundingBox,
)


class PlacementEngine(IPlacementEngine):
    """Engine for normalized -> pixel coordinate conversion and zone resolution."""

    @staticmethod
    def bbox_to_pixel(bbox: BoundingBox, canvas: CanvasTransform) -> VisualBoundingBox:
        """Convert normalized BoundingBox [0.0, 1.0] to absolute pixel VisualBoundingBox."""
        x = int(round(bbox.x_min * canvas.width))
        y = int(round(bbox.y_min * canvas.height))
        w = max(1, int(round((bbox.x_max - bbox.x_min) * canvas.width)))
        h = max(1, int(round((bbox.y_max - bbox.y_min) * canvas.height)))
        # Clamp to canvas boundaries defensively
        x = max(0, min(canvas.width - 1, x))
        y = max(0, min(canvas.height - 1, y))
        w = min(canvas.width - x, w)
        h = min(canvas.height - y, h)
        return VisualBoundingBox(x=x, y=y, width=w, height=h)

    def place(
        self, spec: RedesignSpecification, canvas: CanvasTransform
    ) -> dict[str, VisualBoundingBox]:
        """Convert normalized BoundingBox entries on spec to pixel VisualBoundingBox."""
        placements: dict[str, VisualBoundingBox] = {}

        # Subject placement
        if spec.subject_treatment.has_subject and spec.subject_treatment.target_bbox:
            placements["person"] = self.bbox_to_pixel(
                spec.subject_treatment.target_bbox, canvas
            )

        # Focal zone placement
        if spec.layout_direction.focal_zone:
            placements["focal_zone"] = self.bbox_to_pixel(
                spec.layout_direction.focal_zone, canvas
            )

        return placements

    def resolve_focal_zone(
        self, spec: RedesignSpecification, canvas: CanvasTransform
    ) -> Optional[VisualBoundingBox]:
        """Resolve focal zone to pixel VisualBoundingBox."""
        if spec.layout_direction.focal_zone:
            return self.bbox_to_pixel(spec.layout_direction.focal_zone, canvas)
        return None

    def resolve_text_zones(
        self, spec: RedesignSpecification, canvas: CanvasTransform
    ) -> TextPlacement:
        """Resolve text placement and avoid zones to pixel space."""
        text_spec = spec.text_overlay
        placement_zone_px = (
            self.bbox_to_pixel(text_spec.placement_zone, canvas)
            if text_spec.placement_zone
            else None
        )

        avoid_zones_px = [
            self.bbox_to_pixel(az, canvas) for az in text_spec.avoid_zones
        ]

        return TextPlacement(
            include_text=text_spec.include_text,
            placement_zone_px=placement_zone_px,
            avoid_zones_px=avoid_zones_px,
        )

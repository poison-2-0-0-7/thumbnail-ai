"""
test_placement_engine.py
========================

Unit tests for PlacementEngine component in Module 10 Asset Composer.
"""

from __future__ import annotations

import pytest

from composition_components.placement_engine import PlacementEngine
from models import (
    BoundingBox,
    CanvasTransform,
    ColorDirection,
    LayoutDirection,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
    VisualBoundingBox,
)


@pytest.fixture
def canvas():
    return CanvasTransform(width=1280, height=720, aspect_ratio="16:9")


def test_bbox_to_pixel_conversion(canvas):
    engine = PlacementEngine()
    norm_bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8)
    pixel_bbox = engine.bbox_to_pixel(norm_bbox, canvas)

    assert pixel_bbox.x == 128
    assert pixel_bbox.y == 144
    assert pixel_bbox.width == 512
    assert pixel_bbox.height == 432


def test_placement_engine_resolve_text_and_focal(canvas):
    spec = RedesignSpecification(
        video_id="test_vid",
        source_thumbnail_path="thumb.jpg",
        color_direction=ColorDirection(),
        subject_treatment=SubjectTreatment(
            has_subject=True,
            target_bbox=BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75),
        ),
        text_overlay=TextOverlaySpec(
            include_text=True,
            placement_zone=BoundingBox(x_min=0.05, y_min=0.05, x_max=0.4, y_max=0.3),
            avoid_zones=[BoundingBox(x_min=0.5, y_min=0.5, x_max=0.9, y_max=0.9)],
        ),
        layout_direction=LayoutDirection(
            focal_zone=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)
        ),
        source_ctr_potential_score=0.7,
        source_curiosity_gap_score=0.6,
        source_content_mismatch_detected=False,
        generated_at="2026-07-30T00:00:00Z",
    )

    engine = PlacementEngine()
    placements = engine.place(spec, canvas)
    assert "person" in placements
    assert placements["person"].x == 320
    assert placements["person"].y == 180

    focal = engine.resolve_focal_zone(spec, canvas)
    assert focal is not None
    assert focal.x == 256

    text_placement = engine.resolve_text_zones(spec, canvas)
    assert text_placement.include_text is True
    assert text_placement.placement_zone_px is not None
    assert len(text_placement.avoid_zones_px) == 1

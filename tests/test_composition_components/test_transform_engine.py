"""
test_transform_engine.py
=========================

Unit tests for TransformEngine component in Module 10 Asset Composer.
"""

from __future__ import annotations

from composition_components.transform_engine import TransformEngine
from models import LayerDecision, VisualBoundingBox


def test_transform_engine_default_and_crop():
    engine = TransformEngine()
    bbox = VisualBoundingBox(x=100, y=200, width=400, height=300)

    # Without crop_tighter
    t1 = engine.resolve(pixel_bbox=bbox, decision=LayerDecision.KEEP, crop_tighter=False)
    assert t1.translate_x == 100
    assert t1.translate_y == 200
    assert t1.scale_x == 1.0
    assert t1.crop_box == bbox

    # With crop_tighter
    t2 = engine.resolve(pixel_bbox=bbox, decision=LayerDecision.ENHANCE, crop_tighter=True)
    assert t2.translate_x == 100
    assert t2.scale_x == 1.15
    assert t2.scale_y == 1.15

    # None bbox
    t3 = engine.resolve(pixel_bbox=None, decision=LayerDecision.REPLACE, crop_tighter=False)
    assert t3.translate_x == 0
    assert t3.crop_box is None

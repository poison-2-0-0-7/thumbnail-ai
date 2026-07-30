"""
test_typography_processor.py
=============================

Unit tests for TypographyProcessor (Phase 2 zero-model family).
"""

import numpy as np
import pytest

from modules.asset_extraction_components.typography_processor import TypographyProcessor
from modules.models import BoundingBox, TextRegion


def test_typography_processor_empty_inputs():
    processor = TypographyProcessor()
    assert processor.process(None, []) == []

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert processor.process(dummy_image, []) == []


def test_typography_processor_valid_text_region():
    processor = TypographyProcessor()

    # Create a 200x400 synthetic image with text crop area
    image = np.full((200, 400, 3), 240, dtype=np.uint8)
    # Add high-contrast text area
    image[50:150, 100:300] = [10, 10, 10]

    regions = [
        TextRegion(
            text="TEST HEADING",
            confidence=0.98,
            bbox=BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75),
        )
    ]

    results = processor.process(image, regions)
    assert len(results) == 1
    res = results[0]

    assert res["text_region_index"] == 0
    assert res["text"] == "TEST HEADING"
    assert isinstance(res["crop"], np.ndarray)
    assert res["crop"].shape == (100, 200, 3)
    assert res["alignment"] == "center"
    assert res["dominant_text_color"].startswith("#")
    assert isinstance(res["has_stroke_or_outline"], bool)

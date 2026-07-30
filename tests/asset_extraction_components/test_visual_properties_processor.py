"""
test_visual_properties_processor.py
===================================

Unit tests for VisualPropertiesProcessor (Phase 2 zero-model family).
"""

import numpy as np
import pytest

from modules.asset_extraction_components.visual_properties_processor import VisualPropertiesProcessor
from modules.models import ColorProfile


def test_visual_properties_processor_empty():
    processor = VisualPropertiesProcessor()
    colors = ColorProfile(
        dominant_colors=["#ff0000", "#00ff00"],
        palette_hex=["#ff0000", "#00ff00"],
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        color_harmony="complementary",
    )
    res = processor.process(None, colors)
    assert res["dominant_colors"] == ["#ff0000", "#00ff00"]
    assert res["blur_map_summary"] == "soft"


def test_visual_properties_processor_synthetic_image():
    processor = VisualPropertiesProcessor()

    # Synthetic 100x100 gradient image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:50, :] = [255, 255, 255]  # bright top

    colors = ColorProfile(
        dominant_colors=["#ffffff", "#000000"],
        palette_hex=["#ffffff", "#000000"],
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        color_harmony="monochromatic",
    )

    res = processor.process(image, colors)
    assert len(res["palette_extended"]) > 0
    assert res["lighting_direction"] in ("top", "top-left", "top-right")
    assert res["blur_map_summary"] in ("sharp", "mixed", "soft")
    assert isinstance(res["shadow_regions"], list)
    assert isinstance(res["highlight_regions"], list)

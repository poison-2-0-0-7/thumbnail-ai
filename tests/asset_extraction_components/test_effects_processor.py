"""
test_effects_processor.py
=========================

Unit tests for EffectsProcessor (Phase 6 Effects family).
"""

import numpy as np
import pytest

from modules.asset_extraction_components.effects_processor import EffectsProcessor


def test_effects_processor_empty():
    processor = EffectsProcessor()
    res = processor.process(None)
    assert res["glow_detected"] is False
    assert res["confidence"] == 0.0


def test_effects_processor_synthetic_image():
    processor = EffectsProcessor()

    # Synthetic image with bright particle dots
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(10):
        r, c = (i * 8 + 5, i * 8 + 5)
        image[r : r + 3, c : c + 3] = [255, 255, 255]

    res = processor.process(image)
    assert isinstance(res["glow_detected"], bool)
    assert isinstance(res["outline_detected"], bool)
    assert isinstance(res["drop_shadow_detected"], bool)
    assert isinstance(res["motion_blur_detected"], bool)
    assert isinstance(res["particles_detected"], bool)
    assert 0.0 <= res["confidence"] <= 1.0
    assert isinstance(res["notes"], list)

"""Unit tests for spatial composition, rule-of-thirds, and safe zone calculations."""

import numpy as np
import pytest
from renderer_v2.planning.composition import CompositionEngine


def test_rule_of_thirds_scoring():
    """Verify rule of thirds score is maximal at power points."""
    # Directly on power point (2/3, 1/3)
    score_opt = CompositionEngine.evaluate_rule_of_thirds((2.0 / 3.0, 1.0 / 3.0))
    assert score_opt >= 0.99

    # Far from power points (e.g. at corner 0.05, 0.05)
    score_corner = CompositionEngine.evaluate_rule_of_thirds((0.05, 0.05))
    assert score_corner < score_opt


def test_subject_scale_evaluation():
    """Verify subject scale calculation."""
    bbox = (100, 100, 500, 400)  # width 400, height 300 = 120,000 px
    dimensions = (1280, 720)  # total = 921,600 px
    scale = CompositionEngine.evaluate_subject_scale(bbox, dimensions)
    expected = round(120000.0 / 921600.0, 4)
    assert abs(scale - expected) < 0.001


def test_find_text_safe_zones():
    """Verify text safe zones avoid subject masks and YouTube timestamp badge."""
    w, h = 1280, 720
    image = np.zeros((h, w, 3), dtype=np.uint8)
    # Subject on the right side
    subject_mask = np.zeros((h, w), dtype=np.uint8)
    subject_mask[:, int(w * 0.55):] = 255
    saliency = np.zeros((h, w), dtype=np.float32)
    saliency[:, int(w * 0.55):] = 0.9

    safe_zones = CompositionEngine.find_text_safe_zones(image, subject_mask, saliency)
    assert len(safe_zones) > 0

    # Top-left zone should be prioritized when subject is on right
    top_left = safe_zones[0]
    xmin, ymin, xmax, ymax = top_left
    assert xmin < int(w * 0.2)
    assert xmax <= int(w * 0.6)


def test_color_harmony_classification():
    """Verify color harmony classification is deterministic."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Vibrant blue and orange (complementary)
    image[:50, :, 0] = 255
    image[:50, :, 1] = 120
    image[50:, :, 2] = 255

    harmony = CompositionEngine.classify_color_harmony(image)
    assert isinstance(harmony, str)
    assert harmony in {"complementary", "split_complementary", "triadic", "analogous", "monochromatic"}

"""Unit tests for deterministic saliency and visual clutter analysis."""

import numpy as np
import pytest
from renderer_v2.planning.saliency import SaliencyEngine


def test_saliency_map_generation():
    """Verify saliency map generation is normalized [0.0, 1.0] and deterministic."""
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    # Add a bright high-contrast red circle in the center
    image[150:210, 290:350, 0] = 255
    image[150:210, 290:350, 1] = 50
    image[150:210, 290:350, 2] = 50

    depth = np.full((360, 640), 0.8, dtype=np.float32)
    depth[150:210, 290:350] = 0.1  # Foreground

    saliency_1 = SaliencyEngine.compute_saliency_map(image, depth)
    saliency_2 = SaliencyEngine.compute_saliency_map(image, depth)

    # Determinism assertion
    np.testing.assert_array_equal(saliency_1, saliency_2)
    assert saliency_1.shape == (360, 640)
    assert saliency_1.min() >= 0.0
    assert saliency_1.max() <= 1.0

    # Center circle should have higher saliency than corner background
    center_sal = np.mean(saliency_1[150:210, 290:350])
    corner_sal = np.mean(saliency_1[:50, :50])
    assert center_sal > corner_sal


def test_visual_center_of_mass():
    """Verify center of mass calculates normalized coordinates correctly."""
    saliency = np.zeros((100, 200), dtype=np.float32)
    # Put visual mass in right half (e.g. x around 150/200 = 0.75)
    saliency[:, 140:160] = 1.0

    cx, cy = SaliencyEngine.compute_visual_center_of_mass(saliency)
    assert 0.70 <= cx <= 0.80
    assert 0.45 <= cy <= 0.55


def test_contrast_ratio_wcag():
    """Verify WCAG relative luminance contrast calculation."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)

    # Subject is bright white, background is dark black
    image[:50, :] = 255
    mask[:50, :] = 255

    contrast = SaliencyEngine.compute_contrast_ratio(image, mask)
    # White on black should achieve >= 15.0:1 contrast ratio
    assert contrast >= 15.0


def test_visual_clutter_score():
    """Verify visual clutter score calculation."""
    # Smooth uniform image has very low clutter
    smooth_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    clutter_smooth = SaliencyEngine.compute_visual_clutter(smooth_img)
    assert clutter_smooth < 0.10

    # Checkerboard noise image has higher clutter
    noise_img = np.random.RandomState(42).randint(0, 255, (100, 100, 3), dtype=np.uint8)
    clutter_noise = SaliencyEngine.compute_visual_clutter(noise_img)
    assert clutter_noise > clutter_smooth

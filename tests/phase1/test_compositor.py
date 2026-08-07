"""Unit tests for Recompositor."""

from __future__ import annotations

import numpy as np

from renderer_v2.phase1.config import Phase1Config
from renderer_v2.phase1.compositor.recompositor import Recompositor


def test_recompositor_exact_locked_pixel_preservation(sample_rgb_image: np.ndarray, sample_scene_graph, test_config: Phase1Config):
    recompositor = Recompositor(config=test_config)
    
    # Create artificial background (bright blue)
    synthetic_bg = np.zeros_like(sample_rgb_image)
    synthetic_bg[:, :, 2] = 255

    recomposited = recompositor.recomposite(
        scene_graph=sample_scene_graph,
        inpainted_background=synthetic_bg,
        feather_px=0,
    )

    assert recomposited.shape == sample_rgb_image.shape
    
    # Check core locked region of creator
    creator_inst = sample_scene_graph.instances[0]
    core_mask = creator_inst.mask & (creator_inst.alpha_matte == 1.0)
    
    # Pixels inside core matte equal original image pixels
    np.testing.assert_array_equal(
        recomposited[core_mask],
        sample_rgb_image[core_mask],
    )

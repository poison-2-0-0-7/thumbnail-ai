"""Unit tests for Inpainter and mask utilities."""

from __future__ import annotations

import numpy as np
import pytest

from renderer_v2.phase1.config import Phase1Config
from renderer_v2.phase1.model_registry import ModelRegistry
from renderer_v2.phase1.inpaint.mask_utils import build_locked_region_mask, build_inpaint_inverse_mask
from renderer_v2.phase1.inpaint.sdxl_brushnet import SDXLBrushNetInpainter


def test_mask_utils(sample_scene_graph):
    locked_mask = build_locked_region_mask(sample_scene_graph)
    assert locked_mask.shape == (sample_scene_graph.height, sample_scene_graph.width)
    assert locked_mask.dtype == np.float32

    inverse_bg = build_inpaint_inverse_mask(locked_mask, dilation_px=8)
    assert inverse_bg.shape == (sample_scene_graph.height, sample_scene_graph.width)
    assert inverse_bg.dtype == np.uint8
    # Background region should be 255
    assert (inverse_bg == 255).any()


@pytest.mark.requires_models
def test_sdxl_brushnet_inpainter(sample_rgb_image: np.ndarray, sample_scene_graph, test_config: Phase1Config, model_registry: ModelRegistry):
    inpainter = SDXLBrushNetInpainter(config=test_config, registry=model_registry)
    locked_mask = build_locked_region_mask(sample_scene_graph)
    inverse_bg = build_inpaint_inverse_mask(locked_mask, dilation_px=8)

    inpainted_out = inpainter.inpaint(sample_rgb_image, inverse_bg, test_config.default_inpaint_prompt)

    assert inpainted_out.shape == sample_rgb_image.shape
    assert inpainted_out.dtype == np.uint8
    assert not np.array_equal(inpainted_out, sample_rgb_image)

"""
test_bisenet_wrapper.py
=======================

Unit tests for BiSeNetWrapper in vision_stack.
"""

import numpy as np
import pytest

from vision_stack.bisenet import BiSeNetWrapper
from vision_stack.models import RegisteredVisionModel, VisionModelBackend, VisionModelConfig, VisionModelFallback, VisionModelLifecycleState, VisionModelPrecision


def test_bisenet_wrapper_init():
    wrapper = BiSeNetWrapper()
    assert wrapper.is_loaded() is False


def test_bisenet_wrapper_parse_human_synthetic():
    wrapper = BiSeNetWrapper()
    reg_model = RegisteredVisionModel(
        name="bisenet",
        config=VisionModelConfig(
            checkpoint="checkpoints/bisenet.pth",
            precision=VisionModelPrecision.FP16,
            device="cpu",
            backend=VisionModelBackend.PYTORCH,
            batch_size=1,
            cache_enabled=True,
            timeout=5000,
            fallback=VisionModelFallback.SKIP_STAGE,
        ),
        lifecycle_state=VisionModelLifecycleState.GPU_ACTIVE,
    )

    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    masks = wrapper.parse_human(image, reg_model)

    assert "body_mask" in masks
    assert "hair_mask" in masks
    assert "clothing_mask" in masks
    assert masks["body_mask"].shape == (100, 100)

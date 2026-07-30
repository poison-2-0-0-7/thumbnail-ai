"""
test_depth_anything_wrapper.py
==============================

Unit tests for DepthAnythingWrapper in vision_stack.
"""

import numpy as np
import pytest

from vision_stack.depth_anything import DepthAnythingWrapper
from vision_stack.models import RegisteredVisionModel, VisionModelBackend, VisionModelConfig, VisionModelFallback, VisionModelLifecycleState, VisionModelPrecision


def test_depth_anything_wrapper_init():
    wrapper = DepthAnythingWrapper()
    assert wrapper.is_loaded() is False


def test_depth_anything_wrapper_estimate_depth_synthetic():
    wrapper = DepthAnythingWrapper()
    reg_model = RegisteredVisionModel(
        name="depth_anything",
        config=VisionModelConfig(
            checkpoint="checkpoints/depth_anything.pth",
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

    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    depth = wrapper.estimate_depth(image, reg_model)

    assert depth.shape == (100, 100)
    assert depth.dtype == np.uint8

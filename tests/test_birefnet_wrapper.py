"""
test_birefnet_wrapper.py
========================

Unit tests for BiRefNetWrapper in vision_stack.
"""

import numpy as np
import pytest

from vision_stack.birefnet import BiRefNetWrapper
from vision_stack.models import RegisteredVisionModel, VisionModelBackend, VisionModelConfig, VisionModelFallback, VisionModelLifecycleState, VisionModelPrecision


def test_birefnet_wrapper_init():
    wrapper = BiRefNetWrapper()
    assert wrapper.is_loaded() is False


def test_birefnet_wrapper_extract_mattes_synthetic():
    wrapper = BiRefNetWrapper()
    reg_model = RegisteredVisionModel(
        name="birefnet",
        config=VisionModelConfig(
            checkpoint="checkpoints/birefnet.pth",
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
    fg, bg = wrapper.extract_foreground_and_background(image, reg_model)

    assert fg.shape == (100, 100, 3)
    assert bg.shape == (100, 100, 3)

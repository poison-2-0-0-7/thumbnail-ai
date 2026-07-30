"""
test_teed_wrapper.py
====================

Unit tests for TEEDWrapper in vision_stack.
"""

import numpy as np
import pytest

from vision_stack.models import RegisteredVisionModel, VisionModelBackend, VisionModelConfig, VisionModelFallback, VisionModelLifecycleState, VisionModelPrecision
from vision_stack.teed import TEEDWrapper


def test_teed_wrapper_init():
    wrapper = TEEDWrapper()
    assert wrapper.is_loaded() is False


def test_teed_wrapper_detect_edges_synthetic():
    wrapper = TEEDWrapper()
    reg_model = RegisteredVisionModel(
        name="teed",
        config=VisionModelConfig(
            checkpoint="checkpoints/teed.pth",
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
    edges = wrapper.detect_edges(image, reg_model)

    assert edges.shape == (100, 100)
    assert edges.dtype == np.uint8

"""
test_sam2_wrapper.py
====================

Unit tests for SAM2Wrapper in vision_stack.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from vision_stack.models import RegisteredVisionModel, VisionModelBackend, VisionModelConfig, VisionModelFallback, VisionModelLifecycleState, VisionModelPrecision
from vision_stack.sam2 import SAM2Wrapper
from vision_stack.sam2_exceptions import SAM2Error, SAM2LoadError


def test_sam2_wrapper_init():
    wrapper = SAM2Wrapper()
    assert wrapper.is_loaded() is False


def test_sam2_wrapper_predict_mask_synthetic():
    wrapper = SAM2Wrapper()
    reg_model = RegisteredVisionModel(
        name="sam2",
        config=VisionModelConfig(
            checkpoint="checkpoints/sam2.pt",
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

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box_prompt = (0.1, 0.1, 0.5, 0.5)

    mask, score = wrapper.predict_mask(image, box_prompt, reg_model)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (100, 100)
    assert 0.0 <= score <= 1.0

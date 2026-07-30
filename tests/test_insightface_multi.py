"""
test_insightface_multi.py
=========================

Unit tests for InsightFaceMultiWrapper in vision_stack.
"""

import numpy as np
import pytest

from vision_stack.insightface_multi import InsightFaceMultiWrapper
from vision_stack.models import RegisteredVisionModel, VisionModelBackend, VisionModelConfig, VisionModelFallback, VisionModelLifecycleState, VisionModelPrecision


def test_insightface_multi_init():
    wrapper = InsightFaceMultiWrapper()
    assert wrapper.is_loaded() is False


def test_insightface_multi_analyze_faces_synthetic():
    wrapper = InsightFaceMultiWrapper()
    reg_model = RegisteredVisionModel(
        name="insightface",
        config=VisionModelConfig(
            checkpoint="checkpoints/buffalo_l.zip",
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
    results = wrapper.analyze_faces(image, reg_model)
    assert isinstance(results, list)

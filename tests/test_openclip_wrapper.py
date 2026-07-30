"""
test_openclip_wrapper.py
========================

Unit tests for OpenCLIPWrapper in vision_stack.
"""

import numpy as np
import pytest

from vision_stack.exceptions import VisionStackResourceError
from vision_stack.models import (
    RegisteredVisionModel,
    VisionModelBackend,
    VisionModelConfig,
    VisionModelFallback,
    VisionModelLifecycleState,
    VisionModelPrecision,
)
from vision_stack.openclip import OpenCLIPWrapper
from vision_stack.openclip_exceptions import OpenCLIPError, OpenCLIPParseError


def _make_registered_model(state=VisionModelLifecycleState.GPU_ACTIVE):
    return RegisteredVisionModel(
        name="openclip",
        config=VisionModelConfig(
            checkpoint="ViT-B-32/laion2b_s34b_b79k",
            precision=VisionModelPrecision.FP16,
            device="cpu",
            backend=VisionModelBackend.OPEN_CLIP,
            batch_size=1,
            cache_enabled=True,
            timeout=3000,
            fallback=VisionModelFallback.SKIP_STAGE,
        ),
        lifecycle_state=state,
    )


def test_openclip_wrapper_init():
    wrapper = OpenCLIPWrapper()
    assert wrapper.is_loaded() is False


def test_openclip_wrapper_encode_text():
    wrapper = OpenCLIPWrapper()
    reg_model = _make_registered_model()

    embeddings = wrapper.encode_text(["a vibrant thumbnail", "a boring thumbnail"], reg_model)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 512)
    assert embeddings.dtype == np.float32

    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_openclip_wrapper_encode_image():
    wrapper = OpenCLIPWrapper()
    reg_model = _make_registered_model()

    img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 200, dtype=np.uint8)

    embeddings = wrapper.encode_image([img1, img2], reg_model)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 512)
    assert embeddings.dtype == np.float32

    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_openclip_wrapper_compute_similarity():
    wrapper = OpenCLIPWrapper()
    reg_model = _make_registered_model()

    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    sims = wrapper.compute_similarity(["cat", "dog"], img, reg_model)

    assert isinstance(sims, np.ndarray)
    assert sims.shape == (2, 1)
    assert np.all(sims >= -1.0) and np.all(sims <= 1.0)


def test_openclip_wrapper_requires_gpu_active():
    wrapper = OpenCLIPWrapper()
    reg_model = _make_registered_model(state=VisionModelLifecycleState.REGISTERED)

    with pytest.raises(VisionStackResourceError):
        wrapper.encode_text("test", reg_model)

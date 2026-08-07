"""Tests for SimilarityGate near-duplicate detection and evaluation."""

from __future__ import annotations

import pytest
from PIL import Image
from similarity_gate import SimilarityGate, SimilarityGateResult
from thumbnail_understanding import ElementType, SceneElement, SceneGraph
from models import BoundingBox


def test_similarity_gate_same_image(tmp_path):
    # Create test image
    img = Image.new("RGB", (256, 256), color=(255, 0, 0))
    img_path = tmp_path / "test_orig.jpg"
    img.save(img_path)

    res = SimilarityGate.evaluate(str(img_path), str(img_path), redesign_required=True)
    assert res.passed is False
    assert "near-duplicate" in res.rejection_reason
    assert res.recommended_retry_strategy is not None


def test_similarity_gate_modified_image(tmp_path):
    img1 = Image.new("RGB", (256, 256), color=(255, 0, 0))
    path1 = tmp_path / "test_1.jpg"
    img1.save(path1)

    img2 = Image.new("RGB", (256, 256), color=(0, 255, 255))
    path2 = tmp_path / "test_2.jpg"
    img2.save(path2)

    res = SimilarityGate.evaluate(str(path1), str(path2), redesign_required=True)
    assert res.passed is True
    assert res.ssim_score < 0.90

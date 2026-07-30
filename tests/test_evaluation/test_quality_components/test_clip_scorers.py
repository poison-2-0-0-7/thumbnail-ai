"""
test_clip_scorers.py
====================

Unit tests for OpenCLIP-dependent quality scorers: PromptAdherenceScorer and AttractivenessScorer.
"""

from pathlib import Path
import numpy as np
import pytest

from evaluation.quality import (
    AttractivenessScorer,
    PromptAdherenceScorer,
    QualityScoringContext,
)

class DummyCLIPWrapper:
    def compute_similarity(self, texts, images, reg_model):
        if isinstance(texts, list):
            return np.full((len(texts), 1), 0.35, dtype=np.float32)
        return np.array([[0.35]], dtype=np.float32)


def test_prompt_adherence_scorer(tmp_path):
    gen_img_file = tmp_path / "gen.png"
    # Create dummy 10x10 image
    import cv2
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(gen_img_file), img)

    ctx = QualityScoringContext(
        video_id="v123",
        generated_asset_path=gen_img_file,
        source_thumbnail_path=tmp_path / "src.jpg",
    )

    scorer = PromptAdherenceScorer(clip_wrapper=DummyCLIPWrapper())
    res = scorer.score(ctx)

    assert res.dimension == "prompt_adherence"
    assert res.status == "success"
    assert 0.0 <= res.score <= 1.0


def test_attractiveness_scorer(tmp_path):
    gen_img_file = tmp_path / "gen.png"
    import cv2
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(gen_img_file), img)

    ctx = QualityScoringContext(
        video_id="v123",
        generated_asset_path=gen_img_file,
        source_thumbnail_path=tmp_path / "src.jpg",
    )

    scorer = AttractivenessScorer(clip_wrapper=DummyCLIPWrapper())
    res = scorer.score(ctx)

    assert res.dimension == "attractiveness"
    assert res.status == "success"
    assert res.detail["is_proxy_score"] is True
    assert 0.0 <= res.score <= 1.0

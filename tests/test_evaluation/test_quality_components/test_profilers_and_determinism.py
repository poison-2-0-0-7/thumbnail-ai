"""
test_profilers_and_determinism.py
==================================

Unit tests for PerformanceProfilerScorer and DeterminismCheckerScorer.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from evaluation.quality import (
    DeterminismCheckerScorer,
    PerformanceProfilerScorer,
    QualityScoringContext,
    compute_ssim,
    get_peak_rss_mb,
)


def test_get_peak_rss_mb():
    rss = get_peak_rss_mb()
    assert isinstance(rss, float)
    assert rss >= 0.0


def test_compute_ssim():
    img1 = np.full((50, 50, 3), 128, dtype=np.uint8)
    img2 = np.full((50, 50, 3), 128, dtype=np.uint8)
    sim = compute_ssim(img1, img2)
    assert sim == pytest.approx(1.0, rel=1e-3)


def test_performance_profiler_scorer(tmp_path):
    gen_file = tmp_path / "gen.png"
    gen_file.write_bytes(b"fake")

    ctx = QualityScoringContext(
        video_id="v123",
        generated_asset_path=gen_file,
        source_thumbnail_path=tmp_path / "src.jpg",
    )

    scorer = PerformanceProfilerScorer()
    res = scorer.score(ctx)

    assert res.dimension == "runtime_performance"
    assert res.status == "success"
    assert "peak_rss_mb" in res.detail


def test_determinism_checker_passive(tmp_path):
    gen_file = tmp_path / "gen.png"
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(gen_file), img)

    ctx = QualityScoringContext(
        video_id="v123",
        generated_asset_path=gen_file,
        source_thumbnail_path=tmp_path / "src.jpg",
    )

    scorer = DeterminismCheckerScorer()
    res = scorer.score(ctx)

    assert res.dimension == "determinism"
    assert res.status == "success"
    assert res.score == 1.0


def test_determinism_checker_active(tmp_path):
    gen_file = tmp_path / "gen.png"
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(gen_file), img)

    ctx = QualityScoringContext(
        video_id="v123",
        generated_asset_path=gen_file,
        source_thumbnail_path=tmp_path / "src.jpg",
    )

    def mock_runner(vid):
        return np.full((10, 10, 3), 128, dtype=np.uint8)

    scorer = DeterminismCheckerScorer(repeat_count=3, generation_runner_fn=mock_runner)
    res = scorer.score(ctx)

    assert res.dimension == "determinism"
    assert res.status == "success"
    assert res.score == pytest.approx(1.0, rel=1e-3)

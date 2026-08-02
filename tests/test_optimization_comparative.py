"""
tests/test_optimization_comparative.py
========================================

Unit tests for comparative quality scoring components.
"""

from pathlib import Path
from PIL import Image
import pytest

from evaluation.quality.emotional_ctr_scorer import EmotionalCTRScorer
from evaluation.quality.whitespace_scorer import WhitespaceScorer
from evaluation.quality.scoring_context import QualityScoringContext
from modules.models import QualityAssuranceReport
from optimization.comparative.baseline_scorer import BaselineScore, BaselineScorer
from optimization.comparative.beats_original_scorer import BeatsOriginalScorer, BeatsOriginalVerdict
from optimization.comparative.edit_magnitude_scorer import EditMagnitudeScorer


def test_emotional_ctr_scorer(tmp_path: Path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (256, 256), color=(200, 100, 50)).save(img_path)

    ctx = QualityScoringContext(video_id="v123", generated_asset_path=img_path, source_thumbnail_path=img_path)
    scorer = EmotionalCTRScorer()
    res = scorer.score(ctx)

    assert res.dimension == "emotional_ctr"
    assert 0.0 <= res.score <= 1.0
    assert res.status == "success"


def test_whitespace_scorer(tmp_path: Path):
    img_path = tmp_path / "test_space.png"
    Image.new("RGB", (256, 256), color=(128, 128, 128)).save(img_path)

    ctx = QualityScoringContext(video_id="v123", generated_asset_path=img_path, source_thumbnail_path=img_path)
    scorer = WhitespaceScorer()
    res = scorer.score(ctx)

    assert res.dimension == "whitespace"
    assert 0.0 <= res.score <= 1.0
    assert res.status == "success"


def test_baseline_scorer(tmp_path: Path):
    img_path = tmp_path / "source.png"
    Image.new("RGB", (100, 100), color=(100, 150, 200)).save(img_path)

    scorer = BaselineScorer()
    baseline = scorer.score("v123", img_path)

    assert baseline.video_id == "v123"
    assert 0.0 <= baseline.overall_score <= 1.0
    assert baseline.source_path == str(img_path)


def test_beats_original_scorer():
    baseline = BaselineScore(
        video_id="v123",
        overall_score=0.60,
        dimension_scores={"composition": 0.60},
        source_path="source.png",
    )
    cand_qa = QualityAssuranceReport(
        resolution_passed=True,
        file_integrity_passed=True,
        safety_passed=True,
        overall_score=0.75,
        hard_gate_passed=True,
    )

    scorer = BeatsOriginalScorer(min_win_margin=0.05)
    verdict = scorer.score(
        video_id="v123",
        candidate_index=0,
        candidate_qa_report=cand_qa,
        baseline_score=baseline,
    )

    assert verdict.beats_original is True
    assert pytest.approx(verdict.delta, 0.01) == 0.15


def test_edit_magnitude_scorer(tmp_path: Path):
    src_path = tmp_path / "src.png"
    cand_path = tmp_path / "cand.png"
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(src_path)
    Image.new("RGB", (100, 100), color=(0, 0, 0)).save(cand_path)

    qa = QualityAssuranceReport(
        resolution_passed=True,
        file_integrity_passed=True,
        safety_passed=True,
        identity_score=0.20,
        overall_score=0.50,
        hard_gate_passed=True,
    )

    scorer = EditMagnitudeScorer(min_ssim=0.60, max_identity_drift=0.30)
    res = scorer.score(src_path, cand_path, qa)

    assert res.identity_drift == pytest.approx(0.80, 0.01)
    assert res.over_edited is True

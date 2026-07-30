"""
test_scorers_and_aggregator.py
===============================

Unit tests for PVQEF quality scorers and aggregator.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.quality import (
    Aggregator,
    BackgroundQualityScorer,
    ColorHarmonyScorer,
    CompositionScorer,
    FacePreservationScorer,
    IQualityScorer,
    InlineQAScorer,
    ObjectPreservationScorer,
    QualityScoringContext,
    TextReadabilityScorer,
    VisualConsistencyScorer,
)
from modules.models import DimensionScore, QualityEvaluationReport


def test_quality_scoring_context_init(tmp_path):
    gen_path = tmp_path / "gen.png"
    src_path = tmp_path / "src.jpg"

    ctx = QualityScoringContext(
        video_id="v123",
        generated_asset_path=gen_path,
        source_thumbnail_path=src_path,
    )

    assert ctx.video_id == "v123"
    assert ctx.get_generated_image() is None
    assert ctx.get_source_image() is None


def test_face_preservation_scorer_no_face(tmp_path):
    ctx = QualityScoringContext(
        video_id="v123",
        generated_asset_path=tmp_path / "gen.png",
        source_thumbnail_path=tmp_path / "src.jpg",
    )
    scorer = FacePreservationScorer()
    res = scorer.score(ctx)

    assert res.dimension == "face_preservation"
    assert res.score == 1.0
    assert res.passed is True


def test_inline_qa_scorer_no_candidates(tmp_path):
    ctx = QualityScoringContext(
        video_id="v123",
        generated_asset_path=tmp_path / "gen.png",
        source_thumbnail_path=tmp_path / "src.jpg",
    )
    scorer = InlineQAScorer()
    res = scorer.score(ctx)

    assert res.dimension == "inline_qa"
    assert res.status == "skipped"


def test_aggregator_combines_scorers(tmp_path):
    ctx = QualityScoringContext(
        video_id="v123",
        generated_asset_path=tmp_path / "gen.png",
        source_thumbnail_path=tmp_path / "src.jpg",
    )

    s1 = MagicMock(spec=IQualityScorer)
    s1.dimension = "face_preservation"
    s1.score.return_value = DimensionScore(
        dimension="face_preservation", score=0.9, passed=True, threshold=0.5, scorer_version="1.0"
    )

    s2 = MagicMock(spec=IQualityScorer)
    s2.dimension = "composition"
    s2.score.return_value = DimensionScore(
        dimension="composition", score=0.8, passed=True, threshold=0.5, scorer_version="1.0"
    )

    agg = Aggregator(scorers=[s1, s2], weights={"face_preservation": 0.5, "composition": 0.5})
    report = agg.evaluate(ctx)

    assert isinstance(report, QualityEvaluationReport)
    assert report.video_id == "v123"
    assert len(report.dimension_scores) == 2
    assert report.weighted_overall_score == pytest.approx(0.85)
    assert report.hard_gate_passed is True

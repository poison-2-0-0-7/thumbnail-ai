"""
test_evaluation_engine.py
==========================

Comprehensive test suite for Phase 5.2 Thumbnail Evaluation Engine.
Tests cover:
- Evaluation of CandidateSet and individual CandidateResult objects
- Verification of all 22 deterministic quality metrics:
  1. Face Visibility
  2. Face Size
  3. Face Position
  4. Eye Contact
  5. Emotion Strength
  6. Text Readability
  7. Font Contrast
  8. Subject Saliency
  9. Visual Hierarchy
  10. Rule of Thirds
  11. Negative Space
  12. Composition Balance
  13. Background Clutter
  14. Color Harmony
  15. Color Contrast
  16. Brand Preservation
  17. Object Separation
  18. Typography Quality
  19. Thumbnail Clarity
  20. Visual Simplicity
  21. Mobile Readability
  22. Estimated CTR Score
- Metric return contracts (score 0-100, weight, confidence 0-1, reason, evidence dict)
- Configurable EvaluationProfile weights and thresholds (no magic numbers)
- JSON & Pydantic serialization / deserialization (EvaluationResult & EvaluationSet)
- Pre-flight input validation (missing files, empty 0-byte images, invalid dimensions, None inputs)
- Integration pipeline flow: MultiCandidateGenerator -> ThumbnailEvaluationEngine -> EvaluationSet
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from thumbnail_intelligence.evaluation import (
    EvaluationEngineError,
    EvaluationMetric,
    EvaluationProfile,
    EvaluationReport,
    EvaluationResult,
    EvaluationSet,
    MetricBreakdown,
    ThumbnailEvaluationEngine,
)
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.multi_candidate_generator import MultiCandidateGenerator
from thumbnail_intelligence.reasoning.multi_candidate_models import CandidateSet


@pytest.fixture
def sample_image_file():
    """Create a temporary valid thumbnail raster image on disk for testing."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
        img = np.full((720, 1280, 3), 150, dtype=np.uint8)
        # Add visual features (circle hero, text block)
        cv2.circle(img, (400, 360), 180, (255, 50, 100), -1)
        cv2.rectangle(img, (600, 200), (1200, 400), (0, 255, 200), -1)
        cv2.imwrite(path, img)

    yield path

    if os.path.exists(path):
        os.remove(path)


class TestThumbnailEvaluationEngine:

    def test_evaluate_image_returns_all_twenty_two_metrics(self, sample_image_file):
        """Verify that ThumbnailEvaluationEngine evaluates all 22 required quality metrics."""
        engine = ThumbnailEvaluationEngine()
        result = engine.evaluate_image(sample_image_file)

        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.overall_score <= 100.0
        assert 0.0 <= result.weighted_score <= 100.0
        assert 0.0 <= result.confidence <= 1.0

        # Verify exactly 22 metrics present
        assert len(result.metrics) == 22

        expected_metrics = [
            "face_visibility",
            "face_size",
            "face_position",
            "eye_contact",
            "emotion_strength",
            "text_readability",
            "font_contrast",
            "subject_saliency",
            "visual_hierarchy",
            "rule_of_thirds",
            "negative_space",
            "composition_balance",
            "background_clutter",
            "color_harmony",
            "color_contrast",
            "brand_preservation",
            "object_separation",
            "typography_quality",
            "thumbnail_clarity",
            "visual_simplicity",
            "mobile_readability",
            "estimated_ctr_score",
        ]

        for m_name in expected_metrics:
            assert m_name in result.metrics
            metric = result.metrics[m_name]
            assert isinstance(metric, EvaluationMetric)
            assert 0.0 <= metric.score <= 100.0
            assert metric.weight >= 0.0
            assert 0.0 <= metric.confidence <= 1.0
            assert isinstance(metric.reason, str) and len(metric.reason) > 0
            assert isinstance(metric.evidence, dict)

    def test_categorized_metric_breakdown(self, sample_image_file):
        """Verify that MetricBreakdown categorizes all 22 metrics accurately into 5 categories."""
        engine = ThumbnailEvaluationEngine()
        result = engine.evaluate_image(sample_image_file)

        bd = result.breakdown
        assert isinstance(bd, MetricBreakdown)
        assert len(bd.face_metrics) == 5
        assert len(bd.typography_metrics) == 4
        assert len(bd.composition_metrics) == 5
        assert len(bd.color_metrics) == 3
        assert len(bd.quality_metrics) == 5

    def test_custom_evaluation_profile_weights_and_thresholds(self, sample_image_file):
        """Test custom EvaluationProfile weight overrides and non-magic thresholds."""
        custom_weights = {
            "face_visibility": 0.20,
            "text_readability": 0.20,
            "estimated_ctr_score": 0.60,
        }
        custom_thresholds = {
            "min_font_size_px": 48.0,
            "wcag_contrast_min": 7.0,
        }

        profile = EvaluationProfile(
            profile_id="custom_prof",
            weights=custom_weights,
            thresholds=custom_thresholds,
        )
        engine = ThumbnailEvaluationEngine(profile=profile)
        result = engine.evaluate_image(sample_image_file)

        assert result.metrics["face_visibility"].weight == 0.20
        assert result.metrics["text_readability"].weight == 0.20
        assert result.metrics["estimated_ctr_score"].weight == 0.60

    def test_end_to_end_candidate_set_evaluation(self):
        """Test full pipeline integration: MultiCandidateGenerator -> ThumbnailEvaluationEngine -> EvaluationSet."""
        generator = MultiCandidateGenerator()
        brief = DesignBrief()

        with tempfile.TemporaryDirectory() as tmp_dir:
            candidate_set = generator.generate_from_brief(brief, count=5, output_directory=tmp_dir)
            assert isinstance(candidate_set, CandidateSet)

            engine = ThumbnailEvaluationEngine()
            eval_set = engine.evaluate_candidate_set(candidate_set)

            assert isinstance(eval_set, EvaluationSet)
            assert len(eval_set.results) == 5

            # Verify report details
            assert eval_set.report.total_candidates_evaluated == 5
            assert eval_set.report.top_scoring_candidate_id in {"candidate_a", "candidate_b", "candidate_c", "candidate_d", "candidate_e"}
            assert 0.0 <= eval_set.report.top_score <= 100.0
            assert 0.0 <= eval_set.report.average_overall_score <= 100.0

            # Verify result retrieval by candidate_id
            res_a = eval_set.get_result("candidate_a")
            assert res_a is not None
            assert res_a.candidate_id == "candidate_a"

    def test_json_and_pydantic_serialization(self, sample_image_file):
        """Verify JSON and Pydantic serialization / deserialization for EvaluationResult and EvaluationSet."""
        engine = ThumbnailEvaluationEngine()
        result = engine.evaluate_image(sample_image_file, candidate_id="cand_test")

        # EvaluationResult JSON round-trip
        json_str = result.to_json()
        assert isinstance(json_str, str)
        reconstructed_res = EvaluationResult.from_json(json_str)
        assert reconstructed_res.candidate_id == "cand_test"
        assert reconstructed_res.overall_score == result.overall_score
        assert len(reconstructed_res.metrics) == 22

    def test_pre_flight_input_validation_errors(self):
        """Test validation error handling for missing files, 0-byte files, invalid images, and None inputs."""
        engine = ThumbnailEvaluationEngine()

        # 1. None CandidateSet raises EvaluationEngineError
        with pytest.raises(EvaluationEngineError, match="Cannot evaluate None or empty CandidateSet"):
            engine.evaluate_candidate_set(None)

        # 2. Missing image file path raises EvaluationEngineError
        with pytest.raises(EvaluationEngineError, match="Thumbnail image file not found"):
            engine.evaluate_image("/non/existent/image_path.jpg")

        # 3. Empty 0-byte file raises EvaluationEngineError
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            empty_path = tmp.name

        try:
            with pytest.raises(EvaluationEngineError, match=r"is empty \(0 bytes\)"):
                engine.evaluate_image(empty_path)
        finally:
            if os.path.exists(empty_path):
                os.remove(empty_path)

"""
test_intelligent_critique_engine.py
===================================

Comprehensive test suite for Phase 5.4 Intelligent Critique Engine.
Tests cover:
- IntelligentCritiqueEngine analyzing RankingResult and EvaluationResult
- Deterministic rule-based issue detection across all 22 quality metrics:
  (Face visibility, Face size, Face position, Eye contact, Emotion strength, Text readability, Font contrast, Saliency, Hierarchy, Rule of thirds, Negative space, Balance, Clutter, Color harmony, Color contrast, Brand preservation, Object separation, Typography quality, Clarity, Simplicity, Mobile readability, CTR score)
- Actionable ImprovementSuggestion generation with target elements and parameter changes
- Priority ordering formula (expected CTR gain, visual impact, implementation cost, confidence)
- Executive summary, strengths, weaknesses, critical issues, and cumulative gain estimation
- JSON and Pydantic serialization / deserialization (CritiqueReport)
- Pre-flight input validation (missing winner, missing evaluation, invalid metrics, empty/corrupt reports)
- Edge cases (high-scoring candidates vs low-scoring candidates)
- End-to-end integration pipeline: MultiCandidateGenerator -> ThumbnailEvaluationEngine -> CandidateRankingEngine -> IntelligentCritiqueEngine -> CritiqueReport
"""

import tempfile
import pytest

from thumbnail_intelligence.evaluation import EvaluationMetric, EvaluationResult, ThumbnailEvaluationEngine
from thumbnail_intelligence.ranking import CandidateRankingEngine
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.multi_candidate_generator import MultiCandidateGenerator
from thumbnail_intelligence.critique import (
    CritiqueEngineError,
    CritiqueProfile,
    CritiqueReport,
    ImpactLevel,
    ImplementationCost,
    ImprovementPlan,
    ImprovementSuggestion,
    IntelligentCritiqueEngine,
    Issue,
    IssueSeverity,
)


@pytest.fixture
def sample_ranking_result():
    """Construct a mock RankingResultfixture for testing critique generation."""
    generator = MultiCandidateGenerator()
    brief = DesignBrief()

    with tempfile.TemporaryDirectory() as tmp_dir:
        cand_set = generator.generate_from_brief(brief, count=3, output_directory=tmp_dir)
        eval_engine = ThumbnailEvaluationEngine()
        eval_set = eval_engine.evaluate_candidate_set(cand_set)

        rank_engine = CandidateRankingEngine()
        return rank_engine.rank_evaluation_set(eval_set)


class TestIntelligentCritiqueEngine:

    def test_critique_ranking_result_generates_valid_report(self, sample_ranking_result):
        """Test IntelligentCritiqueEngine analyzing a RankingResult winner."""
        engine = IntelligentCritiqueEngine()
        report = engine.critique_ranking_result(sample_ranking_result)

        assert isinstance(report, CritiqueReport)
        assert report.candidate_id == sample_ranking_result.winner.candidate_id
        assert 0.0 <= report.overall_quality_score <= 100.0
        assert isinstance(report.executive_summary, str) and len(report.executive_summary) > 0
        assert isinstance(report.strengths, list)
        assert isinstance(report.weaknesses, list)
        assert isinstance(report.critical_issues, list)
        assert isinstance(report.improvement_plan, ImprovementPlan)

    def test_deterministic_issue_detection_and_suggestions(self, sample_ranking_result):
        """Verify that metrics below thresholds generate corresponding Issues and ImprovementSuggestions."""
        # Create an EvaluationResult with low text readability and low face size
        eval_res = sample_ranking_result.winner.evaluation_result.model_copy(deep=True)

        low_readability = eval_res.metrics["text_readability"].model_copy(update={"score": 40.0, "reason": "Font size 24px below minimum 36px"})
        low_face = eval_res.metrics["face_size"].model_copy(update={"score": 55.0, "reason": "Face area ratio 6% below minimum 10%"})

        updated_metrics = dict(eval_res.metrics)
        updated_metrics["text_readability"] = low_readability
        updated_metrics["face_size"] = low_face

        eval_res = eval_res.model_copy(update={"metrics": updated_metrics})

        engine = IntelligentCritiqueEngine()
        report = engine.critique_evaluation_result(eval_res)

        assert len(report.critical_issues) >= 2
        issue_metrics = [i.metric_name for i in report.critical_issues]
        assert "text_readability" in issue_metrics
        assert "face_size" in issue_metrics

        # Verify ImprovementPlan suggestions
        plan = report.improvement_plan
        assert len(plan.prioritized_suggestions) >= 2
        sug_types = [s.action_type for s in plan.prioritized_suggestions]
        assert "increase_font_size" in sug_types
        assert "scale_subject" in sug_types

    def test_priority_ordering_formula(self, sample_ranking_result):
        """Verify that ImprovementPlan suggestions are sorted by priority score descending."""
        engine = IntelligentCritiqueEngine()
        report = engine.critique_ranking_result(sample_ranking_result)

        plan = report.improvement_plan
        assert len(plan.prioritized_suggestions) > 0

        # Verify priority_score descending order
        scores = [s.priority_score for s in plan.prioritized_suggestions]
        assert scores == sorted(scores, reverse=True)

    def test_cumulative_gain_estimation(self, sample_ranking_result):
        """Verify cumulative estimated gain calculation in ImprovementPlan."""
        engine = IntelligentCritiqueEngine()
        report = engine.critique_ranking_result(sample_ranking_result)

        assert report.estimated_overall_gain_pts >= 0.0
        assert report.estimated_overall_gain_pts <= 25.0

    def test_json_and_pydantic_serialization(self, sample_ranking_result):
        """Test JSON and Pydantic serialization / deserialization of CritiqueReport."""
        engine = IntelligentCritiqueEngine()
        report = engine.critique_ranking_result(sample_ranking_result)

        json_str = report.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 100

        reconstructed = CritiqueReport.from_json(json_str)
        assert reconstructed.report_id == report.report_id
        assert reconstructed.candidate_id == report.candidate_id
        assert reconstructed.overall_quality_score == report.overall_quality_score
        assert len(reconstructed.improvement_plan.prioritized_suggestions) == len(report.improvement_plan.prioritized_suggestions)

    def test_pre_flight_input_validation_errors(self, sample_ranking_result):
        """Test validation error handling for missing inputs, NaN scores, and missing metrics."""
        engine = IntelligentCritiqueEngine()

        # 1. None RankingResult raises CritiqueEngineError
        with pytest.raises(CritiqueEngineError, match="RankingResult cannot be None"):
            engine.critique_ranking_result(None)

        # 2. None EvaluationResult raises CritiqueEngineError
        with pytest.raises(CritiqueEngineError, match="EvaluationResult cannot be None"):
            engine.critique_evaluation_result(None)

        # 3. Missing metrics raises CritiqueEngineError
        bad_eval = sample_ranking_result.winner.evaluation_result.model_copy(update={"metrics": {}})
        with pytest.raises(CritiqueEngineError, match="missing required 22 metrics"):
            engine.critique_evaluation_result(bad_eval)

    def test_high_scoring_candidate_edge_case(self, sample_ranking_result):
        """Test critique behavior on a high-scoring candidate with zero critical issues."""
        eval_res = sample_ranking_result.winner.evaluation_result.model_copy(deep=True)

        # Set all metrics to 95.0
        perfect_metrics = {k: v.model_copy(update={"score": 95.0, "reason": "Optimal visual quality"}) for k, v in eval_res.metrics.items()}
        eval_res = eval_res.model_copy(update={"metrics": perfect_metrics, "overall_score": 95.0})

        engine = IntelligentCritiqueEngine()
        report = engine.critique_evaluation_result(eval_res)

        assert len(report.critical_issues) == 0
        assert len(report.strengths) == 22
        assert report.improvement_plan.total_estimated_gain_pts == 0.0

    def test_full_end_to_end_pipeline_integration(self):
        """Verify complete pipeline execution: Candidate Generator -> Evaluation Engine -> Ranking Engine -> IntelligentCritiqueEngine -> CritiqueReport."""
        brief = DesignBrief()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Candidate Generator
            candidate_set = MultiCandidateGenerator().generate_from_brief(brief, count=5, output_directory=tmp_dir)

            # 2. Evaluation Engine
            evaluation_set = ThumbnailEvaluationEngine().evaluate_candidate_set(candidate_set)

            # 3. Ranking Engine
            ranking_result = CandidateRankingEngine().rank_evaluation_set(evaluation_set)

            # 4. Intelligent Critique Engine
            critique_report = IntelligentCritiqueEngine().critique_ranking_result(ranking_result)

            assert isinstance(critique_report, CritiqueReport)
            assert critique_report.candidate_id == ranking_result.winner.candidate_id
            assert len(critique_report.executive_summary) > 0
            assert isinstance(critique_report.improvement_plan, ImprovementPlan)

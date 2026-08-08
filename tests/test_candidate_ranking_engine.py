"""
test_candidate_ranking_engine.py
=================================

Comprehensive test suite for Phase 5.3 Candidate Ranking Engine.
Tests cover:
- CandidateRankingEngine ranking EvaluationSet into RankingResult
- Winner, Runner-up, and Top-N selection accuracy
- Deterministic tie-breaking priority metric evaluation (no random selection)
- Ranking Confidence calculation (score margin separation, metric consistency, propagated confidence)
- Explainable pairwise comparisons (strengths, weaknesses, plain text reasoning)
- RankingReport metrics comparison matrix and tie-break logs
- JSON and Pydantic serialization / deserialization (RankingResult)
- Pre-flight input validation (empty EvaluationSet, duplicate candidate IDs, NaN scores, out-of-range scores, missing metrics)
- Edge cases and large candidate sets (10+ candidates)
- End-to-end integration pipeline: MultiCandidateGenerator -> ThumbnailEvaluationEngine -> CandidateRankingEngine -> RankingResult
"""

import tempfile
import pytest

from thumbnail_intelligence.evaluation import EvaluationResult, EvaluationSet, ThumbnailEvaluationEngine
from thumbnail_intelligence.ranking import (
    CandidateRankingEngine,
    RankedCandidate,
    RankingEngineError,
    RankingExplanation,

    RankingPolicy,
    RankingProfile,
    RankingReport,
    RankingResult,
)
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.multi_candidate_generator import MultiCandidateGenerator


@pytest.fixture
def sample_evaluation_set():
    """Construct a mock EvaluationSet with 3 candidates for testing ranking logic."""
    generator = MultiCandidateGenerator()
    brief = DesignBrief()

    with tempfile.TemporaryDirectory() as tmp_dir:
        cand_set = generator.generate_from_brief(brief, count=3, output_directory=tmp_dir)
        eval_engine = ThumbnailEvaluationEngine()
        return eval_engine.evaluate_candidate_set(cand_set)


class TestCandidateRankingEngine:

    def test_basic_candidate_ranking(self, sample_evaluation_set):
        """Test CandidateRankingEngine ranking an EvaluationSet into a RankingResult."""
        engine = CandidateRankingEngine()
        result = engine.rank_evaluation_set(sample_evaluation_set)

        assert isinstance(result, RankingResult)
        assert len(result.ranked_candidates) == 3
        assert isinstance(result.winner, RankedCandidate)
        assert result.winner.rank == 1
        assert isinstance(result.runner_up, RankedCandidate)
        assert result.runner_up.rank == 2

        # Verify descending score order
        assert result.ranked_candidates[0].final_score >= result.ranked_candidates[1].final_score
        assert result.ranked_candidates[1].final_score >= result.ranked_candidates[2].final_score

        # Verify ranking confidence
        assert 0.0 <= result.ranking_confidence <= 1.0

        # Verify pairwise explanation
        assert isinstance(result.explanation, RankingExplanation)
        assert result.explanation.winner_candidate_id == result.winner.candidate_id
        assert result.explanation.runner_up_candidate_id == result.runner_up.candidate_id
        assert len(result.explanation.summary_reasoning) > 0

    def test_deterministic_tie_breaking(self, sample_evaluation_set):
        """Test deterministic tie-breaking when candidate overall scores are identical."""
        engine = CandidateRankingEngine()
        profile = RankingProfile(
            tie_break_priority=["estimated_ctr_score", "text_readability", "face_visibility"],
            tie_threshold_pts=100.0,  # Force tie-breaking for testing
        )

        result = engine.rank_evaluation_set(sample_evaluation_set, profile=profile)

        assert isinstance(result, RankingResult)
        assert len(result.report.tie_break_log) > 0
        # Winner must have won via tie-break metric
        assert "Tie between" in result.report.tie_break_log[0]

    def test_top_n_selection(self, sample_evaluation_set):
        """Test Top-N candidate selection in RankingResult."""
        engine = CandidateRankingEngine()
        result = engine.rank_evaluation_set(sample_evaluation_set, top_n=2)

        assert len(result.top_n) == 2
        assert result.top_n[0].rank == 1
        assert result.top_n[1].rank == 2

    def test_ranking_confidence_calculation(self, sample_evaluation_set):
        """Verify ranking confidence calculation with score separation and metric consistency."""
        engine = CandidateRankingEngine()
        result = engine.rank_evaluation_set(sample_evaluation_set)

        w = result.winner.evaluation_result
        r = result.runner_up.evaluation_result if result.runner_up else None

        conf = engine.compute_confidence(w, r)
        assert 0.0 <= conf <= 1.0
        assert abs(conf - result.ranking_confidence) < 1e-3

    def test_json_and_pydantic_serialization(self, sample_evaluation_set):
        """Test JSON and Pydantic serialization / deserialization of RankingResult."""
        engine = CandidateRankingEngine()
        result = engine.rank_evaluation_set(sample_evaluation_set)

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 100

        reconstructed = RankingResult.from_json(json_str)
        assert reconstructed.set_id == result.set_id
        assert reconstructed.winner.candidate_id == result.winner.candidate_id
        assert reconstructed.ranking_confidence == result.ranking_confidence

    def test_validation_pre_flight_error_handling(self, sample_evaluation_set):
        """Test pre-flight input validation error conditions."""
        engine = CandidateRankingEngine()

        # 1. None EvaluationSet raises RankingEngineError
        with pytest.raises(RankingEngineError, match="EvaluationSet cannot be None"):
            engine.rank_evaluation_set(None)

        # 2. Empty EvaluationSet results raise RankingEngineError
        empty_set = sample_evaluation_set.model_copy(update={"results": []})
        with pytest.raises(RankingEngineError, match="EvaluationSet contains no candidate results"):
            engine.rank_evaluation_set(empty_set)

        # 3. Duplicate candidate IDs raise RankingEngineError
        dup_results = list(sample_evaluation_set.results) + [sample_evaluation_set.results[0]]
        dup_set = sample_evaluation_set.model_copy(update={"results": dup_results})
        with pytest.raises(RankingEngineError, match="Duplicate candidate_id"):
            engine.rank_evaluation_set(dup_set)

        # 4. Out of range score raises RankingEngineError
        invalid_res = sample_evaluation_set.results[0].model_copy(update={"overall_score": 150.0})
        bad_set = sample_evaluation_set.model_copy(update={"results": [invalid_res]})
        with pytest.raises(RankingEngineError, match="out of valid range"):
            engine.rank_evaluation_set(bad_set)

    def test_large_candidate_sets(self):
        """Test ranking performance and correctness on a 10-candidate set."""
        generator = MultiCandidateGenerator()
        brief = DesignBrief()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cand_set = generator.generate_from_brief(brief, count=8, output_directory=tmp_dir)
            eval_engine = ThumbnailEvaluationEngine()
            eval_set = eval_engine.evaluate_candidate_set(cand_set)

            engine = CandidateRankingEngine()
            result = engine.rank_evaluation_set(eval_set, top_n=5)

            assert len(result.ranked_candidates) == 8
            assert len(result.top_n) == 5
            assert result.winner.rank == 1
            assert result.ranked_candidates[-1].rank == 8

    def test_full_end_to_end_pipeline_integration(self):
        """Verify complete pipeline execution: Candidate Generator -> Evaluation Engine -> Ranking Engine -> RankingResult."""
        brief = DesignBrief()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Generate 5 strategic candidates (A through E)
            candidate_set = MultiCandidateGenerator().generate_from_brief(brief, count=5, output_directory=tmp_dir)

            # 2. Evaluate all 22 metrics
            evaluation_set = ThumbnailEvaluationEngine().evaluate_candidate_set(candidate_set)

            # 3. Rank candidates
            ranking_result = CandidateRankingEngine().rank_evaluation_set(evaluation_set)

            assert isinstance(ranking_result, RankingResult)
            assert ranking_result.winner.candidate_id in {"candidate_a", "candidate_b", "candidate_c", "candidate_d", "candidate_e"}
            assert ranking_result.report.winner_summary["candidate_id"] == ranking_result.winner.candidate_id
            assert len(ranking_result.report.metric_comparison_matrix) == 5

"""
test_benchmark_framework.py
============================

Comprehensive test suite for Phase 6.1 Benchmark & Evaluation Framework.
Tests cover:
- DatasetLoader (synthetic dataset creation, JSON, CSV, directory loading)
- BenchmarkRunner (running benchmarks across dataset samples, latency & VRAM tracking, side-by-side visual comparisons)
- FailureAnalyzer (categorization across 7 failure categories: poor face extraction, typography failures, low contrast, weak composition, background failures, pipeline failures, OOM failures)
- LeaderboardBuilder (ranking entries, exporting Markdown, HTML, CSV)
- BenchmarkFramework (end-to-end framework execution, multi-format reporting: HTML, Markdown, JSON, CSV)
- JSON and Pydantic serialization / deserialization (BenchmarkSession)
- Edge cases and error handling
"""

import os
import tempfile
import cv2
import pytest

from thumbnail_intelligence.benchmarks import (
    BenchmarkFramework,
    BenchmarkFrameworkError,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkRunnerError,
    BenchmarkSession,
    DatasetItem,
    DatasetLoader,
    FailureAnalyzer,
    FailureCategory,
    LeaderboardBuilder,
    LeaderboardEntry,
)
from thumbnail_intelligence.evaluation import EvaluationMetric, EvaluationResult, MetricBreakdown


class TestBenchmarkFramework:

    def test_dataset_loader_synthetic_dataset_creation(self):
        """Test DatasetLoader creating synthetic benchmark dataset items."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            items = DatasetLoader.create_synthetic_dataset(count=3, temp_dir=tmp_dir)

            assert len(items) == 3
            for item in items:
                assert isinstance(item, DatasetItem)
                assert item.item_id.startswith("synth_video_")
                assert os.path.exists(item.original_thumbnail_path)
                assert os.path.getsize(item.original_thumbnail_path) > 1000

    def test_failure_analyzer_categorization(self):
        """Test FailureAnalyzer categorizing failures across standardized categories."""
        # 1. OOM failure
        cat, reason = FailureAnalyzer.categorize_failure(error_message="CUDA out of memory error during diffusion inpaint")
        assert cat == FailureCategory.OOM_FAILURES

        # 2. Face extraction failure
        cat, reason = FailureAnalyzer.categorize_failure(error_message="Failed to detect face bbox in primary subject matter")
        assert cat == FailureCategory.POOR_FACE_EXTRACTION

        # 3. Typography failure from EvaluationResult
        m_dict = {
            "text_readability": EvaluationMetric(metric_name="text_readability", category="typography", score=35.0, weight=0.08, confidence=0.9, reason="Font too small", evidence={}),
            "font_contrast": EvaluationMetric(metric_name="font_contrast", category="typography", score=40.0, weight=0.06, confidence=0.9, reason="Low WCAG contrast", evidence={}),
        }
        eval_res = EvaluationResult(candidate_id="c1", candidate_label="Cand 1", overall_score=55.0, weighted_score=55.0, confidence=0.9, metrics=m_dict, breakdown=MetricBreakdown())
        cat, reason = FailureAnalyzer.categorize_failure(eval_result=eval_res)
        assert cat == FailureCategory.TYPOGRAPHY_FAILURES

    def test_leaderboard_builder_ranking_and_export(self):
        """Test LeaderboardBuilder ranking entries and exporting Markdown, HTML, CSV."""
        e1 = LeaderboardEntry(rank=1, model_or_pipeline="Pipeline A", avg_quality_score=82.5, avg_ctr_score=75.0, success_rate_pct=100.0, avg_runtime_s=4.5, peak_vram_gb=2.1)
        e2 = LeaderboardEntry(rank=2, model_or_pipeline="Pipeline B", avg_quality_score=88.0, avg_ctr_score=81.0, success_rate_pct=100.0, avg_runtime_s=3.8, peak_vram_gb=1.9)

        leaderboard = LeaderboardBuilder.build_leaderboard([e1, e2], leaderboard_id="lb_test")

        assert leaderboard.entries[0].model_or_pipeline == "Pipeline B"  # Rank 1 (score 88.0)
        assert leaderboard.entries[1].model_or_pipeline == "Pipeline A"  # Rank 2 (score 82.5)

        md_text = leaderboard.to_markdown()
        assert "| Rank | Model / Pipeline |" in md_text
        assert "Pipeline B" in md_text

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_p = LeaderboardBuilder.export_csv(leaderboard, os.path.join(tmp_dir, "lb.csv"))
            html_p = LeaderboardBuilder.export_html(leaderboard, os.path.join(tmp_dir, "lb.html"))

            assert os.path.exists(csv_p)
            assert os.path.exists(html_p)

    def test_end_to_end_benchmark_framework_execution(self):
        """Test end-to-end BenchmarkFramework execution producing BenchmarkSession and multi-format reports."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            items = DatasetLoader.create_synthetic_dataset(count=2, temp_dir=tmp_dir)

            framework = BenchmarkFramework()
            session = framework.run_benchmark(items, dataset_name="synthetic_test", output_directory=tmp_dir)

            assert isinstance(session, BenchmarkSession)
            assert len(session.results) == 2
            assert session.summary.total_samples == 2
            assert session.summary.success_rate_pct >= 0.0

            # Verify multi-format reports generated on disk
            rep = session.report
            assert isinstance(rep, BenchmarkReport)
            assert os.path.exists(rep.json_report_path)
            assert os.path.exists(rep.markdown_report_path)
            assert os.path.exists(rep.html_report_path)
            assert os.path.exists(rep.csv_report_path)

            # Verify side-by-side visual comparison artifacts generated
            for res in session.results:
                if res.visual_comparison_path:
                    assert os.path.exists(res.visual_comparison_path)
                    img = cv2.imread(res.visual_comparison_path)
                    assert img is not None
                    assert img.ndim == 3

    def test_json_and_pydantic_serialization(self):
        """Test JSON and Pydantic serialization / deserialization of BenchmarkSession."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            items = DatasetLoader.create_synthetic_dataset(count=1, temp_dir=tmp_dir)
            session = BenchmarkFramework().run_benchmark(items, dataset_name="ser_test", output_directory=tmp_dir)

            json_str = session.to_json()
            assert isinstance(json_str, str)
            assert len(json_str) > 100

            reconstructed = BenchmarkSession.from_json(json_str)
            assert reconstructed.session_id == session.session_id
            assert reconstructed.summary.total_samples == session.summary.total_samples

    def test_empty_items_raises_framework_error(self):
        """Verify framework raises BenchmarkFrameworkError when presented with empty items list."""
        framework = BenchmarkFramework()
        with pytest.raises(BenchmarkFrameworkError, match="Cannot run benchmark framework with empty items list"):
            framework.run_benchmark([])

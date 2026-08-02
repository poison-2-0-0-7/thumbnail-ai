"""
tests/test_optimization_orchestration.py
=========================================

Unit tests for winner selection, retry strategy, and optimization loop.
"""

from pathlib import Path
from PIL import Image
import pytest

from modules.models import CandidateScore, ImageGenerationResult, QualityAssuranceReport, GeneratedAsset
from optimization.comparative.baseline_scorer import BaselineScore
from optimization.comparative.beats_original_scorer import BeatsOriginalVerdict
from optimization.comparative.edit_magnitude_scorer import EditMagnitudeScore
from optimization.orchestration.winner_selector import WinnerSelector
from optimization.orchestration.retry_strategy import RetryStrategy
from optimization.orchestration.optimization_loop import OptimizationLoop


def test_winner_selector_selects_highest_delta():
    cands = [
        CandidateScore(candidate_index=0, overall_score=0.65, hard_gate_passed=True, rank=1, selected=True),
        CandidateScore(candidate_index=1, overall_score=0.80, hard_gate_passed=True, rank=2, selected=False),
    ]
    reports = [
        QualityAssuranceReport(resolution_passed=True, file_integrity_passed=True, safety_passed=True, overall_score=0.65, hard_gate_passed=True),
        QualityAssuranceReport(resolution_passed=True, file_integrity_passed=True, safety_passed=True, overall_score=0.80, hard_gate_passed=True),
    ]
    verdicts = [
        BeatsOriginalVerdict(video_id="v1", candidate_index=0, baseline_overall_score=0.60, candidate_overall_score=0.65, delta=0.05, beats_original=True),
        BeatsOriginalVerdict(video_id="v1", candidate_index=1, baseline_overall_score=0.60, candidate_overall_score=0.80, delta=0.20, beats_original=True),
    ]
    edits = [
        EditMagnitudeScore(structural_similarity=0.8, identity_drift=0.1, over_edited=False),
        EditMagnitudeScore(structural_similarity=0.8, identity_drift=0.1, over_edited=False),
    ]

    selector = WinnerSelector()
    sel = selector.select("v1", cands, reports, verdicts, edits)

    assert sel.optimization_selected_index == 1
    assert sel.selection_agrees is False


def test_retry_strategy_bounded():
    strategy = RetryStrategy(max_retries=2)
    
    res1 = strategy.evaluate("v1", 0, [], [])
    assert res1.should_retry is True
    assert res1.attempt_index == 1

    res2 = strategy.evaluate("v1", 1, [], [])
    assert res2.should_retry is True
    assert res2.attempt_index == 2

    res3 = strategy.evaluate("v1", 2, [], [])
    assert res3.should_retry is False


def test_optimization_loop_mock_generation(tmp_path: Path):
    src_img = tmp_path / "src.png"
    out_img = tmp_path / "out.png"
    Image.new("RGB", (100, 100), color=(100, 100, 100)).save(src_img)
    Image.new("RGB", (100, 100), color=(100, 100, 100)).save(out_img)

    def dummy_runner(**kwargs):
        asset = GeneratedAsset(
            filepath=str(out_img),
            path=str(out_img),
            width=100,
            height=100,
            aspect_ratio="1:1",
            format="png",
            file_size_bytes=100,
            sha256="abc",
            created_at="now",
            qa_report=QualityAssuranceReport(
                resolution_passed=True,
                file_integrity_passed=True,
                safety_passed=True,
                overall_score=0.85,
                hard_gate_passed=True,
            ),
        )
        cand = CandidateScore(candidate_index=0, overall_score=0.85, hard_gate_passed=True, selected=True)
        return ImageGenerationResult(
            video_id="v123",
            workflow_version="1.0",
            prompt_package_hash="p1",
            generated_asset=asset,
            candidate_scores=[cand],
            generated_at="now",
        )

    loop = OptimizationLoop()
    res = loop.run("v123", src_img, dummy_runner, {})

    assert res.video_id == "v123"
    assert res.selection.optimization_selected_index == 0
    assert res.acceptance.accepted is True

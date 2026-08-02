"""Tests for CandidateRankingEngine."""

from __future__ import annotations

import pytest
from pathlib import Path
from generation_components import CandidateRankingEngine
from models import CandidateStrategy
from module7_exceptions import NoEligibleCandidateError


class MockQAReport:
    def __init__(self, overall_score: float = 0.8, hard_gate_passed: bool = True, hierarchy: float = 0.8, text_safe: float = 0.9):
        self.overall_score = overall_score
        self.hard_gate_passed = hard_gate_passed
        self.visual_hierarchy_score = hierarchy
        self.text_safe_zone_score = text_safe


class MockFaceMatch:
    def __init__(self, similarity: float = 0.85, passed: bool = True):
        self.similarity = similarity
        self.passed = passed


def test_ranking_engine_multi_dimensional_scoring(tmp_path: Path):
    engine = CandidateRankingEngine()
    strat1 = CandidateStrategy.faithful_default()
    strat2 = CandidateStrategy(name="aggressive_ctr", color_grade_bias=0.2)

    cand0 = (0, tmp_path / "cand0.png", MockQAReport(overall_score=0.85, hierarchy=0.9), MockFaceMatch(similarity=0.9), strat1, None, "hash0", {})
    cand1 = (1, tmp_path / "cand1.png", MockQAReport(overall_score=0.80, hierarchy=0.95), MockFaceMatch(similarity=0.8), strat2, None, "hash1", {})

    winner, scores = engine.rank_candidates([cand0, cand1])

    assert winner[0] in {0, 1}
    assert len(scores) == 2
    assert scores[0].hard_gate_passed is True
    assert scores[1].hard_gate_passed is True


def test_ranking_engine_hard_gate_exclusion(tmp_path: Path):
    engine = CandidateRankingEngine()
    strat = CandidateStrategy.faithful_default()

    # Cand 0 fails hard gate, Cand 1 passes hard gate with lower overall score
    cand0 = (0, tmp_path / "cand0.png", MockQAReport(overall_score=0.95, hard_gate_passed=False), MockFaceMatch(similarity=0.95), strat, None, "hash0", {})
    cand1 = (1, tmp_path / "cand1.png", MockQAReport(overall_score=0.70, hard_gate_passed=True), MockFaceMatch(similarity=0.70), strat, None, "hash1", {})

    winner, scores = engine.rank_candidates([cand0, cand1])

    assert winner[0] == 1
    assert scores[0].hard_gate_passed is False
    assert scores[0].selected is False
    assert scores[1].hard_gate_passed is True
    assert scores[1].selected is True


def test_ranking_engine_all_hard_gate_failures_raises(tmp_path: Path):
    engine = CandidateRankingEngine()
    strat = CandidateStrategy.faithful_default()

    cand0 = (0, tmp_path / "cand0.png", MockQAReport(overall_score=0.90, hard_gate_passed=False), MockFaceMatch(similarity=0.9), strat, None, "hash0", {})

    with pytest.raises(NoEligibleCandidateError):
        engine.rank_candidates([cand0])

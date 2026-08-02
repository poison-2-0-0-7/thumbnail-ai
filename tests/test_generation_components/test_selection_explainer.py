"""Tests for SelectionExplainer."""

from __future__ import annotations

import pytest
from pathlib import Path
from generation_components import SelectionExplainer
from models import CandidateScore, CandidateStrategy


class MockQAReport:
    def __init__(self, overall_score: float = 0.85, hard_gate_passed: bool = True):
        self.overall_score = overall_score
        self.hard_gate_passed = hard_gate_passed


def test_selection_explainer_output_format(tmp_path: Path):
    explainer = SelectionExplainer()
    strat0 = CandidateStrategy.faithful_default()
    strat1 = CandidateStrategy(name="aggressive_ctr")

    cand0 = (0, tmp_path / "cand0.png", MockQAReport(overall_score=0.88), None, strat0, None, "hash0", {})
    cand1 = (1, tmp_path / "cand1.png", MockQAReport(overall_score=0.80), None, strat1, None, "hash1", {})

    scores = [
        CandidateScore(candidate_index=0, overall_score=0.88, identity_similarity=0.9, hard_gate_passed=True, rank=1, selected=True),
        CandidateScore(candidate_index=1, overall_score=0.80, identity_similarity=0.85, hard_gate_passed=True, rank=2, selected=False),
    ]

    exp = explainer.explain(
        winner_candidate=cand0,
        candidate_scores=scores,
        all_candidates=[cand0, cand1],
        clustering_exclusions={1: "near_duplicate_of_0"},
        dimension_scores_map={
            0: {"ctr_score": 0.92, "readability_score": 0.88},
            1: {"ctr_score": 0.80, "readability_score": 0.80},
        },
    )

    assert exp.winner_index == 0
    assert exp.winning_strategy == "faithful"
    assert exp.winning_margin == 0.08
    assert "ctr_score" in exp.dominant_scoring_dimensions
    assert "Candidate 0" in exp.winner_explanation
    assert exp.excluded_candidate_summary["total_candidates"] == 2

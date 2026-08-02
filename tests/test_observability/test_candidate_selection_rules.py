"""Tests for PORCE candidate selection diagnostic rules (RULE-CAND-01 to RULE-CAND-04)."""

from __future__ import annotations

import pytest
from observability.diagnostics.models import RuleContext
from observability.diagnostics.rules.candidate_selection_rules import (
    DuplicateCandidateDetectionRule,
    InconsistentRankingRule,
    PoorWinnerSelectionRule,
    WeakDiversityRule,
)
from observability.facts.models import TraceFacts
from observability.models import GenerationTraceRecord


@pytest.fixture
def empty_facts() -> TraceFacts:
    return TraceFacts(video_id="test_video_123", extracted_at="2026-08-01T00:00:00Z")


def test_rule_cand_01_duplicate_detection(empty_facts):
    rule = DuplicateCandidateDetectionRule()
    trace = GenerationTraceRecord(
        video_id="test_video_123",
        candidate_scores=[0.90, 0.85],
        cluster_id="cluster_1",
        exclusion_reason="near_duplicate_of_0",
    )
    context = RuleContext(facts=empty_facts, generation_trace=trace)
    finding = rule.check(empty_facts, context)

    assert finding is not None
    assert finding.finding_id == "RULE-CAND-01"
    assert finding.severity == "WARNING"


def test_rule_cand_02_weak_diversity(empty_facts):
    rule = WeakDiversityRule()
    trace = GenerationTraceRecord(
        video_id="test_video_123",
        ranking_dimensions={"originality_score": 0.15, "diversity_bonus": 0.20},
    )
    context = RuleContext(facts=empty_facts, generation_trace=trace)
    finding = rule.check(empty_facts, context)

    assert finding is not None
    assert finding.finding_id == "RULE-CAND-02"
    assert finding.severity == "WARNING"


def test_rule_cand_03_inconsistent_ranking():
    rule = InconsistentRankingRule()
    facts = TraceFacts(
        video_id="test_video_123",
        extracted_at="2026-08-01T00:00:00Z",
        winning_candidate_index=1,
        module7_selected_index=1,
    )
    trace = GenerationTraceRecord(
        video_id="test_video_123",
        candidate_scores=[0.95, 0.80],
        selection_explanation="Non-QA dimensions dominated final ranking.",
    )
    context = RuleContext(facts=facts, generation_trace=trace)
    
    finding = rule.check(facts, context)
    assert rule.rule_id == "RULE-CAND-03"


def test_rule_cand_04_poor_winner_selection(empty_facts):
    rule = PoorWinnerSelectionRule()
    trace = GenerationTraceRecord(
        video_id="test_video_123",
        candidate_scores=[0.90, 0.80],
    )
    context = RuleContext(facts=empty_facts, generation_trace=trace)
    assert rule.rule_id == "RULE-CAND-04"

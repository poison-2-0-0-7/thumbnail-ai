"""
test_regression_detector.py
============================

Unit tests for RegressionDetector and IRegressionRule rules.
"""

import pytest

from evaluation.benchmarking import (
    DimensionRegressionRule,
    FailureRateSpikeRule,
    OverallScoreDropRule,
    PerformanceRegressionRule,
    RegressionDetector,
)
from modules.models import BenchmarkRecord, RegressionFinding


def test_overall_score_drop_rule():
    rule = OverallScoreDropRule()

    base = BenchmarkRecord(
        run_id="r1",
        recorded_at="2026-07-30T00:00:00Z",
        total_creators=5,
        succeeded=5,
        skipped=0,
        mean_weighted_overall_score=0.85,
    )

    # No drop
    curr_good = base.model_copy(update={"run_id": "r2", "mean_weighted_overall_score": 0.84})
    assert rule.check(curr_good, base) is None

    # Significant drop (> 0.05)
    curr_bad = base.model_copy(update={"run_id": "r3", "mean_weighted_overall_score": 0.70})
    finding = rule.check(curr_bad, base)
    assert isinstance(finding, RegressionFinding)
    assert finding.rule_name == "overall_score_drop"
    assert finding.severity == "critical"


def test_failure_rate_spike_rule():
    rule = FailureRateSpikeRule()

    base = BenchmarkRecord(
        run_id="r1",
        recorded_at="2026-07-30T00:00:00Z",
        total_creators=10,
        succeeded=10,
        skipped=0,
        mean_weighted_overall_score=0.85,
    )

    curr = base.model_copy(update={"run_id": "r2", "succeeded": 5, "skipped": 5})
    finding = rule.check(curr, base)
    assert isinstance(finding, RegressionFinding)
    assert finding.rule_name == "failure_rate_spike"


def test_regression_detector_aggregate():
    detector = RegressionDetector()

    base = BenchmarkRecord(
        run_id="r1",
        recorded_at="2026-07-30T00:00:00Z",
        total_creators=5,
        succeeded=5,
        skipped=0,
        mean_weighted_overall_score=0.90,
    )

    curr = base.model_copy(update={"run_id": "r2", "mean_weighted_overall_score": 0.75})
    findings = detector.detect(curr, base)
    assert len(findings) >= 1

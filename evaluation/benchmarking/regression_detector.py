"""
regression_detector.py
=======================

Statistical current-vs-baseline regression detection using independent IRegressionRule implementations.
"""

from __future__ import annotations

from typing import Sequence

from loguru import logger

from evaluation.config import (
    EVAL_DETERMINISM_SSIM_THRESHOLD,
    EVAL_QA_DIVERGENCE_THRESHOLD,
    EVAL_REGRESSION_FAILURE_DELTA,
    EVAL_REGRESSION_LATENCY_MULTIPLIER,
    EVAL_REGRESSION_SCORE_DELTA,
)
from modules.models import BenchmarkRecord, PipelineRunReport, RegressionFinding
from .interfaces import IHistoricalStore, IRegressionRule


class OverallScoreDropRule(IRegressionRule):
    """Flags drop in overall weighted score exceeding threshold."""

    @property
    def rule_name(self) -> str:
        return "overall_score_drop"

    def check(self, current: BenchmarkRecord, baseline: BenchmarkRecord) -> RegressionFinding | None:
        delta = current.mean_weighted_overall_score - baseline.mean_weighted_overall_score
        if delta < -EVAL_REGRESSION_SCORE_DELTA:
            severity = "critical" if delta < -0.10 else "warning"
            return RegressionFinding(
                rule_name=self.rule_name,
                severity=severity,
                dimension_or_stage="overall",
                current_value=current.mean_weighted_overall_score,
                baseline_value=baseline.mean_weighted_overall_score,
                delta=delta,
                message=f"Overall score dropped by {abs(delta):.3f} vs baseline ({baseline.mean_weighted_overall_score:.2f} -> {current.mean_weighted_overall_score:.2f})",
            )
        return None


class DimensionRegressionRule(IRegressionRule):
    """Flags drop in any individual dimension mean score."""

    @property
    def rule_name(self) -> str:
        return "dimension_score_drop"

    def check(self, current: BenchmarkRecord, baseline: BenchmarkRecord) -> RegressionFinding | None:
        for dim, base_val in baseline.per_dimension_mean_scores.items():
            curr_val = current.per_dimension_mean_scores.get(dim, base_val)
            delta = curr_val - base_val
            if delta < -EVAL_REGRESSION_SCORE_DELTA:
                return RegressionFinding(
                    rule_name=self.rule_name,
                    severity="warning",
                    dimension_or_stage=dim,
                    current_value=curr_val,
                    baseline_value=base_val,
                    delta=delta,
                    message=f"Dimension {dim} score dropped by {abs(delta):.3f} ({base_val:.2f} -> {curr_val:.2f})",
                )
        return None


class FailureRateSpikeRule(IRegressionRule):
    """Flags spike in creator failure/skip rate."""

    @property
    def rule_name(self) -> str:
        return "failure_rate_spike"

    def check(self, current: BenchmarkRecord, baseline: BenchmarkRecord) -> RegressionFinding | None:
        curr_rate = (current.skipped / current.total_creators) if current.total_creators > 0 else 0.0
        base_rate = (baseline.skipped / baseline.total_creators) if baseline.total_creators > 0 else 0.0
        delta = curr_rate - base_rate
        if delta > EVAL_REGRESSION_FAILURE_DELTA:
            return RegressionFinding(
                rule_name=self.rule_name,
                severity="critical",
                dimension_or_stage="batch_failure_rate",
                current_value=curr_rate,
                baseline_value=base_rate,
                delta=delta,
                message=f"Failure rate spiked by +{delta:.1%} vs baseline ({base_rate:.1%} -> {curr_rate:.1%})",
            )
        return None


class PerformanceRegressionRule(IRegressionRule):
    """Flags latency multiplier spike on any stage."""

    @property
    def rule_name(self) -> str:
        return "performance_latency_spike"

    def check(self, current: BenchmarkRecord, baseline: BenchmarkRecord) -> RegressionFinding | None:
        for stage, base_dur in baseline.mean_stage_durations_seconds.items():
            if base_dur <= 0:
                continue
            curr_dur = current.mean_stage_durations_seconds.get(stage, base_dur)
            multiplier = curr_dur / base_dur
            if multiplier >= EVAL_REGRESSION_LATENCY_MULTIPLIER:
                return RegressionFinding(
                    rule_name=self.rule_name,
                    severity="warning",
                    dimension_or_stage=stage,
                    current_value=curr_dur,
                    baseline_value=base_dur,
                    delta=curr_dur - base_dur,
                    message=f"Stage {stage} latency increased {multiplier:.2f}x ({base_dur:.2f}s -> {curr_dur:.2f}s)",
                )
        return None


class RegressionDetector:
    """Evaluates set of IRegressionRule implementations against historical baselines."""

    def __init__(self, rules: Sequence[IRegressionRule] | None = None) -> None:
        default_rules = [
            OverallScoreDropRule(),
            DimensionRegressionRule(),
            FailureRateSpikeRule(),
            PerformanceRegressionRule(),
        ]
        self.rules = rules if rules is not None else default_rules

    def detect(self, current: BenchmarkRecord, baseline: BenchmarkRecord) -> list[RegressionFinding]:
        findings: list[RegressionFinding] = []
        for rule in self.rules:
            try:
                finding = rule.check(current, baseline)
                if finding:
                    findings.append(finding)
            except Exception as exc:
                logger.error("Regression rule {rule} failed: {exc}", rule=rule.rule_name, exc=exc)
        return findings


def detect_regressions(
    current_report: PipelineRunReport,
    baseline_record: BenchmarkRecord,
    detector: RegressionDetector | None = None,
) -> list[RegressionFinding]:
    """Public function to detect regressions between a current run report and baseline record."""
    from .historical_store import HistoricalStore
    rec = HistoricalStore().create_record_from_run(current_report)
    det = detector or RegressionDetector()
    return det.detect(rec, baseline_record)

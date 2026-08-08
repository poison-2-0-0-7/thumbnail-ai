"""
engine.py
=========

IntelligentCritiqueEngine Implementation for Phase 5.4.
Analyzes the winning thumbnail's EvaluationResult and produces a structured ImprovementPlan & CritiqueReport.
No LLMs, no image rendering, no regeneration. ONLY rule-based visual critique & deterministic improvement planning.

Provides:
- Deterministic rule-based issue detection across all 22 quality metrics
- Actionable ImprovementSuggestion generation with target elements and parameter changes
- Prioritization model combining expected CTR gain, visual impact, implementation cost, and confidence
- Executive summary, strengths, weaknesses, critical issues, and cumulative gain estimation
- Input validation (missing winner, missing evaluation, invalid metrics, empty/corrupt reports)
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Union

from thumbnail_intelligence.evaluation.models import EvaluationResult
from thumbnail_intelligence.ranking.models import RankingResult
from thumbnail_intelligence.critique.models import (
    CritiqueProfile,
    CritiqueReport,
    ImpactLevel,
    ImplementationCost,
    ImprovementPlan,
    ImprovementSuggestion,
    Issue,
    IssueSeverity,
)
from thumbnail_intelligence.critique.rules import DefaultCritiqueRuleSet

logger = logging.getLogger(__name__)


class CritiqueEngineError(RuntimeError):
    """Exception raised for critique engine errors or invalid inputs."""
    pass


class IntelligentCritiqueEngine:
    """Deterministic, rule-based visual critique and improvement planning engine."""

    def __init__(self, profile: Optional[CritiqueProfile] = None) -> None:
        self.profile = profile or CritiqueProfile()
        self.rule_set = DefaultCritiqueRuleSet.get_rules()

    def critique_ranking_result(
        self,
        ranking_result: RankingResult,
        profile: Optional[CritiqueProfile] = None,
    ) -> CritiqueReport:
        """Critique the winning candidate thumbnail from a RankingResult.

        Args:
            ranking_result: Input RankingResult object.
            profile: Optional CritiqueProfile override.

        Returns:
            Master CritiqueReport containing ImprovementPlan and executive summary.
        """
        if not ranking_result:
            raise CritiqueEngineError("RankingResult cannot be None.")

        if not ranking_result.winner or not ranking_result.winner.evaluation_result:
            raise CritiqueEngineError("RankingResult must contain a valid winner with EvaluationResult.")

        winner_eval = ranking_result.winner.evaluation_result
        cand_label = ranking_result.winner.candidate_label or ranking_result.winner.candidate_id

        return self.critique_evaluation_result(
            evaluation_result=winner_eval,
            candidate_label=cand_label,
            profile=profile,
        )

    def critique_evaluation_result(
        self,
        evaluation_result: EvaluationResult,
        candidate_label: str = "",
        profile: Optional[CritiqueProfile] = None,
    ) -> CritiqueReport:
        """Critique a candidate's EvaluationResult and produce a CritiqueReport."""
        prof = profile or self.profile

        # 1. Pre-flight Validation
        self.validate_evaluation_result(evaluation_result)

        cand_id = evaluation_result.candidate_id
        label = candidate_label or evaluation_result.candidate_label or cand_id
        overall_score = evaluation_result.overall_score

        logger.info(f"=== Starting IntelligentCritiqueEngine for candidate '{cand_id}' (overall_score={overall_score:.1f}) ===")

        strengths: List[str] = []
        weaknesses: List[str] = []
        issues: List[Issue] = []
        suggestions: List[ImprovementSuggestion] = []

        # 2. Evaluate all metrics using CritiqueRuleSet
        for m_name, metric in evaluation_result.metrics.items():
            score = metric.score

            if score >= 85.0:
                strengths.append(f"Strong {m_name.replace('_', ' ').title()} ({score:.1f}/100): {metric.reason}")
            elif score < prof.major_threshold_score:
                weaknesses.append(f"Suboptimal {m_name.replace('_', ' ').title()} ({score:.1f}/100): {metric.reason}")

                if m_name in self.rule_set:
                    rule = self.rule_set[m_name]
                    issue, suggestion = rule.evaluate(metric, prof)

                    issues.append(issue)
                    suggestions.append(suggestion)

        # Sort issues: CRITICAL first, then MAJOR, MINOR
        sev_rank = {IssueSeverity.CRITICAL: 0, IssueSeverity.MAJOR: 1, IssueSeverity.MINOR: 2, IssueSeverity.INFO: 3}
        issues.sort(key=lambda i: (sev_rank.get(i.severity, 4), -i.confidence))

        critical_issues = [i for i in issues if i.severity in {IssueSeverity.CRITICAL, IssueSeverity.MAJOR}]

        # 3. Build Prioritized ImprovementPlan
        plan = self.build_improvement_plan(cand_id, suggestions)

        # 4. Generate Executive Summary
        exec_summary = self._generate_executive_summary(
            cand_label=label,
            overall_score=overall_score,
            strengths_cnt=len(strengths),
            critical_cnt=len(critical_issues),
            total_gain=plan.total_estimated_gain_pts,
            top_suggestion=plan.prioritized_suggestions[0].description if plan.prioritized_suggestions else "No major fixes required",
        )

        report = CritiqueReport(
            report_id=f"crit_report_{uuid.uuid4().hex[:8]}",
            schema_version="1.0.0",
            candidate_id=cand_id,
            candidate_label=label,
            overall_quality_score=overall_score,
            executive_summary=exec_summary,
            strengths=strengths,
            weaknesses=weaknesses,
            critical_issues=critical_issues,
            improvement_plan=plan,
            estimated_overall_gain_pts=plan.total_estimated_gain_pts,
        )

        logger.info(f"=== Completed IntelligentCritiqueEngine for candidate '{cand_id}' ({len(issues)} issues detected, estimated gain={plan.total_estimated_gain_pts:.1f} pts) ===")
        return report

    def validate_evaluation_result(self, eval_result: EvaluationResult) -> None:
        """Validate input EvaluationResult integrity before critique."""
        if not eval_result:
            raise CritiqueEngineError("EvaluationResult cannot be None.")

        if not eval_result.candidate_id:
            raise CritiqueEngineError("EvaluationResult has empty candidate_id.")

        if math.isnan(eval_result.overall_score) or math.isinf(eval_result.overall_score):
            raise CritiqueEngineError(f"EvaluationResult has invalid overall score: {eval_result.overall_score}")

        if not eval_result.metrics or len(eval_result.metrics) < 22:
            raise CritiqueEngineError(f"EvaluationResult is missing required 22 metrics (found {len(eval_result.metrics) if eval_result.metrics else 0}).")

    def build_improvement_plan(
        self,
        candidate_id: str,
        suggestions: List[ImprovementSuggestion],
    ) -> ImprovementPlan:
        """Build prioritized ImprovementPlan by sorting suggestions using calculated priority scores."""
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Calculate Priority Score for each suggestion
        impact_map = {ImpactLevel.HIGH: 3.0, ImpactLevel.MEDIUM: 2.0, ImpactLevel.LOW: 1.0}
        cost_map = {ImplementationCost.LOW: 3.0, ImplementationCost.MEDIUM: 2.0, ImplementationCost.HIGH: 1.0}

        scored_suggestions: List[ImprovementSuggestion] = []
        raw_gains: List[float] = []

        for sug in suggestions:
            imp_val = impact_map.get(sug.visual_impact, 2.0)
            cost_val = cost_map.get(sug.implementation_cost, 2.0)

            # Priority Score formula
            p_score = (sug.expected_ctr_gain * 0.4) + (imp_val * 2.5) + (cost_val * 1.5) + (sug.confidence * 2.0)
            updated_sug = sug.model_copy(update={"priority_score": round(p_score, 2)})

            scored_suggestions.append(updated_sug)
            raw_gains.append(sug.expected_ctr_gain)

        # Sort suggestions descending by priority_score
        scored_suggestions.sort(key=lambda s: s.priority_score, reverse=True)

        # Calculate cumulative gain with diminishing returns factor (0.6 multiplier, max 25.0 pts)
        cum_gain = min(25.0, sum(raw_gains) * 0.6)

        return ImprovementPlan(
            plan_id=plan_id,
            candidate_id=candidate_id,
            prioritized_suggestions=scored_suggestions,
            total_estimated_gain_pts=round(cum_gain, 1),
        )

    def _generate_executive_summary(
        self,
        cand_label: str,
        overall_score: float,
        strengths_cnt: int,
        critical_cnt: int,
        total_gain: float,
        top_suggestion: str,
    ) -> str:
        """Generate plain text executive summary for CritiqueReport."""
        status_desc = "excellent" if overall_score >= 85.0 else ("good" if overall_score >= 70.0 else "suboptimal")

        return (
            f"Critique Report for {cand_label}: Current overall quality score is {overall_score:.1f}/100 ({status_desc}). "
            f"Demonstrates {strengths_cnt} major visual strengths and {critical_cnt} critical/major areas for improvement. "
            f"Applying the prioritized improvement plan is estimated to yield +{total_gain:.1f} pts in overall CTR score. "
            f"Top priority recommendation: {top_suggestion}."
        )

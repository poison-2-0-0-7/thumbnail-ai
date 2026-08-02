"""
observability/diagnostics/rules/optimization_diagnostic_rules.py
===================================================================

Diagnostic rules for Module 8 Optimization Layer (PORCE integration).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class GeneratedThumbnailDidNotBeatOriginalRule(IDiagnosticRule):
    """
    RULE-OPT-01: Generated Thumbnail Did Not Beat Original.
    Flags when facts.beats_original is False.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-OPT-01"

    @property
    def rule_name(self) -> str:
        return "Generated Thumbnail Did Not Beat Original"

    @property
    def category(self) -> str:
        return "quality_optimization"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        if facts.beats_original is False:
            baseline = facts.baseline_score or 0.0
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="FAIL",
                confidence=1.0,
                affected_module="optimization_comparative_scoring",
                root_cause=f"Generated candidate thumbnail failed to beat original baseline score (baseline={baseline:.4f})",
                recommended_action="Enable optimization retries (OPTIMIZATION_MAX_RETRIES > 0) or adjust candidate strategy perturbation weights.",
                supporting_facts=[
                    f"video_id={facts.video_id}",
                    f"baseline_score={baseline}",
                    f"beats_original={facts.beats_original}",
                ],
                evaluation_timestamp=now_str,
            )
        return None


class OverEditedAcceptedRule(IDiagnosticRule):
    """
    RULE-OPT-02: Over-Edited Candidate Accepted.
    Flags when facts.over_edited is True.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-OPT-02"

    @property
    def rule_name(self) -> str:
        return "Over-Edited Candidate Accepted"

    @property
    def category(self) -> str:
        return "quality_optimization"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        if facts.over_edited is True:
            edit_mag = facts.edit_magnitude or 0.0
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="FAIL",
                confidence=1.0,
                affected_module="optimization_acceptance_gate",
                root_cause=f"Candidate thumbnail was marked over-edited (structural similarity SSIM={edit_mag:.4f}) but was accepted",
                recommended_action="Tighten OPTIMIZATION_MIN_STRUCTURAL_SIMILARITY threshold or set OPTIMIZATION_ACCEPTANCE_REPORT_ONLY=False.",
                supporting_facts=[
                    f"video_id={facts.video_id}",
                    f"edit_magnitude={edit_mag}",
                    f"over_edited={facts.over_edited}",
                ],
                evaluation_timestamp=now_str,
            )
        return None


class OptimizationSelectionDisagreementRule(IDiagnosticRule):
    """
    RULE-OPT-03: Selection Disagreement Between Module 7 and Optimization Layer.
    Flags when facts.selection_agreed is False.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-OPT-03"

    @property
    def rule_name(self) -> str:
        return "Optimization Selection Disagreement"

    @property
    def category(self) -> str:
        return "quality_optimization"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        if facts.selection_agreed is False:
            m7_idx = facts.module7_selected_index
            opt_idx = facts.winning_candidate_index
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="INFO",
                confidence=0.9,
                affected_module="optimization_winner_selection",
                root_cause=f"Module 7 CandidateRanker selected index {m7_idx} but Optimization WinnerSelector selected index {opt_idx}",
                recommended_action="Review CandidateRanker weights (MODULE7_QA_WEIGHTS) against comparative BeatsOriginal scoring.",
                supporting_facts=[
                    f"video_id={facts.video_id}",
                    f"module7_selected_index={m7_idx}",
                    f"winning_candidate_index={opt_idx}",
                ],
                evaluation_timestamp=now_str,
            )
        return None

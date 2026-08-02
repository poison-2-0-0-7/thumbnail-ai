"""
observability/diagnostics/rules/candidate_selection_rules.py
================================================================

PORCE Diagnostic rules for Multi-Candidate Generation and Selection.
Implements RULE-CAND-01 through RULE-CAND-04 as defined by Module 9 Architecture §10.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class DuplicateCandidateDetectionRule(IDiagnosticRule):
    """
    RULE-CAND-01: Duplicate Candidate Detection.

    Fires WARN when a candidate batch produced near-duplicate candidates
    where cluster count is less than requested candidate count.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-CAND-01"

    @property
    def rule_name(self) -> str:
        return "Duplicate Candidate Detection"

    @property
    def category(self) -> str:
        return "candidate_selection"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        gen_trace = context.generation_trace if context else None
        if not gen_trace:
            return None

        candidate_scores = getattr(gen_trace, "candidate_scores", []) or []
        if len(candidate_scores) <= 1:
            return None

        exclusion_reason = getattr(gen_trace, "exclusion_reason", None)
        cluster_id = getattr(gen_trace, "cluster_id", None)

        has_duplicates = any(
            (isinstance(cs, dict) and cs.get("exclusion_reason", "").startswith("near_duplicate_of_"))
            or (hasattr(cs, "exclusion_reason") and getattr(cs, "exclusion_reason", "").startswith("near_duplicate_of_"))
            for cs in candidate_scores
        ) or (exclusion_reason and "duplicate" in str(exclusion_reason).lower())

        if has_duplicates:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="WARNING",
                confidence=0.9,
                affected_module="candidate_clustering_engine",
                root_cause=f"Candidate batch produced near-duplicate candidates (cluster_id={cluster_id}).",
                recommended_action="Increase strategy perturbation bounds in strategy packs or check prompt diversity.",
                supporting_facts=[f"cluster_id={cluster_id}", f"exclusion_reason={exclusion_reason}"],
                evaluation_timestamp=now_str,
            )
        return None


class WeakDiversityRule(IDiagnosticRule):
    """
    RULE-CAND-02: Weak Diversity.

    Fires WARN when candidates exhibit low strategy perturbation or weak visual diversity across surviving batch.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-CAND-02"

    @property
    def rule_name(self) -> str:
        return "Weak Diversity"

    @property
    def category(self) -> str:
        return "candidate_selection"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        gen_trace = context.generation_trace if context else None
        if not gen_trace:
            return None

        dims = getattr(gen_trace, "ranking_dimensions", None)
        if isinstance(dims, dict):
            orig_score = dims.get("originality_score", 1.0)
            div_bonus = dims.get("diversity_bonus", 1.0)
            if orig_score < 0.25 or div_bonus < 0.25:
                return Finding(
                    finding_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity="WARNING",
                    confidence=0.85,
                    affected_module="candidate_strategy_planner",
                    root_cause=f"Weak candidate diversity detected (originality_score={orig_score:.2f}, diversity_bonus={div_bonus:.2f}).",
                    recommended_action="Use full_spectrum_eight strategy pack or expand lighting/framing bias ranges.",
                    supporting_facts=[f"originality_score={orig_score}", f"diversity_bonus={div_bonus}"],
                    evaluation_timestamp=now_str,
                )
        return None


class InconsistentRankingRule(IDiagnosticRule):
    """
    RULE-CAND-03: Inconsistent Ranking.

    Fires WARN when the selected winner does not have the highest QA overall score,
    indicating non-QA dimensions (CTR, branding, etc.) outweighed pure QA score.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-CAND-03"

    @property
    def rule_name(self) -> str:
        return "Inconsistent Ranking"

    @property
    def category(self) -> str:
        return "candidate_selection"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        gen_trace = context.generation_trace if context else None
        if not gen_trace:
            return None

        candidate_scores = getattr(gen_trace, "candidate_scores", []) or []
        if len(candidate_scores) <= 1:
            return None

        scores_list = []
        for cs in candidate_scores:
            if isinstance(cs, dict):
                scores_list.append((cs.get("candidate_index", 0), cs.get("overall_score", 0.0), cs.get("selected", False)))
            elif hasattr(cs, "candidate_index"):
                scores_list.append((getattr(cs, "candidate_index"), getattr(cs, "overall_score", 0.0), getattr(cs, "selected", False)))

        if not scores_list:
            return None

        max_qa_idx = max(scores_list, key=lambda x: x[1])[0]
        selected_tuple = next((x for x in scores_list if x[2]), None)

        if selected_tuple and selected_tuple[0] != max_qa_idx:
            sel_idx = selected_tuple[0]
            explanation = getattr(gen_trace, "selection_explanation", "") or "Non-QA dimensions dominated final ranking."
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="WARNING",
                confidence=0.9,
                affected_module="candidate_ranking_engine",
                root_cause=f"Algorithmic winner candidate {sel_idx} differed from highest QA overall score candidate {max_qa_idx}.",
                recommended_action="Review SelectionExplanation or adjust MODULE7_RANKING_WEIGHTS if QA score should take higher precedence.",
                supporting_facts=[f"selected_candidate={sel_idx}", f"highest_qa_candidate={max_qa_idx}", f"explanation={explanation}"],
                evaluation_timestamp=now_str,
            )
        return None


class PoorWinnerSelectionRule(IDiagnosticRule):
    """
    RULE-CAND-04: Poor Winner Selection.

    Fires FAIL when the selected candidate failed a QA hard gate while another candidate passed.
    Enforces invariant that hard-gate failures must never be selected.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-CAND-04"

    @property
    def rule_name(self) -> str:
        return "Poor Winner Selection"

    @property
    def category(self) -> str:
        return "candidate_selection"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        gen_trace = context.generation_trace if context else None
        if not gen_trace:
            return None

        candidate_scores = getattr(gen_trace, "candidate_scores", []) or []
        if not candidate_scores:
            return None

        selected_gate_failed = False
        selected_idx = None
        has_passing_candidate = False

        for cs in candidate_scores:
            if isinstance(cs, dict):
                c_idx = cs.get("candidate_index", 0)
                passed = cs.get("hard_gate_passed", True)
                sel = cs.get("selected", False)
            else:
                c_idx = getattr(cs, "candidate_index", 0)
                passed = getattr(cs, "hard_gate_passed", True)
                sel = getattr(cs, "selected", False)

            if passed:
                has_passing_candidate = True
            if sel and not passed:
                selected_gate_failed = True
                selected_idx = c_idx

        if selected_gate_failed and has_passing_candidate:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="FAIL",
                confidence=1.0,
                affected_module="candidate_ranking_engine",
                root_cause=f"Selected candidate {selected_idx} failed QA hard gate, violating hard-gate preservation invariant.",
                recommended_action="Ensure hard-gate failing candidates are excluded prior to candidate ranking.",
                supporting_facts=[f"failed_selected_candidate={selected_idx}"],
                evaluation_timestamp=now_str,
            )
        return None


# Backwards compatibility alias classes
CandidateDiversityRule = DuplicateCandidateDetectionRule
CandidateHardGateRateRule = DuplicateCandidateDetectionRule
CandidateRankingDominanceRule = InconsistentRankingRule
StrategyPackMismatchRule = WeakDiversityRule


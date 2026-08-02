"""
observability/diagnostics/rules/prompt_consistency_rules.py
===========================================================

Diagnostic rules for prompt consistency observations.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class PromptContradictionRule(IDiagnosticRule):
    """
    RULE-PRM-01: Deterministic check for direct term contradictions between positive and negative prompts.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-PRM-01"

    @property
    def rule_name(self) -> str:
        return "Positive/Negative Prompt Contradiction Check"

    @property
    def category(self) -> str:
        return "prompt_consistency"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        pos = (facts.positive_prompt or "").lower()
        neg = (facts.negative_prompt or "").lower()

        if not pos or not neg:
            return None

        # Clean words
        pos_words = set(re.findall(r"\b[a-z]{4,}\b", pos))
        neg_words = set(re.findall(r"\b[a-z]{4,}\b", neg))

        # Exclude common stop words
        stop_words = {"with", "that", "this", "from", "have", "more", "best", "high", "quality", "style"}
        common = (pos_words & neg_words) - stop_words

        if common:
            overlapping_terms = sorted(list(common))
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="WARNING",
                confidence=1.0,
                affected_module="module6",
                root_cause=f"Positive and negative prompts contain overlapping key term(s): {', '.join(overlapping_terms)}.",
                recommended_action="Review Module 6 Prompt Compiler slot filling to avoid placing identical terms in both positive and negative prompts.",
                supporting_facts=[f"overlapping_terms={overlapping_terms}"],
                evaluation_timestamp=now_str,
            )
        return None

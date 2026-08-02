"""
PORCE diagnostic rules for Module 10 Creator Style Learning (RULE-STYLE-01 to RULE-STYLE-03).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class StyleViolationRule(IDiagnosticRule):
    """
    RULE-STYLE-01: Flags when shipped candidate style similarity falls below threshold
    for an established creator profile without an identified intentional style drift.
    """

    rule_id = "RULE-STYLE-01"
    rule_name = "Style Violation Detection"
    category = "creator_style"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        gen_trace = context.generation_trace if context else None
        if not gen_trace:
            return None

        established = getattr(gen_trace, "style_profile_established", None)
        sim = getattr(gen_trace, "style_embedding_similarity", None)
        drift = getattr(gen_trace, "drift_detected", False)

        if established is True and sim is not None and sim < 0.75 and not drift:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="WARNING",
                confidence=0.9,
                affected_module="style_similarity",
                root_cause=f"Candidate thumbnail style similarity ({sim:.2f}) fell below threshold (0.75) for established creator profile without style drift.",
                recommended_action="Increase style guidance prompt weight or adjust candidate ranking style bonus.",
                supporting_facts=[f"style_embedding_similarity={sim}", f"drift_detected={drift}"],
                evaluation_timestamp=now_str,
            )
        return None


class BrandingInconsistencyRule(IDiagnosticRule):
    """
    RULE-STYLE-02: Flags when branding constraints exist but identity/face or branding elements
    were degraded or unpreserved.
    """

    rule_id = "RULE-STYLE-02"
    rule_name = "Branding Inconsistency Detection"
    category = "creator_style"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        gen_trace = context.generation_trace if context else None
        if not gen_trace:
            return None

        selection_agreed = getattr(facts, "selection_agreed", None)
        over_edited = getattr(facts, "over_edited", False)
        ipadapter_enabled = getattr(facts, "ipadapter_enabled", True)
        established = getattr(gen_trace, "style_profile_established", False)

        branding_failed = (selection_agreed is False) or (ipadapter_enabled is False and established) or (over_edited and established)

        if branding_failed:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="WARNING",
                confidence=0.85,
                affected_module="branding_preservation",
                root_cause="Branding or creator identity match failed during candidate generation.",
                recommended_action="Ensure face conditioning IP-Adapter/ControlNet models are enabled with appropriate weights.",
                supporting_facts=[f"selection_agreed={selection_agreed}", f"ipadapter_enabled={ipadapter_enabled}"],
                evaluation_timestamp=now_str,
            )
        return None



class IdentityLossWithoutDriftRule(IDiagnosticRule):
    """
    RULE-STYLE-03: Flags identity loss (very low similarity score < 0.60) when no intentional drift
    was detected on an established creator profile. Distinguishes intentional style change from defect.
    """

    rule_id = "RULE-STYLE-03"
    rule_name = "Identity Loss Without Drift Detection"
    category = "creator_style"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        gen_trace = context.generation_trace if context else None
        if not gen_trace:
            return None

        established = getattr(gen_trace, "style_profile_established", None)
        sim = getattr(gen_trace, "style_embedding_similarity", None)
        drift = getattr(gen_trace, "drift_detected", False)

        if established is True and sim is not None and sim < 0.60 and not drift:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="FAIL",
                confidence=0.95,
                affected_module="style_learning_engine",
                root_cause=f"Severe creator identity loss detected (similarity={sim:.2f} < 0.60) on established profile without style drift.",
                recommended_action="Reject candidate thumbnail and trigger style-aware re-generation with creator style guidance enabled.",
                supporting_facts=[f"style_embedding_similarity={sim}", f"drift_detected={drift}"],
                evaluation_timestamp=now_str,
            )
        return None

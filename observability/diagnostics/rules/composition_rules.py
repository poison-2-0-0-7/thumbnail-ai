"""
observability/diagnostics/rules/composition_rules.py
======================================================

Diagnostic rules for composition workspace and visual structure observations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class CompositionMismatchRule(IDiagnosticRule):
    """
    RULE-CMP-01: Verifies whether composition workspace was generated when asset composer / thumbnail planner ran.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-CMP-01"

    @property
    def rule_name(self) -> str:
        return "Composition Workspace Availability Check"

    @property
    def category(self) -> str:
        return "composition"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        m10_status = facts.module_completion_status.get("module10")
        if m10_status == "success" and not facts.has_composition_workspace:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="WARNING",
                confidence=1.0,
                affected_module="module10",
                root_cause="Module 10 reported successful completion, but no CompositionWorkspace artifact was found in data/composition_workspaces/.",
                recommended_action="Inspect AssetComposer output persistence and workspace serialization.",
                supporting_facts=["module10_status=success", "has_composition_workspace=False"],
                evaluation_timestamp=now_str,
            )
        return None


class MaskOverlapProblemRule(IDiagnosticRule):
    """
    RULE-CMP-02: Checks if layer mask count is high or mask overlaps are detected.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-CMP-02"

    @property
    def rule_name(self) -> str:
        return "Layer Mask Density Check"

    @property
    def category(self) -> str:
        return "composition"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        # High mask count observation
        if facts.mask_count > 5:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="WARNING",
                confidence=0.8,
                affected_module="module10",
                root_cause=f"High layer mask count detected ({facts.mask_count} masks). Excessive layers increase potential mask collision.",
                recommended_action="Review Module 10 mask manager composition boundaries and consolidate overlapping layer targets.",
                supporting_facts=[f"mask_count={facts.mask_count}"],
                evaluation_timestamp=now_str,
            )
        return None


class IdentityDriftRule(IDiagnosticRule):
    """
    RULE-CMP-03: Checks for face identity drift observation markers.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-CMP-03"

    @property
    def rule_name(self) -> str:
        return "Creator Identity Drift Check"

    @property
    def category(self) -> str:
        return "composition"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        # If context contains log errors or candidate identity failure markers
        if context and context.pipeline_trace:
            m7_entries = [m for m in context.pipeline_trace.modules if m.module == "module7"]
            for entry in m7_entries:
                for warn in entry.warnings:
                    if "identity" in warn.lower() or "facematch" in warn.lower():
                        return Finding(
                            finding_id=self.rule_id,
                            rule_name=self.rule_name,
                            category=self.category,
                            severity="WARNING",
                            confidence=0.9,
                            affected_module="module7",
                            root_cause=f"Creator face identity score fell below threshold during QA evaluation: {warn}",
                            recommended_action="Increase IPAdapter weight or supply higher resolution creator face reference crops.",
                            supporting_facts=[f"warning={warn}"],
                            evaluation_timestamp=now_str,
                        )
        return None

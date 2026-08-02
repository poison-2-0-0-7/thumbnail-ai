"""
observability/diagnostics/rules/conditioning_rules.py
======================================================

Diagnostic rules for ControlNet, IPAdapter, and conditioning observations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class ControlNetMissingButExpectedRule(IDiagnosticRule):
    """
    RULE-CND-01: Verifies whether ControlNet was disabled when expected by profile.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-CND-01"

    @property
    def rule_name(self) -> str:
        return "ControlNet Attachment Check"

    @property
    def category(self) -> str:
        return "conditioning"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        profile = facts.generation_profile or ""

        # ControlNet is expected for standard profiles unless explicitly Flux / PROFILE_PREMIUM
        if profile and "premium" not in profile.lower() and "flux" not in profile.lower():
            if not facts.controlnet_enabled and facts.controlnet_count == 0:
                return Finding(
                    finding_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity="WARNING",
                    confidence=1.0,
                    affected_module="module7",
                    root_cause=f"Generation profile '{profile}' was selected but ControlNet conditioning was not enabled.",
                    recommended_action="Verify ControlNet configuration flags and model availability in ComfyUI environment.",
                    supporting_facts=[f"generation_profile={profile}", "controlnet_enabled=False"],
                    evaluation_timestamp=now_str,
                )
        return None


class IPAdapterDisabledButReferenceExistsRule(IDiagnosticRule):
    """
    RULE-CND-02: Checks if IPAdapter was disabled despite reference assets being present.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-CND-02"

    @property
    def rule_name(self) -> str:
        return "IPAdapter Reference Usage Check"

    @property
    def category(self) -> str:
        return "conditioning"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        has_references = bool(facts.conditioning_assets or facts.foreground_assets or facts.background_assets)

        if has_references and not facts.ipadapter_enabled and facts.ipadapter_count == 0:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="WARNING",
                confidence=1.0,
                affected_module="module7",
                root_cause="Visual reference assets exist in pipeline context, but IPAdapter conditioning was not enabled.",
                recommended_action="Enable IPAdapter in Module 7 generation parameters to inject visual references.",
                supporting_facts=[
                    f"conditioning_assets_count={len(facts.conditioning_assets)}",
                    f"foreground_assets_count={len(facts.foreground_assets)}",
                    "ipadapter_enabled=False",
                ],
                evaluation_timestamp=now_str,
            )
        return None


class ConditioningFailureRule(IDiagnosticRule):
    """
    RULE-CND-03: Checks for explicit errors during conditioning attachment.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-CND-03"

    @property
    def rule_name(self) -> str:
        return "Conditioning Attachment Error Check"

    @property
    def category(self) -> str:
        return "conditioning"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        # Inspect pipeline_trace log errors if context is available
        if context and context.pipeline_trace:
            m7_entries = [m for m in context.pipeline_trace.modules if m.module == "module7"]
            for entry in m7_entries:
                for err in entry.errors:
                    if "fragment" in err.lower() or "controlnet" in err.lower() or "conditioning" in err.lower():
                        return Finding(
                            finding_id=self.rule_id,
                            rule_name=self.rule_name,
                            category=self.category,
                            severity="FAIL",
                            confidence=1.0,
                            affected_module="module7",
                            root_cause=f"Conditioning attachment failed during generation assembly: {err}",
                            recommended_action="Inspect Module 7 workflow graph assembler logs and node fragment syntax.",
                            supporting_facts=[f"error={err}"],
                            evaluation_timestamp=now_str,
                        )
        return None

"""
observability/diagnostics/rules/controlnet_capability_rules.py
================================================================

Diagnostic rules for ControlNet capability resolution observations (PORCE).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class ControlNetCapabilityResolutionRule(IDiagnosticRule):
    """
    RULE-EDIT-04: ControlNet Capability Resolution Integrity Diagnostic Rule.

    Evaluates whether ControlNet capability resolution succeeded cleanly (legacy exact match),
    used a fallback model (pattern match), or failed to resolve (unresolved).
    """

    @property
    def rule_id(self) -> str:
        return "RULE-EDIT-04"

    @property
    def rule_name(self) -> str:
        return "ControlNet Capability Resolution Integrity"

    @property
    def category(self) -> str:
        return "conditioning"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        gen_trace = context.generation_trace if context else None
        if not gen_trace or not gen_trace.fragments_attached:
            return None

        controlnet_frags = [
            f for f in gen_trace.fragments_attached
            if (f.fragment_name and f.fragment_name.startswith("controlnet_")) or f.requested_capability
        ]

        if not controlnet_frags:
            return None

        # 1. Unresolved -> FAIL
        unresolved = [f for f in controlnet_frags if getattr(f, "resolution_source", None) == "unresolved"]
        if unresolved:
            names = [f.fragment_name for f in unresolved]
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="FAIL",
                confidence=1.0,
                affected_module="module7_capability_resolution",
                root_cause=f"Generation proceeded with unresolved ControlNet fragment(s): {names}.",
                recommended_action=(
                    "Verify startup capability validation and install required ControlNet models "
                    "into ComfyUI's models/controlnet directory."
                ),
                supporting_facts=[f"unresolved_fragments={names}"],
                evaluation_timestamp=now_str,
            )

        # 2. Fallback pattern match -> WARNING
        pattern_matches = [f for f in controlnet_frags if getattr(f, "resolution_source", None) == "pattern_match"]
        if pattern_matches:
            matches_info = [
                f"{f.fragment_name} -> {getattr(f, 'resolved_model', 'unknown')}"
                for f in pattern_matches
            ]
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="WARNING",
                confidence=1.0,
                affected_module="module7_capability_resolution",
                root_cause=f"Generation used fallback ControlNet model(s): {matches_info}.",
                recommended_action="Verify output visual quality if output differs from standard official reference model.",
                supporting_facts=[f"pattern_matched_fragments={matches_info}"],
                evaluation_timestamp=now_str,
            )

        # 3. Legacy exact match -> INFO
        legacy_matches = [f for f in controlnet_frags if getattr(f, "resolution_source", None) == "legacy_exact_match"]
        if legacy_matches:
            matches_info = [
                f"{f.fragment_name} -> {getattr(f, 'resolved_model', 'unknown')}"
                for f in legacy_matches
            ]
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="INFO",
                confidence=1.0,
                affected_module="module7_capability_resolution",
                root_cause=f"Generation resolved legacy exact match ControlNet model(s): {matches_info}.",
                recommended_action="No action required. Exact match model verified.",
                supporting_facts=[f"legacy_matched_fragments={matches_info}"],
                evaluation_timestamp=now_str,
            )

        return None

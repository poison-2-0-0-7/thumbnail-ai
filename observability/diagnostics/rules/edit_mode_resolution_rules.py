"""
observability/diagnostics/rules/edit_mode_resolution_rules.py
================================================================

Diagnostic rules for edit mode resolution and profile reachability observations (PORCE).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from modules.config import (
    MODULE7_EDIT_CAPABLE_PROFILES,
    MODULE7_GENERATION_PROFILES,
    MODULE7_PROFILE_PREFERENCE,
)
from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class EditCapabilityReachabilityRule(IDiagnosticRule):
    """
    RULE-EDIT-02: Edit Capability Reachability Diagnostic Rule.

    Evaluates whether configured edit-capable profiles (profiles with edit_mode_default='staged_edit')
    are reachable via MODULE7_PROFILE_PREFERENCE for automatic profile selection.

    If edit-capable profiles are configured but omitted from preference ordering,
    edit_mode='auto' will silently resolve to legacy_txt2img across all VRAM tiers.
    """

    def __init__(
        self,
        profiles: Optional[dict[str, Any]] = None,
        preference: Optional[tuple[str, ...]] = None,
    ) -> None:
        self._profiles = profiles
        self._preference = preference

    @property
    def rule_id(self) -> str:
        return "RULE-EDIT-02"

    @property
    def rule_name(self) -> str:
        return "Edit Capability Reachability"

    @property
    def category(self) -> str:
        return "conditioning"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        profiles = self._profiles if self._profiles is not None else MODULE7_GENERATION_PROFILES
        preference = self._preference if self._preference is not None else MODULE7_PROFILE_PREFERENCE

        # Identify edit-capable profiles from configured profiles
        edit_capable = {
            name
            for name, prof in profiles.items()
            if getattr(prof, "edit_mode_default", None) == "staged_edit"
        }

        # Check reachability (intersection of preference tuple and edit_capable set)
        if edit_capable and not (set(preference) & edit_capable):
            excluded = sorted(edit_capable)
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="FAIL",
                confidence=1.0,
                affected_module="module7_profile_selection",
                root_cause=(
                    f"Configured edit-capable profile(s) {excluded} are excluded from "
                    f"MODULE7_PROFILE_PREFERENCE {list(preference)}, causing edit_mode='auto' "
                    "to resolve to legacy_txt2img."
                ),
                recommended_action=(
                    "Include edit-capable profile(s) (e.g. PROFILE_STANDARD_EDIT) in "
                    "MODULE7_PROFILE_PREFERENCE so VRAM-based profile auto-selection can resolve staged_edit."
                ),
                supporting_facts=[
                    f"configured_edit_capable_profiles={excluded}",
                    f"profile_preference={list(preference)}",
                ],
                evaluation_timestamp=now_str,
            )
        return None

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


class StagedEditDenoiseStrengthRule(IDiagnosticRule):
    """
    RULE-EDIT-03: Staged Edit Denoise Strength Diagnostic Rule.

    Evaluates whether a generation configured/executed as 'staged_edit' has a KSampler
    denoise strength that allows conditioning on the source image (denoise < threshold, default 0.95).

    If denoise >= 0.95 (e.g. denoise=1.0), the source image conditioning is fully overwritten
    with pure noise during sampling, causing the renderer to behave like a text-to-image generator.
    """

    def __init__(self, denoise_threshold: float = 0.95) -> None:
        self.denoise_threshold = denoise_threshold

    @property
    def rule_id(self) -> str:
        return "RULE-EDIT-03"

    @property
    def rule_name(self) -> str:
        return "Staged Edit Denoise Strength"

    @property
    def category(self) -> str:
        return "conditioning"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        is_staged_edit = facts.edit_mode == "staged_edit" or (
            bool(facts.workflow_selected and facts.workflow_selected.endswith("_edit"))
        )

        if is_staged_edit and facts.denoise is not None and facts.denoise >= self.denoise_threshold:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="FAIL",
                confidence=1.0,
                affected_module="module7_render_execution",
                root_cause=(
                    f"Generation executed in staged_edit mode (workflow='{facts.workflow_selected}'), "
                    f"but KSampler denoise strength is {facts.denoise:.2f} (>= {self.denoise_threshold}), "
                    "which fully overwrites source image conditioning with pure noise."
                ),
                recommended_action=(
                    "Configure template slot '{{denoise_strength}}' in the edit workflow KSampler node "
                    "with partial denoise (e.g. 0.75) so source image conditioning is preserved during sampling."
                ),
                supporting_facts=[
                    f"edit_mode={facts.edit_mode}",
                    f"workflow_selected={facts.workflow_selected}",
                    f"denoise={facts.denoise}",
                    f"denoise_threshold={self.denoise_threshold}",
                ],
                evaluation_timestamp=now_str,
            )
        return None


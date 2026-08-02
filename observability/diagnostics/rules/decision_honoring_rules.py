"""
observability/diagnostics/rules/decision_honoring_rules.py
============================================================

Diagnostic rules for checking whether Module 9/10 decisions and edit plans were honored.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class EditMaskIgnoredRule(IDiagnosticRule):
    """
    RULE-DEC-01: Checks whether edit masks provided by Module 10 were ignored by renderer.
    On current txt2img renderer, edit masks are present in workspace but no inpainting fragment is attached.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-DEC-01"

    @property
    def rule_name(self) -> str:
        return "Edit Mask Inpainting Check"

    @property
    def category(self) -> str:
        return "decision_honoring"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        if facts.mask_count > 0 and len(facts.edit_mask_paths) == 0:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="INFO",
                confidence=1.0,
                affected_module="module7",
                root_cause="Composition workspace generated sampling masks, but renderer version does not attach inpainting fragments.",
                recommended_action="Upgrade to Module 7 V2 Editing Engine to support masked regional inpainting.",
                supporting_facts=[f"mask_count={facts.mask_count}", "edit_mask_paths_attached=0"],
                evaluation_timestamp=now_str,
            )
        return None


class RendererIgnoredEditPlanRule(IDiagnosticRule):
    """
    RULE-DEC-04 / RULE-DEC-02: Checks if renderer ignored the Module 10.5 / Module 9 edit plan.
    Detects cases such as BackgroundCompositor no-op or generation plan strategy missing from attached fragments.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-DEC-04"

    @property
    def rule_name(self) -> str:
        return "Renderer Edit Plan Honoring Check"

    @property
    def category(self) -> str:
        return "decision_honoring"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        # Check if generation plan reference exists but zero fragments were attached
        if facts.generation_plan_reference and facts.attached_fragment_count == 0:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="FAIL",
                confidence=1.0,
                affected_module="module7",
                root_cause="Generation plan was referenced by Module 10.5, but Module 7 attached 0 workflow graph fragments during assembly.",
                recommended_action="Verify WorkflowGraphAssembler and CandidateStrategyPlanner fragment mapping in image_generator.py.",
                supporting_facts=[
                    f"generation_plan_reference={facts.generation_plan_reference}",
                    "attached_fragment_count=0",
                ],
                evaluation_timestamp=now_str,
            )
        return None


class BackgroundRegeneratedUnnecessarilyRule(IDiagnosticRule):
    """
    RULE-DEC-03: Checks if background decision was KEEP, but unmasked txt2img regenerated entire background.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-DEC-03"

    @property
    def rule_name(self) -> str:
        return "Unnecessary Background Regeneration Check"

    @property
    def category(self) -> str:
        return "decision_honoring"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        if facts.decision_engine_enabled and facts.source_thumbnail_exists and facts.edit_mode == "txt2img":
            if facts.denoise == 1.0 and facts.latent_initialization_mode == "EmptyLatentImage":
                return Finding(
                    finding_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity="WARNING",
                    confidence=0.9,
                    affected_module="module7",
                    root_cause="Decision Engine requested background preservation/keep, but txt2img full-denoise pass regenerated the entire image.",
                    recommended_action="Use regional compositing or inpainting latent mode to preserve existing background pixels.",
                    supporting_facts=["decision_engine_enabled=True", "edit_mode=txt2img", "denoise=1.0"],
                    evaluation_timestamp=now_str,
                )
        return None

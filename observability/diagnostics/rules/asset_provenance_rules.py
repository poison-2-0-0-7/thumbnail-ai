"""
observability/diagnostics/rules/asset_provenance_rules.py
==========================================================

Diagnostic rules for asset extraction and asset provenance observations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, RuleContext
from observability.facts.models import TraceFacts


class AssetExtractionMissingRule(IDiagnosticRule):
    """
    RULE-AST-01: Verifies whether Module 8 asset extraction was enabled but output manifest is missing.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-AST-01"

    @property
    def rule_name(self) -> str:
        return "Asset Extraction Manifest Check"

    @property
    def category(self) -> str:
        return "asset_provenance"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()

        if facts.asset_extraction_enabled:
            m8_status = facts.module_completion_status.get("module8")
            m8_manifest_exists = facts.artifact_availability.get("module8_asset_manifest", False) or facts.artifact_availability.get("module8_manifest", False)

            if m8_status in ("error", "not_run") or (not m8_manifest_exists and "module8" not in facts.persisted_outputs):
                return Finding(
                    finding_id=self.rule_id,
                    rule_name=self.rule_name,
                    category=self.category,
                    severity="FAIL",
                    confidence=1.0,
                    affected_module="module8",
                    root_cause="ASSET_EXTRACTION_ENABLED is True, but Module 8 failed or did not produce an asset extraction manifest.",
                    recommended_action="Check Module 8 logs for extraction wrapper errors (BiRefNet, TEED, SAM2) and verify vision stack dependencies.",
                    supporting_facts=["asset_extraction_enabled=True", f"module8_status={m8_status}"],
                    evaluation_timestamp=now_str,
                )
        return None


class ObjectMappingIncorrectRule(IDiagnosticRule):
    """
    RULE-AST-02: Checks whether referenced foreground/background asset paths exist in artifact availability.
    """

    @property
    def rule_id(self) -> str:
        return "RULE-AST-02"

    @property
    def rule_name(self) -> str:
        return "Asset File Reference Mapping Check"

    @property
    def category(self) -> str:
        return "asset_provenance"

    def check(self, facts: TraceFacts, context: Optional[RuleContext] = None) -> Optional[Finding]:
        now_str = datetime.now(timezone.utc).isoformat()
        missing_assets: list[str] = []

        all_assets = facts.conditioning_assets + facts.foreground_assets + facts.background_assets
        for asset_path in all_assets:
            if not facts.artifact_availability.get(asset_path, True):
                missing_assets.append(asset_path)

        if missing_assets:
            return Finding(
                finding_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity="WARNING",
                confidence=1.0,
                affected_module="module8",
                root_cause=f"Conditioning or composition refers to {len(missing_assets)} asset file(s) that do not exist on disk.",
                recommended_action="Verify asset extraction output directory and check file paths in CompositionWorkspace.",
                supporting_facts=[f"missing_asset_count={len(missing_assets)}", f"missing_sample={missing_assets[0]}"],
                evaluation_timestamp=now_str,
            )
        return None

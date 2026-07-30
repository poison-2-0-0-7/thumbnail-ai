"""
manifest_assembler.py
===================

Assembles final DecisionManifest from resolved decisions and validation report.
Implements IManifestAssembler.
"""

from datetime import datetime, timezone
from typing import Any

from modules.decision_components.confidence import calculate_overall_confidence
from modules.decision_components.interfaces import IManifestAssembler
from modules.models import (
    DecisionAction,
    DecisionManifest,
    DecisionManifestStatus,
    ResolvedDecision,
)


class ManifestAssembler(IManifestAssembler):
    """Assembles DecisionManifest from resolved decisions and validation report."""

    def build(
        self,
        video_id: str,
        source_image_path: str,
        source_image_hash: str,
        decisions: list[ResolvedDecision],
        validation_report: dict[str, Any],
        duration_seconds: float,
    ) -> DecisionManifest:
        """Assemble complete DecisionManifest instance."""
        keep_count = sum(1 for d in decisions if d.action == DecisionAction.KEEP)
        remove_count = sum(1 for d in decisions if d.action == DecisionAction.REMOVE)
        replace_count = sum(1 for d in decisions if d.action == DecisionAction.REPLACE)
        enhance_count = sum(1 for d in decisions if d.action == DecisionAction.ENHANCE)
        add_count = sum(1 for d in decisions if d.action == DecisionAction.ADD)

        confidences = [d.confidence for d in decisions]
        soft_warnings = validation_report.get("soft_warnings", [])
        hard_failures = validation_report.get("hard_failures", [])

        overall_conf = calculate_overall_confidence(
            confidences, soft_warning_count=len(soft_warnings)
        )

        if not validation_report.get("valid", True) or hard_failures:
            status = DecisionManifestStatus.ERROR
        elif soft_warnings:
            status = DecisionManifestStatus.PARTIAL
        else:
            status = DecisionManifestStatus.SUCCESS

        decided_at = datetime.now(timezone.utc).isoformat()

        return DecisionManifest(
            video_id=video_id,
            source_generated_image_path=source_image_path,
            source_generated_image_hash=source_image_hash,
            decisions=decisions,
            keep_count=keep_count,
            remove_count=remove_count,
            replace_count=replace_count,
            enhance_count=enhance_count,
            add_count=add_count,
            overall_confidence=overall_conf,
            conflicts_resolved=sum(len(d.superseded_candidate_ids) for d in decisions),
            status=status,
            partial_failure_reasons=soft_warnings + hard_failures,
            total_duration_seconds=round(duration_seconds, 4),
            decided_at=decided_at,
        )

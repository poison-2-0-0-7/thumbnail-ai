"""
test_manifest_assembler.py
==========================

Unit tests for ManifestAssembler (Phase 5).
"""

import pytest

from modules.decision_components.manifest_assembler import ManifestAssembler
from modules.models import DecisionAction, DecisionManifestStatus, DecisionSource, ResolvedDecision, TargetElement


def test_manifest_assembler_build():
    assembler = ManifestAssembler()

    d_keep = ResolvedDecision(
        decision_id="d1",
        target=TargetElement(element_id="e1", element_type="face", label="face"),
        action=DecisionAction.KEEP,
        confidence=0.90,
        source=DecisionSource.RULE,
        rationale="Keep face",
        priority_rank=1,
    )
    d_add = ResolvedDecision(
        decision_id="d2",
        target=TargetElement(element_id="e2", element_type="text", label="text"),
        action=DecisionAction.ADD,
        confidence=0.80,
        source=DecisionSource.RULE,
        rationale="Add text",
        priority_rank=5,
    )

    validation_report = {"valid": True, "hard_failures": [], "soft_warnings": []}

    manifest = assembler.build(
        video_id="v_100",
        source_image_path="gen.png",
        source_image_hash="a" * 64,
        decisions=[d_keep, d_add],
        validation_report=validation_report,
        duration_seconds=1.23,
    )

    assert manifest.video_id == "v_100"
    assert manifest.keep_count == 1
    assert manifest.add_count == 1
    assert manifest.remove_count == 0
    assert manifest.status == DecisionManifestStatus.SUCCESS
    assert manifest.overall_confidence > 0.8

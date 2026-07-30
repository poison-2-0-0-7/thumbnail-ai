"""
test_validator.py
=================

Unit tests for DecisionValidator (Phase 5).
"""

import pytest

from modules.decision_components.validator import DecisionValidator
from modules.models import BoundingBox, DecisionAction, DecisionSource, ResolvedDecision, TargetElement


def test_validator_clean_pass():
    validator = DecisionValidator()

    resolved = ResolvedDecision(
        decision_id="d1",
        target=TargetElement(element_id="e1", element_type="face", label="face"),
        action=DecisionAction.KEEP,
        confidence=0.90,
        source=DecisionSource.RULE,
        rationale="Keep face",
        priority_rank=1,
    )

    report = validator.validate([resolved])
    assert report["valid"] is True
    assert len(report["hard_failures"]) == 0


def test_validator_hard_failure_invalid_bbox_geometry():
    validator = DecisionValidator()

    bad_target = TargetElement(
        element_id="e1",
        element_type="face",
        label="face",
        bbox=BoundingBox(x_min=0.8, y_min=0.2, x_max=0.2, y_max=0.8),  # x_min > x_max
    )
    resolved = ResolvedDecision(
        decision_id="d1",
        target=bad_target,
        action=DecisionAction.KEEP,
        confidence=0.90,
        source=DecisionSource.RULE,
        rationale="Bad bbox geometry",
        priority_rank=1,
    )

    report = validator.validate([resolved])
    assert report["valid"] is False
    assert len(report["hard_failures"]) == 1

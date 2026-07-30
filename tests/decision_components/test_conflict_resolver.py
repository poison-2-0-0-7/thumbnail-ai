"""
test_conflict_resolver.py
==========================

Unit tests for ConflictResolver (Phase 3).
"""

import pytest

from modules.decision_components.conflict_resolver import ConflictResolver
from modules.models import BoundingBox, CandidateDecision, DecisionAction, DecisionSource, TargetElement


def test_conflict_resolver_priority_ordering():
    resolver = ConflictResolver()

    target = TargetElement(element_id="elem_1", element_type="object", label="car")

    c_replace = CandidateDecision(
        candidate_id="c_replace",
        target=target,
        action=DecisionAction.REPLACE,
        confidence=0.90,
        source=DecisionSource.RULE,
        rationale="Replace car",
    )
    c_keep = CandidateDecision(
        candidate_id="c_keep",
        target=target,
        action=DecisionAction.KEEP,
        confidence=0.80,
        source=DecisionSource.RULE,
        rationale="Keep car",
    )

    resolved = resolver.resolve([c_replace, c_keep])

    assert len(resolved) == 1
    winner = resolved[0]
    # KEEP outranks REPLACE per DECISION_PRIORITY_ORDER
    assert winner.action == DecisionAction.KEEP
    assert winner.superseded_candidate_ids == ["c_replace"]


def test_conflict_resolver_add_deduplication():
    resolver = ConflictResolver()

    target1 = TargetElement(
        element_id="add_arrow_1",
        element_type="effect",
        label="arrow",
        bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3),
    )
    target2 = TargetElement(
        element_id="add_arrow_2",
        element_type="effect",
        label="arrow",
        bbox=BoundingBox(x_min=0.11, y_min=0.11, x_max=0.31, y_max=0.31),  # > 0.7 IoU overlap
    )

    c1 = CandidateDecision(
        candidate_id="c1",
        target=target1,
        action=DecisionAction.ADD,
        confidence=0.85,
        source=DecisionSource.RULE,
        rationale="Add arrow 1",
    )
    c2 = CandidateDecision(
        candidate_id="c2",
        target=target2,
        action=DecisionAction.ADD,
        confidence=0.80,
        source=DecisionSource.RULE,
        rationale="Add arrow 2",
    )

    resolved = resolver.resolve([c1, c2])

    assert len(resolved) == 1
    assert resolved[0].target.element_id == "add_arrow_1"

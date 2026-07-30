"""
test_ambiguity_router.py
========================

Unit tests for AmbiguityRouter (Phase 3).
"""

import pytest

from modules.decision_components.ambiguity_router import AmbiguityRouter
from modules.models import CandidateDecision, DecisionAction, DecisionSource, TargetElement


def test_ambiguity_router_selection():
    router = AmbiguityRouter(threshold=0.65)

    target_high = TargetElement(element_id="elem_1", element_type="face", label="face")
    target_low = TargetElement(element_id="elem_2", element_type="object", label="object")
    target_unplaced_add = TargetElement(element_id="elem_add", element_type="effect", label="arrow", bbox=None)

    c1 = CandidateDecision(
        candidate_id="c1",
        target=target_high,
        action=DecisionAction.KEEP,
        confidence=0.90,
        source=DecisionSource.RULE,
        rationale="High confidence keep",
    )
    c2 = CandidateDecision(
        candidate_id="c2",
        target=target_low,
        action=DecisionAction.REPLACE,
        confidence=0.50,  # Below threshold 0.65 -> needs LLM
        source=DecisionSource.RULE,
        rationale="Low confidence replace",
    )
    c3 = CandidateDecision(
        candidate_id="c3",
        target=target_unplaced_add,
        action=DecisionAction.ADD,
        confidence=0.85,
        source=DecisionSource.RULE,
        rationale="Unplaced ADD recommendation -> needs LLM",
    )

    confident, needs_llm = router.select([c1, c2, c3])

    assert len(confident) == 1
    assert confident[0].candidate_id == "c1"

    assert len(needs_llm) == 2
    needs_llm_ids = {c.candidate_id for c in needs_llm}
    assert "c2" in needs_llm_ids
    assert "c3" in needs_llm_ids


def test_ambiguity_router_conflict_trigger():
    router = AmbiguityRouter(threshold=0.65)

    target = TargetElement(element_id="elem_conflict", element_type="object", label="bg")
    c1 = CandidateDecision(
        candidate_id="c1",
        target=target,
        action=DecisionAction.KEEP,
        confidence=0.80,
        source=DecisionSource.RULE,
        rationale="Keep bg",
    )
    c2 = CandidateDecision(
        candidate_id="c2",
        target=target,
        action=DecisionAction.REPLACE,
        confidence=0.80,
        source=DecisionSource.RULE,
        rationale="Replace bg",
    )

    confident, needs_llm = router.select([c1, c2])

    # Both candidates target same element with conflicting actions -> routed to LLM
    assert len(needs_llm) == 2

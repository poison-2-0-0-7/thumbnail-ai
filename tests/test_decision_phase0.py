"""
test_decision_phase0.py
========================

Unit tests for Module 9 Phase 0 (Data Models, Constants, Exception Hierarchy, Interfaces).
"""

import pytest

from modules.config import AMBIGUITY_CONFIDENCE_THRESHOLD, DECISION_PRIORITY_ORDER, MODULE9_LOG_PATH
from modules.decision_exceptions import DecisionEngineError, InputBundleError, OllamaConnectionError
from modules.models import (
    CandidateDecision,
    DecisionAction,
    DecisionManifest,
    DecisionManifestStatus,
    DecisionSource,
    ReasoningTraceEntry,
    ResolvedDecision,
    TargetElement,
)


def test_exception_hierarchy():
    err = OllamaConnectionError("Failed to reach Ollama")
    assert isinstance(err, DecisionEngineError)
    assert isinstance(err, Exception)


def test_models_instantiation_and_validation():
    target = TargetElement(element_id="face_0", element_type="face", label="creator face")

    candidate = CandidateDecision(
        candidate_id="cand_1",
        target=target,
        action=DecisionAction.KEEP,
        confidence=0.95,
        source=DecisionSource.RULE,
        rationale="Matches primary face",
        rule_ids=["rule_keep_primary_face"],
    )
    assert candidate.action == "keep"
    assert candidate.confidence == 0.95

    resolved = ResolvedDecision(
        decision_id="dec_1",
        target=target,
        action=DecisionAction.KEEP,
        confidence=0.95,
        source=DecisionSource.RULE,
        rationale="Matches primary face",
        priority_rank=1,
    )
    assert resolved.priority_rank == 1

    manifest = DecisionManifest(
        video_id="v_123",
        source_generated_image_path="data/generated_thumbnails/v_123.png",
        source_generated_image_hash="a" * 64,
        decisions=[resolved],
        keep_count=1,
        overall_confidence=0.95,
        status=DecisionManifestStatus.SUCCESS,
        decided_at="2026-07-30T00:00:00Z",
    )
    assert manifest.video_id == "v_123"
    assert manifest.keep_count == 1


def test_confidence_validation_out_of_bounds():
    target = TargetElement(element_id="obj_1", element_type="object", label="car")
    with pytest.raises(ValueError):
        CandidateDecision(
            candidate_id="c_bad",
            target=target,
            action=DecisionAction.REMOVE,
            confidence=1.5,
            source=DecisionSource.RULE,
            rationale="Bad confidence score",
        )


def test_empty_video_id_validation():
    with pytest.raises(ValueError):
        DecisionManifest(
            video_id="   ",
            source_generated_image_path="path.png",
            source_generated_image_hash="a" * 64,
            decided_at="2026-07-30T00:00:00Z",
        )


def test_config_constants():
    assert AMBIGUITY_CONFIDENCE_THRESHOLD == 0.65
    assert DECISION_PRIORITY_ORDER[0] == "keep"
    assert str(MODULE9_LOG_PATH).endswith("module9.log")

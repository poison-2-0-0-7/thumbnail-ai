"""Tests for PORCE Creator Style diagnostic rules (RULE-STYLE-01 to RULE-STYLE-03)."""

from __future__ import annotations

import pytest
from observability.diagnostics.models import RuleContext
from observability.diagnostics.rules.creator_style_rules import (
    BrandingInconsistencyRule,
    IdentityLossWithoutDriftRule,
    StyleViolationRule,
)
from observability.facts.models import TraceFacts
from observability.models import GenerationTraceRecord


@pytest.fixture
def empty_facts() -> TraceFacts:
    return TraceFacts(video_id="test_video_123", extracted_at="2026-08-01T00:00:00Z")


def test_rule_style_01_violation(empty_facts):
    rule = StyleViolationRule()
    trace = GenerationTraceRecord(
        video_id="test_video_123",
        style_profile_established=True,
        style_embedding_similarity=0.65,
        drift_detected=False,
    )
    context = RuleContext(facts=empty_facts, generation_trace=trace)
    finding = rule.check(empty_facts, context)

    assert finding is not None
    assert finding.finding_id == "RULE-STYLE-01"
    assert finding.severity == "WARNING"


def test_rule_style_02_branding_inconsistency(empty_facts):
    rule = BrandingInconsistencyRule()
    facts = TraceFacts(
        video_id="test_video_123",
        extracted_at="2026-08-01T00:00:00Z",
        selection_agreed=False,
    )
    trace = GenerationTraceRecord(
        video_id="test_video_123",
        style_profile_established=True,
    )
    context = RuleContext(facts=facts, generation_trace=trace)
    finding = rule.check(facts, context)

    assert finding is not None
    assert finding.finding_id == "RULE-STYLE-02"
    assert finding.severity == "WARNING"


def test_rule_style_03_identity_loss(empty_facts):
    rule = IdentityLossWithoutDriftRule()
    trace = GenerationTraceRecord(
        video_id="test_video_123",
        style_profile_established=True,
        style_embedding_similarity=0.50,
        drift_detected=False,
    )
    context = RuleContext(facts=empty_facts, generation_trace=trace)
    finding = rule.check(empty_facts, context)

    assert finding is not None
    assert finding.finding_id == "RULE-STYLE-03"
    assert finding.severity == "FAIL"

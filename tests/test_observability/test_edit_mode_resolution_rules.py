"""
tests/test_observability/test_edit_mode_resolution_rules.py
============================================================

Unit and integration tests for Phase 4: PORCE RULE-EDIT-02 (EditCapabilityReachabilityRule).
Verifies:
- RULE-EDIT-02 registration in RuleRegistry
- Rule execution against synthetic pre-fix and post-fix configurations
- Evidence collection and finding schema
- RootCauseReport integration
- Golden-file replay of the 9 historical traces
- No duplicate findings and regression safety
"""

from __future__ import annotations

from pathlib import Path
import pytest

from models import GenerationProfile
from observability.diagnostics import (
    FindingCollection,
    RuleContext,
    RuleEngine,
    RuleExecutionEngine,
    RuleRegistry,
)
from observability.diagnostics.rules import (
    EditCapabilityReachabilityRule,
    StagedEditDenoiseStrengthRule,
)
from types import SimpleNamespace
from modules.config import MODULE7_GENERATION_PROFILES
from observability.facts import FactLoader, FactPersistence
from observability.facts.models import TraceFacts
from observability.models import ArtifactIndex, PipelineTrace
from observability.reporting import RootCauseAssembler


@pytest.fixture
def empty_facts() -> TraceFacts:
    return TraceFacts(
        video_id="vid_test_edit_02",
        extracted_at="2026-08-02T12:00:00Z",
    )


@pytest.fixture
def sample_trace() -> PipelineTrace:
    index = ArtifactIndex(video_id="vid_test_edit_02", refs=[], built_at="2026-08-02T12:00:00Z")
    return PipelineTrace(
        video_id="vid_test_edit_02",
        modules=[],
        artifact_index=index,
        generation_trace=None,
        overall_status="success",
        assembled_at="2026-08-02T12:00:00Z",
    )


def test_rule_edit_02_registry_registration() -> None:
    """Verify RULE-EDIT-02 is properly registered in RuleRegistry defaults."""
    registry = RuleRegistry()
    rule = registry.get_rule("RULE-EDIT-02")
    assert rule is not None
    assert isinstance(rule, EditCapabilityReachabilityRule)
    assert rule.rule_id == "RULE-EDIT-02"
    assert rule.rule_name == "Edit Capability Reachability"
    assert rule.category == "conditioning"


def test_rule_edit_02_prefix_config_fails() -> None:
    """Verify RULE-EDIT-02 produces a FAIL finding when an edit profile is excluded from preference."""
    pre_fix_profiles = {
        "PROFILE_STANDARD": MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"],
        "PROFILE_STANDARD_EDIT": MODULE7_GENERATION_PROFILES["PROFILE_STANDARD_EDIT"],
        "PROFILE_FAST": MODULE7_GENERATION_PROFILES["PROFILE_FAST"],
    }
    pre_fix_preference = ("PROFILE_STANDARD", "PROFILE_FAST")

    rule = EditCapabilityReachabilityRule(
        profiles=pre_fix_profiles, preference=pre_fix_preference
    )
    facts = TraceFacts(video_id="vid_prefix", extracted_at="2026-08-02T12:00:00Z")
    finding = rule.check(facts)

    assert finding is not None
    assert finding.finding_id == "RULE-EDIT-02"
    assert finding.rule_name == "Edit Capability Reachability"
    assert finding.category == "conditioning"
    assert finding.severity == "FAIL"
    assert finding.confidence == 1.0
    assert finding.affected_module == "module7_profile_selection"
    assert "PROFILE_STANDARD_EDIT" in finding.root_cause
    assert "MODULE7_PROFILE_PREFERENCE" in finding.recommended_action
    assert any("PROFILE_STANDARD_EDIT" in fact for fact in finding.supporting_facts)


def test_rule_edit_02_postfix_config_passes() -> None:
    """Verify RULE-EDIT-02 passes (returns None) when an edit profile is included in preference."""
    post_fix_profiles = {
        "PROFILE_STANDARD": MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"],
        "PROFILE_STANDARD_EDIT": MODULE7_GENERATION_PROFILES["PROFILE_STANDARD_EDIT"],
        "PROFILE_FAST": MODULE7_GENERATION_PROFILES["PROFILE_FAST"],
    }
    post_fix_preference = ("PROFILE_STANDARD_EDIT", "PROFILE_STANDARD", "PROFILE_FAST")

    rule = EditCapabilityReachabilityRule(
        profiles=post_fix_profiles, preference=post_fix_preference
    )
    facts = TraceFacts(video_id="vid_postfix", extracted_at="2026-08-02T12:00:00Z")
    finding = rule.check(facts)

    assert finding is None


def test_rule_edit_02_no_edit_capable_profiles_passes() -> None:
    """Verify RULE-EDIT-02 passes if no profiles declare edit_mode_default='staged_edit'."""
    no_edit_profiles = {
        "PROFILE_STANDARD": MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"],
        "PROFILE_FAST": MODULE7_GENERATION_PROFILES["PROFILE_FAST"],
    }
    preference = ("PROFILE_STANDARD", "PROFILE_FAST")

    rule = EditCapabilityReachabilityRule(profiles=no_edit_profiles, preference=preference)
    facts = TraceFacts(video_id="vid_no_edit", extracted_at="2026-08-02T12:00:00Z")
    assert rule.check(facts) is None


def test_rule_engine_execution_and_report_integration(
    empty_facts: TraceFacts, sample_trace: PipelineTrace
) -> None:
    """Verify RULE-EDIT-02 execution via RuleEngine and integration into RootCauseReport."""
    # Under current live config (post-fix), RULE-EDIT-02 passes
    engine = RuleEngine()
    collection = engine.evaluate(empty_facts)
    rule_02_findings = [f for f in collection.findings if f.finding_id == "RULE-EDIT-02"]
    assert len(rule_02_findings) == 0  # Passed, so no FAIL/WARNING finding

    # Now evaluate under custom registry with pre-fix synthetic config rule
    pre_fix_rule = EditCapabilityReachabilityRule(
        profiles={
            "PROFILE_STANDARD_EDIT": MODULE7_GENERATION_PROFILES["PROFILE_STANDARD_EDIT"]
        },
        preference=("PROFILE_STANDARD",),
    )
    custom_registry = RuleRegistry(load_defaults=False)
    custom_registry.register_rule(pre_fix_rule)

    custom_engine = RuleEngine(registry=custom_registry)
    fail_collection = custom_engine.evaluate(empty_facts)

    assert fail_collection.fail_count == 1
    assert len(fail_collection.findings) == 1
    assert fail_collection.findings[0].finding_id == "RULE-EDIT-02"

    # Verify RootCauseReport integration
    assembler = RootCauseAssembler()
    report = assembler.assemble(
        video_id=empty_facts.video_id,
        pipeline_trace=sample_trace,
        finding_collection=fail_collection,
    )

    assert report.fail_count == 1
    assert report.status == "error"
    assert len(report.top_root_causes) >= 1
    assert "PROFILE_STANDARD_EDIT" in report.top_root_causes[0]


def test_golden_file_replay_historical_traces() -> None:
    """Replay the nine historical traces to confirm RULE-EDIT-02 behavior."""
    historical_video_ids = [
        "0EyJaqz8xyw",
        "2zC2viCb_Ck",
        "Ey_SfwEZPR0",
        "I-bnBd5lCew",
        "O0Y-oLarao4",
        "abcdEFGH123",
        "eWzsmjA1vOo",
        "k9Tdx6ddOPQ",
        "vIWkN-2J0ic",
    ]

    traces_dir = Path(__file__).resolve().parent.parent.parent / "data" / "observability" / "facts"

    # Pre-fix rule instance (Cause A configuration defect)
    pre_fix_rule = EditCapabilityReachabilityRule(
        profiles={
            "PROFILE_STANDARD_EDIT": MODULE7_GENERATION_PROFILES["PROFILE_STANDARD_EDIT"]
        },
        preference=("PROFILE_PREMIUM", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM"),
    )

    # Post-fix rule instance (Current system config)
    post_fix_rule = EditCapabilityReachabilityRule()

    persistence = FactPersistence(output_dir=traces_dir)
    loader = FactLoader(persistence=persistence)

    for vid in historical_video_ids:
        coll = loader.load_by_video_id(vid)
        assert coll is not None, f"Historical facts collection for {vid} should exist."
        facts = coll.trace_facts
        assert facts is not None, f"Historical trace facts for {vid} should exist."

        # Replay under pre-fix config: must produce FAIL finding for all 9 traces
        pre_finding = pre_fix_rule.check(facts)
        assert pre_finding is not None, f"Trace {vid} should fail under pre-fix config."
        assert pre_finding.finding_id == "RULE-EDIT-02"
        assert pre_finding.severity == "FAIL"
        assert "PROFILE_STANDARD_EDIT" in pre_finding.root_cause

        # Replay under post-fix config: must pass (None) for all 9 traces
        post_finding = post_fix_rule.check(facts)
        assert post_finding is None, f"Trace {vid} should pass under post-fix config."


def test_rule_edit_03_registry_registration() -> None:
    """Verify RULE-EDIT-03 is properly registered in RuleRegistry defaults."""
    registry = RuleRegistry()
    rule = registry.get_rule("RULE-EDIT-03")
    assert rule is not None
    assert isinstance(rule, StagedEditDenoiseStrengthRule)
    assert rule.rule_id == "RULE-EDIT-03"
    assert rule.rule_name == "Staged Edit Denoise Strength"
    assert rule.category == "conditioning"


def test_rule_edit_03_fails_when_denoise_is_high() -> None:
    """Verify RULE-EDIT-03 produces a FAIL finding when staged_edit has denoise >= 0.95."""
    from observability.diagnostics.rules import StagedEditDenoiseStrengthRule
    rule = StagedEditDenoiseStrengthRule()
    facts = TraceFacts(
        video_id="vid_high_denoise",
        extracted_at="2026-08-02T12:00:00Z",
        edit_mode="staged_edit",
        workflow_selected="general_edit.json",
        denoise=1.0,
    )
    finding = rule.check(facts)

    assert finding is not None
    assert finding.finding_id == "RULE-EDIT-03"
    assert finding.severity == "FAIL"
    assert finding.confidence == 1.0
    assert "denoise strength is 1.00" in finding.root_cause


def test_rule_edit_03_passes_when_denoise_is_low() -> None:
    """Verify RULE-EDIT-03 passes (returns None) when staged_edit has denoise = 0.75."""
    from observability.diagnostics.rules import StagedEditDenoiseStrengthRule
    rule = StagedEditDenoiseStrengthRule()
    facts = TraceFacts(
        video_id="vid_low_denoise",
        extracted_at="2026-08-02T12:00:00Z",
        edit_mode="staged_edit",
        workflow_selected="general_edit.json",
        denoise=0.75,
    )
    finding = rule.check(facts)

    assert finding is None


def test_rule_edit_03_passes_for_txt2img() -> None:
    """Verify RULE-EDIT-03 passes when edit_mode is txt2img even with denoise 1.0."""
    from observability.diagnostics.rules import StagedEditDenoiseStrengthRule
    rule = StagedEditDenoiseStrengthRule()
    facts = TraceFacts(
        video_id="vid_txt2img",
        extracted_at="2026-08-02T12:00:00Z",
        edit_mode="txt2img",
        workflow_selected="general.json",
        denoise=1.0,
    )
    finding = rule.check(facts)

    assert finding is None


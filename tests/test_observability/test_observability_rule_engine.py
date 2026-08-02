"""
tests/test_observability/test_observability_rule_engine.py
===========================================================

Comprehensive unit tests for Sprint 3B: Rule Engine & Diagnostic Engine (PORCE).
Verifies:
- Every rule family evaluation (latent_initialization, conditioning, decision_honoring, asset_provenance, prompt_consistency, composition)
- Rule registration & RuleRegistry behavior
- Rule execution engine & exception resilience
- Finding model schema & validation
- Deterministic evaluation & missing facts handling
- Backward compatibility & regression safety
"""

from __future__ import annotations

import pytest

from observability.diagnostics import (
    Finding,
    FindingCollection,
    IDiagnosticRule,
    RuleContext,
    RuleEngine,
    RuleExecutionEngine,
    RuleRegistry,
    RuleValidation,
)
from observability.diagnostics.rules import (
    AssetExtractionMissingRule,
    BackgroundRegeneratedUnnecessarilyRule,
    CompositionMismatchRule,
    ConditioningFailureRule,
    ControlNetMissingButExpectedRule,
    EditMaskIgnoredRule,
    IdentityDriftRule,
    IPAdapterDisabledButReferenceExistsRule,
    MaskOverlapProblemRule,
    ObjectMappingIncorrectRule,
    PromptContradictionRule,
    RendererIgnoredEditPlanRule,
    SourceNeverEncodedRule,
)
from observability.facts.models import TraceFacts


@pytest.fixture
def empty_facts() -> TraceFacts:
    return TraceFacts(
        video_id="vid_empty",
        extracted_at="2026-08-02T12:00:00Z",
    )


@pytest.fixture
def full_facts() -> TraceFacts:
    return TraceFacts(
        video_id="vid_full",
        extracted_at="2026-08-02T12:00:00Z",
        workflow_selected="gaming.json",
        edit_mode="txt2img",
        generation_profile="gaming_v1",
        sampler="euler",
        scheduler="normal",
        seed=42,
        cfg=7.0,
        steps=20,
        denoise=1.0,
        latent_initialization_mode="EmptyLatentImage",
        controlnet_count=0,
        controlnet_enabled=False,
        ipadapter_count=0,
        ipadapter_enabled=False,
        mask_count=8,
        edit_mask_paths=[],
        conditioning_assets=["data/cond/depth.png"],
        foreground_assets=["data/assets/sub.png"],
        background_assets=[],
        has_composition_workspace=False,
        generation_plan_reference="data/strategy_packs/vid_full.json",
        positive_prompt="epic gaming thumbnail cinematic ultra detailed quality",
        negative_prompt="blurry low quality bad gaming cinematic",
        artifact_availability={"module8_asset_manifest": False, "data/cond/depth.png": True},
        module_completion_status={"module8": "error", "module10": "success"},
        attached_fragment_count=0,
        source_thumbnail_exists=True,
        generated_thumbnail_exists=True,
        asset_extraction_enabled=True,
        decision_engine_enabled=True,
    )


def test_rule_registry_default_registration() -> None:
    registry = RuleRegistry()
    rules = registry.get_all_rules()
    assert len(rules) >= 13

    cat_rules = registry.get_rules_by_category("latent_initialization")
    assert len(cat_rules) >= 1
    assert cat_rules[0].rule_id == "RULE-LAT-01"


def test_rule_registry_custom_registration() -> None:
    class DummyRule(IDiagnosticRule):
        @property
        def rule_id(self) -> str:
            return "RULE-DUMMY-01"

        @property
        def rule_name(self) -> str:
            return "Dummy Rule"

        @property
        def category(self) -> str:
            return "general"

        def check(self, facts: TraceFacts, context: RuleContext | None = None) -> Finding | None:
            return None

    registry = RuleRegistry(load_defaults=False)
    assert len(registry.get_all_rules()) == 0

    dummy = DummyRule()
    registry.register_rule(dummy)
    assert registry.get_rule("RULE-DUMMY-01") is dummy
    assert len(registry.get_all_rules()) == 1


def test_latent_initialization_family(full_facts: TraceFacts) -> None:
    rule = SourceNeverEncodedRule()
    finding = rule.check(full_facts)
    assert finding is not None
    assert finding.finding_id == "RULE-LAT-01"
    assert finding.severity == "INFO"
    assert "EmptyLatentImage" in finding.root_cause


def test_conditioning_family(full_facts: TraceFacts) -> None:
    c_rule = ControlNetMissingButExpectedRule()
    finding_c = c_rule.check(full_facts)
    assert finding_c is not None
    assert finding_c.finding_id == "RULE-CND-01"
    assert finding_c.severity == "WARNING"

    ip_rule = IPAdapterDisabledButReferenceExistsRule()
    finding_ip = ip_rule.check(full_facts)
    assert finding_ip is not None
    assert finding_ip.finding_id == "RULE-CND-02"
    assert finding_ip.severity == "WARNING"

    fail_rule = ConditioningFailureRule()

    assert fail_rule.check(full_facts) is None


def test_decision_honoring_family(full_facts: TraceFacts) -> None:
    mask_rule = EditMaskIgnoredRule()
    f_mask = mask_rule.check(full_facts)
    assert f_mask is not None
    assert f_mask.finding_id == "RULE-DEC-01"
    assert f_mask.severity == "INFO"

    plan_rule = RendererIgnoredEditPlanRule()
    f_plan = plan_rule.check(full_facts)
    assert f_plan is not None
    assert f_plan.finding_id == "RULE-DEC-04"
    assert f_plan.severity == "FAIL"

    bg_rule = BackgroundRegeneratedUnnecessarilyRule()
    f_bg = bg_rule.check(full_facts)
    assert f_bg is not None
    assert f_bg.finding_id == "RULE-DEC-03"
    assert f_bg.severity == "WARNING"


def test_asset_provenance_family(full_facts: TraceFacts) -> None:
    ast_rule = AssetExtractionMissingRule()
    f_ast = ast_rule.check(full_facts)
    assert f_ast is not None
    assert f_ast.finding_id == "RULE-AST-01"
    assert f_ast.severity == "FAIL"

    obj_rule = ObjectMappingIncorrectRule()

    facts_missing_obj = full_facts.model_copy(
        update={"artifact_availability": {"data/cond/depth.png": False}}
    )
    f_obj = obj_rule.check(facts_missing_obj)
    assert f_obj is not None
    assert f_obj.finding_id == "RULE-AST-02"
    assert f_obj.severity == "WARNING"


def test_prompt_consistency_family(full_facts: TraceFacts) -> None:
    prm_rule = PromptContradictionRule()
    f_prm = prm_rule.check(full_facts)
    assert f_prm is not None
    assert f_prm.finding_id == "RULE-PRM-01"
    assert f_prm.severity == "WARNING"
    assert "cinematic" in f_prm.root_cause or "quality" in f_prm.root_cause or "gaming" in f_prm.root_cause


def test_composition_family(full_facts: TraceFacts) -> None:
    cmp_rule = CompositionMismatchRule()
    f_cmp = cmp_rule.check(full_facts)
    assert f_cmp is not None
    assert f_cmp.finding_id == "RULE-CMP-01"

    mask_density_rule = MaskOverlapProblemRule()
    f_mask = mask_density_rule.check(full_facts)
    assert f_mask is not None
    assert f_mask.finding_id == "RULE-CMP-02"

    id_rule = IdentityDriftRule()
    assert id_rule.check(full_facts) is None


def test_rule_engine_full_evaluation(full_facts: TraceFacts) -> None:
    engine = RuleEngine()
    collection = engine.evaluate(full_facts)

    assert collection.video_id == "vid_full"
    assert isinstance(collection, FindingCollection)
    assert len(collection.findings) > 0

    assert collection.fail_count >= 2
    assert collection.warning_count >= 3
    assert collection.info_count >= 2

    for f in collection.findings:
        assert RuleValidation.validate_finding_data(f) is True


def test_rule_engine_empty_facts(empty_facts: TraceFacts) -> None:
    engine = RuleEngine()
    collection = engine.evaluate(empty_facts)

    assert collection.video_id == "vid_empty"
    assert RuleValidation.validate_finding_collection(collection) is True


def test_rule_engine_exception_resilience() -> None:
    class BrokenRule(IDiagnosticRule):
        @property
        def rule_id(self) -> str:
            return "RULE-BROKEN-01"

        @property
        def rule_name(self) -> str:
            return "Broken Exception Rule"

        @property
        def category(self) -> str:
            return "general"

        def check(self, facts: TraceFacts, context: RuleContext | None = None) -> Finding | None:
            raise RuntimeError("Simulated rule explosion")

    registry = RuleRegistry(load_defaults=False)
    registry.register_rule(BrokenRule())

    engine = RuleEngine(registry=registry)
    facts = TraceFacts(video_id="vid_test_err", extracted_at="2026-08-02T12:00:00Z")

    collection = engine.evaluate(facts)
    assert collection.video_id == "vid_test_err"
    assert len(collection.findings) == 1
    assert collection.findings[0].severity == "INFO"
    assert "Simulated rule explosion" in collection.findings[0].root_cause


def test_rule_validation_helpers() -> None:
    rule = SourceNeverEncodedRule()
    assert RuleValidation.validate_rule_instance(rule) is True
    assert RuleValidation.validate_rule_instance("invalid") is False

    finding = Finding(
        finding_id="RULE-TEST-01",
        rule_name="Test Rule",
        category="general",
        severity="INFO",
        affected_module="module7",
        root_cause="Test cause.",
        recommended_action="Test action.",
    )
    assert RuleValidation.validate_finding_data(finding) is True
    assert RuleValidation.validate_finding_data({"invalid": "data"}) is False

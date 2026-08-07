"""
test_reasoning_models_and_context.py
====================================

Test suite for models, contracts, decision tree representations,
and ReasoningContext operations in Phase 3.4A.
"""

from __future__ import annotations

import pytest

from thumbnail_intelligence.knowledge_base.models import (
    EvidenceGrade,
    EvidenceReference,
    EvidenceSourceType,
)
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.models import (
    AudienceReasoningOutput,
    BrandReasoningOutput,
    CreatorReasoningOutput,
    DecisionTree,
    DecisionTreeNode,
    NarrativeReasoningOutput,
    PriorityReasoningOutput,
    RankedStrategy,
    ReasonerContract,
    ReasonerType,
    ReasoningRisk,
    ReasoningTraceStep,
    RiskReasoningOutput,
    StrategyRankingOutput,
)


def _sample_evidence_ref(source_id: str = "src_123") -> EvidenceReference:
    return EvidenceReference(
        source_id=source_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.9,
        grade=EvidenceGrade.STRONG,
        claim_summary="High historical CTR uplift with contrasting subject rim lighting",
    )


def test_reasoner_contract_validation():
    """Verify ReasonerContract fields, validation, and defaults."""
    contract = ReasonerContract(
        name="narrative_v1",
        reasoner_type=ReasonerType.NARRATIVE,
        dependencies=["audience_v1"],
        version="1.2.3",
        description="Extracts visual narrative hooks",
        is_mandatory=True,
        timeout_ms=3000.0,
    )
    assert contract.name == "narrative_v1"
    assert contract.reasoner_type == ReasonerType.NARRATIVE
    assert contract.dependencies == ["audience_v1"]
    assert contract.version == "1.2.3"
    assert contract.timeout_ms == 3000.0

    with pytest.raises(ValueError):
        ReasonerContract(name="", reasoner_type=ReasonerType.NARRATIVE)


def test_narrative_output_model():
    """Verify NarrativeReasoningOutput serialization and bounds."""
    ref = _sample_evidence_ref("ev_nar_1")
    output = NarrativeReasoningOutput(
        story_hook="David vs Goliath showdown",
        narrative_angle="Underdog challenger faces giant opponent",
        emotional_tone="Suspenseful curiosity",
        key_visual_metaphors=["scale contrast", "dim background"],
        scene_framing={"composition": "split_screen"},
        evidence_refs=[ref],
        confidence=0.88,
    )
    assert output.story_hook == "David vs Goliath showdown"
    assert len(output.evidence_refs) == 1
    assert output.confidence == 0.88


def test_audience_output_model():
    """Verify AudienceReasoningOutput fields and defaults."""
    ref = _sample_evidence_ref("ev_aud_1")
    output = AudienceReasoningOutput(
        target_audience_segment="Tech-savvy developers",
        curiosity_triggers=["Hidden benchmark results"],
        psychological_hooks=["Loss aversion"],
        cognitive_load_level="low",
        evidence_refs=[ref],
        confidence=0.92,
    )
    assert output.target_audience_segment == "Tech-savvy developers"
    assert output.cognitive_load_level == "low"
    assert len(output.psychological_hooks) == 1


def test_creator_output_model():
    """Verify CreatorReasoningOutput fields and score bounds."""
    output = CreatorReasoningOutput(
        creator_persona="Analytical educator",
        signature_elements=["bold yellow headline", "expressive headshot"],
        style_alignment_score=0.95,
        channel_voice="Authoritative yet accessible",
        confidence=0.90,
    )
    assert output.creator_persona == "Analytical educator"
    assert output.style_alignment_score == 0.95


def test_brand_output_model():
    """Verify BrandReasoningOutput rules and compliance score."""
    output = BrandReasoningOutput(
        color_palette_rules=["#0055FF", "#FFFFFF", "#111111"],
        typography_rules=["Montserrat Bold only"],
        logo_rules=["Top-right corner with 10% padding"],
        prohibited_elements=["Red arrows", "Generic dollar signs"],
        compliance_score=1.0,
        confidence=0.99,
    )
    assert len(output.color_palette_rules) == 3
    assert len(output.prohibited_elements) == 2
    assert output.compliance_score == 1.0


def test_priority_output_model():
    """Verify PriorityReasoningOutput visual hierarchy and allocations."""
    output = PriorityReasoningOutput(
        focal_element_hierarchy=["creator_face", "headline_text", "background_artifact"],
        visual_weight_allocations={"creator_face": 0.50, "headline_text": 0.35, "background": 0.15},
        composition_style="rule_of_thirds",
        contrast_priorities=["Subject edge isolation"],
        lighting_priorities=["Warm key light", "Cool cyan rim light"],
        confidence=0.94,
    )
    assert output.composition_style == "rule_of_thirds"
    assert sum(output.visual_weight_allocations.values()) == pytest.approx(1.0)


def test_risk_output_model():
    """Verify RiskReasoningOutput and ReasoningRisk representations."""
    risk1 = ReasoningRisk(
        risk_type="fatigue",
        severity="HIGH",
        description="Overused wide-mouth reaction face",
        mitigation="Use subtle micro-expression smirk instead",
    )
    output = RiskReasoningOutput(
        fatigue_risk_score=0.75,
        competitor_convergence_risk=0.20,
        misleading_clickbait_risk=0.05,
        identified_risks=[risk1],
        mitigation_strategies=["Pivot to curiosity gap archetype"],
        confidence=0.91,
    )
    assert output.fatigue_risk_score == 0.75
    assert len(output.identified_risks) == 1
    assert output.identified_risks[0].severity == "HIGH"


def test_strategy_ranking_output_model():
    """Verify StrategyRankingOutput and RankedStrategy structure."""
    strat1 = RankedStrategy(
        strategy_id="strat_001",
        title="Curiosity Gap Reveal",
        description="Obscure final result behind blur sticker",
        expected_ctr_impact=0.85,
        confidence_score=0.90,
        overall_score=0.88,
        pros=["High initial curiosity"],
        cons=["Potential cognitive friction"],
    )
    output = StrategyRankingOutput(
        candidate_strategies=[strat1],
        selected_strategy_id="strat_001",
        ranking_rationale="Highest expected uplift with grounded empirical support",
        confidence=0.89,
    )
    assert output.selected_strategy_id == "strat_001"
    assert len(output.candidate_strategies) == 1
    assert output.candidate_strategies[0].expected_ctr_impact == 0.85


def test_decision_tree_construction_and_traversal():
    """Verify DecisionTree and DecisionTreeNode operations."""
    tree = DecisionTree()
    node_root = DecisionTreeNode(
        node_id="dn_root",
        parent_id=None,
        decision_type="archetype_selection",
        label="Select Archetype",
        chosen_option="before_after_split",
        alternative_options=["big_face_reaction", "curiosity_gap"],
        confidence=0.95,
        rationale="Strongest empirical uplift in benchmark set",
    )
    node_child = DecisionTreeNode(
        node_id="dn_child_1",
        parent_id="dn_root",
        decision_type="palette_selection",
        label="Select Palette",
        chosen_option="high_contrast_cyan_orange",
        confidence=0.90,
    )

    tree.add_node(node_root)
    tree.add_node(node_child)

    assert tree.root_node_id == "dn_root"
    assert tree.get_node("dn_root") is node_root
    assert tree.get_node("dn_child_1") is node_child
    assert tree.get_node("nonexistent") is None

    children = tree.get_children("dn_root")
    assert len(children) == 1
    assert children[0].node_id == "dn_child_1"


def test_reasoning_context_helpers():
    """Verify ReasoningContext helper accessors and validation."""
    ctx = ReasoningContext(graph_id="graph_test_123")
    assert ctx.graph_id == "graph_test_123"
    assert not ctx.is_complete()
    assert ctx.get_evidence_count() == 0
    assert not ctx.has_slot("narrative")

    ref1 = _sample_evidence_ref("ev_1")
    ref2 = _sample_evidence_ref("ev_2")

    ctx.narrative = NarrativeReasoningOutput(story_hook="Hook 1", evidence_refs=[ref1])
    ctx.audience = AudienceReasoningOutput(target_audience_segment="Audience 1", evidence_refs=[ref2])
    ctx.evidence_references = [ref1, ref2]

    assert ctx.has_slot("narrative")
    assert ctx.has_slot("audience")
    assert not ctx.has_slot("brand")
    assert ctx.get_evidence_count() == 2
    assert ctx.get_slot("narrative") is ctx.narrative
    assert ctx.get_slot("brand") is None

    # Slot aliases test
    ctx_slots = ReasoningContext(graph_id="ctx_alias_test")
    ctx_slots.creator_intent = CreatorReasoningOutput(creator_persona="Creator A")
    ctx_slots.brand_constraints = BrandReasoningOutput(compliance_score=1.0)
    ctx_slots.visual_priorities = PriorityReasoningOutput(composition_style="rule_of_thirds")
    ctx_slots.risks = RiskReasoningOutput(fatigue_risk_score=0.1)
    ctx_slots.strategies = StrategyRankingOutput(selected_strategy_id="strat_01")
    ctx_slots.custom_outputs["custom_mod"] = {"status": "ok"}

    assert ctx_slots.has_slot("creator_intent")
    assert ctx_slots.has_slot("creator")
    assert ctx_slots.has_slot("brand_constraints")
    assert ctx_slots.has_slot("brand")
    assert ctx_slots.has_slot("visual_priorities")
    assert ctx_slots.has_slot("priority")
    assert ctx_slots.has_slot("risks")
    assert ctx_slots.has_slot("risk")
    assert ctx_slots.has_slot("strategies")
    assert ctx_slots.has_slot("strategy_ranker")
    assert ctx_slots.has_slot("custom_mod")
    assert not ctx_slots.has_slot("nonexistent_slot")

    assert ctx_slots.get_slot("creator_intent") is ctx_slots.creator_intent
    assert ctx_slots.get_slot("brand") is ctx_slots.brand_constraints
    assert ctx_slots.get_slot("visual_priorities") is ctx_slots.visual_priorities
    assert ctx_slots.get_slot("risks") is ctx_slots.risks
    assert ctx_slots.get_slot("strategies") is ctx_slots.strategies
    assert ctx_slots.get_slot("custom_mod") == {"status": "ok"}
    assert ctx_slots.get_slot("unknown") is None

    assert ctx_slots.get_all_identified_risks() == []
    empty_ctx = ReasoningContext(graph_id="empty_ctx")
    assert empty_ctx.get_all_identified_risks() == []

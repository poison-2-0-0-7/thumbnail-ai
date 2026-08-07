"""
test_brand_models_and_reasoner.py
=================================

Test suite for Brand Reasoning models, preservation directives, visual guardrails,
and production BrandReasoner (Phase 3.4D).
Tests:
- BrandPreservationPriority, VisualElementPreservation, CandidateBrandInterpretation, BrandResult
- BrandReasoner inference, brand pillars synthesis, logo/typography rules, required preservations
- Multi-hypothesis candidate ranking and explainable rejection rationale
- Graph conflict penalty on brand confidence
- ReasonerRegistry auto-registration and ReasoningCoordinator execution
"""

from __future__ import annotations

from typing import Any, Dict, List
import pytest

from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceConflict,
    EvidenceNode,
    EvidenceSummary,
    EvidenceWeight,
    KnowledgeEntryType,
    NormalizedEvidenceGraph,
    ProvenanceRecord,
)
from thumbnail_intelligence.knowledge_base.models import (
    EvidenceGrade,
    EvidenceReference,
    EvidenceSourceType,
)
from thumbnail_intelligence.reasoning.brand_models import (
    BrandPreservationPriority,
    BrandResult,
    CandidateBrandInterpretation,
    VisualElementPreservation,
)
from thumbnail_intelligence.reasoning.brand_reasoner import BrandReasoner
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.creator_models import (
    CreatorArchetype,
    CreatorResult,
    VisualIdentityStyle,
)
from thumbnail_intelligence.reasoning.creator_reasoner import CreatorReasoner
from thumbnail_intelligence.reasoning.models import BrandReasoningOutput
from thumbnail_intelligence.reasoning.narrative_models import (
    NarrativeResult,
    NarrativeType,
)
from thumbnail_intelligence.reasoning.narrative_reasoner import NarrativeReasoner
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry
from thumbnail_intelligence.retrieval.evidence_bundle import (
    RankingMetadata,
    RetrievalScore,
    RetrievedEvidence,
)


def _make_ref(source_id: str = "ev_br_01") -> EvidenceReference:
    return EvidenceReference(
        source_id=source_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.95,
        grade=EvidenceGrade.STRONG,
        claim_summary=f"Brand evidence for {source_id}",
    )


def _build_test_node(
    node_id: str,
    node_type: KnowledgeEntryType,
    payload: Dict[str, Any],
    confidence: float = 0.92,
) -> EvidenceNode:
    ref = _make_ref(node_id)
    score = RetrievalScore(composite_score=confidence)
    item = RetrievedEvidence(
        evidence_id=f"ev_{node_id}",
        entry_id=node_id,
        entry_type=node_type,
        origin=f"origin:{node_id}",
        confidence=confidence,
        reason_retrieved=f"Retrieved for {node_id}",
        score=score,
        ranking=RankingMetadata(rank=1, score=score),
        data_payload=payload,
        evidence_refs=[ref],
    )
    return EvidenceNode(
        node_id=node_id,
        node_type=node_type,
        evidence_item=item,
        confidence=ConfidenceScore(raw_confidence=confidence, propagated_confidence=confidence),
        weight=EvidenceWeight(base_weight=1.0, effective_weight=1.0),
        provenance=ProvenanceRecord(
            origin=f"origin:{node_id}",
            source_id=node_id,
            source_type=EvidenceSourceType.OUTCOME_RECORD,
            retrieval_query_id="query_br",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=True,
    )


def _build_brand_graph(
    creator_name: str = "Studio Velocity",
    colors: List[str] = None,
    conflicts: List[EvidenceConflict] = None,
) -> NormalizedEvidenceGraph:
    nodes = {
        "node_br_creator_01": _build_test_node(
            "node_br_creator_01",
            KnowledgeEntryType.CREATOR_PROFILE_ENTRY,
            {
                "creator_id": "creator_studio_vel",
                "display_name": creator_name,
                "primary_niche": "automotive_engineering",
            },
        ),
        "node_br_hist_02": _build_test_node(
            "node_br_hist_02",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {
                "channel_id": "chan_studio_vel",
                "color_palette": colors or ["#FF0055", "#00F0FF", "#0A0A0E", "#FFFFFF"],
            },
        ),
        "node_br_pat_03": _build_test_node(
            "node_br_pat_03",
            KnowledgeEntryType.DESIGN_PATTERN,
            {"pattern_id": "dual_tone_rim_split"},
        ),
    }
    return NormalizedEvidenceGraph(
        graph_id="graph_brand_test",
        nodes=nodes,
        summary=EvidenceSummary(
            graph_id="graph_brand_test",
            primary_archetype="versus_battle",
            dominant_patterns=["dual_tone_rim_split"],
        ),
        conflicts=conflicts or [],
    )


def test_brand_models_and_preservation_directives():
    """Verify BrandPreservationPriority, VisualElementPreservation, and CandidateBrandInterpretation data contracts."""
    assert BrandPreservationPriority.STRICT_MANDATORY.value == "strict_mandatory"
    assert BrandPreservationPriority.HIGH_RECOMMENDED.value == "high_recommended"

    ref = _make_ref("pres_01")
    pres = VisualElementPreservation(
        element_name="Hero Face Identity",
        element_type="face",
        preservation_priority=BrandPreservationPriority.STRICT_MANDATORY,
        required_treatment="Preserve natural facial features with minimum 30% scale on right third",
        allowed_variation="Facial emotion can vary between astonishment and determination",
        forbidden_change="Do not apply extreme beauty filters or AI distortion",
        evidence_refs=[ref],
    )
    assert pres.element_name == "Hero Face Identity"
    assert pres.preservation_priority == BrandPreservationPriority.STRICT_MANDATORY
    assert len(pres.evidence_refs) == 1

    cand = CandidateBrandInterpretation(
        interpretation_name="Strict Automotive Brand Continuity",
        brand_pillars=["Engineering Precision", "High Velocity Visuals"],
        fit_score=0.96,
        confidence=0.94,
        color_palette=["#FF0055", "#00F0FF"],
        typography_preferences="Bold geometric sans-serif",
        recurring_subjects=["Supercar Profile", "Lead Engineer Face"],
        recurring_layout_patterns=["Two-element split with dynamic speed lines"],
        creator_signature_elements=["Cyan rim lighting"],
        required_preservations=["Hero Face Identity"],
        allowed_variations=["Background road setting"],
        forbidden_changes=["No generic red arrows"],
        pros=["High subscriber brand recall"],
        cons=["Strict composition requirements"],
        evidence_refs=[ref],
        supporting_evidence_ids=["node_br_creator_01"],
    )
    assert cand.fit_score == 0.96
    assert len(cand.brand_pillars) == 2


def test_brand_result_context_compatibility():
    """Verify BrandResult inherits from BrandReasoningOutput and seamlessly stores in ReasoningContext."""
    ref = _make_ref("res_br_01")
    res = BrandResult(
        brand_identity="Studio Velocity: Automotive engineering authority",
        brand_pillars=["Precision", "Clarity"],
        visual_identity={"palette": ["#FF0055", "#00F0FF"]},
        logo_usage="Top-left corner with 10% clearspace",
        color_palette=["#FF0055", "#00F0FF"],
        typography_preferences="Bold sans-serif with 15% outline",
        recurring_subjects=["Lead Engineer", "Supercar"],
        recurring_layout_patterns=["Two-element split"],
        creator_signature_elements=["Cyan rim lighting"],
        brand_constraints=["Hero face must be on outer third"],
        allowed_variations=["Background scene"],
        forbidden_changes=["No facial distortion"],
        color_palette_rules=["Use #FF0055 and #00F0FF"],
        typography_rules=["Bold sans-serif with 15% outline"],
        logo_rules=["Top-left corner with 10% clearspace"],
        prohibited_elements=["No facial distortion"],
        identity_lock_requirements=["Hero face on outer third"],
        compliance_score=0.95,
        brand_confidence=0.94,
        evidence_refs=[ref],
        confidence=0.94,
    )

    assert isinstance(res, BrandReasoningOutput)
    ctx = ReasoningContext(graph_id="ctx_br_test")
    ctx.brand_constraints = res
    assert ctx.has_slot("brand_constraints")
    assert ctx.brand_constraints.brand_identity == "Studio Velocity: Automotive engineering authority"


def test_brand_reasoner_inference():
    """Verify BrandReasoner infers brand identity, pillars, preservations, and candidate rankings."""
    graph = _build_brand_graph()
    context = ReasoningContext(graph_id=graph.graph_id)

    # Narrative context
    context.narrative = NarrativeResult(
        story_hook="Can this $500 car beat a $500,000 Ferrari?",
        narrative_angle="Extreme comparison battle",
        emotional_tone="High stakes and tension",
        narrative_type=NarrativeType.COMPARISON,
        evidence_refs=[_make_ref("nar_br_01")],
        confidence=0.94,
    )

    # Creator context
    context.creator_intent = CreatorResult(
        creator_persona="High-Energy Automotive Engineer",
        signature_elements=["High key rim lighting", "Expressive hero face"],
        style_alignment_score=0.95,
        channel_voice="High energy, technical, authoritative",
        brand_equity_anchors=["Lead Engineer face", "Signature cyan rim"],
        creator_identity="Studio Velocity",
        creator_style="Challenger format with authoritative voice",
        creator_brand="Automotive comparison authority",
        visual_identity=VisualIdentityStyle(
            dominant_color_palette=["#FF0055", "#00F0FF", "#0A0A0E", "#FFFFFF"],
            typography_style="Bold sans-serif with 15% outline",
            face_framing_preference="Outer third hero face",
            lighting_preference="Cyan/magenta split rim lighting",
            composition_rule="Two-element split",
            evidence_refs=[_make_ref("vis_br_01")],
        ),
        brand_consistency=0.95,
        evidence_refs=[_make_ref("cre_br_01")],
        confidence=0.95,
    )

    reasoner = BrandReasoner()
    result = reasoner.reason(graph, context)

    assert isinstance(result, BrandResult)
    assert "Studio Velocity" in result.brand_identity
    assert len(result.brand_pillars) >= 3
    assert len(result.required_preservations) >= 2
    assert any(p.element_type == "face" for p in result.required_preservations)
    assert len(result.candidate_interpretations) >= 2
    assert len(result.rejected_interpretations) >= 1
    assert result.brand_confidence > 0.80
    assert result.compliance_score >= 0.90
    assert len(result.evidence_refs) >= 1
    assert len(result.reasoning_trace) >= 1


def test_brand_reasoner_conflict_penalty():
    """Verify active graph conflicts penalize brand confidence."""
    graph_clean = _build_brand_graph()
    conflict = EvidenceConflict(
        conflict_id="conf_br_01",
        conflict_type="BRAND_CONSTRAINT_VIOLATION",
        conflicting_node_ids=["node_br_creator_01"],
        description="Brand rule violation in color palette",
    )
    graph_conflicted = _build_brand_graph(conflicts=[conflict])

    reasoner = BrandReasoner()
    ctx = ReasoningContext(graph_id="ctx_penalty")

    res_clean = reasoner.reason(graph_clean, ctx)
    res_conflicted = reasoner.reason(graph_conflicted, ctx)

    assert res_conflicted.confidence < res_clean.confidence
    assert res_conflicted.confidence_breakdown["conflict_penalty"] > 0.0


def test_brand_reasoner_registry_and_coordinator_integration():
    """Verify BrandReasoner registers into ReasonerRegistry and executes in ReasoningCoordinator."""
    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(CreatorReasoner())
    registry.register(BrandReasoner())

    coordinator = ReasoningCoordinator(registry=registry)
    graph = _build_brand_graph()

    context = coordinator.coordinate(graph)

    assert context.narrative is not None
    assert context.creator_intent is not None
    assert context.brand_constraints is not None
    assert isinstance(context.brand_constraints, BrandResult)
    assert context.has_slot("brand_constraints")
    assert any("brand_reasoner" in step.reasoner_name for step in context.reasoning_trace)

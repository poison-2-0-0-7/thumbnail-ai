"""
test_creator_models_and_reasoner.py
===================================

Test suite for Creator Reasoning models, visual identity styles, and production CreatorReasoner (Phase 3.4C).
Tests:
- CreatorArchetype, VisualIdentityStyle, CandidateCreatorStyle, CreatorResult data contracts
- CreatorReasoner inference, voice extraction, visual identity synthesis, and brand consistency scoring
- Multi-hypothesis candidate ranking and rejection explanations
- Graph conflict penalties on creator confidence
- ReasonerRegistry auto-registration and ReasoningCoordinator discovery
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
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.creator_models import (
    CandidateCreatorStyle,
    CreatorArchetype,
    CreatorResult,
    VisualIdentityStyle,
)
from thumbnail_intelligence.reasoning.creator_reasoner import CreatorReasoner
from thumbnail_intelligence.reasoning.models import CreatorReasoningOutput
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


def _make_ref(source_id: str = "ev_cre_01") -> EvidenceReference:
    return EvidenceReference(
        source_id=source_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.94,
        grade=EvidenceGrade.STRONG,
        claim_summary=f"Creator evidence for {source_id}",
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
            retrieval_query_id="query_cre",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=True,
    )


def _build_creator_graph(
    creator_name: str = "TechBench Labs",
    colors: List[str] = None,
    patterns: List[str] = None,
    conflicts: List[EvidenceConflict] = None,
) -> NormalizedEvidenceGraph:
    nodes = {
        "node_creator_01": _build_test_node(
            "node_creator_01",
            KnowledgeEntryType.CREATOR_PROFILE_ENTRY,
            {
                "creator_id": "creator_techbench_01",
                "display_name": creator_name,
                "primary_niche": "technology",
            },
        ),
        "node_hist_02": _build_test_node(
            "node_hist_02",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {
                "channel_id": "chan_techbench_01",
                "color_palette": colors or ["#00E5FF", "#FF3366", "#0D0D11", "#FFFFFF"],
            },
        ),
        "node_pattern_03": _build_test_node(
            "node_pattern_03",
            KnowledgeEntryType.DESIGN_PATTERN,
            {"pattern_id": "two_element_split_contrast"},
        ),
    }
    return NormalizedEvidenceGraph(
        graph_id="graph_creator_test",
        nodes=nodes,
        summary=EvidenceSummary(
            graph_id="graph_creator_test",
            primary_archetype="split_comparison",
            dominant_patterns=patterns or ["two_element_split_contrast"],
        ),
        conflicts=conflicts or [],
    )


def test_creator_models_and_visual_identity():
    """Verify CreatorArchetype, VisualIdentityStyle, and CandidateCreatorStyle data contracts."""
    assert CreatorArchetype.ENTERTAINER.value == "entertainer"
    assert CreatorArchetype.CHALLENGER.value == "challenger"
    assert CreatorArchetype.EXPERT_REVIEWER.value == "expert_reviewer"

    ref = _make_ref("vis_01")
    vis = VisualIdentityStyle(
        dominant_color_palette=["#FF0055", "#000000", "#FFFFFF"],
        typography_style="High impact sans-serif with cyan drop shadow",
        face_framing_preference="Hero face on right third, 35% scale",
        lighting_preference="High contrast dual-tone rim light",
        composition_rule="Two-element split with center barrier",
        evidence_refs=[ref],
    )
    assert len(vis.dominant_color_palette) == 3
    assert len(vis.evidence_refs) == 1

    cand = CandidateCreatorStyle(
        persona_name="Signature Tech Reviewer",
        creator_archetype=CreatorArchetype.EXPERT_REVIEWER,
        channel_voice="Analytical, authoritative, precise",
        signature_elements=["Macro hardware close-ups", "Neon cyan accent"],
        brand_equity_anchors=["Creator facial expression", "Benchmark graph overlay"],
        fit_score=0.95,
        confidence=0.92,
        visual_identity=vis,
        evidence_refs=[ref],
        supporting_evidence_ids=["node_creator_01"],
    )
    assert cand.fit_score == 0.95
    assert len(cand.signature_elements) == 2


def test_creator_result_context_compatibility():
    """Verify CreatorResult inherits from CreatorReasoningOutput and seamlessly stores in ReasoningContext."""
    ref = _make_ref("res_01")
    res = CreatorResult(
        creator_persona="Extreme Survival Challenger",
        signature_elements=["Cold blizzard breath", "Orange parka jacket"],
        style_alignment_score=0.94,
        channel_voice="Dramatic high-stakes endurance",
        brand_equity_anchors=["Orange parka", "Challenger face"],
        creator_identity="SurvivalMaster",
        creator_style="Challenger format with dramatic voice",
        creator_brand="Extreme cold wilderness challenges",
        brand_consistency=0.94,
        evidence_refs=[ref],
        confidence=0.94,
    )

    assert isinstance(res, CreatorReasoningOutput)
    ctx = ReasoningContext(graph_id="ctx_cre_test")
    ctx.creator_intent = res
    assert ctx.has_slot("creator_intent")
    assert ctx.creator_intent.creator_persona == "Extreme Survival Challenger"


def test_creator_reasoner_inference():
    """Verify CreatorReasoner infers creator identity, visual identity, and candidate rankings."""
    graph = _build_creator_graph()
    context = ReasoningContext(graph_id=graph.graph_id)

    # Narrative context
    context.narrative = NarrativeResult(
        story_hook="Testing the fastest SSD in the world",
        narrative_angle="Hardware benchmark review",
        emotional_tone="Critical and analytical",
        narrative_type=NarrativeType.REVIEW,
        evidence_refs=[_make_ref("nar_cre_01")],
        confidence=0.95,
    )

    reasoner = CreatorReasoner()
    result = reasoner.reason(graph, context)

    assert isinstance(result, CreatorResult)
    assert result.creator_identity == "TechBench Labs"
    assert len(result.signature_elements) >= 1
    assert result.visual_identity is not None
    assert len(result.visual_identity.dominant_color_palette) >= 1
    assert result.brand_consistency >= 0.80
    assert result.creator_confidence > 0.80
    assert len(result.candidate_creator_styles) >= 2
    assert len(result.rejected_interpretations) >= 1
    assert len(result.evidence_refs) >= 1
    assert len(result.reasoning_trace) >= 1


def test_creator_reasoner_conflict_penalty():
    """Verify active graph conflicts penalize creator confidence."""
    graph_clean = _build_creator_graph()
    conflict = EvidenceConflict(
        conflict_id="conf_cre_01",
        conflict_type="BRAND_CONSTRAINT_VIOLATION",
        conflicting_node_ids=["node_creator_01"],
        description="Brand rule violation in color palette",
    )
    graph_conflicted = _build_creator_graph(conflicts=[conflict])

    reasoner = CreatorReasoner()
    ctx = ReasoningContext(graph_id="ctx_penalty")

    res_clean = reasoner.reason(graph_clean, ctx)
    res_conflicted = reasoner.reason(graph_conflicted, ctx)

    assert res_conflicted.confidence < res_clean.confidence
    assert res_conflicted.confidence_breakdown["conflict_penalty"] > 0.0


def test_creator_reasoner_registry_and_coordinator_integration():
    """Verify CreatorReasoner registers into ReasonerRegistry and executes in ReasoningCoordinator."""
    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(CreatorReasoner())

    coordinator = ReasoningCoordinator(registry=registry)
    graph = _build_creator_graph()

    context = coordinator.coordinate(graph)

    assert context.narrative is not None
    assert context.creator_intent is not None
    assert isinstance(context.creator_intent, CreatorResult)
    assert context.has_slot("creator_intent")
    assert any("creator_reasoner" in step.reasoner_name for step in context.reasoning_trace)

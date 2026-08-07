"""
test_audience_models_and_reasoner.py
====================================

Test suite for Audience Reasoning models, taxonomies, and production AudienceReasoner (Phase 3.4C).
Tests:
- ViewerIntent, ViewerKnowledgeLevel, CognitiveLoadLevel taxonomies
- ViewerPersona and CandidateAudience multi-hypotheses
- AudienceResult backward compatibility with ReasoningContext
- AudienceReasoner inference, intent classification, multi-hypothesis ranking, and calibrated confidence
- Graph conflict penalty on audience confidence
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
from thumbnail_intelligence.reasoning.audience_models import (
    AudienceResult,
    CandidateAudience,
    CognitiveLoadLevel,
    ViewerIntent,
    ViewerKnowledgeLevel,
    ViewerPersona,
)
from thumbnail_intelligence.reasoning.audience_reasoner import AudienceReasoner
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.models import AudienceReasoningOutput
from thumbnail_intelligence.reasoning.narrative_models import (
    CandidateNarrative,
    NarrativeResult,
    NarrativeType,
)
from thumbnail_intelligence.reasoning.narrative_reasoner import NarrativeReasoner
from thumbnail_intelligence.reasoning.pipeline import ReasoningPipeline
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry
from thumbnail_intelligence.retrieval.evidence_bundle import (
    RankingMetadata,
    RetrievalScore,
    RetrievedEvidence,
)


def _make_ref(source_id: str = "ev_aud_01") -> EvidenceReference:
    return EvidenceReference(
        source_id=source_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.92,
        grade=EvidenceGrade.STRONG,
        claim_summary=f"Audience evidence for {source_id}",
    )


def _build_test_node(
    node_id: str,
    node_type: KnowledgeEntryType,
    payload: Dict[str, Any],
    confidence: float = 0.90,
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
            retrieval_query_id="query_aud",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=True,
    )


def _build_evidence_graph(
    title: str = "10 Pro Tips to Build a Custom Mechanical Keyboard",
    transcript: str = "In this step by step tutorial we learn soldering and switch lubing.",
    ocr_text: str = "KEYBOARD GUIDE",
    conflicts: List[EvidenceConflict] = None,
) -> NormalizedEvidenceGraph:
    nodes = {
        "node_01": _build_test_node("node_01", KnowledgeEntryType.HISTORICAL_THUMBNAIL, {"title": title, "transcript": transcript, "ocr_text": ocr_text}),
        "node_02": _build_test_node("node_02", KnowledgeEntryType.DESIGN_PATTERN, {"pattern_id": "step_by_step_visual"}),
    }
    return NormalizedEvidenceGraph(
        graph_id="graph_aud_test",
        nodes=nodes,
        summary=EvidenceSummary(graph_id="graph_aud_test", primary_archetype="step_by_step"),
        conflicts=conflicts or [],
    )


def test_audience_models_taxonomies_and_personas():
    """Verify ViewerIntent, ViewerKnowledgeLevel, CognitiveLoadLevel, and ViewerPersona data contracts."""
    assert ViewerIntent.ENTERTAINMENT.value == "entertainment"
    assert ViewerIntent.LEARNING.value == "learning"
    assert ViewerKnowledgeLevel.INTERMEDIATE.value == "intermediate"
    assert CognitiveLoadLevel.LOW.value == "low"

    ref = _make_ref("per_01")
    persona = ViewerPersona(
        name="The Dedicated DIY Modder",
        demographics_summary="Ages 20-35, tech hobbyist",
        core_interest="Custom keyboard builds and modifications",
        click_trigger="Clean high-contrast switch layout and crisp keycap colors",
        skepticism_level="low",
        evidence_refs=[ref],
    )
    assert persona.name == "The Dedicated DIY Modder"
    assert len(persona.evidence_refs) == 1


def test_candidate_audience_multi_hypothesis():
    """Verify CandidateAudience ranking, pros/cons, and explainable rejection rationale."""
    ref = _make_ref("cand_01")
    cand1 = CandidateAudience(
        audience_segment="Mechanical Keyboard Enthusiasts",
        intent=ViewerIntent.LEARNING,
        knowledge_level=ViewerKnowledgeLevel.INTERMEDIATE,
        cognitive_load=CognitiveLoadLevel.MEDIUM,
        fit_score=0.94,
        confidence=0.91,
        curiosity_triggers=["New switch lubing technique"],
        psychological_hooks=["Skill mastery"],
        pros=["High engagement from core community"],
        cons=["Niche appeal"],
        evidence_refs=[ref],
        supporting_evidence_ids=["node_01"],
    )

    cand2 = CandidateAudience(
        audience_segment="Casual Tech Scrollers",
        intent=ViewerIntent.ENTERTAINMENT,
        knowledge_level=ViewerKnowledgeLevel.BEGINNER,
        cognitive_load=CognitiveLoadLevel.LOW,
        fit_score=0.72,
        confidence=0.75,
        rejection_rationale="Ranked as secondary alternative with lower topic affinity",
        evidence_refs=[ref],
        supporting_evidence_ids=["node_01"],
    )

    assert cand1.fit_score > cand2.fit_score
    assert cand2.rejection_rationale is not None


def test_audience_result_context_compatibility():
    """Verify AudienceResult inherits from AudienceReasoningOutput and seamlessly stores in ReasoningContext."""
    ref = _make_ref("res_01")
    res = AudienceResult(
        target_audience_segment="Passionate PC Builders",
        curiosity_triggers=["Exclusive benchmark results"],
        psychological_hooks=["Curiosity gap"],
        cognitive_load_level="medium",
        viewer_expectations=["Accurate comparisons"],
        audience_confidence=0.92,
        evidence_refs=[ref],
        confidence=0.92,
    )

    assert isinstance(res, AudienceReasoningOutput)
    ctx = ReasoningContext(graph_id="ctx_test")
    ctx.audience = res
    assert ctx.has_slot("audience")
    assert ctx.audience.target_audience_segment == "Passionate PC Builders"


def test_audience_reasoner_learning_intent_inference():
    """Verify AudienceReasoner infers learning intent for educational/tutorial graph."""
    graph = _build_evidence_graph(
        title="How to Solder Mechanical Keyboards Step by Step",
        transcript="In this tutorial we teach beginners the exact solder process.",
        ocr_text="HOW TO SOLDER",
    )
    context = ReasoningContext(graph_id=graph.graph_id)

    # Mock narrative context
    context.narrative = NarrativeResult(
        story_hook="Master soldering in 10 minutes",
        narrative_angle="Instructional guide",
        emotional_tone="Instructional and empowering",
        narrative_type=NarrativeType.TUTORIAL,
        evidence_refs=[_make_ref("nar_01")],
        confidence=0.92,
    )

    reasoner = AudienceReasoner()
    result = reasoner.reason(graph, context)

    assert isinstance(result, AudienceResult)
    assert result.viewer_intent == ViewerIntent.LEARNING
    assert result.primary_audience is not None
    assert len(result.viewer_personas) >= 1
    assert result.audience_confidence > 0.75
    assert len(result.evidence_refs) >= 1
    assert len(result.reasoning_trace) >= 1


def test_audience_reasoner_entertainment_intent_inference():
    """Verify AudienceReasoner infers entertainment intent for challenge video."""
    graph = _build_evidence_graph(
        title="I Survived 24 Hours Inside a Abandoned Mine Challenge",
        transcript="The tunnel collapsed behind us and we were trapped in darkness.",
        ocr_text="TRAPPED 24H",
    )
    context = ReasoningContext(graph_id=graph.graph_id)
    context.narrative = NarrativeResult(
        story_hook="Will they escape the mine?",
        narrative_angle="Extreme survival challenge",
        emotional_tone="Urgency and tension",
        narrative_type=NarrativeType.CHALLENGE,
        evidence_refs=[_make_ref("nar_02")],
        confidence=0.94,
    )

    reasoner = AudienceReasoner()
    result = reasoner.reason(graph, context)

    assert result.viewer_intent == ViewerIntent.ENTERTAINMENT
    assert result.cognitive_load_level == "low"
    assert "Excitement" in result.viewer_emotional_drivers or "Curiosity" in result.viewer_emotional_drivers


def test_audience_reasoner_conflict_penalty():
    """Verify active graph conflicts penalize audience confidence."""
    graph_clean = _build_evidence_graph()
    conflict = EvidenceConflict(
        conflict_id="conf_aud_01",
        conflict_type="CONTRADICTORY_CLAIM",
        conflicting_node_ids=["node_01", "node_02"],
        description="Conflicting audience segment signals",
    )
    graph_conflicted = _build_evidence_graph(conflicts=[conflict])

    reasoner = AudienceReasoner()
    ctx = ReasoningContext(graph_id="ctx_penalty")

    res_clean = reasoner.reason(graph_clean, ctx)
    res_conflicted = reasoner.reason(graph_conflicted, ctx)

    assert res_conflicted.confidence < res_clean.confidence
    assert res_conflicted.confidence_breakdown["conflict_penalty"] > 0.0


def test_audience_reasoner_registry_and_coordinator_integration():
    """Verify AudienceReasoner registers into ReasonerRegistry and executes in ReasoningCoordinator."""
    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(AudienceReasoner())

    coordinator = ReasoningCoordinator(registry=registry)
    graph = _build_evidence_graph()

    context = coordinator.coordinate(graph)

    assert context.narrative is not None
    assert context.audience is not None
    assert isinstance(context.audience, AudienceResult)
    assert context.has_slot("audience")
    assert any("audience_reasoner" in step.reasoner_name for step in context.reasoning_trace)

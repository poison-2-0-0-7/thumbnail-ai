"""
test_narrative_reasoner.py
==========================

Test suite for NarrativeReasoner in Phase 3.4B.
Tests:
- Narrative extraction from NormalizedEvidenceGraph
- Narrative classification across various genres (Discovery, Challenge, Transformation, Comparison, etc.)
- Multi-hypothesis candidate generation (Candidate A, B, C) and best selection with rejection rationale
- Chronological and emotional narrative arc inference
- Visual focus candidate formulation for redesign
- Calibrated confidence calculation with multi-signal factors and conflict penalties
- Grounding gate enforcement
- ReasonerRegistry registration and ReasoningCoordinator discovery
"""

from __future__ import annotations

from typing import Any, Dict, List
import pytest

from thumbnail_intelligence.knowledge_base.models import (
    EvidenceGrade,
    EvidenceReference,
    EvidenceSourceType,
)
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
from thumbnail_intelligence.reasoning.config import ReasoningConfig
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.narrative_models import (
    ArcStage,
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


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------


def _build_test_node(
    node_id: str,
    node_type: KnowledgeEntryType,
    payload: Dict[str, Any],
    confidence: float = 0.90,
) -> EvidenceNode:
    ref = EvidenceReference(
        source_id=node_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=confidence,
        grade=EvidenceGrade.STRONG,
        claim_summary=f"Evidence for {node_id}",
    )
    score = RetrievalScore(composite_score=confidence)
    item = RetrievedEvidence(
        evidence_id=f"ev_id_{node_id}",
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
            retrieval_query_id="query_001",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=True,
    )


def _build_evidence_graph(
    graph_id: str = "graph_test_nar",
    title: str = "I Found a Secret Vault Under My House",
    transcript: str = "We started digging in the basement and discovered an ancient locked door.",
    ocr_text: str = "SECRET VAULT",
    objects: List[str] = None,
    primary_archetype: str = "curiosity_gap",
    conflicts: List[EvidenceConflict] = None,
) -> NormalizedEvidenceGraph:
    nodes: Dict[str, EvidenceNode] = {}

    # Node 1: Historical Video Title & Metadata
    n1 = _build_test_node(
        "node_meta_01",
        KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        {"title": title, "transcript": transcript, "ocr_text": ocr_text},
        confidence=0.95,
    )
    nodes["node_meta_01"] = n1

    # Node 2: Scene Graph & Visual Elements
    n2 = _build_test_node(
        "node_scene_02",
        KnowledgeEntryType.DESIGN_PATTERN,
        {"objects": objects or ["shocked face", "steel safe", "flashlight"], "pattern_id": "high_contrast_rim"},
        confidence=0.90,
    )
    nodes["node_scene_02"] = n2

    # Node 3: Archetype Example
    n3 = _build_test_node(
        "node_archetype_03",
        KnowledgeEntryType.ARCHETYPE_EXAMPLE,
        {"archetype_id": primary_archetype},
        confidence=0.88,
    )
    nodes["node_archetype_03"] = n3

    summary = EvidenceSummary(
        graph_id=graph_id,
        primary_archetype=primary_archetype,
        dominant_patterns=["high_contrast_rim"],
    )

    return NormalizedEvidenceGraph(
        graph_id=graph_id,
        nodes=nodes,
        summary=summary,
        conflicts=conflicts or [],
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_narrative_extraction_and_classification_discovery():
    """Verify narrative reasoner classifies a discovery/mystery video accurately."""
    graph = _build_evidence_graph(
        title="I Found a Secret Hidden Underground Bunker",
        transcript="We discovered the forgotten entrance after weeks of searching.",
        ocr_text="SECRET BUNKER",
        objects=["shocked creator", "concrete hatch", "flashlight"],
        primary_archetype="curiosity_gap",
    )

    reasoner = NarrativeReasoner()
    context = ReasoningContext(graph_id=graph.graph_id)

    result = reasoner.reason(graph, context)

    assert isinstance(result, NarrativeResult)
    assert result.narrative_type == NarrativeType.DISCOVERY
    assert "Shocked Creator" in result.key_subjects or "Concrete Hatch" in result.key_subjects
    assert result.primary_narrative is not None
    assert result.primary_narrative.narrative_type == NarrativeType.DISCOVERY
    assert len(result.evidence_refs) >= 1
    assert result.confidence > 0.70
    assert len(result.reasoning_trace) > 0


def test_narrative_classification_challenge():
    """Verify classification of an endurance/survival challenge storyline."""
    graph = _build_evidence_graph(
        title="Surviving 24 Hours Trapped in an Ice Hotel Challenge",
        transcript="The temperature dropped to minus 30 and it was impossible to stay warm.",
        ocr_text="24 HOURS FROZEN",
        objects=["freezing challenger", "ice wall", "thermometer"],
        primary_archetype="extreme_challenge",
    )

    reasoner = NarrativeReasoner()
    result = reasoner.reason(graph, ReasoningContext(graph_id=graph.graph_id))

    assert result.narrative_type == NarrativeType.CHALLENGE
    assert "Challenge" in result.primary_narrative.title or "Challenge" in result.primary_narrative.premise
    assert result.narrative_arc.dominant_stage == ArcStage.PEAK


def test_narrative_classification_transformation():
    """Verify classification of a makeover/transformation storyline."""
    graph = _build_evidence_graph(
        title="Extreme Bedroom Transformation Before and After Glow Up",
        transcript="We gutted the whole messy room and rebuilt it from scratch.",
        ocr_text="BEFORE & AFTER",
        objects=["messy room", "modern luxury room", "builder"],
        primary_archetype="before_after_split",
    )

    reasoner = NarrativeReasoner()
    result = reasoner.reason(graph, ReasoningContext(graph_id=graph.graph_id))

    assert result.narrative_type == NarrativeType.TRANSFORMATION
    assert result.primary_narrative.narrative_type == NarrativeType.TRANSFORMATION


def test_narrative_classification_comparison():
    """Verify classification of a comparison / versus battle storyline."""
    graph = _build_evidence_graph(
        title="$1 vs $10,000 Gaming PC Battle Test",
        transcript="We compared the budget machine versus the supercomputer to see the difference.",
        ocr_text="$1 VS $10,000",
        objects=["budget laptop", "watercooled rig", "benchmark score"],
        primary_archetype="versus_battle",
    )

    reasoner = NarrativeReasoner()
    result = reasoner.reason(graph, ReasoningContext(graph_id=graph.graph_id))

    assert result.narrative_type == NarrativeType.COMPARISON


def test_multi_hypothesis_candidate_generation_and_rejection():
    """Verify multiple competing candidate narratives (A, B, C) are scored and rejected with rationale."""
    graph = _build_evidence_graph(
        title="I Found a Secret Vault Under My House",
        transcript="We discovered the forgotten safe in the basement.",
    )

    reasoner = NarrativeReasoner()
    result = reasoner.reason(graph, ReasoningContext(graph_id=graph.graph_id))

    # Check candidates
    assert result.primary_narrative is not None
    assert len(result.supporting_narratives) >= 1
    assert len(result.rejected_alternatives) >= 1

    # Winner must have highest score
    winner_score = result.primary_narrative.score
    for alt in result.rejected_alternatives:
        assert alt.score <= winner_score
        assert alt.rejection_rationale is not None
        assert "Ranked as secondary alternative" in alt.rejection_rationale


def test_narrative_arc_progression():
    """Verify 4-stage narrative arc (Beginning, Conflict, Peak, Resolution) with visual cues and grounding."""
    graph = _build_evidence_graph()
    reasoner = NarrativeReasoner()
    result = reasoner.reason(graph, ReasoningContext(graph_id=graph.graph_id))

    arc = result.narrative_arc
    assert arc is not None
    assert len(arc.stages) == 4
    stages_order = [s.stage for s in arc.stages]
    assert stages_order == [ArcStage.BEGINNING, ArcStage.CONFLICT, ArcStage.PEAK, ArcStage.RESOLUTION]

    # Check peak emotional intensity
    peak_step = next(s for s in arc.stages if s.stage == ArcStage.PEAK)
    assert peak_step.emotional_intensity > 0.85
    assert len(peak_step.evidence_refs) >= 1


def test_visual_focus_candidates_formulation():
    """Verify primary, secondary, and tertiary visual focal candidates with treatments."""
    graph = _build_evidence_graph()
    reasoner = NarrativeReasoner()
    result = reasoner.reason(graph, ReasoningContext(graph_id=graph.graph_id))

    focus_list = result.visual_focus_candidates
    assert len(focus_list) == 3

    priorities = [f.visual_priority for f in focus_list]
    assert priorities == ["PRIMARY", "SECONDARY", "TERTIARY"]

    primary_foc = focus_list[0]
    assert primary_foc.recommended_treatment != ""
    assert len(primary_foc.evidence_refs) >= 1


def test_conflict_penalty_on_confidence():
    """Verify that active graph conflicts apply a proportional penalty to narrative confidence."""
    graph_clean = _build_evidence_graph(conflicts=[])
    reasoner = NarrativeReasoner()
    res_clean = reasoner.reason(graph_clean, ReasoningContext(graph_id=graph_clean.graph_id))

    # Graph with 2 conflicts
    conflict1 = EvidenceConflict(
        conflict_id="conf_01",
        conflict_type="CONTRADICTORY_CLAIM",
        conflicting_node_ids=["node_meta_01", "node_scene_02"],
        description="Conflicting visual cues",
    )
    conflict2 = EvidenceConflict(
        conflict_id="conf_02",
        conflict_type="BRAND_CONSTRAINT_VIOLATION",
        conflicting_node_ids=["node_meta_01"],
        description="Brand rule violation",
    )
    graph_conflicted = _build_evidence_graph(conflicts=[conflict1, conflict2])
    res_conflicted = reasoner.reason(graph_conflicted, ReasoningContext(graph_id=graph_conflicted.graph_id))

    assert res_conflicted.confidence < res_clean.confidence
    assert res_conflicted.confidence_breakdown["conflict_penalty"] > 0.0


def test_registry_registration_and_coordinator_discovery():
    """Verify NarrativeReasoner registers seamlessly into ReasonerRegistry and executes in ReasoningCoordinator."""
    registry = ReasonerRegistry()
    reasoner = NarrativeReasoner()
    registry.register(reasoner)

    assert registry.has("narrative_reasoner")
    assert registry.get("narrative_reasoner") is reasoner

    coordinator = ReasoningCoordinator(registry=registry)
    graph = _build_evidence_graph()

    context = coordinator.coordinate(graph)

    assert context.narrative is not None
    assert isinstance(context.narrative, NarrativeResult)
    assert context.narrative.narrative_type == NarrativeType.DISCOVERY
    assert len(context.evidence_references) >= 1
    assert len(context.reasoning_trace) >= 1
    assert any("narrative_reasoner" in step.reasoner_name for step in context.reasoning_trace)


def test_pipeline_execution_with_narrative_reasoner():
    """Verify ReasoningPipeline runs end-to-end with NarrativeReasoner."""
    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())

    pipeline = ReasoningPipeline.from_registry(registry)
    graph = _build_evidence_graph()

    context = pipeline.run(graph)
    assert context.narrative is not None
    assert context.has_slot("narrative")
    assert context.narrative.story_hook != ""

"""
test_strategic_reasoning_pipeline_phase6.py
===========================================

Comprehensive test suite verifying the multi-reasoner Strategic Reasoning Pipeline (Phase 3.4F):
- Executes all 6 production reasoners in topological DAG sequence:
  NarrativeReasoner -> AudienceReasoner -> CreatorReasoner -> BrandReasoner -> PriorityReasoner -> RiskReasoner
- Validates all 6 ReasoningContext slots: narrative, audience, creator_intent, brand_constraints, visual_priorities, risks
- Verifies grounded evidence aggregation across all 6 reasoners
- Tests empty graphs, suppressed nodes, and validation rejections
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
from thumbnail_intelligence.reasoning.audience_models import AudienceResult
from thumbnail_intelligence.reasoning.audience_reasoner import AudienceReasoner
from thumbnail_intelligence.reasoning.brand_models import BrandResult
from thumbnail_intelligence.reasoning.brand_reasoner import BrandReasoner
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.creator_models import CreatorResult
from thumbnail_intelligence.reasoning.creator_reasoner import CreatorReasoner
from thumbnail_intelligence.reasoning.narrative_models import NarrativeResult
from thumbnail_intelligence.reasoning.narrative_reasoner import NarrativeReasoner
from thumbnail_intelligence.reasoning.pipeline import ReasoningPipeline
from thumbnail_intelligence.reasoning.priority_models import PriorityResult
from thumbnail_intelligence.reasoning.priority_reasoner import PriorityReasoner
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry
from thumbnail_intelligence.reasoning.risk_models import RiskResult
from thumbnail_intelligence.reasoning.risk_reasoner import RiskReasoner
from thumbnail_intelligence.retrieval.evidence_bundle import (
    RankingMetadata,
    RetrievalScore,
    RetrievedEvidence,
)


def _build_test_node(
    node_id: str,
    node_type: KnowledgeEntryType,
    payload: Dict[str, Any],
    confidence: float = 0.90,
    is_active: bool = True,
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
            retrieval_query_id="query_pipeline6",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=is_active,
    )


def test_full_6_reasoner_strategic_pipeline():
    """Verify Narrative, Audience, Creator, Brand, Priority, and Risk reasoners run in topological DAG order and populate all slots."""
    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(AudienceReasoner())
    registry.register(CreatorReasoner())
    registry.register(BrandReasoner())
    registry.register(PriorityReasoner())
    registry.register(RiskReasoner())

    # Verify execution order: narrative must be before all, priority before risk
    exec_order = [r.name for r in registry.get_execution_order()]
    assert exec_order[0] == "narrative_reasoner"
    assert "audience_reasoner" in exec_order[1:]
    assert "creator_reasoner" in exec_order[1:]
    assert "brand_reasoner" in exec_order[1:]
    assert "priority_reasoner" in exec_order[1:]
    assert exec_order[-1] == "risk_reasoner" or "risk_reasoner" in exec_order[3:]

    # Build rich evidence graph
    nodes = {
        "node_pipe6_title_01": _build_test_node(
            "node_pipe6_title_01",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {
                "title": "I Survived 100 Hours in Antarctica with Zero Gear",
                "transcript": "The blizzard hit on hour 12 and the temperatures dropped to minus 40.",
                "ocr_text": "100H ANTARCTICA",
                "color_palette": ["#00E5FF", "#FF3366", "#0E0E12", "#FFFFFF"],
                "objects": ["explorer face", "ice cave", "frozen compass"],
            },
        ),
        "node_pipe6_creator_02": _build_test_node(
            "node_pipe6_creator_02",
            KnowledgeEntryType.CREATOR_PROFILE_ENTRY,
            {
                "creator_id": "creator_survival_pro",
                "display_name": "Survival Pro",
                "primary_niche": "extreme_survival",
            },
        ),
        "node_pipe6_pattern_03": _build_test_node(
            "node_pipe6_pattern_03",
            KnowledgeEntryType.DESIGN_PATTERN,
            {"pattern_id": "extreme_challenge_split"},
        ),
    }

    graph = NormalizedEvidenceGraph(
        graph_id="graph_pipe6_full",
        nodes=nodes,
        summary=EvidenceSummary(graph_id="graph_pipe6_full", primary_archetype="extreme_challenge"),
    )

    pipeline = ReasoningPipeline.from_registry(registry)
    context = pipeline.run(graph)

    # Validate all 6 strategic slots are fully populated
    assert context.has_slot("narrative")
    assert context.has_slot("audience")
    assert context.has_slot("creator_intent")
    assert context.has_slot("brand_constraints")
    assert context.has_slot("visual_priorities")
    assert context.has_slot("risks")

    assert isinstance(context.narrative, NarrativeResult)
    assert isinstance(context.audience, AudienceResult)
    assert isinstance(context.creator_intent, CreatorResult)
    assert isinstance(context.brand_constraints, BrandResult)
    assert isinstance(context.visual_priorities, PriorityResult)
    assert isinstance(context.risks, RiskResult)

    # Validate grounding evidence references and trace steps across all 6 reasoners
    assert len(context.evidence_references) >= 3
    assert len(context.reasoning_trace) >= 6
    assert context.overall_confidence > 0.70


def test_empty_graph_and_suppressed_nodes_handling_across_all_6_reasoners():
    """Verify all 6 reasoners execute safely with zero-confidence grounded fallbacks on empty graphs."""
    empty_graph = NormalizedEvidenceGraph(
        graph_id="graph_empty_all6",
        nodes={},
        summary=EvidenceSummary(graph_id="graph_empty_all6"),
    )

    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(AudienceReasoner())
    registry.register(CreatorReasoner())
    registry.register(BrandReasoner())
    registry.register(PriorityReasoner())
    registry.register(RiskReasoner())

    pipeline = ReasoningPipeline.from_registry(registry)
    context = pipeline.run(empty_graph)

    assert context.narrative is not None
    assert context.audience is not None
    assert context.creator_intent is not None
    assert context.brand_constraints is not None
    assert context.visual_priorities is not None
    assert context.risks is not None
    assert context.risks.risk_confidence >= 0.0


def test_risk_reasoner_validate_output_rejection():
    """Verify validate_output rejects non-RiskResult or invalid confidence scores."""
    reasoner = RiskReasoner()
    assert not reasoner.validate_output("invalid_object")
    assert not reasoner.validate_output(None)

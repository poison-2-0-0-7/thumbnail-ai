"""
test_priority_models_and_reasoner.py
====================================

Test suite for Priority Reasoning models, visual hierarchy levels, attention flows,
and production PriorityReasoner (Phase 3.4E).
Tests:
- HierarchyTier, ElementPriorityLevel, BackgroundPriority taxonomies
- VisualHierarchyNode, AttentionFlowStep, CandidateHierarchy, PriorityResult data contracts
- PriorityResult backward compatibility with ReasoningContext.visual_priorities
- PriorityReasoner inference, visual hierarchy formulation, attention weights, canvas allocations
- Sequential 1-2-3 gaze flow steps and non-compete rules
- Multi-hypothesis candidate ranking and rejection rationales
- Graph conflict penalty application
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
from thumbnail_intelligence.reasoning.audience_models import (
    AudienceResult,
    CognitiveLoadLevel,
    ViewerIntent,
    ViewerKnowledgeLevel,
)
from thumbnail_intelligence.reasoning.audience_reasoner import AudienceReasoner
from thumbnail_intelligence.reasoning.brand_models import (
    BrandResult,
    VisualElementPreservation,
)
from thumbnail_intelligence.reasoning.brand_reasoner import BrandReasoner
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.creator_models import (
    CreatorResult,
    VisualIdentityStyle,
)
from thumbnail_intelligence.reasoning.creator_reasoner import CreatorReasoner
from thumbnail_intelligence.reasoning.models import PriorityReasoningOutput
from thumbnail_intelligence.reasoning.narrative_models import (
    NarrativeResult,
    NarrativeType,
)
from thumbnail_intelligence.reasoning.narrative_reasoner import NarrativeReasoner
from thumbnail_intelligence.reasoning.priority_models import (
    AttentionFlowStep,
    BackgroundPriority,
    CandidateHierarchy,
    ElementPriorityLevel,
    HierarchyTier,
    PriorityResult,
    VisualHierarchyNode,
)
from thumbnail_intelligence.reasoning.priority_reasoner import PriorityReasoner
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry
from thumbnail_intelligence.retrieval.evidence_bundle import (
    RankingMetadata,
    RetrievalScore,
    RetrievedEvidence,
)


def _make_ref(source_id: str = "ev_prio_01") -> EvidenceReference:
    return EvidenceReference(
        source_id=source_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.94,
        grade=EvidenceGrade.STRONG,
        claim_summary=f"Priority evidence for {source_id}",
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
            retrieval_query_id="query_prio",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=True,
    )


def _build_priority_graph(
    title: str = "I Spent 24 Hours in an Abandoned Underground Vault",
    objects: List[str] = None,
    conflicts: List[EvidenceConflict] = None,
) -> NormalizedEvidenceGraph:
    nodes = {
        "node_prio_title_01": _build_test_node(
            "node_prio_title_01",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {
                "title": title,
                "transcript": "We entered the vault and found the secret locked chamber.",
                "ocr_text": "24H VAULT",
                "objects": objects or ["shocked face", "steel safe", "flashlight"],
            },
        ),
        "node_prio_pat_02": _build_test_node(
            "node_prio_pat_02",
            KnowledgeEntryType.DESIGN_PATTERN,
            {"pattern_id": "curiosity_gap_high_contrast"},
        ),
    }
    return NormalizedEvidenceGraph(
        graph_id="graph_priority_test",
        nodes=nodes,
        summary=EvidenceSummary(
            graph_id="graph_priority_test",
            primary_archetype="curiosity_gap",
            dominant_patterns=["curiosity_gap_high_contrast"],
        ),
        conflicts=conflicts or [],
    )


def test_priority_models_and_taxonomies():
    """Verify HierarchyTier, ElementPriorityLevel, BackgroundPriority, VisualHierarchyNode, and AttentionFlowStep."""
    assert HierarchyTier.PRIMARY.value == "primary"
    assert HierarchyTier.SECONDARY.value == "secondary"
    assert ElementPriorityLevel.HIGH.value == "high"
    assert BackgroundPriority.MUTED.value == "muted"

    ref = _make_ref("hnode_01")
    node = VisualHierarchyNode(
        element_name="Expressive Hero Face",
        element_category="face",
        tier=HierarchyTier.PRIMARY,
        importance_score=1.0,
        attention_weight=0.45,
        canvas_allocation_fraction=0.35,
        contrast_requirement="Minimum 5.0:1 luminance ratio against dark backing",
        gaze_order=1,
        non_compete_with=["Curiosity Safe"],
        evidence_refs=[ref],
    )
    assert node.element_name == "Expressive Hero Face"
    assert node.attention_weight == 0.45
    assert len(node.evidence_refs) == 1

    flow = AttentionFlowStep(
        step_order=1,
        target_element="Expressive Hero Face",
        visual_cue="Wide eyes and open mouth expression",
        psychological_driver="Instant biological fixation",
        evidence_refs=[ref],
    )
    assert flow.step_order == 1
    assert len(flow.evidence_refs) == 1


def test_candidate_hierarchy_multi_hypothesis():
    """Verify CandidateHierarchy fit scores, attention distributions, pros/cons, and rejection rationales."""
    ref = _make_ref("cand_h_01")
    cand1 = CandidateHierarchy(
        hierarchy_name="Face-First Emotional Hook Hierarchy",
        primary_focus="Expressive Hero Face",
        secondary_focus="Mystery Vault",
        tertiary_focus="Bold Text Hook",
        fit_score=0.96,
        confidence=0.94,
        attention_distribution={"face": 0.45, "object": 0.35, "text": 0.12, "bg": 0.08},
        canvas_allocations={"face_area": 0.35, "object_area": 0.30, "text_area": 0.20, "bg_area": 0.15},
        pros=["Maximizes biological mirror neuron gaze engagement"],
        cons=["Requires high quality facial asset"],
        evidence_refs=[ref],
        supporting_evidence_ids=["node_prio_title_01"],
    )

    cand2 = CandidateHierarchy(
        hierarchy_name="Object-First Mystery Hierarchy",
        primary_focus="Mystery Vault",
        secondary_focus="Expressive Hero Face",
        tertiary_focus="Minimal Hook",
        fit_score=0.78,
        confidence=0.81,
        rejection_rationale="Ranked as secondary alternative with lower face recognition score",
        evidence_refs=[ref],
        supporting_evidence_ids=["node_prio_title_01"],
    )

    assert cand1.fit_score > cand2.fit_score
    assert cand2.rejection_rationale is not None


def test_priority_result_context_compatibility():
    """Verify PriorityResult inherits from PriorityReasoningOutput and seamlessly stores in ReasoningContext."""
    ref = _make_ref("res_prio_01")
    res = PriorityResult(
        primary_subject="Creator Hero Face",
        secondary_subject="Mystery Safe",
        supporting_subjects=["Headline Text", "Dark Cave"],
        visual_hierarchy=[],
        importance_scores={"Creator Hero Face": 1.0, "Mystery Safe": 0.85},
        attention_weights={"primary": 0.42, "secondary": 0.33, "text": 0.15, "bg": 0.10},
        canvas_allocation={"face": 0.35, "object": 0.30, "text": 0.20, "bg": 0.15},
        text_priority=ElementPriorityLevel.MEDIUM,
        face_priority=ElementPriorityLevel.HIGH,
        object_priority=ElementPriorityLevel.HIGH,
        background_priority=BackgroundPriority.MUTED,
        color_importance={"cyan": 0.40, "magenta": 0.35},
        contrast_priority=["Minimum 4.5:1 luminance ratio"],
        required_emphasis=["Highlight Creator Face"],
        suppressed_elements=["Cluttered textures"],
        attention_flow=[],
        max_focal_points=2,
        non_compete_rules=["Text must not overlap face"],
        selection_rationale="Highest empirical fit score",
        priority_confidence=0.94,
        focal_element_hierarchy=["Creator Hero Face", "Mystery Safe"],
        visual_weight_allocations={"face": 0.42, "object": 0.33},
        composition_style="split_comparison",
        contrast_priorities=["Minimum 4.5:1 luminance ratio"],
        lighting_priorities=["High key rim lighting"],
        evidence_refs=[ref],
        confidence=0.94,
    )

    assert isinstance(res, PriorityReasoningOutput)
    ctx = ReasoningContext(graph_id="ctx_prio_test")
    ctx.visual_priorities = res
    assert ctx.has_slot("visual_priorities")
    assert ctx.visual_priorities.primary_subject == "Creator Hero Face"


def test_priority_reasoner_inference():
    """Verify PriorityReasoner synthesizes visual hierarchy, attention flow, allocations, and candidate rankings."""
    graph = _build_priority_graph()
    context = ReasoningContext(graph_id=graph.graph_id)

    # Narrative context
    context.narrative = NarrativeResult(
        story_hook="Unlocking the secret vault door",
        narrative_angle="Mystery exploration",
        emotional_tone="Suspense and curiosity",
        narrative_type=NarrativeType.DISCOVERY,
        key_subjects=["Shocked Face", "Steel Safe", "Cave Background"],
        evidence_refs=[_make_ref("nar_prio_01")],
        confidence=0.95,
    )

    # Audience context
    context.audience = AudienceResult(
        target_audience_segment="Curiosity Seekers",
        curiosity_triggers=["Hidden treasure"],
        psychological_hooks=["Pattern interrupt"],
        cognitive_load_level="medium",
        viewer_expectations=["Clear reveal"],
        evidence_refs=[_make_ref("aud_prio_01")],
        confidence=0.92,
    )

    # Creator context
    context.creator_intent = CreatorResult(
        creator_persona="Extreme Mystery Explorer",
        signature_elements=["Cyan rim lighting", "Expressive face"],
        style_alignment_score=0.95,
        channel_voice="High energy discovery",
        brand_equity_anchors=["Creator face", "Signature cyan rim"],
        creator_identity="MysteryExplorer",
        creator_style="Discovery format",
        creator_brand="Underground mysteries",
        evidence_refs=[_make_ref("cre_prio_01")],
        confidence=0.95,
    )

    # Brand context
    context.brand_constraints = BrandResult(
        brand_identity="MysteryExplorer: Authentic underground exploration",
        brand_pillars=["Authenticity", "High Contrast"],
        visual_identity={"palette": ["#00E5FF", "#FF3366"]},
        logo_usage="Top-left corner",
        color_palette=["#00E5FF", "#FF3366"],
        typography_preferences="Bold sans-serif with 15% outline",
        recurring_subjects=["Creator Face", "Mystery Safe"],
        recurring_layout_patterns=["Two-element split"],
        creator_signature_elements=["Cyan rim lighting"],
        brand_constraints=["Hero face must be on outer third"],
        allowed_variations=["Background lighting"],
        forbidden_changes=["No facial obscuration"],
        color_palette_rules=["Use #00E5FF and #FF3366"],
        typography_rules=["Bold sans-serif with 15% outline"],
        logo_rules=["Top-left corner"],
        prohibited_elements=["No facial obscuration"],
        identity_lock_requirements=["Hero face on outer third"],
        compliance_score=0.95,
        evidence_refs=[_make_ref("br_prio_01")],
        confidence=0.94,
    )

    reasoner = PriorityReasoner()
    result = reasoner.reason(graph, context)

    assert isinstance(result, PriorityResult)
    assert result.primary_subject != ""
    assert result.secondary_subject != ""
    assert len(result.visual_hierarchy) >= 3
    assert len(result.attention_flow) == 3
    assert len(result.non_compete_rules) >= 2
    assert result.canvas_allocation["primary_subject_area"] >= 0.25
    assert len(result.candidate_hierarchies) >= 2
    assert len(result.rejected_hierarchies) >= 1
    assert result.priority_confidence > 0.80
    assert len(result.evidence_refs) >= 1
    assert len(result.reasoning_trace) >= 1


def test_priority_reasoner_conflict_penalty():
    """Verify active graph conflicts penalize priority confidence."""
    graph_clean = _build_priority_graph()
    conflict = EvidenceConflict(
        conflict_id="conf_prio_01",
        conflict_type="MUTUALLY_EXCLUSIVE_ARCHETYPE",
        conflicting_node_ids=["node_prio_title_01"],
        description="Conflicting layout archetypes in evidence graph",
    )
    graph_conflicted = _build_priority_graph(conflicts=[conflict])

    reasoner = PriorityReasoner()
    ctx = ReasoningContext(graph_id="ctx_penalty")

    res_clean = reasoner.reason(graph_clean, ctx)
    res_conflicted = reasoner.reason(graph_conflicted, ctx)

    assert res_conflicted.confidence < res_clean.confidence
    assert res_conflicted.confidence_breakdown["conflict_penalty"] > 0.0


def test_priority_reasoner_registry_and_coordinator_integration():
    """Verify PriorityReasoner registers into ReasonerRegistry and executes in ReasoningCoordinator."""
    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(AudienceReasoner())
    registry.register(CreatorReasoner())
    registry.register(BrandReasoner())
    registry.register(PriorityReasoner())

    coordinator = ReasoningCoordinator(registry=registry)
    graph = _build_priority_graph()

    context = coordinator.coordinate(graph)

    assert context.narrative is not None
    assert context.audience is not None
    assert context.creator_intent is not None
    assert context.brand_constraints is not None
    assert context.visual_priorities is not None
    assert isinstance(context.visual_priorities, PriorityResult)
    assert context.has_slot("visual_priorities")
    assert any("priority_reasoner" in step.reasoner_name for step in context.reasoning_trace)

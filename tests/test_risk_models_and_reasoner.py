"""
test_risk_models_and_reasoner.py
================================

Test suite for Risk Reasoning models, risk taxonomies, severity/likelihood models,
and production RiskReasoner (Phase 3.4F).
Tests:
- RiskCategory, RiskSeverity, RiskLikelihood taxonomies
- DetectedRisk, CandidateRiskProfile, RiskResult data contracts
- RiskResult backward compatibility with ReasoningContext.risks
- RiskReasoner inference, multi-category risk detection, scores, and mitigations
- Multi-hypothesis candidate risk ranking and rejection rationales
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
)
from thumbnail_intelligence.reasoning.audience_reasoner import AudienceReasoner
from thumbnail_intelligence.reasoning.brand_models import BrandResult
from thumbnail_intelligence.reasoning.brand_reasoner import BrandReasoner
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.coordinator import ReasoningCoordinator
from thumbnail_intelligence.reasoning.creator_models import CreatorResult
from thumbnail_intelligence.reasoning.creator_reasoner import CreatorReasoner
from thumbnail_intelligence.reasoning.models import (
    ReasoningRisk,
    RiskReasoningOutput,
)
from thumbnail_intelligence.reasoning.narrative_models import (
    NarrativeResult,
    NarrativeType,
)
from thumbnail_intelligence.reasoning.narrative_reasoner import NarrativeReasoner
from thumbnail_intelligence.reasoning.priority_models import PriorityResult
from thumbnail_intelligence.reasoning.priority_reasoner import PriorityReasoner
from thumbnail_intelligence.reasoning.registry import ReasonerRegistry
from thumbnail_intelligence.reasoning.risk_models import (
    CandidateRiskProfile,
    DetectedRisk,
    RiskCategory,
    RiskLikelihood,
    RiskResult,
    RiskSeverity,
)
from thumbnail_intelligence.reasoning.risk_reasoner import RiskReasoner
from thumbnail_intelligence.retrieval.evidence_bundle import (
    RankingMetadata,
    RetrievalScore,
    RetrievedEvidence,
)


def _make_ref(source_id: str = "ev_risk_01") -> EvidenceReference:
    return EvidenceReference(
        source_id=source_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.93,
        grade=EvidenceGrade.STRONG,
        claim_summary=f"Risk evidence for {source_id}",
    )


def _build_test_node(
    node_id: str,
    node_type: KnowledgeEntryType,
    payload: Dict[str, Any],
    confidence: float = 0.91,
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
            retrieval_query_id="query_risk",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=True,
    )


def _build_risk_graph(
    title: str = "I Spent 50 Hours in Solitary Confinement",
    objects: List[str] = None,
    conflicts: List[EvidenceConflict] = None,
) -> NormalizedEvidenceGraph:
    nodes = {
        "node_risk_title_01": _build_test_node(
            "node_risk_title_01",
            KnowledgeEntryType.HISTORICAL_THUMBNAIL,
            {
                "title": title,
                "transcript": "We locked the metal door for the 50 hour challenge.",
                "ocr_text": "50 HOURS ALONE",
                "objects": objects or ["prison bars", "shocked face", "countdown timer"],
            },
        ),
        "node_risk_pat_02": _build_test_node(
            "node_risk_pat_02",
            KnowledgeEntryType.DESIGN_PATTERN,
            {"pattern_id": "challenge_solitary_split"},
        ),
    }
    return NormalizedEvidenceGraph(
        graph_id="graph_risk_test",
        nodes=nodes,
        summary=EvidenceSummary(
            graph_id="graph_risk_test",
            primary_archetype="extreme_challenge",
            dominant_patterns=["challenge_solitary_split"],
        ),
        conflicts=conflicts or [],
    )


def test_risk_taxonomies_and_models():
    """Verify RiskCategory, RiskSeverity, RiskLikelihood, DetectedRisk, and CandidateRiskProfile."""
    assert RiskCategory.VISUAL_CLUTTER.value == "visual_clutter"
    assert RiskCategory.VIEWER_FATIGUE.value == "viewer_fatigue"
    assert RiskCategory.COMPETITOR_CONVERGENCE.value == "competitor_convergence"
    assert RiskSeverity.CRITICAL.value == "CRITICAL"
    assert RiskSeverity.HIGH.value == "HIGH"
    assert RiskLikelihood.HIGH.value == "high"

    ref = _make_ref("drisk_01")
    risk = DetectedRisk(
        category=RiskCategory.POOR_CONTRAST,
        severity=RiskSeverity.HIGH,
        likelihood=RiskLikelihood.HIGH,
        impact_score=0.45,
        title="Mobile Luminance Contrast Drop",
        description="Background luminance bleeds into face under sunlight",
        affected_element="face_contrast_boundary",
        mitigation_suggestion="Apply minimum 4.5:1 luminance ratio",
        evidence_refs=[ref],
    )
    assert risk.category == RiskCategory.POOR_CONTRAST
    assert risk.impact_score == 0.45
    assert len(risk.evidence_refs) == 1

    cand = CandidateRiskProfile(
        profile_name="Comprehensive Empirical Risk Assessment",
        detected_risks=[risk],
        overall_risk_score=0.35,
        fatigue_score=0.30,
        competitor_convergence_score=0.25,
        clickbait_score=0.15,
        fit_score=0.95,
        confidence=0.94,
        pros=["Holistic diagnostic coverage"],
        cons=["Requires multiple mitigations"],
        evidence_refs=[ref],
        supporting_evidence_ids=["node_risk_title_01"],
    )
    assert cand.fit_score == 0.95
    assert len(cand.detected_risks) == 1


def test_risk_result_context_compatibility():
    """Verify RiskResult inherits from RiskReasoningOutput and stores in ReasoningContext.risks."""
    ref = _make_ref("res_risk_01")
    rrisk = ReasoningRisk(
        risk_type="viewer_fatigue",
        severity="MEDIUM",
        description="Challenge trope fatigue",
        mitigation="Introduce cinematic lighting",
        evidence_refs=[ref],
    )
    res = RiskResult(
        visual_risks=[],
        narrative_risks=[],
        audience_risks=[],
        brand_risks=[],
        ctr_risks=[],
        readability_risks=[],
        policy_risks=[],
        attention_risks=[],
        competition_risks=[],
        cognitive_load_score=0.30,
        overall_severity=RiskSeverity.MEDIUM,
        overall_likelihood=RiskLikelihood.MEDIUM,
        overall_impact=0.30,
        all_detected_risks=[],
        mitigation_suggestions=["Apply 4.5:1 contrast"],
        fatigue_risk_score=0.32,
        competitor_convergence_risk=0.28,
        misleading_clickbait_risk=0.18,
        identified_risks=[rrisk],
        mitigation_strategies=["Apply 4.5:1 contrast"],
        selection_rationale="Comprehensive empirical risk coverage",
        risk_confidence=0.94,
        evidence_refs=[ref],
        confidence=0.94,
    )

    assert isinstance(res, RiskReasoningOutput)
    ctx = ReasoningContext(graph_id="ctx_risk_test")
    ctx.risks = res
    assert ctx.has_slot("risks")
    assert ctx.risks.fatigue_risk_score == 0.32
    assert ctx.risks.competitor_convergence_risk == 0.28


def test_risk_reasoner_inference():
    """Verify RiskReasoner detects risks across all 9 categories, computes scores, and ranks candidates."""
    graph = _build_risk_graph()
    context = ReasoningContext(graph_id=graph.graph_id)

    # Narrative context
    context.narrative = NarrativeResult(
        story_hook="Surviving 50 hours in total isolation",
        narrative_angle="Extreme endurance test",
        emotional_tone="High tension and isolation",
        narrative_type=NarrativeType.CHALLENGE,
        key_subjects=["Shocked Face", "Prison Bars", "Countdown Timer"],
        evidence_refs=[_make_ref("nar_risk_01")],
        confidence=0.95,
    )

    # Audience context
    context.audience = AudienceResult(
        target_audience_segment="Extreme Challenge Enthusiasts",
        curiosity_triggers=["Human limits", "Solitary confinement"],
        cognitive_load_level="medium",
        evidence_refs=[_make_ref("aud_risk_01")],
        confidence=0.92,
    )

    # Creator context
    context.creator_intent = CreatorResult(
        creator_persona="Extreme Challenger",
        signature_elements=["Cyan rim lighting", "Expressive hero face"],
        style_alignment_score=0.95,
        channel_voice="Adrenaline-fueled endurance",
        creator_identity="Challenger Pro",
        creator_style="High stakes endurance",
        creator_brand="Extreme physical challenges",
        evidence_refs=[_make_ref("cre_risk_01")],
        confidence=0.95,
    )

    # Brand context
    context.brand_constraints = BrandResult(
        brand_identity="Challenger Pro: Authentic extreme physical challenges",
        brand_pillars=["Authenticity", "High Contrast"],
        visual_identity={"palette": ["#FF0033", "#00F0FF"]},
        logo_usage="Top-left corner",
        color_palette=["#FF0033", "#00F0FF"],
        typography_preferences="Bold grotesque sans-serif",
        recurring_subjects=["Creator Face", "Tension Prop"],
        recurring_layout_patterns=["Two-element split"],
        creator_signature_elements=["Cyan rim lighting"],
        brand_constraints=["Hero face must be on outer third"],
        allowed_variations=["Background lighting"],
        forbidden_changes=["No facial obscuration"],
        color_palette_rules=["Use #FF0033 and #00F0FF"],
        typography_rules=["Bold grotesque sans-serif"],
        logo_rules=["Top-left corner"],
        prohibited_elements=["No facial obscuration"],
        identity_lock_requirements=["Hero face on outer third"],
        compliance_score=0.95,
        evidence_refs=[_make_ref("br_risk_01")],
        confidence=0.94,
    )

    # Priority context
    context.visual_priorities = PriorityResult(
        primary_subject="Creator Expressive Hero Face",
        secondary_subject="Prison Bars Tension Prop",
        supporting_subjects=["Countdown Timer", "Dark Concrete Cell"],
        visual_hierarchy=[],
        importance_scores={"Creator Face": 1.0, "Prison Bars": 0.85},
        attention_weights={"primary": 0.42, "secondary": 0.33, "text": 0.15, "bg": 0.10},
        canvas_allocation={"face": 0.35, "object": 0.30, "text": 0.20, "bg": 0.15},
        color_importance={"cyan": 0.40, "magenta": 0.35},
        contrast_priority=["Minimum 4.5:1 luminance ratio"],
        required_emphasis=["Creator Face"],
        suppressed_elements=["Cluttered textures"],
        attention_flow=[],
        max_focal_points=2,
        non_compete_rules=["Text must not overlap face"],
        focal_element_hierarchy=["Creator Face", "Prison Bars"],
        visual_weight_allocations={"face": 0.42, "object": 0.33},
        composition_style="split_challenge",
        contrast_priorities=["Minimum 4.5:1 luminance ratio"],
        lighting_priorities=["High key rim lighting"],
        evidence_refs=[_make_ref("prio_risk_01")],
        confidence=0.94,
    )

    reasoner = RiskReasoner()
    result = reasoner.reason(graph, context)

    assert isinstance(result, RiskResult)
    assert len(result.all_detected_risks) >= 3
    assert len(result.visual_risks) >= 1
    assert len(result.audience_risks) >= 1
    assert len(result.readability_risks) >= 1
    assert result.fatigue_risk_score > 0.0
    assert result.competitor_convergence_risk > 0.0
    assert len(result.mitigation_suggestions) >= 2
    assert len(result.candidate_risk_profiles) >= 2
    assert len(result.rejected_risk_profiles) >= 1
    assert result.risk_confidence > 0.80
    assert len(result.evidence_refs) >= 1
    assert len(result.reasoning_trace) >= 1


def test_risk_reasoner_conflict_penalty():
    """Verify active graph conflicts increase fatigue score and penalize risk confidence."""
    graph_clean = _build_risk_graph()
    conflict = EvidenceConflict(
        conflict_id="conf_risk_01",
        conflict_type="MUTUALLY_EXCLUSIVE_ARCHETYPE",
        conflicting_node_ids=["node_risk_title_01"],
        description="Severe trope fatigue in challenge niche",
    )
    graph_conflicted = _build_risk_graph(conflicts=[conflict])

    reasoner = RiskReasoner()
    ctx = ReasoningContext(graph_id="ctx_penalty")

    res_clean = reasoner.reason(graph_clean, ctx)
    res_conflicted = reasoner.reason(graph_conflicted, ctx)

    assert res_conflicted.confidence < res_clean.confidence
    assert res_conflicted.fatigue_risk_score > res_clean.fatigue_risk_score
    assert res_conflicted.confidence_breakdown["conflict_penalty"] > 0.0


def test_risk_reasoner_registry_and_coordinator_integration():
    """Verify RiskReasoner registers into ReasonerRegistry and executes in ReasoningCoordinator."""
    registry = ReasonerRegistry()
    registry.register(NarrativeReasoner())
    registry.register(AudienceReasoner())
    registry.register(CreatorReasoner())
    registry.register(BrandReasoner())
    registry.register(PriorityReasoner())
    registry.register(RiskReasoner())

    coordinator = ReasoningCoordinator(registry=registry)
    graph = _build_risk_graph()

    context = coordinator.coordinate(graph)

    assert context.narrative is not None
    assert context.audience is not None
    assert context.creator_intent is not None
    assert context.brand_constraints is not None
    assert context.visual_priorities is not None
    assert context.risks is not None
    assert isinstance(context.risks, RiskResult)
    assert context.has_slot("risks")
    assert any("risk_reasoner" in step.reasoner_name for step in context.reasoning_trace)

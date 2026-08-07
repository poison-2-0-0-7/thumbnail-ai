"""
test_narrative_models.py
========================

Test suite for Narrative Reasoning domain models and taxonomies (Phase 3.4B).
Tests NarrativeType, ArcStage, ArcStep, NarrativeArc, VisualFocusCandidate,
CandidateNarrative, and NarrativeResult data contracts.
"""

from __future__ import annotations

import pytest

from thumbnail_intelligence.knowledge_base.models import (
    EvidenceGrade,
    EvidenceReference,
    EvidenceSourceType,
)
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.models import NarrativeReasoningOutput
from thumbnail_intelligence.reasoning.narrative_models import (
    ArcStage,
    ArcStep,
    CandidateNarrative,
    NarrativeArc,
    NarrativeResult,
    NarrativeType,
    VisualFocusCandidate,
)


def _make_ref(source_id: str = "ev_ref_101") -> EvidenceReference:
    return EvidenceReference(
        source_id=source_id,
        source_type=EvidenceSourceType.OUTCOME_RECORD,
        confidence=0.92,
        grade=EvidenceGrade.STRONG,
        claim_summary="Empirical evidence backing narrative claim",
    )


def test_narrative_type_taxonomy():
    """Verify standard narrative types and string values."""
    assert NarrativeType.DISCOVERY.value == "discovery"
    assert NarrativeType.CHALLENGE.value == "challenge"
    assert NarrativeType.TRANSFORMATION.value == "transformation"
    assert NarrativeType.COMPARISON.value == "comparison"
    assert NarrativeType.TUTORIAL.value == "tutorial"
    assert NarrativeType.REACTION.value == "reaction"
    assert NarrativeType.REVIEW.value == "review"
    assert NarrativeType.DOCUMENTARY.value == "documentary"
    assert NarrativeType.COMPETITION.value == "competition"
    assert NarrativeType.COMEDY.value == "comedy"
    assert NarrativeType.STORYTELLING.value == "storytelling"
    assert NarrativeType.EDUCATIONAL.value == "educational"
    assert NarrativeType.VLOG.value == "vlog"
    assert NarrativeType.INTERVIEW.value == "interview"
    assert NarrativeType.NEWS.value == "news"
    assert NarrativeType.CUSTOM.value == "custom"


def test_arc_stage_and_arc_step():
    """Verify ArcStage values, ArcStep creation, intensity bounds, and evidence links."""
    ref = _make_ref("step_ref_01")
    step = ArcStep(
        stage=ArcStage.CONFLICT,
        description="The challenger enters an extreme freezing environment with only 3 tools.",
        emotional_intensity=0.85,
        visual_cue="Cold blue blizzard mist with trembling silhouette",
        evidence_ids=["node_01", "node_02"],
        evidence_refs=[ref],
        confidence=0.90,
    )
    assert step.stage == ArcStage.CONFLICT
    assert step.emotional_intensity == 0.85
    assert len(step.evidence_refs) == 1
    assert step.confidence == 0.90


def test_narrative_arc():
    """Verify NarrativeArc structure, stages, and dominant stage assignment."""
    ref = _make_ref("arc_ref_01")
    step1 = ArcStep(
        stage=ArcStage.BEGINNING,
        description="Setup and challenge rules announced.",
        emotional_intensity=0.3,
        evidence_refs=[ref],
    )
    step2 = ArcStep(
        stage=ArcStage.PEAK,
        description="Climactic final minute escape attempt.",
        emotional_intensity=0.98,
        evidence_refs=[ref],
    )

    arc = NarrativeArc(
        arc_name="Survival Against All Odds",
        primary_driver="tension and urgency",
        stages=[step1, step2],
        dominant_stage=ArcStage.PEAK,
        confidence=0.94,
        evidence_refs=[ref],
    )
    assert arc.arc_name == "Survival Against All Odds"
    assert len(arc.stages) == 2
    assert arc.dominant_stage == ArcStage.PEAK
    assert arc.confidence == 0.94


def test_visual_focus_candidate():
    """Verify VisualFocusCandidate priorities and treatment recommendations."""
    ref = _make_ref("foc_ref_01")
    focus = VisualFocusCandidate(
        element_name="Extreme Shocked Face",
        role_in_narrative="Subject experiencing peak revelation",
        visual_priority="PRIMARY",
        recommended_treatment="High key rim light with 15% vignette separation",
        source_node_id="node_subject_01",
        confidence=0.96,
        evidence_refs=[ref],
    )
    assert focus.element_name == "Extreme Shocked Face"
    assert focus.visual_priority == "PRIMARY"
    assert focus.confidence == 0.96
    assert len(focus.evidence_refs) == 1


def test_candidate_narrative_hypotheses():
    """Verify CandidateNarrative multi-hypothesis fields, pros/cons, and rejection rationale."""
    ref = _make_ref("cand_ref_01")
    cand1 = CandidateNarrative(
        title="The Mystery Vault Discovery",
        narrative_type=NarrativeType.DISCOVERY,
        premise="Opening an abandoned underground safe found inside a house",
        hook="What did the previous owner hide inside?",
        emotional_tone="Curiosity, suspense, and surprise",
        score=0.92,
        confidence=0.90,
        evidence_refs=[ref],
        supporting_evidence_ids=["node_vault_01"],
        pros=["High curiosity gap potential", "Backed by strong OCR 'LOCKED' token"],
        cons=["Requires clear mystery box visual in thumbnail"],
    )

    cand2 = CandidateNarrative(
        title="Tool Review Angle",
        narrative_type=NarrativeType.REVIEW,
        premise="Testing tools to cut through the safe",
        hook="Can a $50 grinder cut a bank vault?",
        emotional_tone="Analytical and practical",
        score=0.75,
        confidence=0.72,
        evidence_refs=[ref],
        supporting_evidence_ids=["node_grinder_01"],
        rejection_rationale="Lower curiosity appeal than mystery discovery angle",
    )

    assert cand1.score > cand2.score
    assert cand2.rejection_rationale is not None
    assert len(cand1.pros) == 2


def test_narrative_result_backward_compatibility():
    """Verify NarrativeResult inherits from NarrativeReasoningOutput and seamlessly populates ReasoningContext."""
    ref = _make_ref("res_ref_01")
    cand = CandidateNarrative(
        title="Extreme Survival Challenge",
        narrative_type=NarrativeType.CHALLENGE,
        premise="Surviving 24 hours on a deserted island",
        hook="Can they build shelter before sunset?",
        emotional_tone="Intense urgency",
        score=0.95,
        confidence=0.93,
        evidence_refs=[ref],
        supporting_evidence_ids=["node_01"],
    )

    result = NarrativeResult(
        story_hook=cand.hook,
        narrative_angle=cand.premise,
        emotional_tone=cand.emotional_tone,
        evidence_refs=[ref],
        confidence=0.93,
        primary_narrative=cand,
        narrative_type=NarrativeType.CHALLENGE,
        story_summary="24 hour survival storyline on island.",
        key_subjects=["Challenger", "Wilderness Island"],
        key_events=["Building shelter", "Surviving storm"],
        narrative_confidence=0.93,
        supporting_evidence_ids=["node_01"],
        selection_rationale="Highest empirical evidence support",
    )

    # Contract inheritance checks
    assert isinstance(result, NarrativeReasoningOutput)
    assert isinstance(result, NarrativeResult)
    assert result.story_hook == "Can they build shelter before sunset?"
    assert result.narrative_type == NarrativeType.CHALLENGE

    # Test assignment into ReasoningContext
    ctx = ReasoningContext(graph_id="test_graph_ctx")
    ctx.narrative = result
    assert ctx.has_slot("narrative")
    assert ctx.get_slot("narrative") is result
    assert ctx.narrative.story_summary == "24 hour survival storyline on island."

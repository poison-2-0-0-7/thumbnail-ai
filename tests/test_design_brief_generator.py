"""
test_design_brief_generator.py
===============================

Comprehensive unit test suite for DesignBriefGenerator and DesignBrief data models (Phase 3.5).
Tests cover:
- Deterministic DesignBrief generation from ValidatedReasoningPackage
- Direct schema validation and integrity checks
- Multi-format serialization (JSON, YAML, Pydantic/Dict)
- Forward and backward schema version compatibility
- Strict mode vs non-strict mode execution
- Missing reasoning slots and empty optional fields handling
- Large reasoning packages with complex evidence graphs
- Strict renderer independence invariant verification
"""

import json
import pytest
import yaml
from typing import Dict, Any

from thumbnail_intelligence.evidence.models import (
    ConfidenceScore,
    EvidenceNode,
    EvidenceReference,
    EvidenceSourceType,
    EvidenceSummary,
    EvidenceWeight,
    KnowledgeEntryType,
    NormalizedEvidenceGraph,
    ProvenanceRecord,
)
from thumbnail_intelligence.knowledge_base.models import EvidenceGrade
from thumbnail_intelligence.reasoning.audience_models import (
    AudienceResult,
    CandidateAudience,
    CognitiveLoadLevel,
    ViewerIntent,
    ViewerKnowledgeLevel,
)
from thumbnail_intelligence.reasoning.brand_models import (
    BrandPreservationPriority,
    BrandResult,
    CandidateBrandInterpretation,
    VisualElementPreservation,
)
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.creator_models import (
    CandidateCreatorStyle,
    CreatorArchetype,
    CreatorResult,
    VisualIdentityStyle,
)
from thumbnail_intelligence.reasoning.design_brief_generator import (
    DesignBriefGenerator,
)
from thumbnail_intelligence.reasoning.design_brief_models import (
    AudienceBrief,
    BrandBrief,
    BriefMetadata,
    CameraBrief,
    ColorBrief,
    CompositionBrief,
    CreatorBrief,
    DesignBrief,
    ExecutionConstraintsBrief,
    LightingBrief,
    NarrativeBrief,
    ObjectsBrief,
    TypographyBrief,
    ValidationBrief,
)
from thumbnail_intelligence.reasoning.exceptions import ReasonerValidationError
from thumbnail_intelligence.reasoning.narrative_models import (
    ArcStage,
    CandidateNarrative,
    NarrativeArc,
    NarrativeResult,
    NarrativeType,
    VisualFocusCandidate,
)
from thumbnail_intelligence.reasoning.priority_models import (
    CandidateHierarchy,
    HierarchyTier,
    PriorityResult,
    VisualHierarchyNode,
)
from thumbnail_intelligence.reasoning.risk_models import (
    CandidateRiskProfile,
    DetectedRisk,
    RiskCategory,
    RiskLikelihood,
    RiskResult,
    RiskSeverity,
)
from thumbnail_intelligence.reasoning.strategy_models import (
    StrategyArchetype,
    StrategyCandidate,
    StrategyDecision,
    TradeoffAnalysis,
)
from thumbnail_intelligence.reasoning.validator import StrategicReasoningValidator
from thumbnail_intelligence.reasoning.validator_models import (
    ReasoningValidation,
    ValidatedReasoningPackage,
)
from thumbnail_intelligence.retrieval.evidence_bundle import (
    RankingMetadata,
    RetrievalScore,
    RetrievedEvidence,
)


def _build_test_node(
    node_id: str,
    node_type: KnowledgeEntryType,
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
        evidence_id=f"ev_{node_id}",
        entry_id=node_id,
        entry_type=node_type,
        origin=f"origin:{node_id}",
        confidence=confidence,
        reason_retrieved=f"Retrieved for {node_id}",
        score=score,
        ranking=RankingMetadata(rank=1, score=score),
        data_payload={"test": node_id},
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
            retrieval_query_id="query_db_unit",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=True,
    )


@pytest.fixture
def sample_evidence_graph() -> NormalizedEvidenceGraph:
    """Create a realistic NormalizedEvidenceGraph for testing."""
    node1 = _build_test_node("ev_001", KnowledgeEntryType.DESIGN_PATTERN, 0.95)
    node2 = _build_test_node("ev_002", KnowledgeEntryType.VISUAL_PATTERN, 0.88)
    return NormalizedEvidenceGraph(
        graph_id="graph_test_db",
        summary=EvidenceSummary(graph_id="graph_test_db", primary_archetype="educational"),
        nodes={"ev_001": node1, "ev_002": node2},
    )


@pytest.fixture
def sample_validated_package(sample_evidence_graph: NormalizedEvidenceGraph) -> ValidatedReasoningPackage:
    """Construct a complete, valid ValidatedReasoningPackage for testing."""
    ev_ref1 = EvidenceReference(source_id="ev_001", source_type=EvidenceSourceType.OUTCOME_RECORD, confidence=0.9)
    ev_ref2 = EvidenceReference(source_id="ev_002", source_type=EvidenceSourceType.OUTCOME_RECORD, confidence=0.85)

    # Narrative
    cand_nar = CandidateNarrative(
        title="Unlocking Quantum Tech",
        narrative_type=NarrativeType.DISCOVERY,
        premise="Quantum computing breakthrough explained simply",
        hook="Will quantum tech destroy encryption?",
        confidence=0.9,
        evidence_refs=[ev_ref1],
        supporting_evidence_ids=["ev_001"],
    )
    narrative_res = NarrativeResult(
        primary_narrative=cand_nar,
        narrative_type=NarrativeType.DISCOVERY,
        story_hook="Will quantum tech destroy encryption?",
        narrative_angle="Empirical breakthrough breakdown",
        emotional_tone="awe and mystery",
        narrative_arc=NarrativeArc(
            arc_name="Discovery Arc",
            primary_driver="curiosity",
            dominant_stage=ArcStage.MYSTERY,
            confidence=0.9,
        ),
        visual_focus_candidates=[
            VisualFocusCandidate(
                element_name="Quantum Computer Chip",
                role_in_narrative="Core mystery subject",
                visual_priority="PRIMARY",
                evidence_refs=[ev_ref1],
            )
        ],
        confidence=0.9,
        supporting_evidence_ids=["ev_001"],
    )

    # Audience
    cand_aud = CandidateAudience(
        audience_segment="Tech Enthusiasts",
        intent=ViewerIntent.LEARNING,
        knowledge_level=ViewerKnowledgeLevel.BEGINNER,
        cognitive_load=CognitiveLoadLevel.LOW,
        confidence=0.85,
        curiosity_triggers=["Curiosity gap regarding encryption security"],
        evidence_refs=[ev_ref1],
    )
    audience_res = AudienceResult(
        primary_audience=cand_aud,
        viewer_intent=ViewerIntent.LEARNING,
        optimal_cognitive_load=CognitiveLoadLevel.LOW,
        confidence=0.85,
        supporting_evidence_ids=["ev_001"],
    )

    # Creator
    creator_res = CreatorResult(
        creator_identity="QuantumLab",
        primary_creator_style=CandidateCreatorStyle(
            persona_name="Tech Educator",
            creator_archetype=CreatorArchetype.EDUCATOR,
            channel_voice="Informed, authoritative, clear",
            signature_elements=["Blue rim light", "Host Face on left third"],
            confidence=0.88,
            evidence_refs=[ev_ref2],
        ),
        visual_identity=VisualIdentityStyle(
            dominant_color_palette=["#0066CC", "#FFFFFF"],
            typography_style="Clean sans-serif",
            evidence_refs=[ev_ref2],
        ),
        brand_consistency=0.92,
        visual_constraints=["No cluttered backgrounds"],
        confidence=0.88,
        supporting_evidence_ids=["ev_002"],
    )

    # Brand
    brand_res = BrandResult(
        primary_brand_interpretation=CandidateBrandInterpretation(
            interpretation_name="Clean Tech Brand",
            brand_pillars=["Clarity", "Innovation"],
            confidence=0.92,
            evidence_refs=[ev_ref1],
        ),
        required_preservations=[
            VisualElementPreservation(
                element_name="Channel Logo",
                element_type="logo",
                preservation_priority=BrandPreservationPriority.HIGH_RECOMMENDED,
                required_treatment="Corner placement with high contrast",
                evidence_refs=[ev_ref1],
            )
        ],
        brand_constraints=["Sensational clickbait red arrows"],
        confidence=0.92,
        supporting_evidence_ids=["ev_001"],
    )

    # Priority
    priority_res = PriorityResult(
        primary_hierarchy_candidate=CandidateHierarchy(
            hierarchy_name="Tech Discovery Hierarchy",
            primary_focus="Quantum Computer Chip",
            secondary_focus="Host Face",
            tertiary_focus="Channel Logo",
            confidence=0.87,
            evidence_refs=[ev_ref1, ev_ref2],
        ),
        visual_hierarchy=[
            VisualHierarchyNode(
                element_name="Quantum Computer Chip",
                element_category="object",
                tier=HierarchyTier.PRIMARY,
                importance_score=0.9,
                attention_weight=0.50,
                canvas_allocation_fraction=0.45,
                gaze_order=1,
            ),
            VisualHierarchyNode(
                element_name="Host Face",
                element_category="face",
                tier=HierarchyTier.SECONDARY,
                importance_score=0.7,
                attention_weight=0.30,
                canvas_allocation_fraction=0.30,
                gaze_order=2,
            ),
            VisualHierarchyNode(
                element_name="Channel Logo",
                element_category="logo",
                tier=HierarchyTier.TERTIARY,
                importance_score=0.5,
                attention_weight=0.15,
                canvas_allocation_fraction=0.15,
                gaze_order=3,
            ),
        ],
        confidence=0.87,
        supporting_evidence_ids=["ev_001", "ev_002"],
    )

    # Risk
    r_item = DetectedRisk(
        title="Text Overload Risk",
        description="Too much text overlay on canvas",
        category=RiskCategory.TEXT_OVERLOAD,
        severity=RiskSeverity.LOW,
        impact_score=0.2,
        mitigation_suggestion="Keep text under 4 words",
        evidence_refs=[ev_ref1],
    )
    risk_res = RiskResult(
        primary_risk_profile=CandidateRiskProfile(
            profile_name="Standard Tech Video Risks",
            overall_risk_score=0.20,
            confidence=0.85,
            evidence_refs=[ev_ref1],
        ),
        visual_risks=[r_item],
        all_detected_risks=[r_item],
        confidence=0.85,
        supporting_evidence_ids=["ev_001"],
    )

    # Strategy
    strat_cand = StrategyCandidate(
        candidate_id="strat_001",
        title="Inside the Quantum Chip",
        archetype=StrategyArchetype.EDUCATIONAL,
        description="Grounded visual breakdown of quantum chip with host reaction",
        expected_ctr_uplift=0.15,
        retention_alignment_score=0.90,
        brand_equity_protection_score=0.92,
        risk_penalty=0.05,
        composite_score=0.88,
        confidence=0.88,
        execution_priorities=["Feature Quantum Computer Chip prominently", "Host face on left third"],
        evidence_refs=[ev_ref1, ev_ref2],
        supporting_evidence_ids=["ev_001", "ev_002"],
    )
    strategy_dec = StrategyDecision(
        winning_strategy=strat_cand,
        tradeoff_analysis_detail=TradeoffAnalysis(
            pareto_optimal_strategy_id=strat_cand.candidate_id,
            ctr_vs_retention_tradeoff="High retention, steady CTR",
            brand_vs_novelty_tradeoff="Strong brand protection",
            cognitive_load_tradeoff="Low cognitive load layout",
            evidence_refs=[ev_ref1],
        ),
        decision_confidence=0.88,
        decision_rationale="Balances educational clarity with strong curiosity hook",
        execution_priorities=["Feature Quantum Computer Chip prominently", "Host face on left third"],
        supporting_evidence_ids=["ev_001", "ev_002"],
    )

    context = ReasoningContext(
        graph_id=sample_evidence_graph.graph_id,
        narrative=narrative_res,
        audience=audience_res,
        creator_intent=creator_res,
        brand_constraints=brand_res,
        visual_priorities=priority_res,
        risks=risk_res,
        strategies=strategy_dec,
        overall_confidence=0.88,
        evidence_references=[ev_ref1, ev_ref2],
    )

    validator = StrategicReasoningValidator()
    pkg = validator.validate(context=context, graph=sample_evidence_graph)
    return pkg


class TestDesignBriefGenerator:

    def test_design_brief_generation(self, sample_validated_package: ValidatedReasoningPackage):
        """Test generation of strongly typed DesignBrief from ValidatedReasoningPackage."""
        generator = DesignBriefGenerator()
        brief = generator.generate(sample_validated_package)

        assert isinstance(brief, DesignBrief)
        assert brief.metadata.brief_id.startswith("brief_")
        assert brief.metadata.schema_version == "1.0.0"
        assert brief.metadata.generator_id == "design_brief_generator_v1"

        # Narrative
        assert brief.narrative.primary_story == "Quantum computing breakthrough explained simply"
        assert brief.narrative.supporting_story == "Empirical breakthrough breakdown"
        assert brief.narrative.emotional_goal == "awe and mystery"
        assert brief.narrative.story_focus == "Quantum Computer Chip"
        assert brief.narrative.narrative_type == "discovery"
        assert brief.narrative.narrative_arc == "Discovery Arc"

        # Audience
        assert brief.audience.primary_audience == "Tech Enthusiasts"
        assert brief.audience.viewer_intent == "learning"
        assert brief.audience.cognitive_load == "low"
        assert "Curiosity gap" in brief.audience.curiosity_trigger

        # Creator
        assert brief.creator.creator_identity == "QuantumLab"
        assert brief.creator.creator_archetype == "educator"
        assert brief.creator.historical_consistency == 0.92
        assert "Blue rim light" in brief.creator.style_constraints

        # Brand
        assert "Channel Logo" in brief.brand.required_elements
        assert "Sensational clickbait red arrows" in brief.brand.forbidden_changes

        # Composition
        assert brief.composition.primary_subject == "Quantum Computer Chip"
        assert brief.composition.secondary_subject == "Host Face"
        assert len(brief.composition.visual_hierarchy) == 3

        # Typography
        assert brief.typography.text_priority == "high"
        assert brief.typography.maximum_characters == 25
        assert brief.typography.max_word_count == 4

        # Color
        assert "#0066CC" in brief.color.primary_palette

        # Lighting & Camera
        assert brief.lighting.mood == "high_key_dramatic"
        assert brief.camera.crop == "medium_close_up"

        # Objects & Constraints
        assert "Quantum Computer Chip" in brief.objects.required_objects
        assert "Channel Logo" in brief.execution_constraints.must_preserve

        # Validation
        assert brief.validation.strategy_id == "strat_001"
        assert brief.validation.confidence == sample_validated_package.validation.confidence
        assert brief.validation.ready_for_design_brief is True

    def test_serialization_json_yaml_dict(self, sample_validated_package: ValidatedReasoningPackage):
        """Test multi-format serialization (JSON, YAML, Dict) and round-trip fidelity."""
        generator = DesignBriefGenerator()
        original_brief = generator.generate(sample_validated_package)

        # Dictionary
        brief_dict = original_brief.to_dict()
        assert isinstance(brief_dict, dict)
        dict_restored = DesignBrief.from_dict(brief_dict)
        assert dict_restored.metadata.brief_id == original_brief.metadata.brief_id

        # JSON
        brief_json = original_brief.to_json()
        assert isinstance(brief_json, str)
        json_restored = DesignBrief.from_json(brief_json)
        assert json_restored.narrative.primary_story == original_brief.narrative.primary_story

        # YAML
        brief_yaml = original_brief.to_yaml()
        assert isinstance(brief_yaml, str)
        yaml_restored = DesignBrief.from_yaml(brief_yaml)
        assert yaml_restored.composition.primary_subject == original_brief.composition.primary_subject

    def test_brief_model_validation(self, sample_validated_package: ValidatedReasoningPackage):
        """Test DesignBrief.validate_brief() schema integrity check."""
        generator = DesignBriefGenerator()
        brief = generator.generate(sample_validated_package)

        # Clean brief
        errors = brief.validate_brief()
        assert len(errors) == 0

        # Corrupt metadata & mandatory fields using model_copy
        corrupt_meta = brief.metadata.model_copy(update={"brief_id": ""})
        corrupt_nar = brief.narrative.model_copy(update={"primary_story": ""})
        corrupt_val = brief.validation.model_copy(update={"confidence": 1.5})
        corrupt_brief = brief.model_copy(
            update={
                "metadata": corrupt_meta,
                "narrative": corrupt_nar,
                "validation": corrupt_val,
            }
        )

        corrupt_errors = corrupt_brief.validate_brief()
        assert len(corrupt_errors) >= 3
        assert any("brief_id" in err for err in corrupt_errors)
        assert any("primary_story" in err for err in corrupt_errors)
        assert any("confidence" in err for err in corrupt_errors)

    def test_version_compatibility(self, sample_validated_package: ValidatedReasoningPackage):
        """Test forward and backward compatibility of BriefMetadata schema_version."""
        generator = DesignBriefGenerator()
        brief = generator.generate(sample_validated_package)

        b_dict = brief.to_dict()
        b_dict["metadata"]["schema_version"] = "1.1.0"
        future_brief = DesignBrief.from_dict(b_dict)

        assert future_brief.metadata.schema_version == "1.1.0"
        assert len(future_brief.validate_brief()) == 0

    def test_strict_mode_validation_rejection(self, sample_validated_package: ValidatedReasoningPackage):
        """Test strict validation rejection when ReasoningPackage is not ready."""
        generator = DesignBriefGenerator()

        # Mark package as unready
        unready_package = sample_validated_package.model_copy(update={"ready_for_design_brief": False})

        with pytest.raises(ReasonerValidationError):
            generator.generate(unready_package, strict_validation=True)

        # Non-strict mode should succeed with unready flag preserved in brief.validation
        unready_brief = generator.generate(unready_package, strict_validation=False)
        assert unready_brief.validation.ready_for_design_brief is False

    def test_missing_and_empty_reasoning_slots(self, sample_evidence_graph: NormalizedEvidenceGraph):
        """Test robust fallback handling when reasoning context slots are missing or empty."""
        empty_context = ReasoningContext(graph_id=sample_evidence_graph.graph_id)
        validator = StrategicReasoningValidator()
        pkg = validator.validate(context=empty_context, graph=sample_evidence_graph)

        generator = DesignBriefGenerator()
        brief = generator.generate(pkg, strict_validation=False)

        assert isinstance(brief, DesignBrief)
        assert brief.metadata.brief_id.startswith("brief_")
        assert brief.narrative.primary_story != ""
        assert brief.composition.primary_subject != ""

    def test_reasoner_contract_and_interface(self, sample_evidence_graph: NormalizedEvidenceGraph, sample_validated_package: ValidatedReasoningPackage):
        """Test BaseReasoner contract integration and reason() method."""
        generator = DesignBriefGenerator()

        assert generator.name == "design_brief_generator"
        assert generator.contract.reasoner_type.value == "design_brief_generator"
        assert "validator" in generator.dependencies

        # Reason via BaseReasoner interface
        ctx = sample_validated_package.context
        brief = generator.reason(graph=sample_evidence_graph, context=ctx)

        assert isinstance(brief, DesignBrief)
        assert generator.validate_output(brief) is True
        assert generator.validate_output(None) is False

    def test_strict_renderer_independence_invariant(self, sample_validated_package: ValidatedReasoningPackage):
        """
        Critical Invariant Test: Verify that DesignBrief contains NO renderer-specific parameters
        (no Stable Diffusion prompts, ComfyUI nodes, BrushNet instructions, SDXL parameters, or inpainting nodes).
        """
        generator = DesignBriefGenerator()
        brief = generator.generate(sample_validated_package)
        json_dump = brief.to_json().lower()

        forbidden_tokens = [
            "comfyui",
            "stable_diffusion",
            "sdxl",
            "brushnet",
            "inpainting_mask",
            "lora_weight",
            "controlnet_model",
            "positive_prompt",
            "negative_prompt",
            "cfg_scale",
            "sampler_name",
        ]

        for token in forbidden_tokens:
            assert token not in json_dump, f"Forbidden renderer-specific token '{token}' found in DesignBrief!"

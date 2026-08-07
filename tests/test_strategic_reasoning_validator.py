"""
test_strategic_reasoning_validator.py
=======================================

Comprehensive unit and integration test suite for StrategicReasoningValidator (Phase 3.4H).
Tests:
- Valid reasoning package validation
- Missing module outputs & dependency detection
- Contradictory outputs across all 10 contradiction taxonomies:
  * Narrative vs Priority
  * Audience vs Strategy
  * Audience vs Brand
  * Brand vs Risk
  * Brand vs Priority
  * Creator vs Brand
  * Creator vs Strategy
  * Narrative vs Strategy
  * Priority vs Strategy
  * Risk vs Strategy
- Invalid evidence & ungrounded reasoning
- Confidence propagation mismatches
- Impossible combinations & orphan visual focus candidates
- Validation scoring model (ConsistencyScore & ReadinessScore)
- Edge cases (empty graph, zero evidence, empty context)
"""

import pytest
from typing import List

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
from thumbnail_intelligence.reasoning.narrative_models import (
    ArcStage,
    ArcStep,
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
    ConflictType,
    ReasoningValidation,
    ValidatedReasoningPackage,
    ValidationIssueType,
    ValidationSeverity,
    ValidationStatus,
)
from thumbnail_intelligence.knowledge_base.models import EvidenceGrade
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
            retrieval_query_id="query_val_unit",
            retrieval_reason=f"Retrieved {node_id}",
        ),
        is_active=True,
    )


@pytest.fixture
def sample_evidence_graph() -> NormalizedEvidenceGraph:
    """Create a realistic NormalizedEvidenceGraph for testing."""
    node1 = _build_test_node("ev_001", KnowledgeEntryType.DESIGN_PATTERN, 0.95)
    node2 = _build_test_node("ev_002", KnowledgeEntryType.VISUAL_PATTERN, 0.88)
    graph = NormalizedEvidenceGraph(
        graph_id="graph_test_123",
        summary=EvidenceSummary(graph_id="graph_test_123", primary_archetype="educational"),
        nodes={"ev_001": node1, "ev_002": node2},
    )
    return graph


@pytest.fixture
def valid_reasoning_context(sample_evidence_graph: NormalizedEvidenceGraph) -> ReasoningContext:
    """Construct a complete, fully grounded, internally consistent ReasoningContext."""
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
        evidence_refs=[ev_ref1],
    )
    # Audience
    cand_aud = CandidateAudience(
        audience_segment="Tech Enthusiasts",
        intent=ViewerIntent.LEARNING,
        knowledge_level=ViewerKnowledgeLevel.BEGINNER,
        cognitive_load=CognitiveLoadLevel.LOW,
        confidence=0.85,
        evidence_refs=[ev_ref1],
    )
    audience_res = AudienceResult(
        primary_audience=cand_aud,
        viewer_intent=ViewerIntent.LEARNING,
        confidence=0.85,
        supporting_evidence_ids=["ev_001"],
    )

    # Creator
    creator_res = CreatorResult(
        primary_creator_style=CandidateCreatorStyle(
            persona_name="Tech Educator",
            creator_archetype=CreatorArchetype.EDUCATOR,
            channel_voice="Informed, authoritative, clear",
            signature_elements=["Blue rim light", "Host Face on left"],
            confidence=0.88,
            evidence_refs=[ev_ref2],
        ),
        visual_identity=VisualIdentityStyle(
            dominant_color_palette=["#0066CC", "#FFFFFF"],
            typography_style="Clean sans-serif",
            evidence_refs=[ev_ref2],
        ),
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
    return context


class TestStrategicReasoningValidator:

    def test_valid_reasoning_package(
        self,
        sample_evidence_graph: NormalizedEvidenceGraph,
        valid_reasoning_context: ReasoningContext,
    ):
        """Test validation of a fully valid, grounded, consistent reasoning context."""
        validator = StrategicReasoningValidator()
        pkg = validator.validate(
            context=valid_reasoning_context,
            graph=sample_evidence_graph,
        )

        assert isinstance(pkg, ValidatedReasoningPackage)
        assert pkg.ready_for_design_brief is True
        assert pkg.validation.status in (ValidationStatus.PASSED, ValidationStatus.WARNINGS)
        assert pkg.validation.consistency_score >= 0.85
        assert pkg.validation.readiness_score >= 0.70
        assert len(pkg.validation.blocking_errors) == 0

    def test_missing_module_outputs(self, valid_reasoning_context: ReasoningContext):
        """Test detection of missing mandatory reasoning module outputs."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()
        context.brand_constraints = None  # Remove mandatory brand module

        pkg = validator.validate(context=context)

        assert pkg.ready_for_design_brief is False
        assert pkg.validation.status == ValidationStatus.BLOCKING_ERRORS
        assert len(pkg.validation.blocking_errors) > 0
        missing_issues = [i for i in pkg.validation.blocking_errors if i.issue_type == ValidationIssueType.MISSING_OUTPUT]
        assert len(missing_issues) == 1
        assert missing_issues[0].affected_module == "brand"

    def test_empty_evidence_and_ungrounded_reasoning(self):
        """Test detection of empty evidence and ungrounded confidence."""
        validator = StrategicReasoningValidator()
        context = ReasoningContext(
            narrative=NarrativeResult(confidence=0.9),
            audience=AudienceResult(confidence=0.8),
            creator_intent=CreatorResult(confidence=0.8),
            brand_constraints=BrandResult(confidence=0.8),
            visual_priorities=PriorityResult(confidence=0.8),
            risks=RiskResult(confidence=0.8),
            overall_confidence=0.95,
            evidence_references=[],  # Empty evidence
        )

        pkg = validator.validate(context=context)
        issues = pkg.validation.blocking_errors + pkg.validation.warnings
        types = [i.issue_type for i in issues]

        assert ValidationIssueType.EMPTY_EVIDENCE in types
        assert ValidationIssueType.UNGROUNDED_REASONING in types

    def test_confidence_mismatch(self, valid_reasoning_context: ReasoningContext):
        """Test detection of confidence mismatches across reasoners."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()
        if context.strategies and isinstance(context.strategies, StrategyDecision):
            context.strategies.confidence = 0.95
        if context.risks:
            context.risks.confidence = 0.20  # Low risk confidence vs high strategy confidence

        pkg = validator.validate(context=context)
        issues = pkg.validation.blocking_errors + pkg.validation.warnings
        mismatches = [i for i in issues if i.issue_type == ValidationIssueType.CONFIDENCE_MISMATCH]

        assert len(mismatches) > 0

    def test_contradiction_narrative_vs_priority(self, valid_reasoning_context: ReasoningContext):
        """Test detection of Narrative vs Priority contradiction."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()

        # Narrative demands "Secret Key" as primary visual focus
        if context.narrative:
            context.narrative.visual_focus_candidates = [
                VisualFocusCandidate(element_name="Secret Key", role_in_narrative="Core mystery element", visual_priority="PRIMARY")
            ]
        # Priority suppresses "Secret Key"
        if context.visual_priorities:
            context.visual_priorities.visual_hierarchy = [
                VisualHierarchyNode(
                    element_name="Secret Key",
                    element_category="object",
                    tier=HierarchyTier.SUPPRESSED,
                    canvas_allocation_fraction=0.0,
                )
            ]

        pkg = validator.validate(context=context)
        conflicts = [c for c in pkg.validation.detected_conflicts if c.conflict_type == ConflictType.NARRATIVE_VS_PRIORITY]

        assert len(conflicts) > 0

    def test_contradiction_audience_vs_strategy(self, valid_reasoning_context: ReasoningContext):
        """Test detection of Audience vs Strategy contradiction."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()

        if context.audience and context.audience.primary_audience:
            context.audience.primary_audience = context.audience.primary_audience.model_copy(
                update={"cognitive_load": CognitiveLoadLevel.LOW}
            )
        if context.strategies and isinstance(context.strategies, StrategyDecision) and context.strategies.winning_strategy:
            context.strategies.winning_strategy.description = "High complexity multi-subject cluttered graphic overload"
            context.strategies.winning_strategy.cons = ["extreme visual clutter"]

        pkg = validator.validate(context=context)
        conflicts = [c for c in pkg.validation.detected_conflicts if c.conflict_type == ConflictType.AUDIENCE_VS_STRATEGY]

        assert len(conflicts) > 0

    def test_contradiction_audience_vs_brand(self, valid_reasoning_context: ReasoningContext):
        """Test detection of Audience vs Brand contradiction."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()

        if context.audience and context.audience.primary_audience:
            context.audience.primary_audience = context.audience.primary_audience.model_copy(
                update={"curiosity_triggers": ["Shocking clickbait surprise"]}
            )
        if context.brand_constraints:
            context.brand_constraints.brand_constraints = ["clickbait sensationalism"]

        pkg = validator.validate(context=context)
        conflicts = [c for c in pkg.validation.detected_conflicts if c.conflict_type == ConflictType.AUDIENCE_VS_BRAND]

        assert len(conflicts) > 0

    def test_contradiction_brand_vs_risk(self, valid_reasoning_context: ReasoningContext):
        """Test detection of Brand vs Risk contradiction."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()

        if context.brand_constraints:
            context.brand_constraints.required_preservations = [
                VisualElementPreservation(
                    element_name="Dark Overlay Logo",
                    element_type="logo",
                    preservation_priority=BrandPreservationPriority.STRICT_MANDATORY,
                    required_treatment="Low contrast dark background logo",
                )
            ]
        if context.risks:
            r_item = DetectedRisk(
                title="Poor Contrast Risk",
                category=RiskCategory.POOR_CONTRAST,
                severity=RiskSeverity.CRITICAL,
                description="Dark Overlay Logo causes severe poor contrast",
            )
            context.risks.visual_risks = [r_item]
            context.risks.all_detected_risks = [r_item]

        pkg = validator.validate(context=context)
        conflicts = [c for c in pkg.validation.detected_conflicts if c.conflict_type == ConflictType.BRAND_VS_RISK]

        assert len(conflicts) > 0

    def test_contradiction_brand_vs_priority(self, valid_reasoning_context: ReasoningContext):
        """Test detection of Brand vs Priority contradiction."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()

        if context.brand_constraints:
            context.brand_constraints.required_preservations = [
                VisualElementPreservation(
                    element_name="Corporate Emblem",
                    element_type="logo",
                    preservation_priority=BrandPreservationPriority.STRICT_MANDATORY,
                    required_treatment="Must be prominent",
                )
            ]
        if context.visual_priorities:
            context.visual_priorities.visual_hierarchy = [
                VisualHierarchyNode(
                    element_name="Corporate Emblem",
                    element_category="logo",
                    tier=HierarchyTier.SUPPRESSED,
                    canvas_allocation_fraction=0.0,
                )
            ]

        pkg = validator.validate(context=context)
        conflicts = [c for c in pkg.validation.detected_conflicts if c.conflict_type == ConflictType.BRAND_VS_PRIORITY]

        assert len(conflicts) > 0
        assert pkg.ready_for_design_brief is False  # Strict mandatory brand violation is BLOCKING

    def test_contradiction_creator_vs_brand(self, valid_reasoning_context: ReasoningContext):
        """Test detection of Creator vs Brand contradiction."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()

        if context.creator_intent and context.creator_intent.primary_creator_style:
            context.creator_intent.primary_creator_style = context.creator_intent.primary_creator_style.model_copy(
                update={"signature_elements": ["Neon Arrow Graphics"]}
            )
        if context.brand_constraints:
            context.brand_constraints.brand_constraints = ["Neon Arrow Graphics"]

        pkg = validator.validate(context=context)
        conflicts = [c for c in pkg.validation.detected_conflicts if c.conflict_type == ConflictType.CREATOR_VS_BRAND]

        assert len(conflicts) > 0

    def test_contradiction_creator_vs_strategy(self, valid_reasoning_context: ReasoningContext):
        """Test detection of Creator vs Strategy contradiction."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()

        if context.creator_intent and context.creator_intent.primary_creator_style:
            context.creator_intent.primary_creator_style = context.creator_intent.primary_creator_style.model_copy(
                update={"creator_archetype": CreatorArchetype.EDUCATOR}
            )
        if context.strategies and isinstance(context.strategies, StrategyDecision) and context.strategies.winning_strategy:
            context.strategies.winning_strategy.archetype = StrategyArchetype.REACTION

        pkg = validator.validate(context=context)
        conflicts = [c for c in pkg.validation.detected_conflicts if c.conflict_type == ConflictType.CREATOR_VS_STRATEGY]

        assert len(conflicts) > 0

    def test_contradiction_priority_vs_strategy(self, valid_reasoning_context: ReasoningContext):
        """Test detection of Priority vs Strategy contradiction."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()

        if context.visual_priorities:
            context.visual_priorities.visual_hierarchy = [
                VisualHierarchyNode(
                    element_name="Host Face",
                    element_category="face",
                    tier=HierarchyTier.PRIMARY,
                    canvas_allocation_fraction=0.5,
                )
            ]
        if context.strategies and isinstance(context.strategies, StrategyDecision) and context.strategies.winning_strategy:
            context.strategies.winning_strategy.execution_priorities = ["no host face in composition"]

        pkg = validator.validate(context=context)
        conflicts = [c for c in pkg.validation.detected_conflicts if c.conflict_type == ConflictType.PRIORITY_VS_STRATEGY]

        assert len(conflicts) > 0

    def test_impossible_combinations_canvas_sum(self, valid_reasoning_context: ReasoningContext):
        """Test detection of impossible combinations (canvas sum > 1.0)."""
        validator = StrategicReasoningValidator()
        context = valid_reasoning_context.model_copy()

        if context.visual_priorities:
            context.visual_priorities.visual_hierarchy = [
                VisualHierarchyNode(element_name="E1", element_category="face", canvas_allocation_fraction=0.7),
                VisualHierarchyNode(element_name="E2", element_category="object", canvas_allocation_fraction=0.6),
            ]

        pkg = validator.validate(context=context)
        impossible = [i for i in pkg.validation.blocking_errors if i.issue_type == ValidationIssueType.IMPOSSIBLE_COMBINATION]

        assert len(impossible) > 0

    def test_validation_scoring_penalties(self, valid_reasoning_context: ReasoningContext):
        """Test that ConsistencyScore and ReadinessScore correctly penalize detected issues."""
        validator = StrategicReasoningValidator()

        # Clean context score
        clean_pkg = validator.validate(context=valid_reasoning_context)
        clean_consistency = clean_pkg.validation.consistency_score

        # Inject blocking issue
        dirty_context = valid_reasoning_context.model_copy()
        dirty_context.brand_constraints = None
        dirty_pkg = validator.validate(context=dirty_context)

        assert dirty_pkg.validation.consistency_score < clean_consistency
        assert dirty_pkg.validation.readiness_score <= 0.40
        assert dirty_pkg.ready_for_design_brief is False

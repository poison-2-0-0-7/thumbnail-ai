"""
validator.py
============

Production StrategicReasoningValidator for Thumbnail AI (Phase 3.4H).
Verifies that all strategic reasoning outputs (Narrative, Audience, Creator, Brand, Priority, Risk, StrategyRanker)
are internally consistent, grounded in valid evidence, and free of contradictions prior to DesignBrief generation.

It DOES NOT redesign.
It DOES NOT rerank.
It DOES NOT generate prompts.
It DOES NOT generate images.
It DOES NOT modify any reasoning outputs.
It ONLY validates.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from thumbnail_intelligence.evidence.models import (
    EvidenceReference,
    NormalizedEvidenceGraph,
)
from thumbnail_intelligence.reasoning.audience_models import (
    AudienceResult,
    CognitiveLoadLevel,
    ViewerIntent,
    ViewerKnowledgeLevel,
)
from thumbnail_intelligence.reasoning.brand_models import (
    BrandPreservationPriority,
    BrandResult,
)
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.creator_models import (
    CreatorArchetype,
    CreatorResult,
)
from thumbnail_intelligence.reasoning.interfaces import (
    BaseReasoner,
)
from thumbnail_intelligence.reasoning.models import (
    ReasonerContract,
    ReasonerType,
)
from thumbnail_intelligence.reasoning.narrative_models import (
    ArcStage,
    NarrativeResult,
    NarrativeType,
)
from thumbnail_intelligence.reasoning.priority_models import (
    HierarchyTier,
    PriorityResult,
)
from thumbnail_intelligence.reasoning.risk_models import (
    RiskResult,
    RiskSeverity,
)
from thumbnail_intelligence.reasoning.strategy_models import (
    StrategyArchetype,
    StrategyCandidate,
    StrategyDecision,
)
from thumbnail_intelligence.reasoning.validator_models import (
    ConflictType,
    DetectedConflict,
    ReasoningValidation,
    ValidatedReasoningPackage,
    ValidationIssue,
    ValidationIssueType,
    ValidationSeverity,
    ValidationStatus,
    ValidationTraceStep,
)


class StrategicReasoningValidator(BaseReasoner):
    """
    Strategic Reasoning Verification & Invariant Checking Engine (Phase 3.4H).
    Validates internal consistency, cross-module alignment, evidence grounding,
    and decision readiness across all Phase 3.4 reasoning artifacts.
    """

    def __init__(
        self,
        name: str = "strategic_reasoning_validator",
        version: str = "1.0.0",
        is_mandatory: bool = True,
        timeout_ms: float = 5000.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.CUSTOM,
            dependencies=[
                "narrative_reasoner",
                "audience_reasoner",
                "creator_reasoner",
                "brand_reasoner",
                "priority_reasoner",
                "risk_reasoner",
                "strategy_ranker",
            ],
            version=version,
            description="Verifies cross-module internal consistency, grounding, and readiness before DesignBrief generation",
            is_mandatory=is_mandatory,
            timeout_ms=timeout_ms,
        )
        self.config = config or {}

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(
        self,
        graph: NormalizedEvidenceGraph,
        context: ReasoningContext,
    ) -> ValidatedReasoningPackage:
        """
        BaseReasoner contract method.
        Validates the ReasoningContext against the NormalizedEvidenceGraph and returns a ValidatedReasoningPackage.
        """
        return self.validate(context=context, graph=graph)

    def validate_output(self, output: Any) -> bool:
        """Validate output contract."""
        if isinstance(output, ValidatedReasoningPackage):
            return isinstance(output.validation, ReasoningValidation)
        if isinstance(output, ReasoningValidation):
            return 0.0 <= output.consistency_score <= 1.0
        return False

    def validate(
        self,
        context: ReasoningContext,
        strategy_decision: Optional[StrategyDecision] = None,
        graph: Optional[NormalizedEvidenceGraph] = None,
        narrative: Optional[NarrativeResult] = None,
        audience: Optional[AudienceResult] = None,
        creator: Optional[CreatorResult] = None,
        brand: Optional[BrandResult] = None,
        priority: Optional[PriorityResult] = None,
        risk: Optional[RiskResult] = None,
    ) -> ValidatedReasoningPackage:
        """
        Master validation entrypoint. Accepts ReasoningContext and optional explicit outputs,
        performs exhaustive consistency, grounding, and contradiction checks, and constructs
        the final ValidatedReasoningPackage.
        """
        t_start = time.perf_counter()

        # 1. Resolve effective reasoning outputs
        eff_narrative = narrative or context.narrative
        eff_audience = audience or context.audience
        eff_creator = creator or context.creator_intent
        eff_brand = brand or context.brand_constraints
        eff_priority = priority or context.visual_priorities
        eff_risk = risk or context.risks

        eff_strategy: Optional[StrategyDecision] = strategy_decision
        if eff_strategy is None:
            if context.strategies is not None and isinstance(context.strategies, StrategyDecision):
                eff_strategy = context.strategies
            elif context.strategies is not None:
                # Upcast or wrap if needed
                eff_strategy = getattr(context, "strategies", None)

        trace_steps: List[ValidationTraceStep] = []
        all_issues: List[ValidationIssue] = []
        detected_conflicts: List[DetectedConflict] = []
        all_evidence_refs: List[EvidenceReference] = list(context.evidence_references or [])

        # --- CHECK 1: Missing Outputs & Mandatory Dependencies ---
        c1_start = time.perf_counter()
        c1_issues, c1_conflicts = self._check_missing_outputs_and_dependencies(
            context, eff_narrative, eff_audience, eff_creator, eff_brand, eff_priority, eff_risk, eff_strategy
        )
        c1_dur = (time.perf_counter() - c1_start) * 1000.0
        all_issues.extend(c1_issues)
        detected_conflicts.extend(c1_conflicts)
        trace_steps.append(
            ValidationTraceStep(
                check_name="missing_outputs_and_dependencies",
                status="FAILED" if any(i.severity == ValidationSeverity.BLOCKING for i in c1_issues) else "PASSED",
                duration_ms=c1_dur,
                details=f"Evaluated mandatory module presence. Identified {len(c1_issues)} missing/dependency issues.",
                issues_found=len(c1_issues),
            )
        )

        # --- CHECK 2: Grounding, Evidence Integrity & References ---
        c2_start = time.perf_counter()
        c2_issues = self._check_grounding_and_evidence(
            graph, context, eff_narrative, eff_audience, eff_creator, eff_brand, eff_priority, eff_risk, eff_strategy
        )
        c2_dur = (time.perf_counter() - c2_start) * 1000.0
        all_issues.extend(c2_issues)
        trace_steps.append(
            ValidationTraceStep(
                check_name="grounding_and_evidence_integrity",
                status="WARNING" if c2_issues else "PASSED",
                duration_ms=c2_dur,
                details=f"Verified evidence grounding. Found {len(c2_issues)} evidence/referential issues.",
                issues_found=len(c2_issues),
            )
        )

        # --- CHECK 3: Confidence Propagation & Consistency ---
        c3_start = time.perf_counter()
        c3_issues = self._check_confidence_consistency(
            context, eff_narrative, eff_audience, eff_creator, eff_brand, eff_priority, eff_risk, eff_strategy
        )
        c3_dur = (time.perf_counter() - c3_start) * 1000.0
        all_issues.extend(c3_issues)
        trace_steps.append(
            ValidationTraceStep(
                check_name="confidence_propagation_consistency",
                status="WARNING" if c3_issues else "PASSED",
                duration_ms=c3_dur,
                details=f"Validated confidence propagation. Identified {len(c3_issues)} confidence mismatches.",
                issues_found=len(c3_issues),
            )
        )

        # --- CHECK 4: Circular Reasoning & Dependency Trace ---
        c4_start = time.perf_counter()
        c4_issues = self._check_circular_reasoning(context)
        c4_dur = (time.perf_counter() - c4_start) * 1000.0
        all_issues.extend(c4_issues)
        trace_steps.append(
            ValidationTraceStep(
                check_name="circular_reasoning_detection",
                status="FAILED" if c4_issues else "PASSED",
                duration_ms=c4_dur,
                details=f"Checked trace DAG for circular loops. Identified {len(c4_issues)} circularities.",
                issues_found=len(c4_issues),
            )
        )

        # --- CHECK 5: Cross-Module Contradictions ---
        c5_start = time.perf_counter()
        c5_issues, c5_conflicts = self._check_cross_module_contradictions(
            eff_narrative, eff_audience, eff_creator, eff_brand, eff_priority, eff_risk, eff_strategy
        )
        c5_dur = (time.perf_counter() - c5_start) * 1000.0
        all_issues.extend(c5_issues)
        detected_conflicts.extend(c5_conflicts)
        trace_steps.append(
            ValidationTraceStep(
                check_name="cross_module_contradiction_detection",
                status="FAILED" if any(i.severity in (ValidationSeverity.BLOCKING, ValidationSeverity.CRITICAL) for i in c5_issues) else ("WARNING" if c5_issues else "PASSED"),
                duration_ms=c5_dur,
                details=f"Evaluated 10 cross-module contradiction taxonomies. Found {len(c5_conflicts)} conflicts.",
                issues_found=len(c5_issues),
            )
        )

        # --- CHECK 6: Invariant Bounds, Orphans & Impossible Combinations ---
        c6_start = time.perf_counter()
        c6_issues = self._check_invariants_orphans_impossible(
            eff_narrative, eff_audience, eff_creator, eff_brand, eff_priority, eff_risk, eff_strategy
        )
        c6_dur = (time.perf_counter() - c6_start) * 1000.0
        all_issues.extend(c6_issues)
        trace_steps.append(
            ValidationTraceStep(
                check_name="invariants_orphans_impossible_combinations",
                status="WARNING" if c6_issues else "PASSED",
                duration_ms=c6_dur,
                details=f"Verified canvas bounds and impossible combinations. Found {len(c6_issues)} issues.",
                issues_found=len(c6_issues),
            )
        )

        # Categorize issues by severity & type
        blocking_errors: List[ValidationIssue] = [i for i in all_issues if i.severity == ValidationSeverity.BLOCKING]
        warnings: List[ValidationIssue] = [i for i in all_issues if i.severity != ValidationSeverity.BLOCKING]
        contradiction_issues: List[ValidationIssue] = [i for i in all_issues if i.issue_type == ValidationIssueType.CONTRADICTION]

        # Extract resolution suggestions
        resolution_suggestions: List[str] = []
        for issue in all_issues:
            if issue.suggested_resolution and issue.suggested_resolution not in resolution_suggestions:
                resolution_suggestions.append(issue.suggested_resolution)
        for conflict in detected_conflicts:
            if conflict.suggested_resolution and conflict.suggested_resolution not in resolution_suggestions:
                resolution_suggestions.append(conflict.suggested_resolution)

        # --- SCORING CALCULATION ---
        consistency_score = self._compute_consistency_score(all_issues, detected_conflicts)
        readiness_score = self._compute_readiness_score(
            consistency_score, context, eff_narrative, eff_audience, eff_creator, eff_brand, eff_priority, eff_risk, eff_strategy, blocking_errors
        )
        validation_confidence = self._compute_validation_confidence(graph, context, trace_steps, all_issues)

        # Determine ReadyForDesignBrief gate
        ready_for_design_brief = (readiness_score >= 0.70) and (len(blocking_errors) == 0)

        # Determine overall ValidationStatus
        if len(blocking_errors) > 0:
            status = ValidationStatus.BLOCKING_ERRORS
        elif readiness_score < 0.50 or any(i.severity == ValidationSeverity.CRITICAL for i in all_issues):
            status = ValidationStatus.FAILED
        elif len(warnings) > 0 or len(detected_conflicts) > 0:
            status = ValidationStatus.WARNINGS
        else:
            status = ValidationStatus.PASSED

        validation_report = ReasoningValidation(
            validation_id=f"val_{uuid.uuid4().hex[:8]}",
            status=status,
            consistency_score=consistency_score,
            readiness_score=readiness_score,
            ready_for_design_brief=ready_for_design_brief,
            blocking_errors=blocking_errors,
            warnings=warnings,
            detected_conflicts=detected_conflicts,
            contradictions=contradiction_issues,
            resolution_suggestions=resolution_suggestions,
            validation_trace=trace_steps,
            evidence_references=all_evidence_refs,
            confidence=validation_confidence,
        )

        return ValidatedReasoningPackage(
            package_id=f"pkg_{uuid.uuid4().hex[:8]}",
            context=context,
            strategy_decision=eff_strategy,
            validation=validation_report,
            ready_for_design_brief=ready_for_design_brief,
        )

    # ---------------------------------------------------------------------------
    # Internal Validation Check Implementations
    # ---------------------------------------------------------------------------

    def _check_missing_outputs_and_dependencies(
        self,
        context: ReasoningContext,
        narrative: Optional[NarrativeResult],
        audience: Optional[AudienceResult],
        creator: Optional[CreatorResult],
        brand: Optional[BrandResult],
        priority: Optional[PriorityResult],
        risk: Optional[RiskResult],
        strategy: Optional[StrategyDecision],
    ) -> Tuple[List[ValidationIssue], List[DetectedConflict]]:
        issues: List[ValidationIssue] = []
        conflicts: List[DetectedConflict] = []

        modules: Dict[str, Optional[Any]] = {
            "narrative": narrative,
            "audience": audience,
            "creator": creator,
            "brand": brand,
            "priority": priority,
            "risk": risk,
            "strategy_ranker": strategy,
        }

        # Mandatory module presence
        for mod_name, mod_val in modules.items():
            if mod_val is None:
                issues.append(
                    ValidationIssue(
                        issue_type=ValidationIssueType.MISSING_OUTPUT,
                        severity=ValidationSeverity.BLOCKING,
                        reason=f"Mandatory reasoning output '{mod_name}' is missing from ReasoningContext.",
                        affected_module=mod_name,
                        suggested_resolution=f"Ensure '{mod_name}' reasoner is executed prior to validation.",
                    )
                )

        # Check missing dependencies: if downstream exists without upstream
        if priority is not None and (narrative is None or audience is None):
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.MISSING_DEPENDENCY,
                    severity=ValidationSeverity.BLOCKING,
                    reason="PriorityReasoner executed but upstream Narrative or Audience outputs are missing.",
                    affected_module="priority",
                    suggested_resolution="Execute Narrative and Audience reasoners before PriorityReasoner.",
                )
            )

        if strategy is not None and (risk is None or priority is None or brand is None):
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.MISSING_DEPENDENCY,
                    severity=ValidationSeverity.BLOCKING,
                    reason="StrategyRanker executed but upstream Brand, Priority, or Risk outputs are missing.",
                    affected_module="strategy_ranker",
                    suggested_resolution="Ensure full reasoning DAG (Brand, Priority, Risk) finishes before StrategyRanker.",
                )
            )

        return issues, conflicts

    def _check_grounding_and_evidence(
        self,
        graph: Optional[NormalizedEvidenceGraph],
        context: ReasoningContext,
        narrative: Optional[NarrativeResult],
        audience: Optional[AudienceResult],
        creator: Optional[CreatorResult],
        brand: Optional[BrandResult],
        priority: Optional[PriorityResult],
        risk: Optional[RiskResult],
        strategy: Optional[StrategyDecision],
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # Check for empty evidence in context
        if not context.evidence_references and context.get_evidence_count() == 0:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.EMPTY_EVIDENCE,
                    severity=ValidationSeverity.CRITICAL,
                    reason="ReasoningContext contains zero grounding evidence references.",
                    affected_module="context",
                    suggested_resolution="Populate NormalizedEvidenceGraph and retain evidence references in context.",
                )
            )

        # Check ungrounded reasoning (confidence > 0 with zero evidence)
        if context.overall_confidence > 0.5 and context.get_evidence_count() == 0:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.UNGROUNDED_REASONING,
                    severity=ValidationSeverity.BLOCKING,
                    reason="Overall context confidence is high (>0.5) despite zero supporting evidence references.",
                    affected_module="context",
                    suggested_resolution="Enforce grounding verification or calibrate overall confidence down to 0.0.",
                )
            )

        # Check evidence references in graph if graph provided
        if graph is not None:
            valid_node_ids = set(graph.nodes.keys())
            for ref in context.evidence_references:
                ref_id = getattr(ref, "source_id", None) or getattr(ref, "evidence_id", None)
                if ref_id and ref_id not in valid_node_ids:
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.UNGROUNDED_REASONING,
                            severity=ValidationSeverity.CRITICAL,
                            reason=f"Evidence reference '{ref_id}' is not present in the normalized evidence graph.",
                            affected_module="evidence_graph",
                            suggested_resolution="Remove dangling reference or retrieve node into evidence graph.",
                        )
                    )

        # Per-module evidence checking
        mod_refs: Dict[str, Tuple[List[Any], List[str]]] = {
            "narrative": (narrative.evidence_refs if narrative else [], narrative.supporting_evidence_ids if narrative else []),
            "audience": (audience.evidence_refs if audience else [], audience.supporting_evidence_ids if audience else []),
            "creator": (creator.evidence_refs if creator else [], creator.supporting_evidence_ids if creator else []),
            "brand": (brand.evidence_refs if brand else [], brand.supporting_evidence_ids if brand else []),
            "priority": (priority.evidence_refs if priority else [], priority.supporting_evidence_ids if priority else []),
            "risk": (risk.evidence_refs if risk else [], risk.supporting_evidence_ids if risk else []),
            "strategy": (strategy.evidence_refs if strategy else [], strategy.supporting_evidence_ids if strategy else []),
        }

        for m_name, (refs, ids) in mod_refs.items():
            if (narrative if m_name == "narrative" else True) and not refs and not ids:
                issues.append(
                    ValidationIssue(
                        issue_type=ValidationIssueType.EMPTY_EVIDENCE,
                        severity=ValidationSeverity.WARNING,
                        reason=f"Reasoning module '{m_name}' produced outputs without supporting evidence references.",
                        affected_module=m_name,
                        suggested_resolution=f"Ensure '{m_name}' attaches evidence references to its output model.",
                    )
                )

        return issues

    def _check_confidence_consistency(
        self,
        context: ReasoningContext,
        narrative: Optional[NarrativeResult],
        audience: Optional[AudienceResult],
        creator: Optional[CreatorResult],
        brand: Optional[BrandResult],
        priority: Optional[PriorityResult],
        risk: Optional[RiskResult],
        strategy: Optional[StrategyDecision],
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        mod_conf: Dict[str, Optional[float]] = {
            "narrative": narrative.confidence if narrative else None,
            "audience": audience.confidence if audience else None,
            "creator": creator.confidence if creator else None,
            "brand": brand.confidence if brand else None,
            "priority": priority.confidence if priority else None,
            "risk": risk.confidence if risk else None,
        }

        # Strategy decision confidence vs underlying module confidence
        if strategy is not None and strategy.confidence > 0.85:
            low_mods = [name for name, c in mod_conf.items() if c is not None and c < 0.40]
            if low_mods:
                issues.append(
                    ValidationIssue(
                        issue_type=ValidationIssueType.CONFIDENCE_MISMATCH,
                        severity=ValidationSeverity.CRITICAL,
                        reason=(
                            f"StrategyDecision confidence is high ({strategy.confidence:.2f}), but underlying "
                            f"reasoners ({', '.join(low_mods)}) have low confidence (<0.40)."
                        ),
                        affected_module="strategy_vs_reasoners",
                        suggested_resolution="Penalize StrategyDecision confidence to reflect low upstream reasoner confidence.",
                    )
                )

        # Context overall confidence vs module mean
        valid_confs = [c for c in mod_conf.values() if c is not None]
        if valid_confs:
            mean_conf = sum(valid_confs) / len(valid_confs)
            if context.overall_confidence > mean_conf + 0.35:
                issues.append(
                    ValidationIssue(
                        issue_type=ValidationIssueType.CONFIDENCE_MISMATCH,
                        severity=ValidationSeverity.WARNING,
                        reason=(
                            f"ReasoningContext overall_confidence ({context.overall_confidence:.2f}) significantly "
                            f"exceeds average reasoner confidence ({mean_conf:.2f})."
                        ),
                        affected_module="context",
                        suggested_resolution="Re-aggregate overall context confidence using minimum or weighted mean of active reasoners.",
                    )
                )

        return issues

    def _check_circular_reasoning(self, context: ReasoningContext) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # Check reasoning trace for repeated execute loops
        executed: Set[str] = set()
        for step in context.reasoning_trace:
            if step.action == "execute":
                if step.reasoner_name in executed and step.reasoner_name != "coordinator":
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.CIRCULAR_REASONING,
                            severity=ValidationSeverity.BLOCKING,
                            reason=f"Detected multiple execution steps for reasoner '{step.reasoner_name}', indicating a potential loop.",
                            affected_module=step.reasoner_name,
                            suggested_resolution="Verify topological ordering in ReasonerRegistry to prevent execution cycles.",
                        )
                    )
                executed.add(step.reasoner_name)

        return issues

    def _check_cross_module_contradictions(
        self,
        narrative: Optional[NarrativeResult],
        audience: Optional[AudienceResult],
        creator: Optional[CreatorResult],
        brand: Optional[BrandResult],
        priority: Optional[PriorityResult],
        risk: Optional[RiskResult],
        strategy: Optional[StrategyDecision],
    ) -> Tuple[List[ValidationIssue], List[DetectedConflict]]:
        issues: List[ValidationIssue] = []
        conflicts: List[DetectedConflict] = []

        # -------------------------------------------------------------------
        # 1. Narrative vs Priority
        # -------------------------------------------------------------------
        if narrative and priority:
            # If Narrative demands a visual focal element that Priority suppresses or drops to zero canvas allocation
            nav_focal_names = [f.element_name.lower() for f in narrative.visual_focus_candidates if str(getattr(f, "visual_priority", "")).upper() == "PRIMARY" or str(getattr(f, "priority", "")).upper() == "PRIMARY"]
            if not nav_focal_names and narrative.primary_narrative:
                nav_focal_names.append(narrative.primary_narrative.hook.lower())

            v_nodes = getattr(priority, "visual_hierarchy", None) or getattr(priority, "hierarchy_nodes", [])
            suppressed_names = [s.lower() for s in getattr(priority, "suppressed_elements", [])]

            for s_node in v_nodes:
                if s_node.tier == HierarchyTier.SUPPRESSED or s_node.canvas_allocation_fraction == 0:
                    suppressed_names.append(s_node.element_name.lower())

            for s_name in suppressed_names:
                for n_focal in nav_focal_names:
                    if s_name in n_focal or n_focal in s_name:
                        desc = f"Narrative identifies '{n_focal}' as primary visual focus, but Priority suppresses '{s_name}'."
                        conflicts.append(
                            DetectedConflict(
                                conflict_type=ConflictType.NARRATIVE_VS_PRIORITY,
                                source_module_a="narrative",
                                source_module_b="priority",
                                claim_a=f"Visual Focus Primary: {n_focal}",
                                claim_b=f"Hierarchy Element Suppressed: {s_name}",
                                severity=ValidationSeverity.CRITICAL,
                                description=desc,
                                suggested_resolution="Promote element in Priority hierarchy to Primary or Secondary tier.",
                            )
                        )
                        issues.append(
                            ValidationIssue(
                                issue_type=ValidationIssueType.CONTRADICTION,
                                severity=ValidationSeverity.CRITICAL,
                                reason=desc,
                                affected_module="narrative_vs_priority",
                                suggested_resolution="Align Priority hierarchy nodes with Narrative visual focus candidates.",
                            )
                        )

        # -------------------------------------------------------------------
        # 2. Audience vs Strategy
        # -------------------------------------------------------------------
        if audience and strategy and strategy.winning_strategy:
            win_strat = strategy.winning_strategy
            opt_cog = getattr(audience, "optimal_cognitive_load", None)
            if opt_cog is None and hasattr(audience, "primary_audience") and audience.primary_audience:
                opt_cog = getattr(audience.primary_audience, "cognitive_load", None)

            p_intent = getattr(audience, "primary_intent", None) or getattr(audience, "viewer_intent", None)
            if p_intent is None and hasattr(audience, "primary_audience") and audience.primary_audience:
                p_intent = getattr(audience.primary_audience, "intent", None)

            # Low cognitive load audience vs high complexity strategy
            if opt_cog == CognitiveLoadLevel.LOW:
                cons_text = " ".join(win_strat.cons + win_strat.failure_risks).lower()
                desc_text = win_strat.description.lower()
                if "clutter" in cons_text or "complex" in cons_text or "overload" in desc_text:
                    desc = f"Target audience requires LOW cognitive load, but winning strategy '{win_strat.title}' introduces visual complexity/clutter."
                    conflicts.append(
                        DetectedConflict(
                            conflict_type=ConflictType.AUDIENCE_VS_STRATEGY,
                            source_module_a="audience",
                            source_module_b="strategy_ranker",
                            claim_a="Optimal Cognitive Load: LOW",
                            claim_b=f"Winning Strategy: {win_strat.title} (high complexity/clutter)",
                            severity=ValidationSeverity.CRITICAL,
                            description=desc,
                            suggested_resolution="Select a minimalist or low-complexity strategy candidate for this audience.",
                        )
                    )
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.CONTRADICTION,
                            severity=ValidationSeverity.CRITICAL,
                            reason=desc,
                            affected_module="audience_vs_strategy",
                            suggested_resolution="Simplify visual layout to meet LOW cognitive load requirement.",
                        )
                    )

            # ViewerIntent Learning vs Pure Reaction/Challenge Archetype
            if p_intent in (ViewerIntent.LEARNING, ViewerIntent.PROBLEM_SOLVING):
                if win_strat.archetype in (StrategyArchetype.REACTION, StrategyArchetype.CHALLENGE, StrategyArchetype.HIGH_ENERGY):
                    intent_str = str(p_intent.value) if hasattr(p_intent, "value") else str(p_intent)
                    desc = f"Audience primary intent is {intent_str}, but winning strategy relies on {win_strat.archetype.value} archetype."
                    conflicts.append(
                        DetectedConflict(
                            conflict_type=ConflictType.AUDIENCE_VS_STRATEGY,
                            source_module_a="audience",
                            source_module_b="strategy_ranker",
                            claim_a=f"Primary Intent: {intent_str}",
                            claim_b=f"Strategy Archetype: {win_strat.archetype.value}",
                            severity=ValidationSeverity.WARNING,
                            description=desc,
                            suggested_resolution="Rerank or introduce Educational/Transformation strategy candidate.",
                        )
                    )
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.CONTRADICTION,
                            severity=ValidationSeverity.WARNING,
                            reason=desc,
                            affected_module="audience_vs_strategy",
                            suggested_resolution="Include educational value anchors in strategy execution priorities.",
                        )
                    )

        # -------------------------------------------------------------------
        # 3. Audience vs Brand
        # -------------------------------------------------------------------
        if audience and brand:
            a_trig = getattr(audience, "curiosity_triggers", []) + getattr(audience, "psychological_hooks", [])
            if hasattr(audience, "primary_audience") and audience.primary_audience:
                a_trig.extend(getattr(audience.primary_audience, "curiosity_triggers", []))
                a_trig.extend(getattr(audience.primary_audience, "psychological_hooks", []))

            aud_triggers = " ".join(a_trig).lower()
            brand_prohibited = " ".join(getattr(brand, "prohibited_elements", []) + getattr(brand, "forbidden_changes", []) + getattr(brand, "brand_constraints", [])).lower()
            if ("clickbait" in aud_triggers or "shock" in aud_triggers) and ("clickbait" in brand_prohibited or "sensational" in brand_prohibited):
                desc = "Audience curiosity triggers rely on shock/clickbait tactics prohibited by Brand visual guardrails."
                conflicts.append(
                    DetectedConflict(
                        conflict_type=ConflictType.AUDIENCE_VS_BRAND,
                        source_module_a="audience",
                        source_module_b="brand",
                        claim_a="Curiosity Triggers require shock/clickbait",
                        claim_b="Brand guardrails prohibit clickbait/sensationalism",
                        severity=ValidationSeverity.CRITICAL,
                        description=desc,
                        suggested_resolution="Use intrigue or mystery curiosity gaps instead of prohibited shock tactics.",
                    )
                )
                issues.append(
                    ValidationIssue(
                        issue_type=ValidationIssueType.CONTRADICTION,
                        severity=ValidationSeverity.CRITICAL,
                        reason=desc,
                        affected_module="audience_vs_brand",
                        suggested_resolution="Replace sensational triggers with brand-compliant curiosity hooks.",
                    )
                )

        # -------------------------------------------------------------------
        # 4. Brand vs Risk
        # -------------------------------------------------------------------
        if brand and risk:
            b_directives = getattr(brand, "preservation_directives", None) or getattr(brand, "required_preservations", [])
            brand_mandates = [p.element_name.lower() for p in b_directives if hasattr(p, "preservation_priority") and p.preservation_priority == BrandPreservationPriority.STRICT_MANDATORY]
            r_list = getattr(risk, "all_detected_risks", None) or getattr(risk, "detected_risks", [])
            high_risks = [r for r in r_list if hasattr(r, "severity") and r.severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH)]

            for r in high_risks:
                r_desc = r.description.lower()
                for mandate in brand_mandates:
                    if mandate in r_desc:
                        r_mit = getattr(r, "mitigation", None) or getattr(r, "mitigation_suggestion", "")
                        desc = f"Strict mandatory brand element '{mandate.title()}' causes {r.severity.value} risk ({r.description})."
                        conflicts.append(
                            DetectedConflict(
                                conflict_type=ConflictType.BRAND_VS_RISK,
                                source_module_a="brand",
                                source_module_b="risk",
                                claim_a=f"Mandatory Brand Element: {mandate.title()}",
                                claim_b=f"Detected Risk: {r.description}",
                                severity=ValidationSeverity.CRITICAL,
                                description=desc,
                                suggested_resolution=f"Apply risk mitigation ({r_mit}) while retaining brand core asset.",
                            )
                        )
                        issues.append(
                            ValidationIssue(
                                issue_type=ValidationIssueType.CONTRADICTION,
                                severity=ValidationSeverity.CRITICAL,
                                reason=desc,
                                affected_module="brand_vs_risk",
                                suggested_resolution="Enforce visual treatment mitigation for mandatory brand element.",
                            )
                        )

        # -------------------------------------------------------------------
        # 5. Brand vs Priority
        # -------------------------------------------------------------------
        if brand and priority:
            b_directives = getattr(brand, "preservation_directives", None) or getattr(brand, "required_preservations", [])
            mandatory_elements = [p.element_name.lower() for p in b_directives if hasattr(p, "preservation_priority") and p.preservation_priority == BrandPreservationPriority.STRICT_MANDATORY]
            v_nodes = getattr(priority, "visual_hierarchy", None) or getattr(priority, "hierarchy_nodes", [])
            suppressed_in_priority = [s.element_name.lower() for s in v_nodes if s.tier == HierarchyTier.SUPPRESSED or s.canvas_allocation_fraction == 0]

            for m_elem in mandatory_elements:
                if m_elem in suppressed_in_priority:
                    desc = f"Mandatory brand element '{m_elem.title()}' is suppressed in Priority visual hierarchy."
                    conflicts.append(
                        DetectedConflict(
                            conflict_type=ConflictType.BRAND_VS_PRIORITY,
                            source_module_a="brand",
                            source_module_b="priority",
                            claim_a=f"Strict Mandatory Element: {m_elem.title()}",
                            claim_b=f"Priority Status: Suppressed / Tier 0",
                            severity=ValidationSeverity.BLOCKING,
                            description=desc,
                            suggested_resolution="Promote mandatory brand element to non-zero canvas allocation in Priority hierarchy.",
                        )
                    )
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.CONTRADICTION,
                            severity=ValidationSeverity.BLOCKING,
                            reason=desc,
                            affected_module="brand_vs_priority",
                            suggested_resolution="Mandatory brand elements cannot be suppressed.",
                        )
                    )

        # -------------------------------------------------------------------
        # 6. Creator vs Brand
        # -------------------------------------------------------------------
        if creator and brand:
            c_style = getattr(creator, "primary_creator_style", None) or getattr(creator, "primary_style", None)
            c_sigs = (getattr(c_style, "signature_elements", []) if c_style else []) + getattr(creator, "signature_elements", [])
            c_sig = " ".join(c_sigs + getattr(creator, "brand_equity_anchors", [])).lower()
            b_pro = " ".join(getattr(brand, "prohibited_elements", []) + getattr(brand, "forbidden_changes", []) + getattr(brand, "brand_constraints", [])).lower()
            for sig in c_sigs:
                sig_l = sig.lower()
                if sig_l in b_pro:
                    desc = f"Creator signature element '{sig}' is strictly prohibited by Brand guidelines ({b_pro})."
                    conflicts.append(
                        DetectedConflict(
                            conflict_type=ConflictType.CREATOR_VS_BRAND,
                            source_module_a="creator",
                            source_module_b="brand",
                            claim_a=f"Creator Signature Element: {sig}",
                            claim_b=f"Brand Prohibited Element: {sig}",
                            severity=ValidationSeverity.CRITICAL,
                            description=desc,
                            suggested_resolution="Modify creator signature element to comply with corporate brand rules.",
                        )
                    )
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.CONTRADICTION,
                            severity=ValidationSeverity.CRITICAL,
                            reason=desc,
                            affected_module="creator_vs_brand",
                            suggested_resolution="Reconcile creator persona tropes with brand prohibited directives.",
                        )
                    )

        # -------------------------------------------------------------------
        # 7. Creator vs Strategy
        # -------------------------------------------------------------------
        if creator and strategy and strategy.winning_strategy:
            win_strat = strategy.winning_strategy
            c_arch = getattr(creator, "creator_archetype", None)
            if c_arch is None and hasattr(creator, "primary_style") and creator.primary_style:
                c_arch = getattr(creator.primary_style, "creator_archetype", None)
            if c_arch is None and hasattr(creator, "primary_creator_style") and creator.primary_creator_style:
                c_arch = getattr(creator.primary_creator_style, "creator_archetype", None)

            if c_arch in (CreatorArchetype.EDUCATOR, CreatorArchetype.INVESTIGATOR):
                if win_strat.archetype in (StrategyArchetype.REACTION, StrategyArchetype.CHALLENGE):
                    arch_str = str(c_arch.value) if hasattr(c_arch, "value") else str(c_arch)
                    desc = f"Creator archetype is {arch_str}, which conflicts with winning strategy archetype {win_strat.archetype.value}."
                    conflicts.append(
                        DetectedConflict(
                            conflict_type=ConflictType.CREATOR_VS_STRATEGY,
                            source_module_a="creator",
                            source_module_b="strategy_ranker",
                            claim_a=f"Creator Archetype: {arch_str}",
                            claim_b=f"Winning Strategy Archetype: {win_strat.archetype.value}",
                            severity=ValidationSeverity.CRITICAL,
                            description=desc,
                            suggested_resolution="Select a strategy archetype aligned with creator authority and channel voice.",
                        )
                    )
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.CONTRADICTION,
                            severity=ValidationSeverity.CRITICAL,
                            reason=desc,
                            affected_module="creator_vs_strategy",
                            suggested_resolution="Align strategy archetype with established creator channel voice.",
                        )
                    )

        # -------------------------------------------------------------------
        # 8. Narrative vs Strategy
        # -------------------------------------------------------------------
        if narrative and strategy and strategy.winning_strategy:
            win_strat = strategy.winning_strategy
            if narrative.narrative_arc and narrative.narrative_arc.dominant_stage == ArcStage.MYSTERY:
                s_title = win_strat.title.lower()
                s_desc = win_strat.description.lower()
                if "answer" in s_title or "revealed" in s_title or "explicit" in s_desc:
                    desc = f"Narrative emphasizes MYSTERY arc stage, but winning strategy '{win_strat.title}' reveals explicit resolution."
                    conflicts.append(
                        DetectedConflict(
                            conflict_type=ConflictType.NARRATIVE_VS_STRATEGY,
                            source_module_a="narrative",
                            source_module_b="strategy_ranker",
                            claim_a="Dominant Arc Stage: MYSTERY",
                            claim_b=f"Winning Strategy: {win_strat.title} (explicit reveal)",
                            severity=ValidationSeverity.WARNING,
                            description=desc,
                            suggested_resolution="Preserve curiosity gap in visual framing without spoiling mystery outcome.",
                        )
                    )
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.CONTRADICTION,
                            severity=ValidationSeverity.WARNING,
                            reason=desc,
                            affected_module="narrative_vs_strategy",
                            suggested_resolution="Adjust visual treatment to maintain mystery framing.",
                        )
                    )

        # -------------------------------------------------------------------
        # 9. Priority vs Strategy
        # -------------------------------------------------------------------
        if priority and strategy and strategy.winning_strategy:
            win_strat = strategy.winning_strategy
            p_nodes = getattr(priority, "visual_hierarchy", None) or getattr(priority, "hierarchy_nodes", [])
            p_primary_nodes = [n for n in p_nodes if n.tier == HierarchyTier.PRIMARY]
            exec_text = " ".join(win_strat.execution_priorities).lower()
            for p_node in p_primary_nodes:
                p_name = p_node.element_name.lower()
                if "no " + p_name in exec_text or "without " + p_name in exec_text or "avoid " + p_name in exec_text or "exclude " + p_name in exec_text:
                    desc = f"Priority assigns Primary visual tier to '{p_node.element_name}', but Strategy execution priorities omit or negate it ({exec_text})."
                    conflicts.append(
                        DetectedConflict(
                            conflict_type=ConflictType.PRIORITY_VS_STRATEGY,
                            source_module_a="priority",
                            source_module_b="strategy_ranker",
                            claim_a=f"Primary Visual Tier: {p_node.element_name}",
                            claim_b=f"Strategy Execution Priorities: {exec_text}",
                            severity=ValidationSeverity.CRITICAL,
                            description=desc,
                            suggested_resolution="Reconcile visual focal point priorities with strategy composition layout.",
                        )
                    )
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.CONTRADICTION,
                            severity=ValidationSeverity.CRITICAL,
                            reason=desc,
                            affected_module="priority_vs_strategy",
                            suggested_resolution="Ensure primary visual focus element is supported in strategy layout.",
                        )
                    )

        # -------------------------------------------------------------------
        # 10. Risk vs Strategy
        # -------------------------------------------------------------------
        if risk and strategy and strategy.winning_strategy:
            win_strat = strategy.winning_strategy
            r_list = getattr(risk, "all_detected_risks", None) or getattr(risk, "detected_risks", [])
            crit_risks = [r for r in r_list if hasattr(r, "severity") and r.severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH)]
            for c_risk in crit_risks:
                c_cat = c_risk.category.value
                if c_cat in " ".join(win_strat.failure_risks).lower() or c_cat in win_strat.description.lower():
                    r_mit = getattr(c_risk, "mitigation", None) or getattr(c_risk, "mitigation_suggestion", None)
                    if win_strat.risk_penalty < 0.05 and not r_mit:
                        desc = f"Winning strategy '{win_strat.title}' contains {c_risk.severity.value} risk '{c_cat}' without active mitigation or risk penalty."
                        conflicts.append(
                            DetectedConflict(
                                conflict_type=ConflictType.RISK_VS_STRATEGY,
                                source_module_a="risk",
                                source_module_b="strategy_ranker",
                                claim_a=f"{c_risk.severity.value} Risk: {c_cat}",
                                claim_b=f"Winning Strategy Penalty: {win_strat.risk_penalty}",
                                severity=ValidationSeverity.CRITICAL,
                                description=desc,
                                suggested_resolution="Apply risk penalty to candidate score or mandate risk mitigation plan.",
                            )
                        )
                        issues.append(
                            ValidationIssue(
                                issue_type=ValidationIssueType.CONTRADICTION,
                                severity=ValidationSeverity.CRITICAL,
                                reason=desc,
                                affected_module="risk_vs_strategy",
                                suggested_resolution="Enforce mitigation step for critical unmitigated risk.",
                            )
                        )

        return issues, conflicts

    def _check_invariants_orphans_impossible(
        self,
        narrative: Optional[NarrativeResult],
        audience: Optional[AudienceResult],
        creator: Optional[CreatorResult],
        brand: Optional[BrandResult],
        priority: Optional[PriorityResult],
        risk: Optional[RiskResult],
        strategy: Optional[StrategyDecision],
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # Check visual priority allocations sum > 1.05
        if priority:
            p_nodes = getattr(priority, "visual_hierarchy", None) or getattr(priority, "hierarchy_nodes", [])
            if p_nodes:
                total_canvas = sum(n.canvas_allocation_fraction for n in p_nodes)
                if total_canvas > 1.05:
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.IMPOSSIBLE_COMBINATION,
                            severity=ValidationSeverity.BLOCKING,
                            reason=f"Visual hierarchy canvas allocations sum to {total_canvas:.2f}, exceeding max canvas limit (1.0).",
                            affected_module="priority",
                            suggested_resolution="Normalize canvas allocation fractions so their sum is <= 1.0.",
                        )
                    )

                total_weight = sum(n.attention_weight for n in p_nodes)
                if total_weight > 1.05:
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.IMPOSSIBLE_COMBINATION,
                            severity=ValidationSeverity.BLOCKING,
                            reason=f"Visual hierarchy attention weights sum to {total_weight:.2f}, exceeding max weight limit (1.0).",
                            affected_module="priority",
                            suggested_resolution="Normalize attention weights so their sum is <= 1.0.",
                        )
                    )

        # Strategy uplift out of bounds
        if strategy and strategy.winning_strategy:
            uplift = strategy.winning_strategy.expected_ctr_uplift
            if uplift < 0.0 or uplift > 1.0:
                issues.append(
                    ValidationIssue(
                        issue_type=ValidationIssueType.IMPOSSIBLE_COMBINATION,
                        severity=ValidationSeverity.BLOCKING,
                        reason=f"Strategy expected_ctr_uplift ({uplift}) is outside valid range [0.0, 1.0].",
                        affected_module="strategy_ranker",
                        suggested_resolution="Clamp CTR uplift scores to [0.0, 1.0].",
                    )
                )

        # Orphan visual focus candidates (produced by narrative but missing in priority/strategy)
        if narrative and priority:
            p_nodes = getattr(priority, "visual_hierarchy", None) or getattr(priority, "hierarchy_nodes", [])
            p_node_names = [n.element_name.lower() for n in p_nodes]
            for vf in narrative.visual_focus_candidates:
                vf_name = vf.element_name.lower()
                if not any(vf_name in p_n or p_n in vf_name for p_n in p_node_names):
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.ORPHAN_OUTPUT,
                            severity=ValidationSeverity.WARNING,
                            reason=f"Visual focus candidate '{vf.element_name}' from Narrative is not represented in Priority hierarchy nodes.",
                            affected_module="narrative_vs_priority",
                            suggested_resolution="Add visual focus candidate to Priority hierarchy or document intentional omission.",
                        )
                    )

        return issues

    # ---------------------------------------------------------------------------
    # Scoring Models
    # ---------------------------------------------------------------------------

    def _compute_consistency_score(
        self,
        issues: List[ValidationIssue],
        conflicts: List[DetectedConflict],
    ) -> float:
        score = 1.00

        penalties = {
            ValidationSeverity.BLOCKING: 0.35,
            ValidationSeverity.CRITICAL: 0.20,
            ValidationSeverity.WARNING: 0.10,
            ValidationSeverity.INFO: 0.02,
        }

        for issue in issues:
            score -= penalties.get(issue.severity, 0.10)

        # Extra penalty for explicit contradictions
        for conflict in conflicts:
            score -= 0.05

        return max(0.0, min(1.0, round(score, 4)))

    def _compute_readiness_score(
        self,
        consistency_score: float,
        context: ReasoningContext,
        narrative: Optional[NarrativeResult],
        audience: Optional[AudienceResult],
        creator: Optional[CreatorResult],
        brand: Optional[BrandResult],
        priority: Optional[PriorityResult],
        risk: Optional[RiskResult],
        strategy: Optional[StrategyDecision],
        blocking_errors: List[ValidationIssue],
    ) -> float:
        # Completeness score: fraction of 7 core modules present
        core_mods = [narrative, audience, creator, brand, priority, risk, strategy]
        present_count = sum(1 for m in core_mods if m is not None)
        completeness_score = present_count / 7.0

        # Grounding score: ratio of non-empty evidence
        grounding_score = 1.0 if context.get_evidence_count() > 0 else 0.0

        # Confidence score: mean of active reasoner confidences
        active_confs = [m.confidence for m in core_mods if m is not None and hasattr(m, "confidence")]
        confidence_score = sum(active_confs) / len(active_confs) if active_confs else 0.50

        raw_readiness = (
            0.50 * consistency_score
            + 0.25 * completeness_score
            + 0.15 * grounding_score
            + 0.10 * confidence_score
        )

        # Cap readiness score to 0.40 if blocking errors exist
        if len(blocking_errors) > 0:
            raw_readiness = min(0.40, raw_readiness)

        return max(0.0, min(1.0, round(raw_readiness, 4)))

    def _compute_validation_confidence(
        self,
        graph: Optional[NormalizedEvidenceGraph],
        context: ReasoningContext,
        trace_steps: List[ValidationTraceStep],
        issues: List[ValidationIssue],
    ) -> float:
        base_conf = 1.0

        if graph is None or not graph.nodes:
            base_conf -= 0.15
        if context.get_evidence_count() == 0:
            base_conf -= 0.25

        if any(step.status == "FAILED" for step in trace_steps):
            base_conf -= 0.10

        return max(0.0, min(1.0, round(base_conf, 4)))

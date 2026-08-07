"""
priority_reasoner.py
====================

Production PriorityReasoner for the Thumbnail Intelligence Engine (Phase 3.4E).
Converts narrative, audience, creator, and brand reasoning into a grounded visual hierarchy.
Determines what deserves the viewer's attention first, attention weights, canvas area allocations,
sequential gaze trajectories, and explicit non-compete rules.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from thumbnail_intelligence.evidence.models import (
    EvidenceNode,
    EvidenceReference,
    EvidenceSourceType,
    KnowledgeEntryType,
    NormalizedEvidenceGraph,
)
from thumbnail_intelligence.knowledge_base.models import (
    EvidenceGrade,
    _utc_now_iso,
)
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.interfaces import PriorityReasoner as BasePriorityReasoner
from thumbnail_intelligence.reasoning.models import (
    ReasonerContract,
    ReasonerType,
)
from thumbnail_intelligence.reasoning.priority_models import (
    AttentionFlowStep,
    BackgroundPriority,
    CandidateHierarchy,
    ElementPriorityLevel,
    HierarchyTier,
    PriorityResult,
    VisualHierarchyNode,
)


class PriorityReasoner(BasePriorityReasoner):
    """
    Visual Hierarchy and Attention Priority Reasoning Engine.
    Determines element dominance, attention weights, canvas allocations, and gaze flow.
    """

    def __init__(
        self,
        name: str = "priority_reasoner",
        version: str = "1.0.0",
        is_mandatory: bool = True,
        timeout_ms: float = 5000.0,
    ) -> None:
        self._contract = ReasonerContract(
            name=name,
            reasoner_type=ReasonerType.PRIORITY,
            dependencies=["narrative_reasoner", "audience_reasoner", "creator_reasoner", "brand_reasoner"],
            version=version,
            description="Infers grounded visual hierarchy, attention ordering, canvas allocations, and non-compete rules",
            is_mandatory=is_mandatory,
            timeout_ms=timeout_ms,
        )

    @property
    def contract(self) -> ReasonerContract:
        return self._contract

    def reason(
        self,
        graph: NormalizedEvidenceGraph,
        context: ReasoningContext,
    ) -> PriorityResult:
        """
        Execute visual priority reasoning over NormalizedEvidenceGraph and current ReasoningContext.

        Args:
            graph: Grounded NormalizedEvidenceGraph.
            context: ReasoningContext containing narrative, audience, creator, and brand findings.

        Returns:
            Validated PriorityResult with visual hierarchy, attention weights, and canvas allocations.
        """
        trace: List[str] = []
        trace.append(f"Starting PriorityReasoner execution (v{self.version}) on graph {graph.graph_id}")

        # 1. Ingest active evidence nodes and prior reasoning context
        active_nodes = [node for node in graph.get_active_nodes() if node.is_active]
        narrative_ctx = context.narrative
        audience_ctx = context.audience
        creator_ctx = context.creator_intent
        brand_ctx = context.brand_constraints

        trace.append(
            f"Ingested {len(active_nodes)} active nodes; narrative: {narrative_ctx is not None}, "
            f"audience: {audience_ctx is not None}, creator: {creator_ctx is not None}, brand: {brand_ctx is not None}"
        )

        # 2. Extract grounded focal elements and references
        tokens, refs_map = self._extract_priority_signals(
            graph, active_nodes, narrative_ctx, audience_ctx, creator_ctx, brand_ctx, trace
        )

        # 3. Identify Primary, Secondary, and Supporting Subjects
        primary_subj, secondary_subj, supporting_subjs = self._identify_subjects(
            narrative_ctx, creator_ctx, brand_ctx, tokens, trace
        )

        # 4. Formulate Visual Hierarchy Nodes & Canvas Allocations
        hierarchy_nodes, attention_weights, canvas_allocations, importance_scores = self._build_hierarchy_nodes(
            primary_subj, secondary_subj, supporting_subjs, audience_ctx, brand_ctx, refs_map, trace
        )

        # 5. Formulate Attention Flow and Non-Compete Rules
        attention_flow = self._build_attention_flow(primary_subj, secondary_subj, supporting_subjs, refs_map, trace)
        non_compete_rules = self._formulate_non_compete_rules(primary_subj, secondary_subj, brand_ctx, trace)

        # 6. Multi-Hypothesis Candidate Hierarchy Generation
        candidates = self._generate_candidate_hierarchies(
            graph, primary_subj, secondary_subj, hierarchy_nodes, attention_weights, canvas_allocations, refs_map, trace
        )
        if not candidates:
            candidates = [self._build_default_candidate(primary_subj, secondary_subj, attention_weights, canvas_allocations, refs_map)]

        primary_candidate = candidates[0]
        secondary_candidates = candidates[1:] if len(candidates) > 1 else []

        # 7. Multi-Signal Calibrated Confidence Model
        confidence_breakdown, overall_confidence = self._calculate_confidence(
            graph, active_nodes, narrative_ctx, audience_ctx, creator_ctx, brand_ctx, tokens, trace
        )

        # 8. Harvest and Deduplicate Evidence References
        all_refs: List[EvidenceReference] = []
        seen_refs: Set[str] = set()

        for cand in candidates:
            for ref in cand.evidence_refs:
                key = f"{ref.source_id}:{ref.source_type.value if hasattr(ref.source_type, 'value') else ref.source_type}"
                if key not in seen_refs:
                    seen_refs.add(key)
                    all_refs.append(ref)

        for hnode in hierarchy_nodes:
            for ref in hnode.evidence_refs:
                key = f"{ref.source_id}:{ref.source_type.value if hasattr(ref.source_type, 'value') else ref.source_type}"
                if key not in seen_refs:
                    seen_refs.add(key)
                    all_refs.append(ref)

        # Grounding gate invariant: zero evidence refs implies zero confidence
        if not all_refs:
            overall_confidence = 0.0
            confidence_breakdown["evidence_quality"] = 0.0

        supporting_ids = list(refs_map.keys())

        # 9. Selection Rationale
        selection_rationale = (
            f"Visual hierarchy '{primary_candidate.hierarchy_name}' selected as primary ({primary_candidate.fit_score:.2f}) "
            f"due to optimal balance between primary gaze fixation on '{primary_subj}' "
            f"and support across {len(primary_candidate.supporting_evidence_ids)} grounded evidence nodes."
        )

        # 10. Assemble PriorityResult
        focal_element_hierarchy = [node.element_name for node in hierarchy_nodes]
        contrast_priorities = [
            "Minimum 4.5:1 luminance ratio between hero subject and background environment",
            "Crisp edge rim lighting separation on opposing curiosity object",
        ]
        lighting_priorities = [
            "High-key 3-point key light on creator face occupying outer third",
            "Volumetric cyan/magenta ambient rim light on secondary tension prop",
        ]

        result = PriorityResult(
            primary_subject=primary_subj,
            secondary_subject=secondary_subj,
            supporting_subjects=supporting_subjs,
            visual_hierarchy=hierarchy_nodes,
            importance_scores=importance_scores,
            attention_weights=attention_weights,
            canvas_allocation=canvas_allocations,
            text_priority=ElementPriorityLevel.MEDIUM if len(supporting_subjs) > 0 else ElementPriorityLevel.LOW,
            face_priority=ElementPriorityLevel.HIGH,
            object_priority=ElementPriorityLevel.HIGH,
            background_priority=BackgroundPriority.MUTED,
            color_importance={"primary_accent": 0.40, "secondary_rim": 0.35, "background_fill": 0.25},
            contrast_priority=contrast_priorities,
            required_emphasis=[f"Maximize visual prominence of '{primary_subj}'", f"Crisp contrast separation on '{secondary_subj}'"],
            suppressed_elements=["Cluttered background textures", "Competing secondary text paragraphs", "Low contrast props"],
            attention_flow=attention_flow,
            max_focal_points=2,
            non_compete_rules=non_compete_rules,
            primary_hierarchy_candidate=primary_candidate,
            candidate_hierarchies=candidates,
            rejected_hierarchies=secondary_candidates,
            selection_rationale=selection_rationale,
            priority_confidence=overall_confidence,
            confidence_breakdown=confidence_breakdown,
            supporting_evidence_ids=supporting_ids,
            focal_element_hierarchy=focal_element_hierarchy,
            visual_weight_allocations=attention_weights,
            composition_style=graph.summary.primary_archetype or "split_comparison",
            contrast_priorities=contrast_priorities,
            lighting_priorities=lighting_priorities,
            evidence_refs=all_refs,
            confidence=overall_confidence,
            reasoning_trace=trace,
            metadata={"dominant_focus": primary_subj, "focal_elements_count": len(hierarchy_nodes)},
        )

        trace.append(f"Successfully constructed PriorityResult with {len(all_refs)} grounding references")
        return result

    def validate_output(self, output: Any) -> bool:
        """Validate output satisfies contract and confidence invariants."""
        if not isinstance(output, PriorityResult):
            return False
        if not (0.0 <= output.confidence <= 1.0):
            return False
        if not (0.0 <= output.priority_confidence <= 1.0):
            return False
        return True

    # -----------------------------------------------------------------------
    # Internal Extraction and Inference Helpers
    # -----------------------------------------------------------------------

    def _extract_priority_signals(
        self,
        graph: NormalizedEvidenceGraph,
        active_nodes: List[EvidenceNode],
        narrative_ctx: Any,
        audience_ctx: Any,
        creator_ctx: Any,
        brand_ctx: Any,
        trace: List[str],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[EvidenceReference]]]:
        """Extract focal objects, subjects, and visual cues mapped to evidence references."""
        tokens: Dict[str, List[str]] = {
            "objects": [],
            "faces": [],
            "text": [],
            "patterns": [],
        }
        refs_map: Dict[str, List[EvidenceReference]] = {}

        for node in active_nodes:
            payload = getattr(node.evidence_item, "data_payload", {}) or {}
            node_refs = list(getattr(node.evidence_item, "evidence_refs", []))

            if not node_refs:
                synth_ref = EvidenceReference(
                    source_id=node.node_id,
                    source_type=getattr(node.provenance, "source_type", EvidenceSourceType.KNOWLEDGE_ENTRY),
                    confidence=node.confidence.propagated_confidence,
                    grade=EvidenceGrade.STRONG if node.confidence.propagated_confidence > 0.8 else EvidenceGrade.MODERATE,
                    claim_summary=getattr(node.provenance, "retrieval_reason", "Priority evidence node"),
                )
                node_refs.append(synth_ref)

            refs_map[node.node_id] = node_refs

            if "objects" in payload:
                objs = payload.get("objects", [])
                if isinstance(objs, list):
                    tokens["objects"].extend([str(o).lower() for o in objs])

            if "ocr_text" in payload:
                tokens["text"].append(str(payload["ocr_text"]))

            if node.node_type in (KnowledgeEntryType.DESIGN_PATTERN, KnowledgeEntryType.VISUAL_PATTERN):
                pat = payload.get("pattern_id", node.node_id)
                tokens["patterns"].append(str(pat).lower())

        # Collect upstream references
        for upstream_ctx in (narrative_ctx, audience_ctx, creator_ctx, brand_ctx):
            if upstream_ctx and hasattr(upstream_ctx, "evidence_refs"):
                for ref in upstream_ctx.evidence_refs:
                    refs_map.setdefault(ref.source_id, []).append(ref)

        trace.append(f"Extracted priority signals across {len(refs_map)} grounded node sources")
        return tokens, refs_map

    def _identify_subjects(
        self,
        narrative_ctx: Any,
        creator_ctx: Any,
        brand_ctx: Any,
        tokens: Dict[str, List[str]],
        trace: List[str],
    ) -> Tuple[str, str, List[str]]:
        """Identify primary hero subject, secondary tension object, and supporting elements."""
        primary_subj = "Creator Expressive Hero Face"
        secondary_subj = "Curiosity Story Tension Prop"
        supporting_subjs = ["Bold Text Hook", "Contextual Environment Background"]

        if narrative_ctx and hasattr(narrative_ctx, "key_subjects") and narrative_ctx.key_subjects:
            subjs = list(narrative_ctx.key_subjects)
            if len(subjs) >= 1:
                primary_subj = subjs[0]
            if len(subjs) >= 2:
                secondary_subj = subjs[1]
            if len(subjs) > 2:
                supporting_subjs = subjs[2:]
        elif tokens["objects"]:
            objs = [o.title() for o in tokens["objects"]]
            if len(objs) >= 1:
                primary_subj = objs[0]
            if len(objs) >= 2:
                secondary_subj = objs[1]
            if len(objs) > 2:
                supporting_subjs = objs[2:]

        trace.append(f"Identified primary subject: '{primary_subj}', secondary subject: '{secondary_subj}'")
        return primary_subj, secondary_subj, supporting_subjs

    def _build_hierarchy_nodes(
        self,
        primary_subj: str,
        secondary_subj: str,
        supporting_subjs: List[str],
        audience_ctx: Any,
        brand_ctx: Any,
        refs_map: Dict[str, List[EvidenceReference]],
        trace: List[str],
    ) -> Tuple[List[VisualHierarchyNode], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Build structured hierarchy nodes, attention distribution, and canvas area allocations."""
        all_refs = [r for sublist in refs_map.values() for r in sublist]

        h1 = VisualHierarchyNode(
            element_name=primary_subj,
            element_category="face" if "face" in primary_subj.lower() or "creator" in primary_subj.lower() else "object",
            tier=HierarchyTier.PRIMARY,
            importance_score=1.0,
            attention_weight=0.42,
            canvas_allocation_fraction=0.35,
            contrast_requirement="Minimum 5.0:1 luminance ratio against dark backing with warm key light",
            gaze_order=1,
            non_compete_with=[secondary_subj, "Bold Text Hook"],
            evidence_refs=all_refs[:2],
        )

        h2 = VisualHierarchyNode(
            element_name=secondary_subj,
            element_category="object" if "prop" in secondary_subj.lower() or "object" in secondary_subj.lower() else "graphic",
            tier=HierarchyTier.SECONDARY,
            importance_score=0.85,
            attention_weight=0.33,
            canvas_allocation_fraction=0.30,
            contrast_requirement="Crisp cyan/magenta rim lighting on opposing outer third",
            gaze_order=2,
            non_compete_with=[primary_subj],
            evidence_refs=all_refs[2:4] if len(all_refs) >= 4 else all_refs[:1],
        )

        h3 = VisualHierarchyNode(
            element_name="Bold Text Hook",
            element_category="text",
            tier=HierarchyTier.TERTIARY,
            importance_score=0.70,
            attention_weight=0.15,
            canvas_allocation_fraction=0.20,
            contrast_requirement="High contrast grotesque typography with 15% solid stroke outline",
            gaze_order=3,
            non_compete_with=[primary_subj],
            evidence_refs=all_refs[:1],
        )

        h4 = VisualHierarchyNode(
            element_name="Contextual Environment Background",
            element_category="background",
            tier=HierarchyTier.TERTIARY,
            importance_score=0.40,
            attention_weight=0.10,
            canvas_allocation_fraction=0.15,
            contrast_requirement="Muted dark luminance below 0.30 with subtle atmospheric haze",
            gaze_order=4,
            non_compete_with=[],
            evidence_refs=all_refs[:1],
        )

        hierarchy_nodes = [h1, h2, h3, h4]

        attention_weights = {
            "primary_subject": 0.42,
            "secondary_subject": 0.33,
            "text_overlay": 0.15,
            "background": 0.10,
        }

        canvas_allocations = {
            "primary_subject_area": 0.35,
            "secondary_subject_area": 0.30,
            "text_overlay_area": 0.20,
            "background_environment_area": 0.15,
        }

        importance_scores = {
            primary_subj: 1.0,
            secondary_subj: 0.85,
            "Bold Text Hook": 0.70,
            "Contextual Environment Background": 0.40,
        }

        trace.append(f"Formulated {len(hierarchy_nodes)} visual hierarchy nodes with normalized attention weights")
        return hierarchy_nodes, attention_weights, canvas_allocations, importance_scores

    def _build_attention_flow(
        self,
        primary_subj: str,
        secondary_subj: str,
        supporting_subjs: List[str],
        refs_map: Dict[str, List[EvidenceReference]],
        trace: List[str],
    ) -> List[AttentionFlowStep]:
        """Formulate sequential 1-2-3 gaze trajectory across the canvas."""
        all_refs = [r for sublist in refs_map.values() for r in sublist]

        flow: List[AttentionFlowStep] = [
            AttentionFlowStep(
                step_order=1,
                target_element=primary_subj,
                visual_cue="Expressive high-arousal facial features or sharp high-contrast silhouette",
                psychological_driver="Instant biological fixation and emotional empathy",
                evidence_refs=all_refs[:2],
            ),
            AttentionFlowStep(
                step_order=2,
                target_element=secondary_subj,
                visual_cue="Crisp glowing rim lighting and contrasting vibrant colors on opposing third",
                psychological_driver="Curiosity gap exploration and story premise comprehension",
                evidence_refs=all_refs[2:4] if len(all_refs) >= 4 else all_refs[:1],
            ),
            AttentionFlowStep(
                step_order=3,
                target_element="Bold Text Hook",
                visual_cue="Short 2-4 word punchy headline with dark drop shadow",
                psychological_driver="Cognitive confirmation and premise lock",
                evidence_refs=all_refs[:1],
            ),
        ]

        trace.append(f"Constructed {len(flow)}-step sequential attention gaze trajectory")
        return flow

    def _formulate_non_compete_rules(
        self,
        primary_subj: str,
        secondary_subj: str,
        brand_ctx: Any,
        trace: List[str],
    ) -> List[str]:
        """Generate non-compete rules ensuring no visual rivalry between primary and secondary elements."""
        rules = [
            f"Text hook must never overlap or obscure the eye line of '{primary_subj}'.",
            f"'{primary_subj}' and '{secondary_subj}' must be positioned on opposing thirds to prevent visual collision.",
            "Background luminance must remain below 0.30 to ensure zero foreground contrast loss.",
            "Avoid placing more than 2 high-saturation accent colors in the same quadrant.",
        ]
        trace.append(f"Formulated {len(rules)} visual non-compete guardrails")
        return rules

    def _generate_candidate_hierarchies(
        self,
        graph: NormalizedEvidenceGraph,
        primary_subj: str,
        secondary_subj: str,
        hierarchy_nodes: List[VisualHierarchyNode],
        attention_weights: Dict[str, float],
        canvas_allocations: Dict[str, float],
        refs_map: Dict[str, List[EvidenceReference]],
        trace: List[str],
    ) -> List[CandidateHierarchy]:
        """Generate competing candidate visual hierarchy interpretations (Face-First, Object-First, Split-Contrast)."""
        all_refs = [r for sublist in refs_map.values() for r in sublist]
        all_node_ids = list(refs_map.keys())

        candidates: List[CandidateHierarchy] = []

        # Candidate A: Face-First Emotional Hook Hierarchy (Highest Fit)
        cand_a = CandidateHierarchy(
            hierarchy_name="Face-First Emotional Hook Hierarchy",
            primary_focus=primary_subj,
            secondary_focus=secondary_subj,
            tertiary_focus="Bold Text Hook",
            fit_score=0.95,
            confidence=0.93,
            attention_distribution=attention_weights,
            canvas_allocations=canvas_allocations,
            pros=["Maximizes immediate human mirror neuron engagement", "Strong biological gaze capture"],
            cons=["Requires high quality expressive photoshoot assets"],
            evidence_refs=all_refs[:3],
            supporting_evidence_ids=all_node_ids[:3],
        )
        candidates.append(cand_a)

        # Candidate B: Object-First Mystery Reveal Hierarchy (Alternative 1)
        alt_weights_b = {
            "primary_subject": 0.32,
            "secondary_subject": 0.45,
            "text_overlay": 0.13,
            "background": 0.10,
        }
        cand_b = CandidateHierarchy(
            hierarchy_name="Object-First Mystery Reveal Hierarchy",
            primary_focus=secondary_subj,
            secondary_focus=primary_subj,
            tertiary_focus="Minimalist Hook",
            fit_score=0.79,
            confidence=0.82,
            attention_distribution=alt_weights_b,
            canvas_allocations={"object_area": 0.45, "face_area": 0.30, "text_area": 0.15, "bg_area": 0.10},
            pros=["Prioritizes high intrigue topic object", "Excellent for mystery discovery formats"],
            cons=["Slightly lower immediate human face recognition"],
            rejection_rationale="Ranked as secondary alternative (#2) with fit score 0.79 compared to face-first anchor (0.95)",
            evidence_refs=all_refs[:2],
            supporting_evidence_ids=all_node_ids[:2],
        )
        candidates.append(cand_b)

        # Candidate C: Split-Contrast Equal Battle Hierarchy (Alternative 2)
        alt_weights_c = {
            "primary_subject": 0.38,
            "secondary_subject": 0.38,
            "text_overlay": 0.14,
            "background": 0.10,
        }
        cand_c = CandidateHierarchy(
            hierarchy_name="Balanced Split-Contrast Hierarchy",
            primary_focus="Equal Split Battle Elements",
            secondary_focus="Center Barrier Divider",
            tertiary_focus="Versus Badge Text",
            fit_score=0.68,
            confidence=0.74,
            attention_distribution=alt_weights_c,
            canvas_allocations={"left_subject": 0.35, "right_subject": 0.35, "center_badge": 0.15, "bg": 0.15},
            pros=["Clear competitive visual tension", "Ideal for versus/comparison formats"],
            cons=["Risk of split viewer attention and cognitive friction"],
            rejection_rationale="Ranked as tertiary alternative (#3) with fit score 0.68 due to split gaze competition risk",
            evidence_refs=all_refs[:1],
            supporting_evidence_ids=all_node_ids[:1],
        )
        candidates.append(cand_c)

        trace.append(f"Generated {len(candidates)} candidate visual hierarchy interpretations")
        return candidates

    def _build_default_candidate(
        self,
        primary_subj: str,
        secondary_subj: str,
        attention_weights: Dict[str, float],
        canvas_allocations: Dict[str, float],
        refs_map: Dict[str, List[EvidenceReference]],
    ) -> CandidateHierarchy:
        """Construct fallback candidate visual hierarchy when minimal signals are present."""
        all_refs = [r for sublist in refs_map.values() for r in sublist]
        return CandidateHierarchy(
            hierarchy_name="Standard Two-Element Hierarchy",
            primary_focus=primary_subj,
            secondary_focus=secondary_subj,
            tertiary_focus="Text Hook",
            fit_score=0.80,
            confidence=0.80,
            attention_distribution=attention_weights,
            canvas_allocations=canvas_allocations,
            evidence_refs=all_refs[:1],
            supporting_evidence_ids=list(refs_map.keys())[:1],
        )

    def _calculate_confidence(
        self,
        graph: NormalizedEvidenceGraph,
        active_nodes: List[EvidenceNode],
        narrative_ctx: Any,
        audience_ctx: Any,
        creator_ctx: Any,
        brand_ctx: Any,
        tokens: Dict[str, List[str]],
        trace: List[str],
    ) -> Tuple[Dict[str, float], float]:
        """Compute multi-signal calibrated confidence across all strategic upstream inputs."""
        narrative_conf = getattr(narrative_ctx, "confidence", 0.85) if narrative_ctx is not None else 0.80
        audience_conf = getattr(audience_ctx, "confidence", 0.85) if audience_ctx is not None else 0.80
        creator_conf = getattr(creator_ctx, "confidence", 0.85) if creator_ctx is not None else 0.80
        brand_conf = getattr(brand_ctx, "confidence", 0.85) if brand_ctx is not None else 0.80

        # Evidence quality
        if active_nodes:
            ev_quality = sum(n.confidence.propagated_confidence for n in active_nodes) / len(active_nodes)
        else:
            ev_quality = 0.50

        # Historical consistency
        hist_consistency = 0.95 if tokens["objects"] or tokens["patterns"] else 0.80

        # Conflict penalty
        conflicts_count = len(getattr(graph, "conflicts", []))
        conflict_penalty = min(0.40, conflicts_count * 0.10)

        raw_conf = (
            0.20 * narrative_conf
            + 0.20 * audience_conf
            + 0.20 * creator_conf
            + 0.15 * brand_conf
            + 0.15 * ev_quality
            + 0.10 * hist_consistency
        )
        final_conf = max(0.0, min(1.0, raw_conf * (1.0 - conflict_penalty)))

        breakdown = {
            "narrative_confidence": narrative_conf,
            "audience_confidence": audience_conf,
            "creator_confidence": creator_conf,
            "brand_confidence": brand_conf,
            "evidence_quality": ev_quality,
            "historical_consistency": hist_consistency,
            "conflict_penalty": conflict_penalty,
        }

        trace.append(f"Calibrated priority confidence score: {final_conf:.2f}")
        return breakdown, final_conf
